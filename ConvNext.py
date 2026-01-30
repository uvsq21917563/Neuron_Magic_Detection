import os
import pandas as pd
import torch
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch import nn, optim
from torch.optim import lr_scheduler
from PIL import Image

# --------------------------
# 1. Transformations
# --------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --------------------------
# 2. Dataset personnalisé
# --------------------------
class MagicCardDataset(Dataset):
    def __init__(self, csv_file, base_img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.base_img_dir = base_img_dir # C'est le dossier "mtg_dataset"
        self.transform = transform
        self.types = {'Creature': 0, 'Instant': 1, 'Land': 2, 'Artifact': 3, 'Sorcery': 4, 'Enchantment': 5, 'Planeswalker': 6}
        self.rarities = {'common': 0, 'uncommon': 1, 'rare': 2, 'mythic': 3, 'special': 4}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            # Le CSV contient déjà "train/id.jpg" ou "test/id.jpg"
            relative_path = self.data.iloc[idx, 0]
            img_path = os.path.join(self.base_img_dir, relative_path)
            
            if not os.path.exists(img_path):
                return None

            image = Image.open(img_path).convert('RGB')
            label_type = self.data.iloc[idx, 1]
            label_rarity = self.data.iloc[idx, 2]
            label_colors = self.data.iloc[idx, 3:].values.astype('float32')

            if self.transform:
                image = self.transform(image)

            label_type = torch.tensor(self.types[label_type], dtype=torch.long)
            label_rarity = torch.tensor(self.rarities[label_rarity], dtype=torch.long)
            label_colors = torch.tensor(label_colors, dtype=torch.float32)

            return image, (label_type, label_rarity, label_colors)
        except Exception as e:
            return None

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0: return None
    images, labels = zip(*batch)
    return torch.stack(images), (torch.stack([l[0] for l in labels]), 
                                 torch.stack([l[1] for l in labels]), 
                                 torch.stack([l[2] for l in labels]))

# --------------------------
# 3. Chemins et Chargement
# --------------------------
base_dir = os.path.dirname(__file__)
mtg_dir = os.path.join(base_dir, "mtg_dataset")

# On charge les deux séparément
train_csv = os.path.join(mtg_dir, "Train.csv")
test_csv = os.path.join(mtg_dir, "Test.csv")
val_csv = os.path.join(mtg_dir, "Val.csv")

train_dataset = MagicCardDataset(csv_file=train_csv, base_img_dir=mtg_dir, transform=transform)
test_dataset = MagicCardDataset(csv_file=test_csv, base_img_dir=mtg_dir, transform=transform)
val_dataset = MagicCardDataset(csv_file=val_csv, base_img_dir=mtg_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

# --------------------------
# 4. Modèle
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pour éviter le bug du None au début, on définit les tailles manuellement 
# ou on boucle jusqu'à trouver un item valide
num_colors = 6 # White, Blue, Black, Red, Green, Colorless
num_types = 7
num_rarities = 5

class MultiOutputModel(nn.Module):
    def __init__(self, num_types, num_rarities, num_colors):
        super().__init__()
        # Chargement correct du backbone (CONVNEXT)
        self.backbone = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        )

        # ConvNeXt: classifier = Sequential(LN, Flatten, Linear)
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Identity()

        self.fc_type = nn.Linear(in_features, num_types)
        self.fc_rarity = nn.Linear(in_features, num_rarities)
        self.fc_colors = nn.Linear(in_features, num_colors)

    def forward(self, x):
        features = self.backbone(x)
        return self.fc_type(features), self.fc_rarity(features), torch.sigmoid(self.fc_colors(features))

model = MultiOutputModel(num_types, num_rarities, num_colors).to(device)

# --------------------------
# 5. Entraînement
# --------------------------
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)
criterion_ce = nn.CrossEntropyLoss()
criterion_bce = nn.BCELoss()
best_val_loss = float('inf')

best_model_path = os.path.join("saved_models", "best_model_convnext.pth")
loss_save_path = os.path.join("saved_losses", "convnext_history.json")
history = {"train_loss": [], "val_loss": []}

for epoch in range(30):
    model.train()
    train_loss = 0.0
    num_batches = 0
    
    for batch in train_loader:
        if batch is None: continue
        images, (types, rarities, colors) = batch
        images, types, rarities, colors = images.to(device), types.to(device), rarities.to(device), colors.to(device)

        optimizer.zero_grad()
        out_type, out_rarity, out_colors = model(images)
        
        loss = criterion_ce(out_type, types) + criterion_ce(out_rarity, rarities) + criterion_bce(out_colors, colors)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        num_batches += 1
    
    avg_loss = train_loss / num_batches if num_batches > 0 else 0
    
    # Étape de validation
    model.eval()
    val_loss = 0.0
    val_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            if batch is None: continue
            images, (types, rarities, colors) = batch
            images, types, rarities, colors = images.to(device), types.to(device), rarities.to(device), colors.to(device)
            
            out_type, out_rarity, out_colors = model(images)
            loss = criterion_ce(out_type, types) + criterion_ce(out_rarity, rarities) + criterion_bce(out_colors, colors)
            val_loss += loss.item()
            val_batches += 1
    
    avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
    
    history["train_loss"].append(avg_loss)
    history["val_loss"].append(avg_val_loss)
    
    with open(loss_save_path, 'w') as f:
        json.dump(history, f)

    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Sauvegarder le meilleur modèle
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e} ---New High Score !!!---")
    else:
        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e}")

# --------------------------
# 6. Test
# --------------------------
print("\n--- ÉVALUATION SUR LE DATASET DE TEST ---")
model.eval()

inv_types = {v: k for k, v in train_dataset.types.items()}
inv_rarities = {v: k for k, v in train_dataset.rarities.items()}
color_names = ['Blanc', 'Bleu', 'Noir', 'Rouge', 'Vert', 'Incolore']

type_ok = 0
rarity_ok = 0
color_ok = 0
total = 0

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        if batch is None: continue
        
        images, (types, rarities, colors) = batch
        images = images.to(device)
        types = types.to(device)
        rarities = rarities.to(device)
        colors = colors.to(device)
        
        out_type, out_rarity, out_colors = model(images)
        
        pred_types = torch.argmax(out_type, dim=1)
        pred_rarities = torch.argmax(out_rarity, dim=1)
        pred_colors_batch = (out_colors > 0.3).int().cpu().numpy()
        
        true_types = types.cpu().numpy()
        true_rarities = rarities.cpu().numpy()
        true_colors_batch = colors.cpu().numpy()

        batch_size = len(images)
        for j in range(batch_size):
            pred_type_idx = pred_types[j].item()
            pred_rarity_idx = pred_rarities[j].item()
            pred_colors = pred_colors_batch[j]
            
            true_type_idx = true_types[j]
            true_rarity_idx = true_rarities[j]
            true_colors = true_colors_batch[j]

            total += 1
            type_ok += int(pred_type_idx == true_type_idx)
            rarity_ok += int(pred_rarity_idx == true_rarity_idx)
            color_ok += int((pred_colors == true_colors).all())

print(f"Précision sur le type    : {100 * type_ok / total:.2f}%")
print(f"Précision sur la rareté : {100 * rarity_ok / total:.2f}%")
print(f"Précision sur la couleur: {100 * color_ok / total:.2f}%")
