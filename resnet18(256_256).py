import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch import nn, optim
from PIL import Image

# --------------------------
# 1. Transformations
# --------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),              # Un peu plus grand pour voir les symboles
    transforms.RandomHorizontalFlip(p=0.5),     # Augmentation (miroir)
    transforms.RandomRotation(degrees=10),      # Augmentation (légère rotation)
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
        self.base_img_dir = base_img_dir
        self.transform = transform
        self.types = {'Creature': 0, 'Instant': 1, 'Land': 2, 'Artifact': 3, 'Sorcery': 4, 'Enchantment': 5, 'Planeswalker': 6}
        self.rarities = {'common': 0, 'uncommon': 1, 'rare': 2, 'mythic': 3, 'special': 4}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
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

train_csv = os.path.join(mtg_dir, "Train.csv")
test_csv = os.path.join(mtg_dir, "Test.csv")

train_dataset = MagicCardDataset(csv_file=train_csv, base_img_dir=mtg_dir, transform=transform)
test_dataset = MagicCardDataset(csv_file=test_csv, base_img_dir=mtg_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# --------------------------
# 4. Modèle (Correction ResNet)
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_colors = 6
num_types = 7
num_rarities = 5

class MultiOutputModel(nn.Module):
    def __init__(self, num_types, num_rarities, num_colors):
        super().__init__()
        # Chargement de ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # --- CORRECTION ICI ---
        # 1. On récupère la taille de sortie de la couche 'fc' (512 pour ResNet18)
        in_features = self.backbone.fc.in_features
        
        # 2. On remplace la couche 'fc' (spécifique à ResNet) par Identity
        self.backbone.fc = nn.Identity()

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
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion_ce = nn.CrossEntropyLoss()
criterion_bce = nn.BCELoss()

print(f"Démarrage de l'entraînement sur {device}...")

for epoch in range(20):
    model.train()
    total_loss = 0
    for batch in train_loader:
        if batch is None: continue
        images, (types, rarities, colors) = batch
        images, types, rarities, colors = images.to(device), types.to(device), rarities.to(device), colors.to(device)

        optimizer.zero_grad()
        out_type, out_rarity, out_colors = model(images)
        
        loss = criterion_ce(out_type, types) + criterion_ce(out_rarity, rarities) + criterion_bce(out_colors, colors)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1} terminée. Loss moyenne: {total_loss / len(train_loader):.4f}")

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
        
        out_type, out_rarity, out_colors = model(images)
        
        pred_type_idx = torch.argmax(out_type, dim=1).item()
        pred_rarity_idx = torch.argmax(out_rarity, dim=1).item()
        pred_colors = (out_colors > 0.3).int().cpu().numpy()[0]
        
        true_type_idx = types.item()
        true_rarity_idx = rarities.item()
        true_colors = colors.cpu().numpy()[0]

        total += 1
        type_ok += int(pred_type_idx == true_type_idx)
        rarity_ok += int(pred_rarity_idx == true_rarity_idx)
        color_ok += int((pred_colors == true_colors).all())

        print(f"\n--- Carte n°{i+1} ---")
        p_t = inv_types[pred_type_idx]
        r_t = inv_types[true_type_idx]
        print(f"Type    : Prédit [{p_t:12}] | Réel [{r_t:12}] {'✅' if p_t == r_t else '❌'}")
        
        p_r = inv_rarities[pred_rarity_idx]
        r_r = inv_rarities[true_rarity_idx]
        print(f"Rareté  : Prédit [{p_r:12}] | Réel [{r_r:12}] {'✅' if p_r == r_r else '❌'}")
        
        p_c = [color_names[j] for j, val in enumerate(pred_colors) if val == 1]
        t_c = [color_names[j] for j, val in enumerate(true_colors) if val == 1]
        print(f"Couleurs: Prédit {p_c} | Réel   {t_c} {'✅' if p_c == t_c else '❌'}")

if total > 0:
    print(f"\nPrécision sur le type    : {100 * type_ok / total:.2f}%")
    print(f"Précision sur la rareté : {100 * rarity_ok / total:.2f}%")
    print(f"Précision sur la couleur: {100 * color_ok / total:.2f}%")
else:
    print("\nAucune donnée de test trouvée.")