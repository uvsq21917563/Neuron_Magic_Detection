import os
import pandas as pd
import torch
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch import nn, optim
from PIL import Image

# --------------------------
# CONFIGURATION
# --------------------------
# Hyperparamètres pour ConvNeXt
BATCH_SIZE = 32
LEARNING_RATE = 4e-5  # Un peu plus bas pour ConvNeXt finetuning
WEIGHT_DECAY = 1e-2   # Important pour AdamW
EPOCHS = 2           # ConvNeXt apprend vite, mais a besoin d'un peu de temps

# --------------------------
# 1. Transformations (Data Augmentation)
# --------------------------
# Pour l'entraînement : on bouge un peu l'image pour forcer le modèle à être robuste
train_transform = transforms.Compose([
    transforms.Resize((232, 232)),           # ConvNeXt aime être un peu plus grand que 224 avant crop
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),  # Miroir
    transforms.RandomRotation(degrees=15),   # Rotation légère
    transforms.ColorJitter(brightness=0.1, contrast=0.1), # Variation légère de lumière
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Pour le test/validation : on ne touche à rien, juste resize standard
val_transform = transforms.Compose([
    transforms.Resize((232, 232)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --------------------------
# 2. Dataset Robuste
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
        # Récupération sécurisée de la ligne
        row = self.data.iloc[idx]
        
        # Chemin de l'image (colonne 'image_path')
        relative_path = row['image_path']
        img_path = os.path.join(self.base_img_dir, relative_path)
        
        if not os.path.exists(img_path):
            # On retourne None pour que collate_fn l'ignore sans crasher
            return None

        try:
            image = Image.open(img_path).convert('RGB')
            
            # --- RECUPERATION DES LABELS PAR NOM DE COLONNE ---
            # C'est ici que ça plantait avant. Maintenant c'est sécurisé.
            l_type = row['type']
            l_rarity = row['rarity']
            
            # On construit la liste des couleurs manuellement
            # pour éviter de lire la colonne 'name' par erreur
            colors = [
                row['is_white'], row['is_blue'], row['is_black'], 
                row['is_red'], row['is_green'], row['is_colorless']
            ]

            if self.transform:
                image = self.transform(image)

            # Conversion en Tensors
            label_type = torch.tensor(self.types.get(l_type, 0), dtype=torch.long) # .get() évite le crash si type inconnu
            label_rarity = torch.tensor(self.rarities.get(l_rarity, 0), dtype=torch.long)
            label_colors = torch.tensor(colors, dtype=torch.float32)

            return image, (label_type, label_rarity, label_colors)
            
        except Exception as e:
            print(f"[Warning] Erreur lecture image {relative_path}: {e}")
            return None

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0: return None
    images, labels = zip(*batch)
    return torch.stack(images), (torch.stack([l[0] for l in labels]), 
                                 torch.stack([l[1] for l in labels]), 
                                 torch.stack([l[2] for l in labels]))

# --------------------------
# 3. Chemins
# --------------------------
# Utilisation de chemin absolu pour la fiabilité
base_dir = os.path.dirname(os.path.abspath(__file__))
mtg_dir = os.path.join(base_dir, "mtg_dataset")

train_csv = os.path.join(mtg_dir, "Train.csv")
test_csv = os.path.join(mtg_dir, "Test.csv")
val_csv = os.path.join(mtg_dir, "Val.csv")

if not os.path.exists(train_csv):
    raise FileNotFoundError(f"Impossible de trouver : {train_csv}. Lancez d'abord Recup_img_cartes.py !")

# Chargement des datasets
# Note: On utilise train_transform pour l'entrainement et val_transform pour les autres
train_dataset = MagicCardDataset(csv_file=train_csv, base_img_dir=mtg_dir, transform=train_transform)
val_dataset = MagicCardDataset(csv_file=val_csv, base_img_dir=mtg_dir, transform=val_transform)
test_dataset = MagicCardDataset(csv_file=test_csv, base_img_dir=mtg_dir, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# --------------------------
# 4. Modèle ConvNeXt Multi-Têtes
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation du périphérique : {device}")

class MultiOutputConvNext(nn.Module):
    def __init__(self, num_types=7, num_rarities=5, num_colors=6):
        super().__init__()
        # Chargement du backbone
        self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        
        # ConvNeXt Classifier structure :
        # [0] LayerNorm2d
        # [1] Flatten
        # [2] Linear (C'est celle-ci qu'on veut enlever)
        
        in_features = self.backbone.classifier[2].in_features
        
        # CORRECTION : On ne remplace QUE la couche linéaire finale.
        # On garde [0] et [1] pour que les dimensions soient correctes (B, 768).
        self.backbone.classifier[2] = nn.Identity()

        # Création des 3 têtes
        self.fc_type = nn.Linear(in_features, num_types)
        self.fc_rarity = nn.Linear(in_features, num_rarities)
        self.fc_colors = nn.Linear(in_features, num_colors)

    def forward(self, x):
        features = self.backbone(x)
        return self.fc_type(features), self.fc_rarity(features), torch.sigmoid(self.fc_colors(features))

model = MultiOutputConvNext().to(device)

# --------------------------
# 5. Entraînement
# --------------------------
# AdamW est recommandé pour ConvNeXt
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

criterion_ce = nn.CrossEntropyLoss()
criterion_bce = nn.BCELoss()

os.makedirs("saved_models", exist_ok=True)
best_model_path = os.path.join("saved_models", "best_convnext.pth")
best_val_loss = float('inf')

print(f"Démarrage de l'entraînement pour {EPOCHS} époques...")

for epoch in range(EPOCHS):
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
    
    avg_train_loss = train_loss / num_batches if num_batches > 0 else 0
    
    # Validation
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
    
    # Gestion Learning Rate
    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Sauvegarde
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Epoch {epoch+1:02d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {current_lr:.1e} [SAVED] 🏆")
    else:
        print(f"Epoch {epoch+1:02d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {current_lr:.1e}")

# --------------------------
# 6. Test Final
# --------------------------
print("\n--- ÉVALUATION FINALE SUR LE TEST SET ---")
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path))
    print("Meilleur modèle rechargé.")

model.eval()
inv_types = {v: k for k, v in train_dataset.types.items()}
inv_rarities = {v: k for k, v in train_dataset.rarities.items()}
color_names = ['W', 'U', 'B', 'R', 'G', 'C']

type_ok = 0
rarity_ok = 0
color_ok = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        if batch is None: continue
        images, (types, rarities, colors) = batch
        images = images.to(device)
        
        # Prédictions
        out_type, out_rarity, out_colors = model(images)
        pred_types = torch.argmax(out_type, dim=1).cpu().numpy()
        pred_rarities = torch.argmax(out_rarity, dim=1).cpu().numpy()
        pred_colors = (out_colors > 0.5).int().cpu().numpy()
        
        # Vérité terrain
        true_types = types.cpu().numpy()
        true_rarities = rarities.cpu().numpy()
        true_colors = colors.cpu().numpy()
        
        for i in range(len(images)):
            total += 1
            type_ok += 1 if pred_types[i] == true_types[i] else 0
            rarity_ok += 1 if pred_rarities[i] == true_rarities[i] else 0
            # Pour les couleurs, on exige une correspondance exacte (toutes les couleurs justes)
            color_ok += 1 if (pred_colors[i] == true_colors[i]).all() else 0

if total > 0:
    print(f"Précision Type    : {100 * type_ok / total:.2f}%")
    print(f"Précision Rareté  : {100 * rarity_ok / total:.2f}%")
    print(f"Précision Couleur : {100 * color_ok / total:.2f}%")
else:
    print("Aucune image de test n'a pu être chargée.")