import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# --- CONFIGURATION ---
DATA_DIR = "mtg_dataset"
TRAIN_CSV = os.path.join(DATA_DIR, "OCR_Train.csv")
VAL_CSV = os.path.join(DATA_DIR, "OCR_Val.csv")
MODEL_NAME = "microsoft/trocr-small-printed"
SAVE_PATH = os.path.join("sauvegarde_modeles", "ocr")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATASET PYTORCH ---
class MTGOCRDataset(Dataset):
    def __init__(self, csv_file, root_dir, processor, max_target_length=32):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['image_path'])
        image = Image.open(img_path).convert("RGB")
        
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        labels = self.processor.tokenizer(
            row['label'], 
            padding="max_length", 
            max_length=self.max_target_length
        ).input_ids
        
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        return {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels)
        }

# --- FONCTION DE TEST RAPIDE ---
def test_prediction(model, processor, dataset, num_samples=3):
    """Affiche quelques prédictions pour vérifier le comportement du modèle"""
    model.eval()
    print(f"\n--- TEST DE PRÉDICTION (Sur {num_samples} échantillons) ---")
    
    for i in range(num_samples):
        # Récupération d'un exemple
        item = dataset[i]
        pixel_values = item["pixel_values"].unsqueeze(0).to(DEVICE)
        
        # Récupération du vrai label (on ignore les -100)
        true_ids = [l for l in item["labels"].tolist() if l != -100]
        true_text = processor.tokenizer.decode(true_ids, skip_special_tokens=True)
        
        # Génération par le modèle
        with torch.no_grad():
            generated_ids = model.generate(
            pixel_values, 
            max_new_tokens=32, 
            min_length=5,      # Force le modèle à écrire au moins 5 caractères
            num_beams=5,       # Utilise une recherche par faisceau pour explorer plus de possibilités
            early_stopping=False
        )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"Vrai nom : {true_text}")
        print(f"Prédit   : {generated_text}")
        print("-" * 30)

# --- ÉTAPE D'ENTRAÎNEMENT ---
def train_ocr():
    print(f"Utilisation de : {DEVICE}")
    
    # 1. Chargement
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(DEVICE)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # 2. Datasets
    train_dataset = MTGOCRDataset(TRAIN_CSV, DATA_DIR, processor)
    val_dataset = MTGOCRDataset(VAL_CSV, DATA_DIR, processor)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # 3. Optimiseur
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # 4. Entraînement
    model.train()
    for epoch in range(2): # Tu peux augmenter le nombre d'époques
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Époque {epoch+1}")
        
        for batch in loop:
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

    # 5. Sauvegarde organisée
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    
    model.save_pretrained(SAVE_PATH)
    processor.save_pretrained(SAVE_PATH)
    print(f"\nModèle sauvegardé dans : {SAVE_PATH}")

    # 6. Test immédiat
    test_prediction(model, processor, val_dataset)

if __name__ == "__main__":
    train_ocr()