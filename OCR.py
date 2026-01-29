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
        # Nettoyage du chemin pour Linux/Colab
        clean_path = row['image_path'].replace('\\', '/')
        img_path = os.path.join(self.root_dir, clean_path)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new('RGB', (384, 384)) # Image vide en cas d'erreur

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

# --- FONCTION DE VALIDATION ---
def calcul_validation(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(loader)

# --- FONCTION DE TEST RAPIDE ---
def test_prediction(model, processor, dataset, num_samples=100):
    model.eval()
    print(f"\n--- TEST DE PRÉDICTION (Sur {num_samples} échantillons) ---")
    
    for i in range(min(num_samples, len(dataset))):
        item = dataset[i]
        pixel_values = item["pixel_values"].unsqueeze(0).to(DEVICE)
        
        true_ids = [l for l in item["labels"].tolist() if l != -100]
        true_text = processor.tokenizer.decode(true_ids, skip_special_tokens=True)
        
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values, 
                max_new_tokens=32, 
                min_length=5,
                num_beams=5,
                early_stopping=False
            )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"Vrai nom : {true_text}")
        print(f"Prédit   : {generated_text} {'✅' if true_text.lower() == generated_text.lower() else '❌'}")
        print("-" * 30)

# --- ÉTAPE D'ENTRAÎNEMENT ---
def train_ocr():
    print(f"Utilisation de : {DEVICE}")
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # 1. Chargement
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(DEVICE)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # 2. Datasets
    train_dataset = MTGOCRDataset(TRAIN_CSV, DATA_DIR, processor)
    val_dataset = MTGOCRDataset(VAL_CSV, DATA_DIR, processor)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # 3. Optimiseur et Scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2)

    # 4. Entraînement
    epochs = 30 # Ajustable
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Époque {epoch+1}/{epochs}")
        
        for batch in loop:
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        
        # --- VALIDATION ---
        avg_val_loss = calcul_validation(model, val_loader)
        scheduler.step(avg_val_loss)
        
        status = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(SAVE_PATH)
            processor.save_pretrained(SAVE_PATH)
            status = "--- Nouveau meilleur score ! Modèle sauvegardé ---"

        print(f"Fin Époque {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} {status}")

    # 5. Test final avec le meilleur modèle rechargé
    print("\nRechargement du meilleur modèle pour test final...")
    model = VisionEncoderDecoderModel.from_pretrained(SAVE_PATH).to(DEVICE)
    test_prediction(model, processor, val_dataset)

if __name__ == "__main__":
    train_ocr()