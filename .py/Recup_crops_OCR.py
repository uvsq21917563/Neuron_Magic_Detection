import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_DIR = "mtg_dataset" # Dossier unique pour tout
# On crée un sous-dossier pour les vignettes afin de ne pas mélanger avec les cartes entières
CROP_SUBDIR = os.path.join(DATA_DIR, "crops") 
CSV_FILES = ["Train.csv", "Val.csv", "Test.csv"]

# Proportions du crop (12% du haut)
TOP_PERCENT = 0.12 

def prepare_ocr_data():
    # Création du dossier pour les images découpées
    if not os.path.exists(CROP_SUBDIR):
        os.makedirs(CROP_SUBDIR)
        for sub in ['train', 'val', 'test']:
            os.makedirs(os.path.join(CROP_SUBDIR, sub))

    for csv_name in CSV_FILES:
        csv_path = os.path.join(DATA_DIR, csv_name)
        if not os.path.exists(csv_path):
            print(f"Fichier {csv_name} introuvable dans {DATA_DIR}, on passe...")
            continue

        print(f"\nTraitement de {csv_name}...")
        df = pd.read_csv(csv_path)
        ocr_data = []

        for _, row in tqdm(df.iterrows(), total=len(df)):
            # Chemin de l'image originale (ex: mtg_dataset/train/id.jpg)
            img_path = os.path.join(DATA_DIR, row['image_path'])
            
            if not os.path.exists(img_path):
                continue

            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    
                    # Zone du titre : 5% à 95% en largeur, 4% à 12% en hauteur
                    left = width * 0.05
                    top = height * 0.04
                    right = width * 0.95
                    bottom = height * TOP_PERCENT
                    
                    name_crop = img.crop((left, top, right, bottom)).convert("RGB")
                    
                    # Sauvegarde dans mtg_dataset/crops/train/id.jpg par exemple
                    crop_rel_path = os.path.join("crops", row['image_path'])
                    save_path = os.path.join(DATA_DIR, crop_rel_path)
                    
                    name_crop.save(save_path)
                    
                    # Label pour le nouveau CSV
                    ocr_data.append({
                        'image_path': crop_rel_path,
                        'label': row['name']
                    })
            except Exception as e:
                print(f"Erreur sur {img_path}: {e}")

        # Sauvegarde du CSV OCR dans le dossier principal mtg_dataset
        ocr_df = pd.DataFrame(ocr_data)
        output_csv_name = f"OCR_{csv_name}" # Ex: OCR_Train.csv
        ocr_df.to_csv(os.path.join(DATA_DIR, output_csv_name), index=False)
        print(f"Fini ! {output_csv_name} sauvegardé dans {DATA_DIR}.")

if __name__ == "__main__":
    prepare_ocr_data()