import os
import json
import time
import requests
import pandas as pd
import shutil  # Pour supprimer les dossiers

# --- CONFIGURATION ---
DATA_DIR = "mtg_dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
JSON_FILE = "default_cards.json"

TRAIN_CSV = os.path.join(DATA_DIR, "Train.csv")
TEST_CSV = os.path.join(DATA_DIR, "Test.csv")

LIMIT_TRAIN = 100
LIMIT_TEST = 20
IMAGE_SIZE = 'normal'

TARGET_TYPES = ['Creature', 'Instant', 'Sorcery', 'Enchantment', 'Artifact', 'Planeswalker', 'Land']

def reset_dataset():
    """Supprime les anciens dossiers et fichiers pour repartir à zéro"""
    print("Nettoyage du dataset existant...")
    # Supprimer les dossiers d'images
    for folder in [TRAIN_DIR, TEST_DIR]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f" Dossier {folder} supprimé.")
        os.makedirs(folder)

    # Supprimer les anciens CSV
    for csv in [TRAIN_CSV, TEST_CSV]:
        if os.path.exists(csv):
            os.remove(csv)
            print(f" Fichier {csv} supprimé.")

def get_main_type(type_line):
    for t in TARGET_TYPES:
        if t in type_line:
            return t
    return "Other"

def process_cards():
    # --- ÉTAPE DE NETTOYAGE ---
    reset_dataset()
    
    print("\nChargement du JSON...")
    try:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            cards = json.load(f)
    except FileNotFoundError:
        print(f"Erreur: {JSON_FILE} introuvable.")
        return

    train_data = []
    test_data = []
    count_total = 0

    print(f"Début de la récupération ({LIMIT_TRAIN} Train + {LIMIT_TEST} Test)...")

    for card in cards:
        if count_total >= (LIMIT_TRAIN + LIMIT_TEST):
            break

        if 'image_uris' not in card or 'color_identity' not in card or 'type_line' not in card:
            continue
        
        main_type = get_main_type(card['type_line'])
        if main_type == "Other": 
            continue

        if count_total < LIMIT_TRAIN:
            target_folder = TRAIN_DIR
            current_list = train_data
        else:
            target_folder = TEST_DIR
            current_list = test_data

        identity = card['color_identity']
        file_name = f"{card['id']}.jpg"
        file_path = os.path.join(target_folder, file_name)

        try:
            # On télécharge systématiquement puisque le dossier est vide
            time.sleep(0.05) 
            img_url = card['image_uris'][IMAGE_SIZE]
            response = requests.get(img_url, timeout=10)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
            else:
                continue
        except Exception:
            continue

        row = {
            'image_path': os.path.join(os.path.basename(target_folder), file_name),
            'type': main_type,
            'rarity': card['rarity'],
            'is_white': 1 if 'W' in identity else 0,
            'is_blue': 1 if 'U' in identity else 0,
            'is_black': 1 if 'B' in identity else 0,
            'is_red': 1 if 'R' in identity else 0,
            'is_green': 1 if 'G' in identity else 0,
            'is_colorless': 1 if len(identity) == 0 else 0
        }
        current_list.append(row)
        
        count_total += 1
        if count_total % 50 == 0:
            print(f"Cartes récupérées : {count_total}")

    # Sauvegarde des CSV
    pd.DataFrame(train_data).to_csv(TRAIN_CSV, index=False)
    pd.DataFrame(test_data).to_csv(TEST_CSV, index=False)

    print(f"\nTerminé ! Dossiers propres et nouveaux CSV créés.")

if __name__ == "__main__":
    process_cards()