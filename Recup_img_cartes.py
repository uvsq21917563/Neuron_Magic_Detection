import os
import json
import time
import requests
import pandas as pd
import shutil
import random

# --- CONFIGURATION ---
DATA_DIR = "mtg_dataset"
DIRS = {
    'train': os.path.join(DATA_DIR, "train"),
    'val': os.path.join(DATA_DIR, "val"),
    'test': os.path.join(DATA_DIR, "test")
}
FILES = {
    'train': os.path.join(DATA_DIR, "Train.csv"),
    'val': os.path.join(DATA_DIR, "Val.csv"),
    'test': os.path.join(DATA_DIR, "Test.csv")
}
JSON_FILE = "default_cards.json"

# Taille des datasets
LIMIT_TRAIN = 3000
LIMIT_VAL = 600
LIMIT_TEST = 600
TOTAL_NEEDED = LIMIT_TRAIN + LIMIT_VAL + LIMIT_TEST

IMAGE_SIZE = 'normal'

# Répartition demandée
QUOTAS = {
    'W': 0.15,       # Blanc
    'U': 0.15,       # Bleu
    'B': 0.15,       # Noir
    'R': 0.15,       # Rouge
    'G': 0.15,       # Vert
    'Multi': 0.20,   # Multicolore
    'Colorless': 0.05 # Incolore
}

TARGET_TYPES = ['Creature', 'Instant', 'Sorcery', 'Enchantment', 'Artifact', 'Planeswalker', 'Land']

def reset_dataset():
    """Supprime les anciens dossiers et fichiers"""
    print("Nettoyage du dataset existant...")
    for key, folder in DIRS.items():
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)
        print(f" Dossier {folder} recréé.")

    for key, csv_file in FILES.items():
        if os.path.exists(csv_file):
            os.remove(csv_file)

def get_main_type(type_line):
    if "Token" in type_line: 
        return "Other"
    for t in TARGET_TYPES:
        if t in type_line:
            return t
    return "Other"

def get_color_category(identity):
    if len(identity) == 0:
        return 'Colorless'
    elif len(identity) > 1:
        return 'Multi'
    else:
        return identity[0] # 'W', 'U', 'B', 'R', 'G'

def format_time(seconds):
    """Convertit les secondes en format MM:SS"""
    return time.strftime("%M:%S", time.gmtime(seconds))

def download_and_save(card_list, dataset_name):
    """Télécharge les images avec Timer et ETA"""
    folder = DIRS[dataset_name]
    data_rows = []
    total_cards = len(card_list)
    
    print(f"\n--- Démarrage du dataset : {dataset_name.upper()} ({total_cards} cartes) ---")
    start_time = time.time()
    
    for i, card in enumerate(card_list):
        card_id = card['id']
        file_name = f"{card_id}.jpg"
        file_path = os.path.join(folder, file_name)
        
        # Téléchargement
        try:
            time.sleep(0.05) # Pause API
            img_url = card['image_uris'][IMAGE_SIZE]
            
            if not os.path.exists(file_path):
                response = requests.get(img_url, timeout=10)
                if response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                else:
                    print(f" [!] Erreur téléchargement {card_id}")
                    continue
        except Exception as e:
            print(f" [!] Exception sur {card_id}: {e}")
            continue

        # Préparation CSV
        identity = card['color_identity']
        main_type = get_main_type(card['type_line'])
        
        row = {
            'image_path': os.path.join(dataset_name, file_name),
            'type': main_type,
            'rarity': card['rarity'],
            'is_white': 1 if 'W' in identity else 0,
            'is_blue': 1 if 'U' in identity else 0,
            'is_black': 1 if 'B' in identity else 0,
            'is_red': 1 if 'R' in identity else 0,
            'is_green': 1 if 'G' in identity else 0,
            'is_colorless': 1 if len(identity) == 0 else 0
        }
        data_rows.append(row)
        
        # --- TIMER / LOGGING TOUTES LES 25 CARTES ---
        count = i + 1
        if count % 25 == 0 or count == total_cards:
            elapsed = time.time() - start_time
            speed = count / elapsed # Cartes par seconde
            remaining = total_cards - count
            eta = remaining / speed if speed > 0 else 0
            
            print(f"   -> {count}/{total_cards} | "
                  f"Temps: {format_time(elapsed)} | "
                  f"Reste env: {format_time(eta)}")

    # Sauvegarde CSV
    df = pd.DataFrame(data_rows)
    df.to_csv(FILES[dataset_name], index=False)
    print(f" -> {FILES[dataset_name]} sauvegardé ({len(df)} entrées).")


def process_cards():
    reset_dataset()
    
    print("\nChargement du JSON...")
    try:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            all_cards = json.load(f)
    except FileNotFoundError:
        print(f"Erreur: {JSON_FILE} introuvable.")
        return

    # --- ÉTAPE 1 : FILTRAGE ---
    print("Filtrage et tri des cartes...")
    pools = {k: [] for k in QUOTAS.keys()}
    random.shuffle(all_cards)

    for card in all_cards:
        if 'image_uris' not in card or 'color_identity' not in card or 'type_line' not in card:
            continue
        
        layout = card.get('layout', '')
        if layout in ['token', 'double_faced_token', 'art_series', 'emblem', 'planar', 'vanguard', 'scheme']:
            continue
            
        main_type = get_main_type(card['type_line'])
        if main_type == "Other":
            continue
            
        cat = get_color_category(card['color_identity'])
        pools[cat].append(card)

    # --- ÉTAPE 2 : SÉLECTION ---
    final_selection = []
    print("\nSélection selon quotas :")
    
    for cat, ratio in QUOTAS.items():
        count_needed = int(TOTAL_NEEDED * ratio)
        available = len(pools[cat])
        
        if available < count_needed:
            print(f"⚠️  Manque de cartes '{cat}' (Dispo: {available}). On prend tout.")
            taken = pools[cat]
        else:
            taken = pools[cat][:count_needed]
            
        final_selection.extend(taken)
        print(f" - {cat}: {len(taken)} cartes.")

    random.shuffle(final_selection)
    total_selected = len(final_selection)
    
    # --- ÉTAPE 3 : DISTRIBUTION ---
    train_end = LIMIT_TRAIN
    val_end = LIMIT_TRAIN + LIMIT_VAL
    
    if total_selected < TOTAL_NEEDED:
        print("Ajustement des tailles suite au manque de cartes...")
        train_end = int(total_selected * 0.7)
        val_end = int(total_selected * 0.85)
    
    train_list = final_selection[:train_end]
    val_list = final_selection[train_end:val_end]
    test_list = final_selection[val_end:]

    # --- ÉTAPE 4 : TÉLÉCHARGEMENT ---
    download_and_save(train_list, 'train')
    download_and_save(val_list, 'val')
    download_and_save(test_list, 'test')

    print("\n--- Opération Terminée ---")

if __name__ == "__main__":
    process_cards()