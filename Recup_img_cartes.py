import os
import json
import time
import requests
import pandas as pd

# --- CONFIGURATION ---
DATA_DIR = "mtg_dataset"
IMG_DIR = os.path.join(DATA_DIR, "images")
JSON_FILE = "default_cards.json"
CSV_FILE = os.path.join(DATA_DIR, "metadata.csv")
LIMIT = 100
IMAGE_SIZE = 'normal'

# Types principaux que nous voulons classifier
TARGET_TYPES = ['Creature', 'Instant', 'Sorcery', 'Enchantment', 'Artifact', 'Planeswalker', 'Land']

def setup():
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)

def get_main_type(type_line):
    """Extrait le type principal d'une ligne de type (ex: 'Legendary Creature — Elf' -> 'Creature')"""
    for t in TARGET_TYPES:
        if t in type_line:
            return t
    return "Other"

def download_dataset():
    setup()
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        cards = json.load(f)

    data_list = []
    count = 0

    print(f"Extraction des données (Objectif : {LIMIT} images)...")

    for card in cards:
        if count >= LIMIT:
            break

        # On saute les cartes sans image ou sans identité de couleur
        if 'image_uris' not in card or 'color_identity' not in card or 'type_line' not in card:
            continue

        # 1. Extraction de l'Identité Couleur (Multi-label)
        # On crée une colonne par couleur (1 si présente, 0 sinon)
        identity = card['color_identity']
        colors = {
            'is_white': 1 if 'W' in identity else 0,
            'is_blue': 1 if 'U' in identity else 0,
            'is_black': 1 if 'B' in identity else 0,
            'is_red': 1 if 'R' in identity else 0,
            'is_green': 1 if 'G' in identity else 0,
            'is_colorless': 1 if len(identity) == 0 else 0
        }

        # 2. Extraction du Type
        main_type = get_main_type(card['type_line'])
        if main_type == "Other": continue # On ignore les types marginaux pour la propreté

        # 3. Extraction de la Rareté
        rarity = card['rarity']

        # Téléchargement de l'image
        img_url = card['image_uris'][IMAGE_SIZE]
        file_name = f"{card['id']}.jpg"
        file_path = os.path.join(IMG_DIR, file_name)

        if not os.path.exists(file_path):
            try:
                time.sleep(0.1)
                img_data = requests.get(img_url).content
                with open(file_path, 'wb') as h:
                    h.write(img_data)
                
                # Ajout aux métadonnées
                row = {
                    'image_path': file_name,
                    'type': main_type,
                    'rarity': rarity,
                    **colors
                }
                data_list.append(row)
                count += 1
                if count % 50 == 0: print(f"Cartes téléchargées : {count}")
            except Exception as e:
                print(f"Erreur sur {card['name']}: {e}")

    # Sauvegarde des métadonnées en CSV
    df = pd.DataFrame(data_list)
    df.to_csv(CSV_FILE, index=False)
    print(f"\nTerminé ! {len(df)} cartes enregistrées dans {CSV_FILE}")

if __name__ == "__main__":
    # Assurez-vous d'avoir déjà le fichier default_cards.json (voir scripts précédents)
    download_dataset()