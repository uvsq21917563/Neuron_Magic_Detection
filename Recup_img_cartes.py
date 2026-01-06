import os
import json
import time
import requests

# --- CONFIGURATION ---
DATA_DIR = "mtg_dataset"
JSON_FILE = "default_cards.json"
# On limite le nombre d'images pour le test (mettez None pour tout télécharger)
LIMIT_PER_COLOR = 100 
# Tailles possibles : 'small', 'normal', 'large', 'png', 'art_crop'
IMAGE_SIZE = 'normal' 

def download_bulk_metadata():
    """Télécharge le fichier JSON global de Scryfall s'il n'existe pas déjà."""
    if not os.path.exists(JSON_FILE):
        print("Récupération de l'URL du bulk data...")
        resp = requests.get("https://api.scryfall.com/bulk-data")
        bulk_url = next(item for item in resp.json()['data'] if item['type'] == 'default_cards')['download_uri']
        
        print(f"Téléchargement du JSON global (cela peut prendre du temps)...")
        r = requests.get(bulk_url, stream=True)
        with open(JSON_FILE, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return JSON_FILE

def setup_folders(colors):
    """Crée un dossier pour chaque couleur."""
    for color in colors:
        path = os.path.join(DATA_DIR, color)
        if not os.path.exists(path):
            os.makedirs(path)

def download_images():
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        cards = json.load(f)

    # On définit les couleurs cibles (W=Blanc, U=Bleu, B=Noir, R=Rouge, G=Vert)
    color_map = {'W': 'White', 'U': 'Blue', 'B': 'Black', 'R': 'Red', 'G': 'Green'}
    setup_folders(color_map.values())
    
    counts = {c: 0 for c in color_map.keys()}

    print("Début du téléchargement des images...")
    for card in cards:
        # On ne prend que les cartes d'une seule couleur et qui ont une image
        if 'colors' in card and len(card['colors']) == 1:
            color_code = card['colors'][0]
            
            if color_code in color_map and counts[color_code] < LIMIT_PER_COLOR:
                if 'image_uris' in card and IMAGE_SIZE in card['image_uris']:
                    img_url = card['image_uris'][IMAGE_SIZE]
                    color_name = color_map[color_code]
                    
                    # Nettoyage du nom de fichier
                    file_name = f"{card['id']}.jpg"
                    file_path = os.path.join(DATA_DIR, color_name, file_name)
                    
                    if not os.path.exists(file_path):
                        # Respecter le rate limit de Scryfall (50-100ms entre chaque requête)
                        time.sleep(0.1) 
                        img_data = requests.get(img_url).content
                        with open(file_path, 'wb') as handler:
                            handler.write(img_data)
                        
                        counts[color_code] += 1
                        if sum(counts.values()) % 10 == 0:
                            print(f"Progression : {counts}")

    print("Téléchargement terminé !")

if __name__ == "__main__":
    download_bulk_metadata()
    download_images()