# Neuron_Magic_Detection

Guide de navigation et utilisation dans le repository.

---

## 1. Organisation du repository


* **mtg_dataset :** contient les images et méta-données correspondantes aux différentes classes
* **saved_lossses :** dossier où l'évolution des loss est stocké
* **saved_models :** dossier où le meilleur modèle de chaque entraînement est stocké
* **.py :** trace des modèles non-adaptés au format notebook 

---

## 2. Les modèles implémentés

1. **CNN_ENBO :** modèle EfficientNet-B0
2. **ViT** 
3. **OCR**

---

Pour les notebook compiler les cellules pré-cellule-d'entraînement
puis lancer au choix l'entraînement, la phase de test ou l'affiche de l'évolution de la loss
Pour lancer la phase de test et/ou l'affiche de l'évolution de la loss bien renseigner le modèle ou la loss que l'on veut tester/afficher