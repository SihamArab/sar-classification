## Classification d'Images Radar SAR — MSTAR-10

**ARAB Siham & OURZIK Koceila** | Master Signal et Télécommunication, UBO 2025-2026  
**Encadrant :** Mr. TOUMI Abdelmalek

---

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Structure du projet

```
mstar-sar-classification/
├── mstar_sar_classification.py   # Script principal 
├── requirements.txt
├── mstar_data/                   # Dataset MSTAR-10 (à fournir)
    ├── train/                    # 10 classes, dépression 17°
    └── test/                     # 10 classes, dépression 15°
```

## Configuration

Tous les paramètres sont centralisés dans la classe `Config` (ligne ~75) :

```python
DATA_ROOT = Path('./mstar_data')   # Chemin du dataset
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.5
PATIENCE = 15                      # Early stopping
USE_AUGMENTATION = True            # Speckle noise + rotations SAR
PRETRAINED_MODELS = {'AlexNet': True, 'VGG16': True, 'ResNet50': True, 'DenseNet121': True}
```

## Utilisation

```bash
# Lancer l'entraînement complet (~12h CPU)
python mstar_sar_classification.py
```

Le script exécute séquentiellement : chargement des données → prétraitement (échelle log + normalisation speckle) → augmentation → entraînement avec early stopping → évaluation → export des sorties.

Pour entraîner un seul modèle, mettre les autres à `False` dans `PRETRAINED_MODELS`.
