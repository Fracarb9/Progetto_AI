from preprocessing import load_dataset
from features import extract_features
from models import train_decision_tree, train_mlp, train_knn

import numpy as np

# Varianti delle immagini presenti nel dataset (3 formati diversi)
VARIANTS = [
    "template",
    "template_unwrapped",
    "template_unwrapped_boxed"
]

# Directory di base del dataset
BASE_PATH = "newhandpdV3"

# Ciclo su ciascuna variante del dataset
for variant in VARIANTS:
    print(f"\n=== PROCESSING VARIANT: {variant} ===")

    # Percorsi train e test per la variante corrente
    train_path = f"{BASE_PATH}/train_{variant}"
    test_path = f"{BASE_PATH}/test_{variant}"

    # 1. Carica immagini e label
    X_train_images, y_train = load_dataset(train_path)
    X_test_images, y_test = load_dataset(test_path)

    # 2. Estrai feature dalle immagini
    # Ogni immagine viene trasformata in un vettore di feature numeriche
    X_train = np.array([extract_features(img) for img in X_train_images])
    X_test = np.array([extract_features(img) for img in X_test_images])

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # 3. Addestra e valuta i modelli
    # Ogni modello viene addestrato e valutato separatamente
    print("\n-- Decision Tree --")
    train_decision_tree(X_train, y_train, X_test, y_test, variant)

    print("\n-- MLP --")
    train_mlp(X_train, y_train, X_test, y_test, variant)

    print("\n-- k-NN --")
    train_knn(X_train, y_train, X_test, y_test, variant)
