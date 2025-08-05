from preprocessing import load_dataset
from features import extract_features
from models import train_decision_tree, train_mlp, train_knn

import numpy as np

VARIANTS = [
    "template",
    "template_unwrapped",
    "template_unwrapped_boxed"
]

BASE_PATH = "newhandpdV3"

for variant in VARIANTS:
    print(f"\n=== PROCESSING VARIANT: {variant} ===")

    train_path = f"{BASE_PATH}/train_{variant}"
    test_path = f"{BASE_PATH}/test_{variant}"

    # Carica immagini e label
    X_train_images, y_train = load_dataset(train_path)
    X_test_images, y_test = load_dataset(test_path)

    # Estrai feature
    X_train = np.array([extract_features(img) for img in X_train_images])
    X_test = np.array([extract_features(img) for img in X_test_images])

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Addestra e valuta
    print("\n-- Decision Tree --")
    train_decision_tree(X_train, y_train, X_test, y_test, variant)

    print("\n-- MLP --")
    train_mlp(X_train, y_train, X_test, y_test, variant)

    print("\n-- k-NN --")
    train_knn(X_train, y_train, X_test, y_test, variant)
