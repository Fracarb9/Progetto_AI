from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


"""Fase 3: Addestramento dei modelli"""


def scale_features(X_train, X_test):
    """
    Applica StandardScaler a train e test set.
    Lo scaling facilita la convergenza (soprattutto per MLP e k-NN).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Debug: mostra i primi valori originali e scalati
    print("Valori originali (prime 5 righe, prime 5 colonne):")
    print(X_train[:5, :5])
    print("Valori scalati:")
    print(X_train_scaled[:5, :5])

    return X_train_scaled, X_test_scaled


def train_decision_tree(X_train_feat, y_train, X_test_feat, y_test, variant):
    """
    Addestra e valuta un Decision Tree con GridSearchCV.
    Parametri ottimizzati:
        - max_depth
        - criterion (gini/entropy)
    """
    print(">>> Decision Tree")
    X_train_feat, X_test_feat = scale_features(X_train_feat, X_test_feat)

    params = {'max_depth': [5, 10, 15, 20], 'criterion': ['gini', 'entropy']}
    clf_dt = GridSearchCV(DecisionTreeClassifier(), params, cv=5, scoring='accuracy')
    clf_dt.fit(X_train_feat, y_train)

    print("Migliori parametri:", clf_dt.best_params_)
    y_pred_dt = clf_dt.predict(X_test_feat)

    # Report di classificazione
    print(classification_report(y_test, y_pred_dt, target_names=['healthy', 'parkinson']))

    # Matrice di confusione
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred_dt,
        display_labels=["healthy", "parkinson"]
    )
    plt.title(f"Decision Tree ({variant})")
    plt.show()


def train_mlp(X_train, y_train, X_test, y_test, variant):
    """
    Addestra e valuta una rete neurale MLP con GridSearchCV.
    Parametri ottimizzati:
        - hidden_layer_sizes (architettura dei layer)
        - alpha (fattore di regolarizzazione L2)
        - max_iter (numero massimo di iterazioni)
    """
    print(f"\nTraining MLP for variant: {variant}")
    X_train, X_test = scale_features(X_train, X_test)

    # Spazio degli iperparametri
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50)],
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [200, 300]
    }

    mlp = MLPClassifier(
        random_state=42,
        early_stopping=True,   # usa parte del train come validation interna
        validation_fraction=0.1
    )

    grid = GridSearchCV(
        estimator=mlp,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    # Addestramento con ricerca degli iperparametri
    grid.fit(X_train, y_train)

    print(f"Best parameters: {grid.best_params_}")
    print(f"Best cross-validation accuracy: {grid.best_score_:.4f}")

    best_mlp = grid.best_estimator_

    # --- Curva di apprendimento (loss per epoca) ---
    plt.figure(figsize=(8, 5))
    plt.plot(best_mlp.loss_curve_, marker='o')
    plt.title(f"MLP Loss Curve - {variant}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()

    # --- Valutazione sul test set ---
    y_pred = best_mlp.predict(X_test)
    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_pred, target_names=['healthy', 'parkinson']))

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['healthy', 'parkinson'],
                yticklabels=['healthy', 'parkinson'])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"MLP Confusion Matrix - {variant}")
    plt.show()


def train_knn(X_train_feat, y_train, X_test_feat, y_test, variant):
    """
    Addestra e valuta un k-Nearest Neighbors con GridSearchCV.
    Parametri ottimizzati:
        - n_neighbors (numero di vicini)
        - weights (uniform o distance)
    """
    print(">>> k-NN")
    X_train_feat, X_test_feat = scale_features(X_train_feat, X_test_feat)

    params = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
    clf_knn = GridSearchCV(KNeighborsClassifier(), params, cv=5, scoring='accuracy')
    clf_knn.fit(X_train_feat, y_train)

    print("Migliori parametri:", clf_knn.best_params_)
    y_pred_knn = clf_knn.predict(X_test_feat)

    # Report di classificazione
    print(classification_report(y_test, y_pred_knn, target_names=['healthy', 'parkinson']))

    # Matrice di confusione
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred_knn,
        display_labels=["healthy", "parkinson"]
    )
    plt.title(f"k-NN ({variant})")
    plt.show()

