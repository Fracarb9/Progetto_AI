import os
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt



def train_decision_tree(X_train_feat, y_train, X_test_feat, y_test, variant):
    print(">>> Decision Tree")
    params = {'max_depth': [5, 10, 15, 20], 'criterion': ['gini', 'entropy']}
    clf_dt = GridSearchCV(DecisionTreeClassifier(), params, cv=5, scoring='accuracy')
    clf_dt.fit(X_train_feat, y_train)

    print("Migliori parametri:", clf_dt.best_params_)
    y_pred_dt = clf_dt.predict(X_test_feat)
    print(classification_report(y_test, y_pred_dt))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_dt)
    plt.title(f"Decision Tree ({variant})")
    plt.show()


def train_mlp(X_train_feat, y_train, X_test_feat, y_test, variant):
    print(">>> MLP")
    params = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50)],
        'activation': ['relu', 'tanh'],
        'max_iter': [1000],
    }
    clf_mlp = GridSearchCV(MLPClassifier(), params, cv=5, scoring='accuracy')
    clf_mlp.fit(X_train_feat, y_train)

    print("Migliori parametri:", clf_mlp.best_params_)
    y_pred_mlp = clf_mlp.predict(X_test_feat)
    print(classification_report(y_test, y_pred_mlp))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_mlp)
    plt.title(f"MLP ({variant})")
    plt.show()

def train_knn(X_train_feat, y_train, X_test_feat, y_test, variant):
    print(">>> k-NN")
    params = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
    clf_knn = GridSearchCV(KNeighborsClassifier(), params, cv=5, scoring='accuracy')
    clf_knn.fit(X_train_feat, y_train)

    print("Migliori parametri:", clf_knn.best_params_)
    y_pred_knn = clf_knn.predict(X_test_feat)
    print(classification_report(y_test, y_pred_knn))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_knn)
    plt.title(f"k-NN ({variant})")
    plt.show()
