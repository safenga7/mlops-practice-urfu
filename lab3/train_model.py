"""
Model training and artifact serialization for Lab 3 microservice.
Trains a RandomForestClassifier on the Iris dataset with StandardScaler preprocessing,
matching the architecture and features from Lab 2.
"""

import os
import pickle
import sys
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

RANDOM_STATE = 42
TEST_SIZE = 0.2


def train_and_save(output_dir: str = "models"):
    """Trains the model and saves model.pkl and scaler.pkl into output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    print("Loading Iris dataset...", file=sys.stderr)
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("Fitting StandardScaler...", file=sys.stderr)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training RandomForestClassifier...", file=sys.stderr)
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model test accuracy: {acc:.4f}", file=sys.stderr)
    print(classification_report(y_test, y_pred, target_names=target_names), file=sys.stderr)

    # Save artifacts
    model_path = os.path.join(output_dir, "model.pkl")
    scaler_path = os.path.join(output_dir, "scaler.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Artifacts successfully saved to {output_dir}/", file=sys.stderr)
    return acc


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    train_and_save(models_dir)
