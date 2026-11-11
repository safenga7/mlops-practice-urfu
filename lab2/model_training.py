"""
Обучение модели (Random Forest) на обучающей выборке и сохранение в pickle.
"""

import os
import sys
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    proc_dir = os.path.join(base, "data", "processed")

    train = pd.read_csv(os.path.join(proc_dir, "train.csv"))
    X_train = train[FEATURES]
    y_train = train[TARGET]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    model_path = os.path.join(base, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Модель обучена и сохранена в {model_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
