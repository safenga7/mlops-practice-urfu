"""
Обучение модели машинного обучения (случайный лес)
на предобработанных обучающих данных и сохранение в файл model.pkl.
"""

import os
import pickle
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

FEATURES = ["month", "day", "humidity", "wind_speed", "cloud_cover"]
TARGET = "is_warm"


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base, "train", "preprocessed.csv")

    df = pd.read_csv(train_path)
    X_train = df[FEATURES]
    y_train = df[TARGET]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    model_path = os.path.join(base, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("model_preparation: модель обучена и сохранена", file=sys.stderr)


if __name__ == "__main__":
    main()
