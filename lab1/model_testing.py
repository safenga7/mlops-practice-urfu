"""
Загрузка сохранённой модели и проверка её качества на тестовой выборке.

Выводит в stdout одну строку вида:
  Model test accuracy is: 0.876
"""

import os
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

FEATURES = ["month", "day", "humidity", "wind_speed", "cloud_cover"]
TARGET = "is_warm"


def main():
    base = os.path.dirname(os.path.abspath(__file__))

    # Загрузка модели
    with open(os.path.join(base, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    # Загрузка тестовых данных
    test_path = os.path.join(base, "test", "preprocessed.csv")
    df = pd.read_csv(test_path)
    X_test = df[FEATURES]
    y_test = df[TARGET]

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Единственная строка в stdout
    print(f"Model test accuracy is: {accuracy:.3f}")


if __name__ == "__main__":
    main()
