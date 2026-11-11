"""
Загрузка сохранённой модели и оценка на тестовых данных.

Выводит метрики accuracy, precision, recall и classification report.
Последняя строка stdout — итоговая accuracy в формате:
  Model test accuracy is: 0.967
"""

import os
import sys
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    proc_dir = os.path.join(base, "data", "processed")

    # Загрузка модели
    with open(os.path.join(base, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    # Загрузка тестовых данных
    test = pd.read_csv(os.path.join(proc_dir, "test.csv"))
    X_test = test[FEATURES]
    y_test = test[TARGET]

    y_pred = model.predict(X_test)

    # Подробный отчёт — в stderr
    print(classification_report(y_test, y_pred), file=sys.stderr)

    # Единственная строка в stdout
    acc = accuracy_score(y_test, y_pred)
    print(f"Model test accuracy is: {acc:.3f}")


if __name__ == "__main__":
    main()
