"""
Предобработка данных с помощью StandardScaler.

1. Объединяет исходные CSV из train/ → обучающая выборка.
2. Читает исходные CSV из test/  → тестовая выборка.
3. Подгоняет (fit) скейлер ТОЛЬКО на обучающей выборке.
4. Трансформирует (transform) обе выборки.
5. Сохраняет результаты в train/preprocessed.csv и test/preprocessed.csv.

Файл preprocessed.csv (результат предыдущего запуска) намеренно исключается
из чтения, чтобы повторный запуск давал идентичный результат.
"""

import os
import glob
import pickle
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES = ["month", "day", "humidity", "wind_speed", "cloud_cover"]
TARGET = "is_warm"

# Имя файла, который создаёт этот скрипт — его нужно исключить при чтении.
OUTPUT_NAME = "preprocessed.csv"


def load_csvs(directory):
    """Читает все CSV из каталога, кроме preprocessed.csv."""
    files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    files = [f for f in files if os.path.basename(f) != OUTPUT_NAME]
    frames = [pd.read_csv(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base, "train")
    test_dir = os.path.join(base, "test")

    train_df = load_csvs(train_dir)
    test_df = load_csvs(test_dir)

    scaler = StandardScaler()
    train_df[FEATURES] = scaler.fit_transform(train_df[FEATURES])
    test_df[FEATURES] = scaler.transform(test_df[FEATURES])

    # Сохраняем скейлер, чтобы при необходимости можно было использовать повторно
    with open(os.path.join(base, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    train_df.to_csv(os.path.join(train_dir, "preprocessed.csv"), index=False)
    test_df.to_csv(os.path.join(test_dir, "preprocessed.csv"), index=False)

    print("data_preprocessing: данные нормализованы", file=sys.stderr)


if __name__ == "__main__":
    main()
