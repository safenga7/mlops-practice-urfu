"""
Предобработка данных и разделение на обучающую/тестовую выборки.

1. Читает сырые данные iris.data (CSV без заголовка).
2. Назначает имена столбцов.
3. Удаляет пустые строки (в конце файла UCI есть пустая строка).
4. Кодирует целевой класс числами (LabelEncoder).
5. Нормализует числовые признаки (StandardScaler), fit только на train.
6. Сохраняет train.csv и test.csv в data/processed/.
"""

import os
import sys
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species",
]
FEATURES = COLUMNS[:4]
TARGET = "species"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(base, "data", "raw", "iris.data")
    proc_dir = os.path.join(base, "data", "processed")
    os.makedirs(proc_dir, exist_ok=True)

    # --- чтение ---
    df = pd.read_csv(raw_path, header=None, names=COLUMNS)
    df = df.dropna()
    # Убираем строки, где species пустой (бывает в конце файла)
    df = df[df[TARGET].str.strip() != ""]
    df = df.reset_index(drop=True)

    print(f"Загружено {len(df)} записей", file=sys.stderr)

    # --- кодирование целевой переменной ---
    le = LabelEncoder()
    df[TARGET] = le.fit_transform(df[TARGET])

    # Сохраняем маппинг классов
    with open(os.path.join(proc_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    # --- разделение ---
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET]
    )

    # --- масштабирование признаков ---
    scaler = StandardScaler()
    train_df[FEATURES] = scaler.fit_transform(train_df[FEATURES])
    test_df[FEATURES] = scaler.transform(test_df[FEATURES])

    with open(os.path.join(proc_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # --- сохранение ---
    train_df.to_csv(os.path.join(proc_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(proc_dir, "test.csv"), index=False)

    print(
        f"Train: {len(train_df)} строк, Test: {len(test_df)} строк → "
        f"data/processed/",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
