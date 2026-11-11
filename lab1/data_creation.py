"""
Создание синтетических наборов данных, описывающих дневную температуру.

Генерируются несколько CSV-файлов:
  - train/  — обучающая выборка (3 набора: чистый, с шумом, с аномалиями)
  - test/   — тестовая выборка (1 набор)

Признаки:
  month       — месяц (1–12)
  day         — день месяца (1–31)
  humidity    — относительная влажность (%)
  wind_speed  — скорость ветра (м/с)
  cloud_cover — облачность (0–1)
Целевая переменная (классификация):
  is_warm     — 1 если температура ≥ 15 °C, иначе 0
"""

import os
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)


def generate_temperature_data(n_samples, noise_std=0.0, anomaly_fraction=0.0):
    """Генерирует данные о дневной температуре."""
    month = np.random.randint(1, 13, size=n_samples)
    day = np.random.randint(1, 32, size=n_samples)
    humidity = np.random.uniform(20, 100, size=n_samples)
    wind_speed = np.random.uniform(0, 15, size=n_samples)
    cloud_cover = np.random.uniform(0, 1, size=n_samples)

    # Базовая температура: зависит от месяца (синусоида с пиком летом)
    temp = 10 * np.sin(np.pi * (month - 3) / 6) + 12
    temp -= wind_speed * 0.3
    temp -= cloud_cover * 3
    temp += humidity * 0.02

    if noise_std > 0:
        temp += np.random.normal(0, noise_std, size=n_samples)

    if anomaly_fraction > 0:
        n_anomalies = int(n_samples * anomaly_fraction)
        idx = np.random.choice(n_samples, n_anomalies, replace=False)
        temp[idx] += np.random.uniform(-20, 20, size=n_anomalies)

    is_warm = (temp >= 15).astype(int)

    return pd.DataFrame({
        "month": month,
        "day": day,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "cloud_cover": cloud_cover,
        "is_warm": is_warm,
    })


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base, "train")
    test_dir = os.path.join(base, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Обучающие наборы
    df1 = generate_temperature_data(400, noise_std=0.0)
    df1.to_csv(os.path.join(train_dir, "data_clean.csv"), index=False)

    df2 = generate_temperature_data(400, noise_std=1.0)
    df2.to_csv(os.path.join(train_dir, "data_noisy.csv"), index=False)

    df3 = generate_temperature_data(200, noise_std=0.5, anomaly_fraction=0.05)
    df3.to_csv(os.path.join(train_dir, "data_anomalies.csv"), index=False)

    # Тестовый набор
    df_test = generate_temperature_data(300, noise_std=0.5)
    df_test.to_csv(os.path.join(test_dir, "data_test.csv"), index=False)

    import sys
    print("data_creation: наборы данных созданы", file=sys.stderr)


if __name__ == "__main__":
    main()
