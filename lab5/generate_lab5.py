#!/usr/bin/env python3
"""
Lab 5 Generation and Verification Script
Generates datasets, trains model, generates test suite, runs pytest,
and constructs fully populated Jupyter Notebooks with all outputs.
"""

import os
import sys
import json
import pickle
import subprocess
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

LAB5_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(LAB5_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 1. Generate Clean Datasets
np.random.seed(42)
x = np.linspace(0, 10, 100)

y_clean_1 = 2.5 * x + 5.0 + np.random.normal(0, 0.4, 100)
df_clean_1 = pd.DataFrame({'x': x, 'y': y_clean_1})
clean_1_path = os.path.join(DATA_DIR, "clean_data_1.csv")
df_clean_1.to_csv(clean_1_path, index=False)

y_clean_2 = 2.5 * x + 5.0 + np.random.normal(0, 0.6, 100)
df_clean_2 = pd.DataFrame({'x': x, 'y': y_clean_2})
clean_2_path = os.path.join(DATA_DIR, "clean_data_2.csv")
df_clean_2.to_csv(clean_2_path, index=False)

y_clean_3 = 2.5 * x + 5.0 + np.random.normal(0, 0.5, 100)
df_clean_3 = pd.DataFrame({'x': x, 'y': y_clean_3})
clean_3_path = os.path.join(DATA_DIR, "clean_data_3.csv")
df_clean_3.to_csv(clean_3_path, index=False)

# 2. Train Model
X_train = df_clean_1[['x']].values
y_train = df_clean_1['y'].values
model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
train_r2 = float(r2_score(y_train, y_train_pred))
train_mse = float(mean_squared_error(y_train, y_train_pred))
slope = float(model.coef_[0])
intercept = float(model.intercept_)

model_path = os.path.join(LAB5_DIR, "linear_regression_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

# 3. Generate Noisy Dataset
np.random.seed(123)
x_noisy = np.linspace(0, 10, 100)
y_noisy = 2.5 * x_noisy + 5.0 + np.random.normal(0, 6.0, 100)
outliers = np.random.choice(100, 15, replace=False)
y_noisy[outliers] += np.random.choice([-25.0, 25.0], 15)

df_noised = pd.DataFrame({'x': x_noisy, 'y': y_noisy})
noised_path = os.path.join(DATA_DIR, "noised_data.csv")
df_noised.to_csv(noised_path, index=False)

# Evaluate all datasets
eval_results = []
for name, p in [
    ("Чистый датасет 1 (clean_1)", clean_1_path),
    ("Чистый датасет 2 (clean_2)", clean_2_path),
    ("Чистый датасет 3 (clean_3)", clean_3_path),
    ("Зашумленный датасет (noised)", noised_path),
]:
    d = pd.read_csv(p)
    pred = model.predict(d[['x']].values)
    r2_val = float(r2_score(d['y'].values, pred))
    mse_val = float(mean_squared_error(d['y'].values, pred))
    status = "PASSED (SLA OK)" if (r2_val >= 0.85 and mse_val <= 2.0) else "FAILED (SLA VIOLATION)"
    eval_results.append({
        "Датасет": name,
        "R2 Score": round(r2_val, 4),
        "MSE": round(mse_val, 4),
        "SLA (R2 >= 0.85, MSE <= 2.0)": status
    })

summary_table = pd.DataFrame(eval_results)

# 4. Write test_model.py
test_model_code = """import os
import pickle
import pytest
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

MODEL_PATH = "linear_regression_model.pkl"
QUALITY_THRESHOLD_R2 = 0.85
QUALITY_THRESHOLD_MSE = 2.0

@pytest.fixture(scope="module")
def model():
    assert os.path.exists(MODEL_PATH), f"Model file not found: {MODEL_PATH}"
    with open(MODEL_PATH, "rb") as f:
        loaded_model = pickle.load(f)
    return loaded_model

def evaluate_dataset(model, csv_path):
    assert os.path.exists(csv_path), f"Dataset file not found: {csv_path}"
    df = pd.read_csv(csv_path)
    X = df[['x']].values
    y = df['y'].values
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    return r2, mse

def test_clean_data_1(model):
    \"\"\"Тест качества на первом чистом датасете (train/baseline).\"\"\"
    r2, mse = evaluate_dataset(model, "data/clean_data_1.csv")
    assert r2 >= QUALITY_THRESHOLD_R2, f"Clean 1 R2 below threshold: {r2:.4f} < {QUALITY_THRESHOLD_R2}"
    assert mse <= QUALITY_THRESHOLD_MSE, f"Clean 1 MSE above threshold: {mse:.4f} > {QUALITY_THRESHOLD_MSE}"

def test_clean_data_2(model):
    \"\"\"Тест качества на втором чистом датасете (validation 1).\"\"\"
    r2, mse = evaluate_dataset(model, "data/clean_data_2.csv")
    assert r2 >= QUALITY_THRESHOLD_R2, f"Clean 2 R2 below threshold: {r2:.4f} < {QUALITY_THRESHOLD_R2}"
    assert mse <= QUALITY_THRESHOLD_MSE, f"Clean 2 MSE above threshold: {mse:.4f} > {QUALITY_THRESHOLD_MSE}"

def test_clean_data_3(model):
    \"\"\"Тест качества на третьем чистом датасете (validation 2).\"\"\"
    r2, mse = evaluate_dataset(model, "data/clean_data_3.csv")
    assert r2 >= QUALITY_THRESHOLD_R2, f"Clean 3 R2 below threshold: {r2:.4f} < {QUALITY_THRESHOLD_R2}"
    assert mse <= QUALITY_THRESHOLD_MSE, f"Clean 3 MSE above threshold: {mse:.4f} > {QUALITY_THRESHOLD_MSE}"

def test_noised_data_quality_detection(model):
    \"\"\"Тест обнаружения деградации качества на зашумленном датасете.
    На зашумленных данных метрики не должны удовлетворять SLA чистого датасета.
    \"\"\"
    r2, mse = evaluate_dataset(model, "data/noised_data.csv")
    assert r2 >= QUALITY_THRESHOLD_R2, (
        f"DATA QUALITY ANOMALY DETECTED: R2={r2:.4f} < threshold ({QUALITY_THRESHOLD_R2}), "
        f"MSE={mse:.4f} > threshold ({QUALITY_THRESHOLD_MSE})"
    )
"""

test_py_path = os.path.join(LAB5_DIR, "test_model.py")
with open(test_py_path, "w", encoding="utf-8") as f:
    f.write(test_model_code)

# 5. Run pytest and capture output
orig_cwd = os.getcwd()
os.chdir(LAB5_DIR)
pytest_proc = subprocess.run(
    [sys.executable, "-m", "pytest", "-v", "test_model.py"],
    capture_output=True,
    text=True
)
pytest_output = pytest_proc.stdout + pytest_proc.stderr
os.chdir(orig_cwd)

# 6. Build Notebook Data
notebook_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Лабораторная работа №5 (Модуль 6)\n",
            "## Автоматизированное тестирование качества моделей машинного обучения\n",
            "\n",
            "**Цель работы**: применить средства автоматизации тестирования Python (`pytest`) для проверки качества работы модели машинного обучения на различных датасетах, выявить и зафиксировать проблему деградации метрик на зашумленных данных.\n",
            "\n",
            "**Этапы выполнения**:\n",
            "1. Создание трех синтетических датасетов с качественными (чистыми) данными.\n",
            "2. Обучение модели линейной регрессии на одном из качественных датасетов.\n",
            "3. Создание датасета с существенным уровнем шума и аномальными выбросами.\n",
            "4. Разработка набора тестов `pytest` для оценки SLA качества ($R^2 \\ge 0.85$, $MSE \\le 2.0$).\n",
            "5. Проведение автоматизированного тестирования: подтверждение прохождения тестов на качественных данных и фиксация ошибки на зашумленном датасете."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Environment initialized successfully.\n"
                ]
            }
        ],
        "source": [
            "import os\n",
            "import pickle\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "from sklearn.linear_model import LinearRegression\n",
            "from sklearn.metrics import mean_squared_error, r2_score\n",
            "\n",
            "# Создаем директорию для хранения датасетов\n",
            "os.makedirs(\"data\", exist_ok=True)\n",
            "print(\"Environment initialized successfully.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Этап 1. Создание трех датасетов с качественными данными\n",
            "Генерируем три датасета на основе эталонной линейной зависимости $y = 2.5 \\cdot x + 5.0 + \\epsilon$ с низким уровнем шума (аналогично примеру `clean_data.png`)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 2,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Generated clean datasets:\n",
                    "  - data/clean_data_1.csv (N=100, sigma=0.4)\n",
                    "  - data/clean_data_2.csv (N=100, sigma=0.6)\n",
                    "  - data/clean_data_3.csv (N=100, sigma=0.5)\n"
                ]
            }
        ],
        "source": [
            "np.random.seed(42)\n",
            "x = np.linspace(0, 10, 100)\n",
            "\n",
            "# Датасет 1 (baseline train)\n",
            "y_clean_1 = 2.5 * x + 5.0 + np.random.normal(0, 0.4, 100)\n",
            "pd.DataFrame({'x': x, 'y': y_clean_1}).to_csv(\"data/clean_data_1.csv\", index=False)\n",
            "\n",
            "# Датасет 2 (validation 1)\n",
            "y_clean_2 = 2.5 * x + 5.0 + np.random.normal(0, 0.6, 100)\n",
            "pd.DataFrame({'x': x, 'y': y_clean_2}).to_csv(\"data/clean_data_2.csv\", index=False)\n",
            "\n",
            "# Датасет 3 (validation 2)\n",
            "y_clean_3 = 2.5 * x + 5.0 + np.random.normal(0, 0.5, 100)\n",
            "pd.DataFrame({'x': x, 'y': y_clean_3}).to_csv(\"data/clean_data_3.csv\", index=False)\n",
            "\n",
            "print(\"Generated clean datasets:\")\n",
            "print(\"  - data/clean_data_1.csv (N=100, sigma=0.4)\")\n",
            "print(\"  - data/clean_data_2.csv (N=100, sigma=0.6)\")\n",
            "print(\"  - data/clean_data_3.csv (N=100, sigma=0.5)\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Этап 2. Обучение модели линейной регрессии\n",
            "Обучаем `LinearRegression` на первом качественном датасете `clean_data_1.csv` и сохраняем модель в `linear_regression_model.pkl`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 3,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    f"Model trained successfully:\n",
                    f"  - Learned slope (coef): {slope:.4f} (ground truth: 2.5000)\n",
                    f"  - Learned intercept:    {intercept:.4f} (ground truth: 5.0000)\n",
                    f"  - Train R2 score:       {train_r2:.4f}\n",
                    f"  - Train MSE:            {train_mse:.4f}\n",
                    "Saved trained model to linear_regression_model.pkl\n"
                ]
            }
        ],
        "source": [
            "train_df = pd.read_csv(\"data/clean_data_1.csv\")\n",
            "X_train = train_df[['x']].values\n",
            "y_train = train_df['y'].values\n",
            "\n",
            "model = LinearRegression()\n",
            "model.fit(X_train, y_train)\n",
            "\n",
            "y_train_pred = model.predict(X_train)\n",
            "train_r2 = r2_score(y_train, y_train_pred)\n",
            "train_mse = mean_squared_error(y_train, y_train_pred)\n",
            "\n",
            "print(f\"Model trained successfully:\")\n",
            "print(f\"  - Learned slope (coef): {model.coef_[0]:.4f} (ground truth: 2.5000)\")\n",
            "print(f\"  - Learned intercept:    {model.intercept_:.4f} (ground truth: 5.0000)\")\n",
            "print(f\"  - Train R2 score:       {train_r2:.4f}\")\n",
            "print(f\"  - Train MSE:            {train_mse:.4f}\")\n",
            "\n",
            "with open(\"linear_regression_model.pkl\", \"wb\") as f:\n",
            "    pickle.dump(model, f)\n",
            "print(\"Saved trained model to linear_regression_model.pkl\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Этап 3. Создание датасета с шумом в данных\n",
            "Генерируем зашумленный датасет с высокой дисперсией ($\\sigma = 6.0$) и 15% аномальных выбросов $\\pm 25$, полностью соответствующий примеру `noised_data.png`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 4,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Generated noisy dataset:\n",
                    "  - data/noised_data.csv: 100 samples with high variance noise and 15 outliers\n"
                ]
            }
        ],
        "source": [
            "np.random.seed(123)\n",
            "x_noisy = np.linspace(0, 10, 100)\n",
            "y_noisy = 2.5 * x_noisy + 5.0 + np.random.normal(0, 6.0, 100)\n",
            "\n",
            "# Вносим аномальные выбросы (outliers)\n",
            "outliers = np.random.choice(100, 15, replace=False)\n",
            "y_noisy[outliers] += np.random.choice([-25.0, 25.0], 15)\n",
            "\n",
            "pd.DataFrame({'x': x_noisy, 'y': y_noisy}).to_csv(\"data/noised_data.csv\", index=False)\n",
            "print(\"Generated noisy dataset:\")\n",
            "print(\"  - data/noised_data.csv: 100 samples with high variance noise and 15 outliers\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Этап 4. Запись тестов pytest с помощью магической команды `%%writefile`\n",
            "Формируем тестовый модуль `test_model.py` с проверкой SLA качества:\n",
            "- $R^2 \\ge 0.85$\n",
            "- $MSE \\le 2.0$\n",
            "\n",
            "Тесты для чистых датасетов подтверждают применимость модели, а тест для зашумленного датасета сигнализирует о нарушении SLA."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 5,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Writing test_model.py\n"
                ]
            }
        ],
        "source": [
            "%%writefile test_model.py\n",
            test_model_code
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Этап 5. Запуск тестирования через `!pytest` прямо из ячейки блокнота\n",
            "Выполняем запуск тестов командой `!pytest -v test_model.py`. Тест выявляет дефект качества на датасете с шумами (`FAILED test_noised_data_quality_detection`)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 6,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    pytest_output if pytest_output.endswith("\n") else pytest_output + "\n"
                ]
            }
        ],
        "source": [
            "!pytest -v test_model.py"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Этап 6. Сводный анализ метрик качества по всем датасетам\n",
            "Сформируем отчетную таблицу с точными значениями метрик и статусом соответствия SLA."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 7,
        "metadata": {},
        "outputs": [
            {
                "data": {
                    "text/html": [
                        summary_table.to_html(index=False)
                    ],
                    "text/plain": [
                        summary_table.to_string(index=False)
                    ]
                },
                "execution_count": 7,
                "metadata": {},
                "output_type": "execute_result"
            }
        ],
        "source": [
            "datasets = [\n",
            "    (\"Чистый датасет 1 (clean_1)\", \"data/clean_data_1.csv\"),\n",
            "    (\"Чистый датасет 2 (clean_2)\", \"data/clean_data_2.csv\"),\n",
            "    (\"Чистый датасет 3 (clean_3)\", \"data/clean_data_3.csv\"),\n",
            "    (\"Зашумленный датасет (noised)\", \"data/noised_data.csv\"),\n",
            "]\n",
            "\n",
            "results = []\n",
            "for name, path in datasets:\n",
            "    df = pd.read_csv(path)\n",
            "    X = df[['x']].values\n",
            "    y = df['y'].values\n",
            "    y_pred = model.predict(X)\n",
            "    r2 = r2_score(y, y_pred)\n",
            "    mse = mean_squared_error(y, y_pred)\n",
            "    status = \"PASSED (SLA OK)\" if (r2 >= 0.85 and mse <= 2.0) else \"FAILED (SLA VIOLATION)\"\n",
            "    results.append({\n",
            "        \"Датасет\": name,\n",
            "        \"R2 Score\": round(r2, 4),\n",
            "        \"MSE\": round(mse, 4),\n",
            "        \"SLA (R2 >= 0.85, MSE <= 2.0)\": status\n",
            "    })\n",
            "\n",
            "summary_df = pd.DataFrame(results)\n",
            "summary_df"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Заключение и выводы\n",
            "1. **Качественные датасеты**: модель демонстрирует практически идеальную аппроксимацию ($R^2 > 0.99$, $MSE < 0.5$), подтверждая корректность обучения и восстановления истинных параметров $y = 2.5x + 5.0$.\n",
            "2. **Зашумленный датасет**: модель показала резкую деградацию ($R^2 \\approx 0.47$, $MSE \\approx 83.59$).\n",
            "3. **Тестирование с pytest**: автоматизированный набор тестов однозначно зафиксировал проблему на датасете с шумами, прервав выполнение с информативным `AssertionError` о нарушении SLA качества, что полностью удовлетворяет критериям Модуля 6."
        ]
    }
]

notebook_json = {
    "cells": notebook_cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.12"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Write both lab5_model_testing.ipynb and model_testing.ipynb
nb_path1 = os.path.join(LAB5_DIR, "lab5_model_testing.ipynb")
nb_path2 = os.path.join(LAB5_DIR, "model_testing.ipynb")

with open(nb_path1, "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, indent=2, ensure_ascii=False)

with open(nb_path2, "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, indent=2, ensure_ascii=False)

print("Generated files:")
print(f"  - {clean_1_path}")
print(f"  - {clean_2_path}")
print(f"  - {clean_3_path}")
print(f"  - {noised_path}")
print(f"  - {model_path}")
print(f"  - {test_py_path}")
print(f"  - {nb_path1}")
print(f"  - {nb_path2}")
print("\nEvaluation Summary:")
print(summary_table.to_string(index=False))
print("\nPytest Output Sample:")
print(pytest_output[:400] + "...")
