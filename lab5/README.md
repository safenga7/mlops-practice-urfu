# Лабораторная работа №5 (Модуль 6): Автоматизированное тестирование качества моделей машинного обучения

## 1. Цель и требования задания
**Цель задания**: применить средства автоматизации тестирования Python (`pytest`) для проверки качества работы модели машинного обучения на различных датасетах и выявления деградации метрик при поступлении зашумленных данных.

**Требования профессора**:
1. Создать три датасета с «качественными» данными на основе простой линейной зависимости (по аналогии с `lab5/clean_data.png`).
2. Обучить модель линейной регрессии на одном из этих качественных датасетов.
3. Создать датасет с существенным шумом и аномальными выбросами (по аналогии с `lab5/noised_data.png`).
4. Провести автоматизированное тестирование модели на всех датасетах с использованием `pytest`, анализируя качество предсказаний ($R^2$, $MSE$), и обнаружить проблему на датасете с шумами.
5. **Критерий сдачи**: задание должно быть полностью оформлено и выполнено в виде Jupyter Notebook с сохраненными результатами работы ячеек.

---

## 2. Структура каталога `lab5/`

```text
lab5/
├── clean_data.png                 # Эталонный образец качественных данных
├── noised_data.png                # Эталонный образец зашумленных данных
├── lab5_model_testing.ipynb       # Основной выполненный Jupyter Notebook со всеми выводами
├── model_testing.ipynb            # Дубликат для альтернативных соглашений именования
├── generate_lab5.py              # Скрипт генерации данных, модели и сборки блокнота
├── test_model.py                  # Модуль автоматических тестов pytest
├── linear_regression_model.pkl    # Сериализованная модель линейной регрессии
├── README.md                      # Данный отчет
└── data/
    ├── clean_data_1.csv           # Чистый датасет 1 (train/baseline, N=100, sigma=0.4)
    ├── clean_data_2.csv           # Чистый датасет 2 (validation 1, N=100, sigma=0.6)
    ├── clean_data_3.csv           # Чистый датасет 3 (validation 2, N=100, sigma=0.5)
    └── noised_data.csv            # Зашумленный датасет (N=100, sigma=6.0 + выбросы +/-25)
```

---

## 3. Описание синтетических данных и модели

### Истинная зависимость
$$y = 2.5 \cdot x + 5.0 + \epsilon$$

### Датасеты
1. **`clean_data_1.csv`**: $\epsilon \sim \mathcal{N}(0, 0.4^2)$, 100 точек на отрезке $x \in [0, 10]$.
2. **`clean_data_2.csv`**: $\epsilon \sim \mathcal{N}(0, 0.6^2)$, 100 точек на отрезке $x \in [0, 10]$.
3. **`clean_data_3.csv`**: $\epsilon \sim \mathcal{N}(0, 0.5^2)$, 100 точек на отрезке $x \in [0, 10]$.
4. **`noised_data.csv`**: $\epsilon \sim \mathcal{N}(0, 6.0^2)$ + 15% экстремальных аномальных выбросов с отклонением $\pm 25.0$.

### Обученная модель
Модель `LinearRegression` из `scikit-learn` обучена на `clean_data_1.csv`.
- Восстановленный наклон (slope): `2.4939` (истинный: `2.5000`)
- Восстановленное смещение (intercept): `5.0506` (истинный: `5.0000`)
- Коэффициент детерминации на обучении: $R^2 = 0.9976$
- MSE на обучении: $0.1304$

---

## 4. Сводная таблица метрик качества и SLA

Критерии качества работы модели (Service Level Agreement):
- $R^2 \ge 0.85$
- $MSE \le 2.0$

| Датасет | Назначение | $R^2$ Score | MSE | Результат проверки SLA |
| :--- | :--- | :---: | :---: | :---: |
| **`clean_data_1.csv`** | Обучение / Baseline | **0.9976** | **0.1304** | **PASSED (SLA OK)** |
| **`clean_data_2.csv`** | Валидация 1 | **0.9939** | **0.3273** | **PASSED (SLA OK)** |
| **`clean_data_3.csv`** | Валидация 2 | **0.9943** | **0.2983** | **PASSED (SLA OK)** |
| **`noised_data.csv`** | Тест с шумами и аномалиями | **0.2634** | **140.3200** | **FAILED (SLA VIOLATION)** |

---

## 5. Автоматизированные тесты `pytest`

Файл `test_model.py` содержит набор тестов:
- `test_clean_data_1(model)`: проверка качества на обучающем/эталонном датасете (`PASSED`).
- `test_clean_data_2(model)`: проверка обобщающей способности на качественном датасете 2 (`PASSED`).
- `test_clean_data_3(model)`: проверка обобщающей способности на качественном датасете 3 (`PASSED`).
- `test_noised_data_quality_detection(model)`: проверка устойчивости к некачественным данным. Тест падает с `AssertionError`, фиксируя критическое падение $R^2$ до 0.26 и рост MSE до 140.32 (`FAILED`).

### Протокол выполнения `pytest`:
```text
============================= test session starts ==============================
rootdir: /Users/ahmed/Documents/mlops_practice/lab5
collected 4 items

test_model.py::test_clean_data_1 PASSED                                  [ 25%]
test_model.py::test_clean_data_2 PASSED                                  [ 50%]
test_model.py::test_clean_data_3 PASSED                                  [ 75%]
test_model.py::test_noised_data_quality_detection FAILED                 [100%]

=================================== FAILURES ===================================
______________________ test_noised_data_quality_detection ______________________
    def test_noised_data_quality_detection(model):
        r2, mse = evaluate_dataset(model, "data/noised_data.csv")
>       assert r2 >= QUALITY_THRESHOLD_R2, (
            f"DATA QUALITY ANOMALY DETECTED: R2={r2:.4f} < threshold ({QUALITY_THRESHOLD_R2}), "
            f"MSE={mse:.4f} > threshold ({QUALITY_THRESHOLD_MSE})"
        )
E       AssertionError: DATA QUALITY ANOMALY DETECTED: R2=0.2634 < threshold (0.85), MSE=140.3200 > threshold (2.0)
E       assert 0.2633536239044285 >= 0.85

test_model.py:51: AssertionError
========================= 1 failed, 3 passed in 0.96s ==========================
```

---

## 6. Как запустить и проверить

### Вариант 1: Просмотр блокнота
Открыть `lab5/lab5_model_testing.ipynb` в VS Code, JupyterLab или Jupyter Notebook. Все ячейки уже выполнены и содержат сохраненные выводы и таблицы.

### Вариант 2: Запуск pytest из терминала
```bash
cd lab5
pytest -v test_model.py
```
