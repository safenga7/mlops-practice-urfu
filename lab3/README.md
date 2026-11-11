# Лабораторная работа 3 — Микросервис машинного обучения в Docker

Развертывание микросервиса инференса модели машинного обучения на базе **FastAPI** и **Docker** (соответствует Модулю 3 курса «Автоматизация машинного обучения»).

---

## 1. Цель и требования задания

Согласно заданию Модуля 3:
1. Подготовить python-код для модели и микросервиса.
2. Создать `Dockerfile` (в репозитории используется оригинальное имя файла `dockerfile`).
3. Создать Docker-образ.
4. Запустить Docker-контейнер и проверить его работу через API.
5. *(Дополнительный плюс)* Использование `docker-compose`.

---

## 2. Архитектура и структура проекта

Микросервис использует модель **RandomForestClassifier** и скейлер **StandardScaler** для классификации ирисов (Iris dataset: Setosa, Versicolor, Virginica), наследуя логику и схему признаков из Лабораторной работы 2.

```text
lab3/
├── app.py                # FastAPI-приложение (эндпоинты /health, /predict, /predict/batch)
├── train_model.py        # Скрипт обучения и сериализации модели и скейлера
├── requirements.txt      # Зависимости (FastAPI, Uvicorn, Pydantic, Scikit-learn, NumPy)
├── dockerfile            # Dockerfile для сборки контейнера (на базе python:3.11-slim)
├── docker-compose.yml    # Конфигурация для оркестрации сервиса
└── README.md             # Данная документация с примерами и протоколом тестирования
```

---

## 3. Описание API

Базовый адрес: `http://localhost:8000` (документация Swagger UI доступна по адресу `http://localhost:8000/docs`).

### Эндпоинты

| Метод | Путь | Описание | Ожидаемый статус |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Информация о сервисе | `200 OK` |
| `GET` | `/health` | Проверка жизнеспособности (health check) | `200 OK` |
| `POST` | `/predict` | Инференс для одного объекта | `200 OK` / `422 Unprocessable` |
| `POST` | `/predict/batch` | Пакетный инференс для нескольких объектов | `200 OK` / `422 Unprocessable` |

### Схема входных данных (`/predict`)

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Все поля строго валидируются через Pydantic (`float`, значения строго > 0 и < 25 см). При передаче некорректных типов или выходе за границы диапазона сервис возвращает ошибку валидации `422 Unprocessable Entity`.

---

## 4. Сборка и запуск

### Вариант 1: С помощью Docker Compose (рекомендуется)

```bash
cd lab3
docker compose up -d --build
```

Остановка контейнера:
```bash
docker compose down
```

### Вариант 2: С помощью стандартного Docker CLI

Сборка образа:
```bash
docker build -t lab3-ml-service:latest -f dockerfile .
```

Запуск контейнера:
```bash
docker run -d --name lab3-ml-service -p 127.0.0.1:8000:8000 lab3-ml-service:latest
```

---

## 5. Протокол фактического тестирования

Все проверки выполнены внутри рабочего Docker-контейнера.

### 5.1. Проверка состояния сервиса (`GET /health`)

**Запрос**:
```bash
curl -s http://127.0.0.1:8000/health
```

**Фактический ответ**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "classes": [
    "setosa",
    "versicolor",
    "virginica"
  ]
}
```

---

### 5.2. Предсказание для класса Setosa

**Запрос**:
```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

**Фактический ответ**:
```json
{
  "prediction": 0,
  "species": "setosa",
  "probabilities": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  }
}
```

---

### 5.3. Предсказание для класса Versicolor

**Запрос**:
```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.9, "sepal_width": 3.0, "petal_length": 4.2, "petal_width": 1.5}'
```

**Фактический ответ**:
```json
{
  "prediction": 1,
  "species": "versicolor",
  "probabilities": {
    "setosa": 0.0,
    "versicolor": 0.98,
    "virginica": 0.02
  }
}
```

---

### 5.4. Предсказание для класса Virginica

**Запрос**:
```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 6.9, "sepal_width": 3.1, "petal_length": 5.4, "petal_width": 2.1}'
```

**Фактический ответ**:
```json
{
  "prediction": 2,
  "species": "virginica",
  "probabilities": {
    "setosa": 0.0,
    "versicolor": 0.0,
    "virginica": 1.0
  }
}
```

---

### 5.5. Пакетное предсказание (`POST /predict/batch`)

**Запрос**:
```bash
curl -s -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
      {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3}
    ]
  }'
```

**Фактический ответ**:
```json
{
  "predictions": [
    {
      "prediction": 0,
      "species": "setosa",
      "probabilities": {
        "setosa": 1.0,
        "versicolor": 0.0,
        "virginica": 0.0
      }
    },
    {
      "prediction": 2,
      "species": "virginica",
      "probabilities": {
        "setosa": 0.0,
        "versicolor": 0.0,
        "virginica": 1.0
      }
    }
  ]
}
```

---

### 5.6. Валидация некорректных входных данных

**Отрицательное значение признака (`sepal_length = -5.0`)**:
```bash
curl -s -w "\nHTTP_STATUS: %{http_code}\n" -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": -5.0, "sepal_width": 3.0, "petal_length": 1.4, "petal_width": 0.2}'
```
**Результат**:
```json
{"detail":[{"type":"greater_than","loc":["body","sepal_length"],"msg":"Input should be greater than 0","input":-5.0,"ctx":{"gt":0.0}}]}
HTTP_STATUS: 422
```

**Строковое значение вместо числа (`sepal_length = "abc"`)**:
```bash
curl -s -w "\nHTTP_STATUS: %{http_code}\n" -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": "abc", "sepal_width": 3.0, "petal_length": 1.4, "petal_width": 0.2}'
```
**Результат**:
```json
{"detail":[{"type":"float_parsing","loc":["body","sepal_length"],"msg":"Input should be a valid number, unable to parse string as a number","input":"abc"}]}
HTTP_STATUS: 422
```

---

## 6. Ресурсы и производительность
* **Базовый образ**: `python:3.11-slim`
* **Потребление памяти контейнером**: ~105 МиБ
* **CPU в режиме ожидания**: < 0.3%
* **Время отклика `/predict`**: < 5 мс
