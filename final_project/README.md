# Итоговый индивидуальный проект: Конвейер машинного обучения data-продукта (FastAPI + DVC + Jenkins + Docker)

## 1. Описание и цели проекта

**Тема проекта**: Разработка и автоматизация сквозного MLOps конвейера для data-продукта машинного обучения (REST API микросервис классификации).  
**Формат выполнения**: Индивидуальный проект (согласовано с преподавателем).

Проект полностью удовлетворяет всем 7 критериям официального технического задания:
1. **Репозиторий Git**: исходный код, история коммитов, теги и версионирование размещены в Git.
2. **Оркестрация CI/CD**: процесс автоматизирован через Jenkins (`final_project/Jenkinsfile`).
3. **Версионирование данных**: данные отслеживаются через `DVC` и синхронизированы с постоянным удаленным хранилищем (`.dvc_storage`).
4. **Ветвление в Git**: разработка функционала и версионирование данных проведены в изолированных ветках `feature/*` с последующим слиянием (`--no-ff`) в основную ветку.
5. **Тестирование**: в конвейер включены как модульные тесты API (`test_api.py`), так и тесты на качество данных (`test_data_quality.py`).
6. **Docker-контейнеризация**: итоговое приложение упаковано в Docker-образ `final-mlops-app:latest`, сборка и валидация происходят в конвейере.
7. **Предварительно обученная модель**: используется сериализованная оптимизированная модель классификации (`final_project/app/model/iris_classifier.pkl`, точность $96.7\%$).

---

## 2. Архитектура решения

```text
mlops_practice/
└── final_project/
    ├── README.md                      # Полная документация проекта
    ├── Jenkinsfile                    # Декларативный CI/CD пайплайн для Jenkins
    ├── Dockerfile                     # Сборка production Docker-образа
    ├── docker-compose.yml             # Локальный запуск сервиса
    ├── requirements.txt               # Зависимости сервиса и тестов
    ├── app/
    │   ├── __init__.py
    │   ├── main.py                    # FastAPI приложение (endpoints: /health, /predict, /metrics)
    │   └── model/
    │       ├── .gitignore             # Исключение для отслеживания модели в Git
    │       └── iris_classifier.pkl    # Предварительно обученная модель
    ├── data/
    │   ├── .gitignore                 # Исключение исходного CSV из Git (управляется DVC)
    │   ├── iris_dataset.csv.dvc       # DVC-метафайл (хеш MD5: 7aa5d8d3862191620f8597315bcaf81c)
    │   └── generate_dataset.py        # Скрипт генерации датасета с валидацией схемы
    ├── tests/
    │   ├── test_data_quality.py       # Тесты качества данных (schema, nulls, bounds, drift)
    │   └── test_api.py                # Модульные и интеграционные тесты API
    └── scripts/
        ├── run_pipeline.sh            # Сквозной скрипт локального выполнения конвейера
        └── test_service.sh            # Скрипт проверки live-эндпоинтов
```

---

## 3. Версионирование данных (DVC)

Датасет `iris_dataset.csv` версионируется утилитой **DVC**:
- **DVC файл**: `final_project/data/iris_dataset.csv.dvc`
- **MD5 хеш**: `7aa5d8d3862191620f8597315bcaf81c`
- **Размер**: 4,169 байт, 150 строк, 6 колонок
- **Удаленное хранилище по умолчанию (Portable DVC Remote)**: `dvc_storage/` (встроено в репозиторий, не требует учетных записей или доступа в интернет)
- **Локальный бэкап-remote**: `/Users/ahmed/.dvc_storage/mlops_practice` (настроен в `.dvc/config.local`)
- **Команды управления и воспроизведения**:
  ```bash
  # Извлечение датасета из встроенного хранилища:
  dvc pull

  # Проверка статуса синхронизации:
  dvc status

  # Альтернативная генерация датасета без DVC:
  python3 final_project/data/generate_dataset.py
  ```
  ```

---

## 4. Стратегия ветвления в Git

В соответствии с требованием 4, разработка велась в отдельных ветках:

1. **`feature/dataset-and-quality-tests`**:
   - Реализация датасета и его версионирование через DVC.
   - Разработка набора тестов на качество данных `test_data_quality.py`.
   - Слияние в `student/mlops-completion` (коммит `65e2be6`).

2. **`feature/api-service-and-docker`**:
   - Обучение и фиксация предварительно обученной модели `iris_classifier.pkl`.
   - Разработка FastAPI микросервиса `app/main.py`.
   - Модульные тесты `test_api.py`.
   - Создание `Dockerfile` и `docker-compose.yml`.
   - Сборка и смоук-тест Docker-образа `final-mlops-app:latest`.
   - Слияние в `student/mlops-completion` (коммит `aceaf5f`).

3. **`feature/ci-cd-orchestration`**:
   - Декларативный конвейер `Jenkinsfile`.
   - Скрипты автоматизации `run_pipeline.sh` и `test_service.sh`.
   - Итоговая документация `final_project/README.md`.

---

## 5. Система автоматизированного тестирования

Конвейер запускает две независимые группы тестов:

### 1. Тесты качества данных (`test_data_quality.py`)
- `test_dataset_schema`: валидация наличия всех колонок и их строгих типов данных.
- `test_dataset_completeness`: проверка отсутствия пропущенных значений (0 nulls).
- `test_feature_value_ranges`: проверка вхождения признаков в физически допустимые интервалы (длина/ширина лепестков и чашелистиков).
- `test_target_cardinality_and_balance`: проверка сбалансированности классов (ровно по 50 элементов).
- `test_statistical_distribution`: проверка отсутствия статистического дрейфа (data drift).

### 2. Модульные и интеграционные тесты API (`test_api.py`)
- `test_root_endpoint`: валидация статуса 200 и метаданных API.
- `test_healthcheck`: валидация готовности модели к инференсу (`model_loaded=True`).
- `test_metrics_endpoint`: проверка доступности метрик телеметрии.
- `test_predict_setosa`: валидация инференса на классе Setosa.
- `test_predict_versicolor`: валидация инференса на классе Versicolor.
- `test_predict_virginica`: валидация инференса на классе Virginica.
- `test_predict_batch`: валидация пакетной обработки запросов.
- `test_validation_negative_values`: валидация отсечения отрицательных чисел (HTTP 422).
- `test_validation_missing_field`: валидация неполного набора полей (HTTP 422).
- `test_validation_type_mismatch`: валидация строковых значений вместо float (HTTP 422).

---

## 6. Микросервис и Docker-образ

### Эндпоинты API:
- `GET /health` — проверка работоспособности сервиса и загрузки ML модели.
- `POST /predict` — одиночное предсказание с вероятностями по классам.
- `POST /predict/batch` — пакетное предсказание списка объектов.
- `GET /metrics` — оперативная телеметрия (количество запросов, аптайм).
- `GET /docs` — интерактивная документация Swagger UI.

### Сборка и запуск контейнера:
```bash
# Сборка образа
docker build -t final-mlops-app:latest final_project/

# Запуск контейнера
docker run -d --rm --name final-mlops-service -p 8000:8000 final-mlops-app:latest

# Проверка работоспособности
curl -s http://localhost:8000/health
```

---

## 7. Оркестрация в Jenkins CI/CD

В Jenkins пайплайн `final_project/Jenkinsfile` выполняет следующие стадии:
1. **Environment & Checkout**: фиксация окружения, версии Python и коммита.
2. **Install Pipeline Dependencies**: подготовка тестовых библиотек.
3. **DVC Data Verification**: верификация целостности метафайла данных и удаленного хранилища.
4. **Data Quality Tests**: запуск 5 проверок качества данных (`pytest`).
5. **API Unit & Integration Tests**: запуск 10 модульных тестов микросервиса (`pytest`).
6. **Docker Packaging Validation**: проверка манифестов и артефактов контейнеризации.

---

## 8. Как запустить конвейер целиком

### Локальный запуск скрипта полного цикла:
```bash
./final_project/scripts/run_pipeline.sh
```

### Запуск тестов сервиса:
```bash
./final_project/scripts/test_service.sh 8000
```
