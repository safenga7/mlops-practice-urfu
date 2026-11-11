# Лабораторная работа 2 — Jenkins ML-конвейер

Автоматизированный конвейер машинного обучения, управляемый через Jenkins.

## Набор данных

**Iris** — классический набор данных о трёх видах ирисов (Setosa, Versicolor, Virginica), 150 записей, 4 числовых признака.

Источник: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)

Если источник недоступен, скрипт `data_acquisition.py` завершится с ошибкой и подскажет, куда положить файл вручную (`data/raw/iris.data`).

## Структура проекта

```
lab2/
├── Jenkinsfile              # Декларативный Jenkins Pipeline
├── requirements.txt         # Зависимости Python
├── data_acquisition.py      # Скачивание данных из UCI
├── data_preprocessing.py    # Очистка, кодирование, масштабирование, split
├── model_training.py        # Обучение RandomForest, сохранение в pickle
├── model_evaluation.py      # Оценка модели на тестовых данных
└── README.md                # Этот файл
```

После запуска конвейера появятся:

```
lab2/
├── data/
│   ├── raw/iris.data                # Сырые данные
│   └── processed/
│       ├── train.csv                # Обучающая выборка
│       ├── test.csv                 # Тестовая выборка
│       ├── scaler.pkl               # Подогнанный StandardScaler
│       └── label_encoder.pkl        # Кодировщик меток
└── model.pkl                        # Обученная модель
```

## Что делает каждый скрипт

| Скрипт | Назначение |
|--------|------------|
| `data_acquisition.py` | Скачивает `iris.data` из UCI с помощью `urllib` (стандартная библиотека). Никакие ключи или токены не нужны. |
| `data_preprocessing.py` | Парсит CSV, кодирует виды ирисов числами, делит данные 80/20 (стратифицированно), масштабирует признаки StandardScaler (fit только на train). |
| `model_training.py` | Обучает `RandomForestClassifier` на train-данных, сохраняет модель в `model.pkl`. |
| `model_evaluation.py` | Загружает модель из `model.pkl`, предсказывает на test-данных, выводит classification report и accuracy. |

## Подготовка виртуальной машины Ubuntu (Apple Silicon Mac)

Курс предполагает виртуальную машину с Jenkins. На Mac с Apple Silicon можно использовать [UTM](https://mac.getutm.app/) (бесплатно) или [Parallels](https://www.parallels.com/) для запуска Ubuntu ARM64.

### 1. Создать ВМ

1. Скачать UTM из App Store или с сайта.
2. Скачать ISO-образ Ubuntu Server 22.04 ARM64 с [ubuntu.com](https://ubuntu.com/download/server/arm).
3. В UTM создать новую ВМ → «Virtualize» → Linux → выбрать ISO → выделить 2+ ГБ RAM, 20+ ГБ диск.
4. Установить Ubuntu, перезагрузить, убрать ISO из настроек ВМ.

### 2. Установить ПО в ВМ

```bash
sudo apt update && sudo apt upgrade -y

# Java (для Jenkins)
sudo apt install -y openjdk-17-jdk

# Jenkins
curl -fsSL https://pkg.jenkins.io/debian-stable/jenkins.io-2023.key | sudo tee /usr/share/keyrings/jenkins-keyring.asc > /dev/null
echo "deb [signed-by=/usr/share/keyrings/jenkins-keyring.asc] https://pkg.jenkins.io/debian-stable binary/" | sudo tee /etc/apt/sources.list.d/jenkins.list > /dev/null
sudo apt update
sudo apt install -y jenkins

# Python
sudo apt install -y python3 python3-pip git

# Запуск Jenkins
sudo systemctl enable jenkins
sudo systemctl start jenkins
```

Jenkins будет доступен на `http://<ip-адрес-вм>:8080`.

### 3. Начальная настройка Jenkins (выполняется вручную в браузере)

1. Открыть `http://<ip-адрес-вм>:8080` в браузере на Mac.
2. Ввести начальный пароль из файла: `sudo cat /var/lib/jenkins/secrets/initialAdminPassword`
3. Установить рекомендуемые плагины.
4. Создать администратора.

### 4. Создать Pipeline-задание в Jenkins (вручную)

1. **New Item** → ввести имя (например, `lab2-ml-pipeline`) → **Pipeline** → OK.
2. В настройках задания:
   - **Pipeline → Definition**: выбрать **Pipeline script from SCM**.
   - **SCM**: Git.
   - **Repository URL**: URL вашего репозитория (например, `https://github.com/<user>/mlops_practice.git`).
   - **Branch**: `*/main` (или ваша ветка).
   - **Script Path**: `lab2/Jenkinsfile`.
3. Нажать **Save**.
4. Нажать **Build Now**.

> Файлы `Jenkinsfile`, скрипты и `requirements.txt` уже подготовлены в этом репозитории. В Jenkins нужно только указать откуда их брать (шаги выше).

## Локальная проверка скриптов на macOS (без Jenkins)

```bash
cd lab2

# Установить зависимости
pip3 install -r requirements.txt

# Запустить скрипты по порядку
python3 data_acquisition.py
python3 data_preprocessing.py
python3 model_training.py
python3 model_evaluation.py
```

Ожидаемый вывод последнего скрипта:

```
Model test accuracy is: 1.000
```

## Примечание

Код написан с помощью AI-инструмента (Antigravity / Gemini). Jenkins не запускался в рамках генерации — Python-скрипты проверены только локально.
