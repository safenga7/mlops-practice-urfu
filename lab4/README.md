# Лабораторная работа 4 — Версионирование данных с помощью DVC

Практическое применение утилиты **Data Version Control (DVC)** для отслеживания и версионирования наборов данных (соответствует Модулю 4 курса «Автоматизация машинного обучения»).

---

## 1. Цель и требования задания

Согласно заданию Модуля 4:
1. Создать папку `lab4` в корне проекта.
2. Установить `git` и `dvc`. Настроить проект для работы с `git` и `dvc`.
3. Настроить удаленное хранилище файлов (remote storage).
4. Создать датасет о пассажирах «Титаника» (Titanic dataset).
5. **Версия 1**: Модифицировать датасет, оставив признаки класс (`Pclass`), пол (`Sex`) и возраст (`Age`). Сделать коммит в git и push в dvc.
6. **Версия 2**: Заполнить пропущенные значения (`NaN`) в признаке `Age` средним значением. Сделать коммит в git и push в dvc.
7. **Версия 3**: Создать новые признаки с использованием one-hot-encoding для строкового признака `Sex`. Сделать коммит в git и push в dvc.
8. Выполнить переключение между всеми созданными версиями датасета.
9. Подготовить подробный отчет.

---

## 2. Структура файлов

```text
lab4/
├── data/
│   ├── .gitignore             # Исключает titanic.csv из Git (создан DVC)
│   ├── titanic.csv            # Фактический файл данных (контролируется DVC)
│   └── titanic.csv.dvc        # Метафайл DVC с хешем и размером данных (отслеживается Git)
├── prepare_dataset.py         # Скрипт подготовки версий данных
├── .gitignore                 # Исключает локальный кэш сырых данных (raw_titanic.csv)
├── readme                     # Исходный файл-заглушка из репозитория
└── README.md                  # Данная документация с протоколом выполнения
```

---

## 3. Настройка окружения и DVC Remote

1. **Инициализация DVC**:
   ```bash
   dvc init
   dvc config core.analytics false
   ```

2. **Настройка удаленных хранилищ (Portable Storage & Local Backup)**:
   Для гарантированной воспроизводимости проекта при клонировании репозитория настроено встроенное переносимое хранилище `portable_remote`, а персональный путь к хосту вынесен в локальный конфигурационный файл `.dvc/config.local` (игнорируемый Git):
   
   * **Основной переносимый репозиторий (`.dvc/config`, зафиксирован в Git)**:
     ```ini
     [core]
         analytics = false
         remote = portable_remote
     ['remote "portable_remote"']
         url = ../dvc_storage
     ```
   * **Локальный разработческий бэкап (`.dvc/config.local`, не попадает в Git)**:
     ```ini
     ['remote "local_remote"']
         url = /Users/ahmed/.dvc_storage/mlops_practice
     ```

---

## 3.1. Инструкция по воспроизведению версий данных для проверяющего

После клонирования репозитория все версии данных восстанавливаются локально без необходимости внешних учетных записей или доступа в облако:

1. **Восстановление текущей версии данных (`v3.0`)**:
   ```bash
   dvc pull
   ```
   DVC автоматически извлечет `lab4/data/titanic.csv` и `final_project/data/iris_dataset.csv` из встроенного хранилища `dvc_storage/`.

2. **Переключение на исторические версии данных (`v1.0`, `v2.0`, `v3.0`)**:
   Для переключения между историческими тегами Git рекомендуется один раз наполнить локальный кэш DVC из встроенного хранилища:
   ```bash
   mkdir -p .dvc/cache/files
   rsync -a --ignore-existing dvc_storage/files/ .dvc/cache/files/
   # Либо через cp:
   # cp -Rf dvc_storage/files/* .dvc/cache/files/
   ```
   После этого переключение между версиями осуществляется стандартными командами:
   * **Версия 1** (`v1.0`):
     ```bash
     git checkout v1.0
     dvc checkout
     ```
   * **Версия 2** (`v2.0`):
     ```bash
     git checkout v2.0
     dvc checkout
     ```
   * **Версия 3** (`v3.0`):
     ```bash
     git checkout v3.0
     dvc checkout
     ```

3. **Альтернативное детерминированное воспроизведение без DVC**:
   Скрипт `lab4/prepare_dataset.py` позволяет напрямую сгенерировать идентичный байт-в-байт датасет для любой версии:
   ```bash
   python3 lab4/prepare_dataset.py --version 1
   python3 lab4/prepare_dataset.py --version 2
   python3 lab4/prepare_dataset.py --version 3
   ```

---

## 4. Версии датасета и фиксация в DVC / Git

### Версия 1: Базовый набор признаков (`Pclass`, `Sex`, `Age`)
* **Команда генерации**:
  ```bash
  python lab4/prepare_dataset.py --version 1
  ```
* **Характеристики данных**:
  * Столбцы: `['Pclass', 'Sex', 'Age']`
  * Количество строк: 891
  * Пропусков в `Age`: 177
* **DVC хеш**: `bbadc75974dd530969b353f83ca6f29c` (10 572 байт)
* **Git Commit**: `45cd94e` (`feat(lab4): v1 - initialize DVC and track initial Titanic dataset with Pclass, Sex, Age`)
* **Git Tag**: `v1.0`

---

### Версия 2: Заполнение пропусков в `Age` средним значением
* **Команда генерации**:
  ```bash
  python lab4/prepare_dataset.py --version 2
  ```
* **Характеристики данных**:
  * Столбцы: `['Pclass', 'Sex', 'Age']`
  * Пропуски заполнены средним: `mean(Age) = 29.70`
  * Пропусков в `Age`: 0
* **DVC хеш**: `3c38d48847fc0b553ef337bef299ee91` (13 581 байт)
* **Git Commit**: `7a65d92` (`feat(lab4): v2 - impute missing Age values with column mean`)
* **Git Tag**: `v2.0`

---

### Версия 3: One-Hot Encoding категориального признака `Sex`
* **Команда генерации**:
  ```bash
  python lab4/prepare_dataset.py --version 3
  ```
* **Характеристики данных**:
  * Столбцы: `['Pclass', 'Age', 'Sex_female', 'Sex_male']`
  * Пропусков в `Age`: 0
* **DVC хеш**: `6c4826385958fb72f7f52a6bbe7e3bb6` (12 078 байт)
* **Git Commit**: `c1c7eb5` (`feat(lab4): v3 - add one-hot encoding for categorical Sex feature`)
* **Git Tag**: `v3.0`

---

## 5. Протокол проверки операций DVC

### 5.1. Проверка `dvc push`
Все три версии успешно отправлены в хранилище `/Users/ahmed/.dvc_storage/mlops_practice/files/md5/`:
```text
/Users/ahmed/.dvc_storage/mlops_practice/files/md5/bb/adc75974dd530969b353f83ca6f29c
/Users/ahmed/.dvc_storage/mlops_practice/files/md5/3c/38d48847fc0b553ef337bef299ee91
/Users/ahmed/.dvc_storage/mlops_practice/files/md5/6c/4826385958fb72f7f52a6bbe7e3bb6
```

### 5.2. Проверка `dvc pull`
Удаление локального файла и успешное восстановление из удаленного хранилища:
```bash
rm lab4/data/titanic.csv
dvc pull
```
**Вывод**:
```text
A       lab4/data/titanic.csv
1 file added
```

### 5.3. Переключение между версиями (`git checkout` + `dvc checkout`)

#### Переключение на Версию 1:
```bash
git checkout v1.0
dvc checkout
```
**Фактическая проверка содержимого**:
```bash
python -c "import pandas as pd; df=pd.read_csv('lab4/data/titanic.csv'); print(df.columns.tolist(), 'NaN Age:', df['Age'].isna().sum())"
```
**Результат**:
```text
['Pclass', 'Sex', 'Age'] NaN Age: 177
```

#### Переключение на Версию 2:
```bash
git checkout v2.0
dvc checkout
```
**Результат**:
```text
['Pclass', 'Sex', 'Age'] NaN Age: 0
```

#### Переключение на Версию 3:
```bash
git checkout v3.0
dvc checkout
```
**Результат**:
```text
['Pclass', 'Age', 'Sex_female', 'Sex_male'] NaN Age: 0
```

---

## 6. Воспроизведение эксперимента

Для проверки преподавателем или проверяющим:
```bash
# 1. Клонировать репозиторий и перейти в ветку
git checkout student/mlops-completion

# 2. Восстановить нужную версию данных через DVC
git checkout v1.0
dvc checkout
head -n 5 lab4/data/titanic.csv

git checkout v2.0
dvc checkout
head -n 5 lab4/data/titanic.csv

git checkout v3.0
dvc checkout
head -n 5 lab4/data/titanic.csv
```
