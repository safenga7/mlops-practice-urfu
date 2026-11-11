#!/usr/bin/env bash
#
# Конвейер: последовательный запуск всех этапов ML-пайплайна.
# Промежуточные сообщения идут в stderr, в stdout — только итоговая метрика.

set -euo pipefail

cd "$(dirname "$0")"

echo "=== Этап 1: создание данных ===" >&2
python3 data_creation.py

echo "=== Этап 2: предобработка ===" >&2
python3 data_preprocessing.py

echo "=== Этап 3: обучение модели ===" >&2
python3 model_preparation.py

echo "=== Этап 4: тестирование модели ===" >&2
python3 model_testing.py
