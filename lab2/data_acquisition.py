"""
Получение данных: скачивание набора данных Iris из UCI ML Repository.

Источник: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
Набор содержит 150 записей с 4 числовыми признаками и 1 целевым классом
(вид ириса: Setosa, Versicolor, Virginica).

Если источник недоступен, скрипт завершается с кодом 1 и сообщением об ошибке.
В таком случае можно положить файл iris.data вручную в lab2/data/raw/
и перезапустить пайплайн начиная со следующего этапа.
"""

import os
import sys
import urllib.request
import urllib.error

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    dest = os.path.join(raw_dir, "iris.data")

    print(f"Скачивание данных из {URL} ...", file=sys.stderr)
    try:
        urllib.request.urlretrieve(URL, dest)
    except (urllib.error.URLError, OSError) as e:
        print(
            f"ОШИБКА: не удалось скачать данные.\n"
            f"Причина: {e}\n"
            f"Убедитесь, что есть доступ в интернет, или положите iris.data "
            f"вручную в {raw_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Проверяем, что файл не пустой
    if os.path.getsize(dest) == 0:
        print("ОШИБКА: скачанный файл пуст.", file=sys.stderr)
        sys.exit(1)

    print(f"Данные сохранены в {dest}", file=sys.stderr)


if __name__ == "__main__":
    main()
