"""Загрузка базы CHB-MIT с PhysioNet.

Скрипт будет реализован на следующем шаге фазы 1. Пока файлы кладутся
в ``data/raw/chbXX/`` вручную, проверка запускается через
``scripts/preprocess_patient.py --quick``.
"""

from __future__ import annotations


def main() -> None:
    """Сообщить, что шаг конвейера ещё не реализован."""
    print(
        "Этот шаг ещё не реализован. Сейчас доступна проверка одного пациента:\n"
        "  python scripts/preprocess_patient.py --quick --patient chb01 --data-root data/raw"
    )


if __name__ == "__main__":
    main()
