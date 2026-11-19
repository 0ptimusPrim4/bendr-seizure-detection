"""Запуск тонкой настройки одного фолда LOPO-CV.

Скрипт будет реализован на фазе 3.
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
