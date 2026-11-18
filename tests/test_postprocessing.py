"""Заготовки тестов постобработки. Реализация — фаза 4."""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Сглаживание, группировка ложных срабатываний и событийные метрики реализуются на фазе 4.")
def test_postprocessing_placeholder() -> None:
    """Файл подключён к pytest и пока намеренно пропускается."""
