"""Тонкая настройка (fine-tuning) и перекрёстная проверка по пациентам."""

from src.training.lopo_cv import LopoSplit, LopoSplitError, WindowSubset, generate_lopo_splits

__all__ = [
    "LopoSplit",
    "LopoSplitError",
    "WindowSubset",
    "generate_lopo_splits",
]
