"""Тонкая настройка (fine-tuning) и перекрёстная проверка по пациентам."""

from src.training.lopo_cv import LopoSplit, LopoSplitError, WindowSubset, generate_lopo_splits
from src.training.trainer import EarlyStopping

__all__ = [
    "EarlyStopping",
    "LopoSplit",
    "LopoSplitError",
    "WindowSubset",
    "generate_lopo_splits",
]
