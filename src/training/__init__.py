"""Тонкая настройка (fine-tuning) и перекрёстная проверка по пациентам."""

from src.training.lopo_cv import LopoSplit, LopoSplitError, WindowSubset, generate_lopo_splits
from src.training.loss import SensitivitySpecificityWeightedCrossEntropy
from src.training.trainer import EarlyStopping, Trainer

__all__ = [
    "EarlyStopping",
    "LopoSplit",
    "LopoSplitError",
    "SensitivitySpecificityWeightedCrossEntropy",
    "Trainer",
    "WindowSubset",
    "generate_lopo_splits",
]
