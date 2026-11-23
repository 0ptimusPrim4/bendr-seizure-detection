# -*- coding: utf-8 -*-
"""
Назначение: функция потерь SSWCE (Sensitivity-Specificity Weighted Cross-Entropy).
Дата: 23.11.2026

Формула из диссертации Benfenati (2023), ур. (5.9):
    L = L_CE + α (1 − specificity) + β (1 − sensitivity),
α = 0.3, β = 0.5. Метрики считаются мягко по вероятностям, чтобы шёл градиент.
"""

import torch
import torch.nn.functional as F
from torch import nn


class SensitivitySpecificityWeightedCrossEntropy(nn.Module):
    """CE с добавкой по специфичности и чувствительности."""

    def __init__(self, alpha=0.3, beta=0.5):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)

    @classmethod
    def from_config(cls, cfg):
        sec = cfg.get("loss", cfg)
        return cls(alpha=sec["alpha"], beta=sec["beta"])

    def forward(self, probs, target):
        # Голова уже выдаёт softmax, второй раз его не вызываем.
        log_probs = torch.log(probs.clamp(min=1e-7))
        ce = F.nll_loss(log_probs, target)
        positive = probs[:, 1]
        y = target.float()
        n_pos = y.sum().clamp(min=1.0)
        n_neg = (1.0 - y).sum().clamp(min=1.0)
        sensitivity = (positive * y).sum() / n_pos
        specificity = ((1.0 - positive) * (1.0 - y)).sum() / n_neg
        return ce + self.alpha * (1.0 - specificity) + self.beta * (1.0 - sensitivity)
