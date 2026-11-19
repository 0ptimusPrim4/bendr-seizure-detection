"""CE плюс штраф за плохую чувствительность и специфичность.

Loss = CE + alpha * (1 - specificity) + beta * (1 - sensitivity)
alpha и beta лежат в configs/default.yaml, сейчас 0.3 и 0.5.
"""

import torch
import torch.nn.functional as F
from torch import nn


class SensitivitySpecificityWeightedCrossEntropy(nn.Module):
    def __init__(self, config):
        super().__init__()
        section = config.get("loss", config)
        self.alpha = float(section["alpha"])
        self.beta = float(section["beta"])

    def forward(self, probs, target):
        """probs — уже softmax, [B, 2]. target — индексы классов, [B]."""
        # NOTE: голова уже сделала softmax, второй раз его не зовём
        log_probs = torch.log(probs.clamp(min=1e-7))
        ce = F.nll_loss(log_probs, target)

        # hack: считаем soft-metrics через вероятности, чтобы градиент шёл
        positive = probs[:, 1]
        y = target.float()
        n_pos = y.sum().clamp(min=1.0)
        n_neg = (1.0 - y).sum().clamp(min=1.0)
        sensitivity = (positive * y).sum() / n_pos
        specificity = ((1.0 - positive) * (1.0 - y)).sum() / n_neg

        # TODO: пустой класс в батче даёт знаменатель 1 и нулевой числитель. На батче 256 почти не всплывает
        return ce + self.alpha * (1.0 - specificity) + self.beta * (1.0 - sensitivity)
