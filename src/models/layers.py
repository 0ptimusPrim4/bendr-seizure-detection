"""Мелкие куски BENDR, которые неудобно держать внутри энкодера."""

import torch
from torch import nn


class ChannelDropout(nn.Module):
    """С вероятностью p зануляет канал целиком, по всей длине времени.

    Так модель привыкает к отвалившемуся электроду, а не к дырке в одном отсчёте.
    В eval ничего не трогает.
    """

    def __init__(self, p):
        super().__init__()
        if not 0.0 <= float(p) < 1.0:
            raise ValueError(f"channel dropout p должен быть в [0, 1), пришло {p}")
        self.p = float(p)

    def forward(self, x):
        # NOTE: маска формы [B, C, 1] — один бросок на весь временной ряд канала
        if not self.training or self.p == 0.0:
            return x
        keep = torch.bernoulli(x.new_full((x.shape[0], x.shape[1], 1), 1.0 - self.p))
        return x * keep


class RelativePosition(nn.Module):
    """Относительные позиции как в BENDR: групповая свёртка по времени и GELU.

    На вход и выход [B, T, C]. Ядро и groups берутся из конфига (25 и 16).
    """

    def __init__(self, channels, kernel=25, groups=16):
        super().__init__()
        if channels % groups != 0:
            raise ValueError(f"d_model={channels} не делится на groups={groups}")
        # TODO: padding=kernel//2 держит длину только для нечётного ядра. 25 — нечётное, ок
        self.conv = nn.Conv1d(channels, channels, kernel, padding=kernel // 2, groups=groups)
        self.act = nn.GELU()

    def forward(self, tokens):
        # tokens: [B, T, C] -> свёртка хочет [B, C, T]
        encoded = self.act(self.conv(tokens.transpose(1, 2)))
        # FIXME: на ~10 токенах после downsample 96x ядро 25 видит всю последовательность сразу
        if encoded.shape[-1] != tokens.shape[1]:
            encoded = encoded[..., : tokens.shape[1]]
        return encoded.transpose(1, 2)
