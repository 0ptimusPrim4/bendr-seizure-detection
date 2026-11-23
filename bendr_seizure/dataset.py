# -*- coding: utf-8 -*-
"""
Назначение: набор окон ЭЭГ и взвешенная выборка для несбалансированных классов.
Дата: 23.11.2026

Вход тензора — [N, 20, 1024]. Балансировка — Weighted Random Sampling
(как в конфиге исходного репозитория). Доля обучающих данных 25/50/100 %
берётся стратифицированно.
"""

from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit


class WindowDataset(Dataset):
    """Окна [C, T] и метки 0/1."""

    def __init__(self, windows, labels):
        self.windows = np.ascontiguousarray(windows, dtype=np.float32)
        self.labels = np.ascontiguousarray(labels, dtype=np.int64)
        if len(self.windows) != len(self.labels):
            raise ValueError("Число окон и меток должно совпадать.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.windows[idx]), torch.tensor(self.labels[idx])


def take_fraction(windows, labels, fraction, seed):
    """Оставляет долю объектов, стратифицируя по метке, если это возможно."""
    fraction = float(fraction)
    if fraction >= 1.0:
        return windows, labels
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"Доля данных должна быть в (0, 1], получено {fraction}.")
    n = len(labels)
    if n < 2:
        return windows, labels
    counts = np.bincount(labels.astype(int), minlength=2)
    if counts.min() >= 2:
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=fraction, random_state=seed)
        idx, _ = next(splitter.split(np.zeros(n), labels))
    else:
        splitter = ShuffleSplit(n_splits=1, train_size=max(fraction, 1.0 / n), random_state=seed)
        idx, _ = next(splitter.split(np.zeros(n)))
    return windows[idx], labels[idx]


def make_loader(windows, labels, batch_size, shuffle=False, weighted=False, seed=42):
    """DataLoader; при weighted=True — выборка с весами, обратными частоте класса."""
    ds = WindowDataset(windows, labels)
    if len(ds) == 0:
        raise ValueError("Пустой набор окон.")
    generator = torch.Generator().manual_seed(int(seed))
    if weighted:
        counts = np.bincount(labels.astype(int), minlength=2).astype(np.float64)
        counts[counts == 0] = 1.0
        weights = 1.0 / counts[labels.astype(int)]
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(ds),
            replacement=True,
            generator=generator,
        )
        return DataLoader(ds, batch_size=batch_size, sampler=sampler)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=generator)
