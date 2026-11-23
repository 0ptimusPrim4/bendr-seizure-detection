# -*- coding: utf-8 -*-
"""
Назначение: постобработка меток окон — голосование, группировка ЛС, событийные метрики.
Дата: 23.11.2026

Реализация фазы 4 исходного репозитория: мажоритарное голосование в окне
нечётной длины (по умолчанию 3), группировка подряд идущих ложных тревог
и событийная чувствительность (приступ найден, если хотя бы одно его окно
получило метку «приступ»).
"""

from __future__ import annotations
import numpy as np


def majority_voting(labels, width=3):
    """Сглаживание: центр заменяется преобладающим классом в окне ширины width."""
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if width % 2 == 0 or width < 1:
        raise ValueError(f"Ширина окна голосования должна быть нечётной и > 0, получено {width}.")
    if len(y) == 0:
        return y.copy()
    half = width // 2
    padded = np.pad(y, (half, half), mode="edge")
    out = np.empty_like(y)
    for i in range(len(y)):
        window = padded[i:i + width]
        # При равенстве голосов оставляем исходную метку центра.
        ones = int(window.sum())
        zeros = width - ones
        if ones > zeros:
            out[i] = 1
        elif zeros > ones:
            out[i] = 0
        else:
            out[i] = y[i]
    return out


def group_false_positives(pred, true):
    """Подряд идущие ЛС без паузы из TN считаются одним событием.

    Возвращает число сгруппированных ложных тревог.
    """
    pred = np.asarray(pred, dtype=np.int64).reshape(-1)
    true = np.asarray(true, dtype=np.int64).reshape(-1)
    if pred.shape != true.shape:
        raise ValueError("Длины pred и true должны совпадать.")
    fp = (pred == 1) & (true == 0)
    groups = 0
    in_run = False
    for flag in fp:
        if flag and not in_run:
            groups += 1
            in_run = True
        elif not flag:
            in_run = False
    return groups


def event_hits(pred, window_starts_sec, window_sec, seizures):
    """Для каждого приступа: True, если хотя бы одно окно внутри интервала положительно.

    seizures — список пар (start_sec, end_sec).
    """
    pred = np.asarray(pred, dtype=np.int64).reshape(-1)
    starts = np.asarray(window_starts_sec, dtype=np.float64).reshape(-1)
    ends = starts + float(window_sec)
    hits = []
    for s0, s1 in seizures:
        overlap = (starts < float(s1)) & (ends > float(s0))
        hits.append(bool(np.any(pred[overlap] == 1)) if np.any(overlap) else False)
    return hits


def event_sensitivity(pred, window_starts_sec, window_sec, seizures):
    """Доля приступов, у которых есть хотя бы одно положительное окно."""
    hits = event_hits(pred, window_starts_sec, window_sec, seizures)
    if not hits:
        return None
    return float(np.mean(hits))
