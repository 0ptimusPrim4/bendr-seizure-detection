"""Нарезка ЭЭГ на окна и разметка приступов.

Окна не перекрываются. Длина окна и длина постиктального исключения
(post-ictal exclusion) берутся из раздела ``data`` конфига. Окно с приступом
сохраняется с меткой 1, даже если оно задевает начало постиктального периода:
иначе короткий хвост приступа пропал бы. Окно без приступа, попавшее в
15 минут после конца приступа, выбрасывается и не становится фоном.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.config_lookup import require_section
from src.preprocessing.annotations import SeizureInterval

_SFREQ_ATOL = 1e-3


class SegmentationError(ValueError):
    """Запись нельзя разрезать на окна с текущим конфигом."""


def segment_and_label(
    data: np.ndarray,
    sfreq: float,
    seizures: Sequence[SeizureInterval],
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Разрезать сигнал на окна и разметить пересечение с приступами.

    Args:
        data: Сигнал ``[C, T]`` после сопоставления каналов и фильтра.
        sfreq: Частота дискретизации, Гц.
        seizures: Интервалы приступов этого файла, в секундах от его начала.
        config: Полный конфиг или раздел ``data``.

    Returns:
        Кортеж ``(windows, labels)``. ``windows`` имеет форму ``[N, C, window]``,
        ``labels`` — форму ``[N]`` со значениями 0 (фон) и 1 (приступ).
        Неполное последнее окно отбрасывается. Если целых окон не осталось,
        ``N = 0``.

    Raises:
        SegmentationError: Частота, длина окна или форма сигнала не подходят.
    """
    section = require_section(config, "data")
    signal = np.asarray(data, dtype=np.float32)
    if signal.ndim != 2:
        raise SegmentationError(f"Ожидался массив [C, T], получено ndim={signal.ndim}.")
    n_channels, n_samples = signal.shape
    if n_channels == 0 or n_samples == 0:
        raise SegmentationError("В сигнале нет каналов или отсчётов.")

    _check_sampling_rate(sfreq, section)
    window = _window_samples(section, float(sfreq))
    exclusion = int(round(float(section["postictal_exclusion_min"]) * 60.0 * float(sfreq)))
    if exclusion < 0:
        raise SegmentationError("postictal_exclusion_min не может быть отрицательным.")

    seizure_ranges = _sample_ranges(seizures, float(sfreq), n_samples)
    exclusion_ranges = _exclusion_ranges(seizure_ranges, exclusion, n_samples)

    windows: list[np.ndarray] = []
    labels: list[int] = []
    for start in range(0, n_samples - window + 1, window):
        stop = start + window
        hits_seizure = _overlaps_any(start, stop, seizure_ranges)
        hits_exclusion = _overlaps_any(start, stop, exclusion_ranges)
        if hits_seizure:
            label = 1
        elif hits_exclusion:
            continue
        else:
            label = 0
        windows.append(signal[:, start:stop])
        labels.append(label)

    if not windows:
        empty = np.empty((0, n_channels, window), dtype=np.float32)
        return empty, np.empty((0,), dtype=np.int64)
    return np.stack(windows, axis=0), np.asarray(labels, dtype=np.int64)


def _window_samples(section: dict, sfreq: float) -> int:
    window_sec = float(section["window_sec"])
    if window_sec <= 0:
        raise SegmentationError(f"window_sec должен быть положительным, получено: {window_sec}.")
    exact = window_sec * sfreq
    window = int(round(exact))
    if window < 1 or abs(exact - window) > 1e-6:
        raise SegmentationError(
            f"Окно {window_sec} с при частоте {sfreq} Гц не даёт целое число отсчётов ({exact})."
        )
    return window


def _check_sampling_rate(sfreq: float, section: dict) -> None:
    if "sampling_rate_hz" not in section:
        if sfreq <= 0:
            raise SegmentationError(f"Частота дискретизации должна быть положительной, получено: {sfreq}.")
        return
    expected = float(section["sampling_rate_hz"])
    if abs(float(sfreq) - expected) > _SFREQ_ATOL:
        raise SegmentationError(
            f"Частота {sfreq} Гц не совпадает с sampling_rate_hz={expected} из конфига. "
            "Сначала приведите запись к целевой частоте."
        )


def _sample_ranges(
    seizures: Sequence[SeizureInterval],
    sfreq: float,
    n_samples: int,
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for seizure in seizures:
        start = int(np.floor(float(seizure.start_sec) * sfreq))
        end = int(np.ceil(float(seizure.end_sec) * sfreq))
        start = min(max(start, 0), n_samples)
        end = min(max(end, 0), n_samples)
        if end <= start:
            raise SegmentationError(
                f"Приступ {seizure.start_sec}–{seizure.end_sec} с не попадает в запись."
            )
        ranges.append((start, end))
    return ranges


def _exclusion_ranges(
    seizure_ranges: Sequence[tuple[int, int]],
    exclusion_samples: int,
    n_samples: int,
) -> list[tuple[int, int]]:
    """Постиктальные отрезки без отсчётов, которые сами принадлежат приступу."""
    raw = [(end, min(n_samples, end + exclusion_samples)) for _, end in seizure_ranges]
    raw = [(start, end) for start, end in raw if end > start]
    if not raw or not seizure_ranges:
        return raw
    return _subtract(raw, seizure_ranges)


def _subtract(
    source: Sequence[tuple[int, int]],
    cuts: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    pieces = list(source)
    for cut_start, cut_end in cuts:
        updated: list[tuple[int, int]] = []
        for start, end in pieces:
            if cut_end <= start or cut_start >= end:
                updated.append((start, end))
                continue
            if start < cut_start:
                updated.append((start, cut_start))
            if cut_end < end:
                updated.append((cut_end, end))
        pieces = updated
    return pieces


def _overlaps_any(start: int, stop: int, ranges: Sequence[tuple[int, int]]) -> bool:
    return any(start < end and stop > begin for begin, end in ranges)
