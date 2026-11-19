"""Нарезка на окна без перекрытия и метки 0/1.

Постиктал после приступа выкидываем, но само окно с приступом оставляем,
даже если оно чуть залезает за end_sec. Иначе короткий хвост пропадёт.
"""

import numpy as np

from src.config_lookup import require_section

_SFREQ_ATOL = 1e-3


class SegmentationError(ValueError):
    """Окна из этой записи при таком конфиге не собрать."""


def make_windows(data, sfreq, seizures, config):
    """Режет [C, T] на [N, C, window] и метки [N]. Хвост короче окна выбрасывает."""
    section = require_section(config, "data")
    signal = np.asarray(data, dtype=np.float32)
    if signal.ndim != 2:
        raise SegmentationError(f"Ожидался массив [C, T], получено ndim={signal.ndim}.")
    n_channels, n_samples = signal.shape
    if n_channels == 0 or n_samples == 0:
        raise SegmentationError("В сигнале нет каналов или отсчётов.")

    _check_sampling_rate(sfreq, section)
    window = _window_samples(section, float(sfreq))
    # 15 минут в конфиге, здесь только перевод в отсчёты
    exclusion = int(round(float(section["postictal_exclusion_min"]) * 60.0 * float(sfreq)))
    if exclusion < 0:
        raise SegmentationError("postictal_exclusion_min не может быть отрицательным.")

    seizure_ranges = _sample_ranges(seizures, float(sfreq), n_samples)
    exclusion_ranges = _exclusion_ranges(seizure_ranges, exclusion, n_samples)

    windows = []
    labels = []
    for start in range(0, n_samples - window + 1, window):
        stop = start + window
        # NOTE: пересечение с приступом важнее постиктала, иначе метка 1 теряется
        if _overlaps_any(start, stop, seizure_ranges):
            label = 1
        elif _overlaps_any(start, stop, exclusion_ranges):
            continue
        else:
            label = 0
        windows.append(signal[:, start:stop])
        labels.append(label)

    if not windows:
        # FIXME: пустой результат — это (0, C, window), а не ошибка. Вызывающий может этого не ждать
        return np.empty((0, n_channels, window), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(windows, axis=0), np.asarray(labels, dtype=np.int64)


def _window_samples(section, sfreq):
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


def _check_sampling_rate(sfreq, section):
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


def _sample_ranges(seizures, sfreq, n_samples):
    ranges = []
    for seizure in seizures:
        start = min(max(int(np.floor(float(seizure.start_sec) * sfreq)), 0), n_samples)
        end = min(max(int(np.ceil(float(seizure.end_sec) * sfreq)), 0), n_samples)
        if end <= start:
            raise SegmentationError(
                f"Приступ {seizure.start_sec}–{seizure.end_sec} с не попадает в запись."
            )
        ranges.append((start, end))
    return ranges


def _exclusion_ranges(seizure_ranges, exclusion_samples, n_samples):
    """Постиктал после end, но куски, которые сами являются приступом, не выкидываем."""
    raw = [(end, min(n_samples, end + exclusion_samples)) for _, end in seizure_ranges]
    raw = [(start, end) for start, end in raw if end > start]
    if not raw or not seizure_ranges:
        return raw
    return _subtract(raw, seizure_ranges)


def _subtract(source, cuts):
    pieces = list(source)
    for cut_start, cut_end in cuts:
        updated = []
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


def _overlaps_any(start, stop, ranges):
    return any(start < end and stop > begin for begin, end in ranges)


# TODO: дырки после выкинутого постиктала не склеиваем, иначе секунды приступов поедут
