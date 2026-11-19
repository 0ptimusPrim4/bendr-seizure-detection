"""Полосовая фильтрация ЭЭГ (EEG) фильтром Баттерворта.

Используется двунаправленная фильтрация ``filtfilt``: у неё нулевой фазовый
сдвиг, поэтому метки приступов не съезжают относительно сигнала.
Параметры полосы и порядок читаются из раздела ``filter`` конфига.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

from src.config_lookup import require_section

_SFREQ_ATOL = 1e-3


class FilterError(ValueError):
    """Сигнал нельзя пропустить через полосовой фильтр из конфига."""


def apply_bandpass_filter(data: np.ndarray, sfreq: float, config: dict) -> np.ndarray:
    """Отфильтровать сигнал по оси времени.

    Args:
        data: Массив, последняя ось которого — время. Обычно ``[C, T]``.
        sfreq: Частота дискретизации, Гц.
        config: Полный конфиг или раздел ``filter``.

    Returns:
        Отфильтрованный массив той же формы, ``float32``.

    Raises:
        FilterError: Частота, полоса или длина сигнала не подходят для фильтра.
    """
    params = require_section(config, "filter")
    signal = np.asarray(data, dtype=np.float64)
    if signal.size == 0 or signal.shape[-1] == 0:
        raise FilterError("Нечего фильтровать: по оси времени нет отсчётов.")

    _check_sampling_rate(sfreq, config)
    order = int(params["order"])
    low_hz = float(params["low_hz"])
    high_hz = float(params["high_hz"])
    nyquist = float(sfreq) / 2.0
    if order < 1:
        raise FilterError(f"Порядок фильтра должен быть положительным, получено: {order}.")
    if not (0.0 < low_hz < high_hz < nyquist):
        raise FilterError(
            f"Полоса {low_hz}–{high_hz} Гц недопустима при частоте {sfreq} Гц "
            f"(Найквист {nyquist} Гц)."
        )

    coefficients_b, coefficients_a = butter(order, [low_hz, high_hz], btype="bandpass", fs=float(sfreq))
    padlen = 3 * max(len(coefficients_a), len(coefficients_b))
    if signal.shape[-1] <= padlen:
        raise FilterError(
            f"Для filtfilt нужно больше {padlen} отсчётов, в записи {signal.shape[-1]}."
        )

    filtered = filtfilt(coefficients_b, coefficients_a, signal, axis=-1)
    return np.ascontiguousarray(filtered, dtype=np.float32)


def _check_sampling_rate(sfreq: float, config: dict) -> None:
    expected = _expected_sfreq(config)
    if expected is None:
        if sfreq <= 0:
            raise FilterError(f"Частота дискретизации должна быть положительной, получено: {sfreq}.")
        return
    if abs(float(sfreq) - expected) > _SFREQ_ATOL:
        raise FilterError(
            f"Частота {sfreq} Гц не совпадает с sampling_rate_hz={expected} из конфига. "
            "Сначала приведите запись к целевой частоте."
        )


def _expected_sfreq(config: dict) -> float | None:
    if "sampling_rate_hz" in config and "window_sec" in config:
        return float(config["sampling_rate_hz"])
    data = config.get("data")
    if isinstance(data, dict) and "sampling_rate_hz" in data:
        return float(data["sampling_rate_hz"])
    return None
