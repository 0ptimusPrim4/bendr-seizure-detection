"""Полосовой Баттерворт. filtfilt, чтобы фаза не уехала и метки остались на месте."""

import numpy as np
from scipy.signal import butter, filtfilt

from src.config_lookup import require_section

# magic number, но в диссертации Benfenati так, не трогаем: допуск на частоту после ресемпла
_SFREQ_ATOL = 1e-3


class FilterError(ValueError):
    """Полосу из конфига на этот сигнал не наложить."""


def bandpass_filter(data, sfreq, config):
    """Фильтрует по последней оси. Полоса и порядок берутся из config['filter']."""
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

    b, a = butter(order, [low_hz, high_hz], btype="bandpass", fs=float(sfreq))
    # filtfilt просит запас по краям, иначе scipy кидает своё исключение без контекста
    padlen = 3 * max(len(a), len(b))
    if signal.shape[-1] <= padlen:
        raise FilterError(f"Для filtfilt нужно больше {padlen} отсчётов, в записи {signal.shape[-1]}.")

    # TODO: на часовых записях это заметно тормозит, потом можно резать по кускам
    # FIXME: гоняем float64 и сразу кладём обратно в float32 — лишняя копия
    return np.ascontiguousarray(filtfilt(b, a, signal, axis=-1), dtype=np.float32)


def _check_sampling_rate(sfreq, config):
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


def _expected_sfreq(config):
    # раздел filter иногда приходит уже вынутым, а частота лежит рядом в data
    if "sampling_rate_hz" in config and "window_sec" in config:
        return float(config["sampling_rate_hz"])
    data = config.get("data")
    if isinstance(data, dict) and "sampling_rate_hz" in data:
        return float(data["sampling_rate_hz"])
    return None
