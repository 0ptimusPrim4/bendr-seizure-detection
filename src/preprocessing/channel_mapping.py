"""Сопоставление каналов CHB-MIT с 20 слотами предобучения BENDR.

Из записи выбрасываются каналы из ``channels.drop``. При
``drop_later_duplicates: true`` у повторяющегося имени остаётся первое
вхождение: в chb01 второй ``T8-P8`` не попадает в модель. Оставшиеся строки
раскладываются по ``channels.bendr_order`` через словарь ``channels.mapping``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.config_lookup import require_section


class ChannelMappingError(ValueError):
    """Каналы записи нельзя разложить по словарю BENDR."""


def map_channels(
    data: np.ndarray,
    original_names: tuple[str, ...] | list[str],
    config: dict,
) -> np.ndarray:
    """Привести сигнал к порядку каналов предобучения BENDR.

    Args:
        data: Массив ``[C, T]`` в исходном порядке электродов.
        original_names: Имена каналов, по одному на строку ``data``.
        config: Полный конфиг или раздел ``channels``.

    Returns:
        Массив ``[len(bendr_order), T]``. Для конфига по умолчанию это ``[20, T]``.

    Raises:
        ChannelMappingError: Несовпадение формы, пустой словарь или отсутствующий
            канал при ``on_missing: error``.
    """
    channels = require_section(config, "channels")
    signal = np.asarray(data)
    if signal.ndim != 2:
        raise ChannelMappingError(
            f"Ожидался массив каналов формы [C, T], получено ndim={signal.ndim}."
        )
    names = tuple(str(name) for name in original_names)
    if signal.shape[0] != len(names):
        raise ChannelMappingError(
            f"Число строк сигнала ({signal.shape[0]}) не совпало с числом имён ({len(names)})."
        )
    if signal.shape[1] == 0:
        raise ChannelMappingError("В записи нет временных отсчётов.")

    order = tuple(str(name) for name in channels["bendr_order"])
    mapping = {str(slot): str(source) for slot, source in dict(channels["mapping"]).items()}
    missing_slots = [slot for slot in order if slot not in mapping]
    if missing_slots:
        raise ChannelMappingError(
            "В channels.mapping нет слотов BENDR: " + ", ".join(missing_slots) + "."
        )

    available = _available_indexes(names, channels)
    on_missing = str(channels.get("on_missing", "error")).strip().lower()
    if on_missing not in {"error", "zero"}:
        raise ChannelMappingError(
            f"channels.on_missing должен быть «error» или «zero», получено: {on_missing}."
        )

    rows: list[np.ndarray] = []
    absent: list[str] = []
    for slot in order:
        source = mapping[slot]
        index = available.get(_normalize_name(source))
        if index is None:
            absent.append(f"{slot} ← {source}")
            rows.append(np.zeros(signal.shape[1], dtype=np.float32))
        else:
            rows.append(np.asarray(signal[index], dtype=np.float32))

    if absent and on_missing == "error":
        present = ", ".join(names) if names else "нет каналов"
        raise ChannelMappingError(
            "Не найдены каналы для слотов: "
            + "; ".join(absent)
            + f". В записи есть: {present}."
        )

    mapped = np.stack(rows, axis=0)
    expected = _expected_channels(config)
    if expected is not None and mapped.shape[0] != expected:
        raise ChannelMappingError(
            f"После сопоставления каналов {mapped.shape[0]}, а data.n_channels_out = {expected}."
        )
    return mapped


def _available_indexes(names: Sequence[str], channels: dict) -> dict[str, int]:
    drop = {_normalize_name(name) for name in channels.get("drop", [])}
    drop_duplicates = bool(channels.get("drop_later_duplicates", False))
    available: dict[str, int] = {}
    for index, name in enumerate(names):
        key = _normalize_name(name)
        if not key or key in drop:
            continue
        if drop_duplicates and key in available:
            continue
        available[key] = index
    return available


def _expected_channels(config: dict) -> int | None:
    data = config.get("data")
    if isinstance(data, dict) and "n_channels_out" in data:
        return int(data["n_channels_out"])
    if "n_channels_out" in config:
        return int(config["n_channels_out"])
    return None


def _normalize_name(name: str) -> str:
    return " ".join(str(name).strip().upper().split())
