"""20 слотов BENDR из каналов CHB-MIT.

Кого выкинуть и куда положить — только из конфига, здесь нет своей таблицы.
"""

import numpy as np

from src.config_lookup import require_section


class MappingError(ValueError):
    """Каналы записи не лезут в словарь BENDR."""


def map_channels(data, original_names, config):
    """Выкидывает лишние каналы и раскладывает остальные по bendr_order.
    На выходе [20, T], если в конфиге 20 слотов.
    """
    channels = require_section(config, "channels")
    signal = np.asarray(data)
    if signal.ndim != 2:
        raise MappingError(f"Ожидался массив каналов формы [C, T], получено ndim={signal.ndim}.")
    names = tuple(str(name) for name in original_names)
    if signal.shape[0] != len(names):
        raise MappingError(
            f"Число строк сигнала ({signal.shape[0]}) не совпало с числом имён ({len(names)})."
        )
    if signal.shape[1] == 0:
        raise MappingError("В записи нет временных отсчётов.")

    order = tuple(str(name) for name in channels["bendr_order"])
    mapping = {str(slot): str(source) for slot, source in dict(channels["mapping"]).items()}
    missing_slots = [slot for slot in order if slot not in mapping]
    if missing_slots:
        raise MappingError("В channels.mapping нет слотов BENDR: " + ", ".join(missing_slots) + ".")

    available = _available_indexes(names, channels)
    on_missing = str(channels.get("on_missing", "error")).strip().lower()
    if on_missing not in {"error", "zero"}:
        raise MappingError(f"channels.on_missing должен быть «error» или «zero», получено: {on_missing}.")

    rows = []
    absent = []
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
        raise MappingError(
            "Не найдены каналы для слотов: " + "; ".join(absent) + f". В записи есть: {present}."
        )

    mapped = np.stack(rows, axis=0)
    expected = _expected_channels(config)
    if expected is not None and mapped.shape[0] != expected:
        raise MappingError(
            f"После сопоставления каналов {mapped.shape[0]}, а data.n_channels_out = {expected}."
        )
    return mapped


def _available_indexes(names, channels):
    drop = {_normalize_name(name) for name in channels.get("drop", [])}
    # NOTE: в chb01 T8-P8 записан дважды, берём первое вхождение
    drop_duplicates = bool(channels.get("drop_later_duplicates", False))
    available = {}
    for index, name in enumerate(names):
        key = _normalize_name(name)
        if not key or key in drop:
            continue
        if drop_duplicates and key in available:
            continue
        available[key] = index
    return available


def _expected_channels(config):
    data = config.get("data")
    if isinstance(data, dict) and "n_channels_out" in data:
        return int(data["n_channels_out"])
    if "n_channels_out" in config:
        return int(config["n_channels_out"])
    return None


def _normalize_name(name):
    # hack: в заголовках EDF пробелы и регистр гуляют, сравниваем сжатую строку
    return " ".join(str(name).strip().upper().split())


# TODO: SCALE в диссертации — не электрод, а статистика амплитуды. Пока кладём то, что прописано в mapping
# FIXME: пара FT9-FT10 выкидывается по имени из конфига. Другой монтаж с теми же электродами не поймаем
