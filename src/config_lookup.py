"""Доступ к разделам ``configs/default.yaml``.

Функции предобработки принимают весь конфиг. Фильтр, каналы и нормализация
лежат внутри ``data``, схема LOPO-CV — внутри ``training``. Тот же вызов
понимает и словарь одного раздела, если в нём уже есть нужные поля.
"""

from __future__ import annotations

_MARKERS: dict[str, tuple[str, ...]] = {
    "channels": ("mapping", "bendr_order"),
    "filter": ("order", "low_hz", "high_hz"),
    "normalization": ("method",),
    "lopo": ("val_fraction",),
    "data": ("window_sec", "sampling_rate_hz"),
}


def require_section(config: dict, name: str) -> dict:
    """Вернуть раздел конфига по имени.

    Args:
        config: Полный конфиг или уже вынутый раздел.
        name: Имя раздела: ``channels``, ``filter``, ``normalization``, ``lopo``, ``data``.

    Raises:
        KeyError: Раздел не найден.
        TypeError: Вместо словаря передано другое значение.
    """
    if not isinstance(config, dict):
        raise TypeError(f"Конфиг раздела «{name}» должен быть словарём.")
    markers = _MARKERS.get(name, ())
    if markers and any(marker in config for marker in markers):
        return config
    direct = config.get(name)
    if isinstance(direct, dict):
        return direct
    for parent in ("data", "training"):
        nested = config.get(parent)
        if isinstance(nested, dict):
            found = nested.get(name)
            if isinstance(found, dict):
                return found
    raise KeyError(f"В конфиге нет раздела «{name}».")
