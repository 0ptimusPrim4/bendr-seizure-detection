# -*- coding: utf-8 -*-
"""
Назначение: загрузка YAML-конфигурации эксперимента и разрешение путей.
Дата: 23.11.2026
"""

from pathlib import Path
import yaml

# Корень проекта — каталог на уровень выше пакета bendr_seizure.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = PROJECT_ROOT / "configs" / "cpu_quick.yaml"


def load_config(path=None):
    """Читает YAML. По умолчанию — configs/cpu_quick.yaml в корне проекта."""
    cfg_path = Path(path) if path else DEFAULT_PATH
    if not cfg_path.is_absolute():
        cand = Path(cfg_path)
        if not cand.is_file():
            cand = PROJECT_ROOT / cfg_path
        cfg_path = cand
    with cfg_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Конфиг {cfg_path} должен быть словарём.")
    cfg["_config_path"] = str(cfg_path.resolve())
    resolve_paths(cfg, project_root=PROJECT_ROOT)
    return cfg


def resolve_paths(cfg, project_root=None):
    """Относительные пути из раздела paths приводятся к корню проекта."""
    root = Path(project_root or PROJECT_ROOT)
    paths = cfg.get("paths") or {}
    for key, val in list(paths.items()):
        if val is None or val == "":
            continue
        p = Path(val)
        if not p.is_absolute():
            paths[key] = str((root / p).resolve())
        else:
            paths[key] = str(p)
    cfg["paths"] = paths
    return cfg


def section(cfg, *names):
    """Достаёт вложенный раздел, например section(cfg, 'data', 'filter')."""
    cur = cfg
    for name in names:
        if not isinstance(cur, dict) or name not in cur:
            raise KeyError(f"В конфиге нет раздела «{'.'.join(names)}».")
        cur = cur[name]
    return cur
