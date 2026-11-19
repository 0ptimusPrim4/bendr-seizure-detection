"""Полный конвейер одного пациента: каналы, фильтр, окна, нормализация.

    python scripts/run_preprocessing.py --patient chb01 --data-root data/raw

Статистики здесь считаются по окнам этого же пациента — только чтобы
посмотреть форму. В LOPO fit надо звать на train, не на тесте.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.annotations import parse_summary
from src.preprocessing.channel_mapping import map_channels
from src.preprocessing.edf_loader import load_edf
from src.preprocessing.filtering import bandpass_filter
from src.preprocessing.normalization import Normalizer
from src.preprocessing.segmentation import make_windows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Предобработка одного пациента CHB-MIT.")
    parser.add_argument("--patient", default="chb01")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--edf", default=None, help="Один файл. Без флага берутся все EDF, которые нашлись.")
    args = parser.parse_args(argv)

    config = _load_config(args.config)
    target_sfreq = float(config["data"]["sampling_rate_hz"])
    raw_dir = _raw_dir(args.data_root, config)
    patient = args.patient.strip().lower()

    summary_path = _find_summary(raw_dir, patient)
    if summary_path is None:
        print(
            f"Сводка не найдена для {patient}. Ждал {raw_dir / patient / (patient + '-summary.txt')}.",
            file=sys.stderr,
        )
        return 2

    try:
        summary = parse_summary(summary_path, patient_id=patient)
    except (OSError, ValueError) as e:
        print(f"Ошибка разбора сводки: {e}", file=sys.stderr)
        return 1

    jobs = _jobs(summary, raw_dir, patient, args.edf)
    if not jobs:
        print(f"Для {patient} не найден ни один EDF в {raw_dir / patient}.", file=sys.stderr)
        return 1

    windows_list = []
    labels_list = []
    try:
        for record, edf_path in jobs:
            recording = load_edf(edf_path, target_sfreq)
            mapped = map_channels(recording.data, recording.channel_names, config)
            filtered = bandpass_filter(mapped, recording.sfreq, config)
            windows, labels = make_windows(filtered, recording.sfreq, record.seizures, config)
            if len(labels):
                windows_list.append(windows)
                labels_list.append(labels)
    except (OSError, ValueError, RuntimeError) as e:
        # FIXME: один битый EDF роняет весь прогон, пропускать файл пока не умеем
        print(f"Ошибка предобработки: {e}", file=sys.stderr)
        return 1

    if not labels_list:
        print(f"Пациент {patient}: после исключения постиктального периода не осталось целых окон.")
        return 0

    windows = np.concatenate(windows_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    # TODO: нормализатор учится на самом пациенте. Для LOPO это утечка, тут только проверка формы
    try:
        normalized = Normalizer(config).fit_transform(windows)
    except RuntimeError as e:
        print(f"Ошибка нормализации: {e}", file=sys.stderr)
        return 1
    channel_mean = normalized.mean(axis=(0, 2))
    channel_std = normalized.std(axis=(0, 2))

    print(f"Пациент: {patient}")
    print(f"Файлов прочитано: {len(jobs)}")
    print(f"Окна [N, C, T]: {list(windows.shape)}")
    print(f"Метки [N]: {list(labels.shape)}")
    print(f"Класс 0 (фон): {int(np.sum(labels == 0))}")
    print(f"Класс 1 (приступ): {int(np.sum(labels == 1))}")
    print(
        "После нормализации по этим окнам: "
        f"среднее по каналам от {channel_mean.min():.4f} до {channel_mean.max():.4f}, "
        f"стандартное отклонение от {channel_std.min():.4f} до {channel_std.max():.4f}."
    )
    print("В LOPO-CV fit нормализатора вызывается только на train-фолде, не на пациенте из теста.")
    return 0


def _load_config(path):
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"конфиг {config_path} должен быть словарём.")
    return loaded


def _raw_dir(data_root, config):
    if data_root is not None:
        path = Path(data_root)
        return path if path.is_absolute() else Path.cwd() / path
    configured = Path(str(config["paths"]["raw_dir"]))
    return configured if configured.is_absolute() else ROOT / configured


def _find_summary(raw_dir, patient):
    # NOTE: сводка лежит либо в каталоге пациента, либо рядом с edf, если скачали плоско
    for candidate in (raw_dir / patient / f"{patient}-summary.txt", raw_dir / f"{patient}-summary.txt"):
        if candidate.is_file():
            return candidate
    return None


def _jobs(summary, raw_dir, patient, edf_argument):
    if edf_argument:
        given = Path(edf_argument)
        candidates = (
            given if given.is_absolute() else Path.cwd() / given,
            raw_dir / patient / given.name,
            raw_dir / given.name,
        )
        path = next((item for item in candidates if item.is_file()), None)
        record = next((item for item in summary.files if path and item.file_name == path.name), None)
        if record is None or path is None:
            return []
        return [(record, path)]

    found = []
    for record in summary.files:
        for candidate in (raw_dir / patient / record.file_name, raw_dir / record.file_name):
            if candidate.is_file():
                found.append((record, candidate))
                break
    return found


if __name__ == "__main__":
    sys.exit(main())
