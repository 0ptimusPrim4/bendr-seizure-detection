"""Полный конвейер предобработки одного пациента CHB-MIT.

Цепочка: чтение EDF, сопоставление каналов, полосовой фильтр, окна с метками,
нормализация. Статистики нормализации в этом прогоне считаются по окнам
показанного пациента, чтобы проверить форму тензора. В LOPO-CV тот же
нормализатор обучается только на train-фолде.

Запуск из корня репозитория:

    python scripts/run_preprocessing.py --patient chb01 --edf data/raw/chb01/chb01_03.edf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.annotations import AnnotationParseError, EdfRecord, PatientSummary, parse_summary
from src.preprocessing.channel_mapping import ChannelMappingError, map_channels
from src.preprocessing.edf_loader import EdfLoadError, load_edf
from src.preprocessing.filtering import FilterError, apply_bandpass_filter
from src.preprocessing.normalization import EEGNormalizer, NormalizerError
from src.preprocessing.segmentation import SegmentationError, segment_and_label


def main(argv: list[str] | None = None) -> int:
    """Прогнать предобработку пациента и напечатать формы и баланс классов.

    Returns:
        0, если хотя бы один EDF обработан. 1 при ошибке чтения или конфига.
        2, если сводка не найдена.
    """
    parser = argparse.ArgumentParser(description="Полная предобработка одного пациента CHB-MIT.")
    parser.add_argument("--patient", default="chb01", help="Идентификатор пациента, например chb01.")
    parser.add_argument("--data-root", default=None, help="Каталог с пациентами. По умолчанию paths.raw_dir.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"), help="Путь к YAML-конфигу.")
    parser.add_argument("--edf", default=None, help="Один EDF вместо всех файлов пациента, которые есть на диске.")
    args = parser.parse_args(argv)

    try:
        config = _load_config(Path(args.config))
        target_sfreq = float(config["data"]["sampling_rate_hz"])
        raw_dir = _resolve_raw_dir(args.data_root, config)
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"Не удалось прочитать конфиг: {exc}", file=sys.stderr)
        return 1

    patient = args.patient.strip().lower()
    summary_path = _locate_summary(raw_dir, patient)
    if summary_path is None:
        print(
            f"Сводка не найдена для {patient}. Ожидался файл "
            f"{raw_dir / patient / (patient + '-summary.txt')}.",
            file=sys.stderr,
        )
        return 2

    try:
        summary = parse_summary(summary_path, patient_id=patient)
    except (AnnotationParseError, OSError) as exc:
        print(f"Ошибка разбора сводки: {exc}", file=sys.stderr)
        return 1

    jobs = _jobs(summary, raw_dir, patient, args.edf)
    if not jobs:
        print(f"Для {patient} не найден ни один EDF в {raw_dir / patient}.", file=sys.stderr)
        return 1

    windows_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    iterator = tqdm(jobs, desc=f"Предобработка {patient}", unit="файл") if len(jobs) > 1 else jobs
    try:
        for record, edf_path in iterator:
            recording = load_edf(edf_path, target_sfreq)
            mapped = map_channels(recording.data, recording.channel_names, config)
            filtered = apply_bandpass_filter(mapped, recording.sfreq, config)
            windows, labels = segment_and_label(filtered, recording.sfreq, record.seizures, config)
            if len(labels):
                windows_list.append(windows)
                labels_list.append(labels)
    except (ChannelMappingError, FilterError, SegmentationError, EdfLoadError, OSError, ValueError) as exc:
        print(f"Ошибка предобработки: {exc}", file=sys.stderr)
        return 1

    if not labels_list:
        print(f"Пациент {patient}: после исключения постиктального периода не осталось целых окон.")
        return 0

    windows = np.concatenate(windows_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    try:
        normalized = EEGNormalizer(config).fit_transform(windows)
    except NormalizerError as exc:
        print(f"Ошибка нормализации: {exc}", file=sys.stderr)
        return 1

    background = int(np.sum(labels == 0))
    seizure = int(np.sum(labels == 1))
    channel_mean = normalized.mean(axis=(0, 2))
    channel_std = normalized.std(axis=(0, 2))
    print(f"Пациент: {patient}")
    print(f"Файлов прочитано: {len(jobs)}")
    print(f"Окна [N, C, T]: {list(windows.shape)}")
    print(f"Метки [N]: {list(labels.shape)}")
    print(f"Класс 0 (фон): {background}")
    print(f"Класс 1 (приступ): {seizure}")
    print(
        "После нормализации по этим окнам: "
        f"среднее по каналам от {channel_mean.min():.4f} до {channel_mean.max():.4f}, "
        f"стандартное отклонение от {channel_std.min():.4f} до {channel_std.max():.4f}."
    )
    print(
        "В LOPO-CV fit нормализатора вызывается только на train-фолде, "
        "не на пациенте, оставленном для теста."
    )
    return 0


def _load_config(path: Path) -> dict:
    config_path = path if path.is_absolute() else ROOT / path
    with config_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"конфиг {config_path} должен быть словарём.")
    return loaded


def _resolve_raw_dir(data_root: str | None, config: dict) -> Path:
    if data_root is not None:
        path = Path(data_root)
        return path if path.is_absolute() else Path.cwd() / path
    configured = Path(str(config["paths"]["raw_dir"]))
    return configured if configured.is_absolute() else ROOT / configured


def _locate_summary(raw_dir: Path, patient: str) -> Path | None:
    for candidate in (raw_dir / patient / f"{patient}-summary.txt", raw_dir / f"{patient}-summary.txt"):
        if candidate.is_file():
            return candidate
    return None


def _jobs(
    summary: PatientSummary,
    raw_dir: Path,
    patient: str,
    edf_argument: str | None,
) -> list[tuple[EdfRecord, Path]]:
    if edf_argument:
        given = Path(edf_argument)
        candidates = (
            given if given.is_absolute() else Path.cwd() / given,
            raw_dir / patient / given.name,
            raw_dir / given.name,
        )
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        record = next((item for item in summary.files if item.file_name == path.name), None)
        if record is None or not path.is_file():
            return []
        return [(record, path)]

    found: list[tuple[EdfRecord, Path]] = []
    for record in summary.files:
        for candidate in (raw_dir / patient / record.file_name, raw_dir / record.file_name):
            if candidate.is_file():
                found.append((record, candidate))
                break
    return found


if __name__ == "__main__":
    sys.exit(main())
