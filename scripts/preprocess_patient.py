"""Проверка и предобработка одного пациента CHB-MIT.

Один скрипт закрывает два прежних входа.

Быстрое чтение сводки и одного EDF:

    python scripts/preprocess_patient.py --quick --patient chb01 --data-root data/raw

Полный конвейер одного файла: каналы, фильтр, окна, нормализация:

    python scripts/preprocess_patient.py --patient chb01 --edf data/raw/chb01/chb01_03.edf

Полный конвейер всех EDF пациента, которые лежат рядом со сводкой:

    python scripts/preprocess_patient.py --all --patient chb01 --data-root data/raw

В полном прогоне статистики нормализации считаются по окнам показанного
пациента, чтобы проверить форму тензора. В LOPO-CV тот же нормализатор
обучается только на train-фолде.
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
from src.preprocessing.edf_loader import EdfLoadError, EdfRecording, load_edf
from src.preprocessing.filtering import FilterError, apply_bandpass_filter
from src.preprocessing.normalization import EEGNormalizer, NormalizerError
from src.preprocessing.segmentation import SegmentationError, segment_and_label


def main(argv: list[str] | None = None) -> int:
    """Запустить быструю проверку или полный конвейер одного пациента.

    Returns:
        0, если запрошенное чтение удалось. 1 при ошибке разбора, чтения,
        флагов или отсутствующем обязательном EDF. 2, если сводки нет на диске.
    """
    parser = argparse.ArgumentParser(description="Проверка и предобработка одного пациента CHB-MIT.")
    parser.add_argument("--patient", default="chb01", help="Идентификатор пациента, например chb01.")
    parser.add_argument("--data-root", default=None, help="Каталог с пациентами. По умолчанию paths.raw_dir.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"), help="Путь к YAML-конфигу.")
    parser.add_argument("--quick", action="store_true", help="Только сводка и один EDF, без фильтра и окон.")
    parser.add_argument("--edf", default=None, help="Конкретный EDF: с --quick только чтение, иначе полный конвейер.")
    parser.add_argument("--all", action="store_true", help="Полный конвейер для всех найденных EDF пациента.")
    parser.add_argument(
        "--require-edf",
        action="store_true",
        help="В режиме --quick завершаться с ошибкой, если EDF не найден.",
    )
    args = parser.parse_args(argv)

    try:
        mode = _mode(args.quick, args.all, args.edf)
        config = _load_config(Path(args.config))
        target_sfreq = float(config["data"]["sampling_rate_hz"])
        raw_dir = _resolve_raw_dir(args.data_root, config)
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"Не удалось прочитать конфиг или режим: {exc}", file=sys.stderr)
        return 1

    patient = args.patient.strip().lower()
    summary_path = _locate_summary(raw_dir, patient)
    if summary_path is None:
        print(
            f"Сводка не найдена для {patient}. Ожидался файл "
            f"{raw_dir / patient / (patient + '-summary.txt')}.\n"
            "Скачайте CHB-MIT с https://physionet.org/content/chbmit/1.0.0/ "
            "и распакуйте каталог пациента в data/raw/.",
            file=sys.stderr,
        )
        return 2

    try:
        summary = parse_summary(summary_path, patient_id=patient)
    except (AnnotationParseError, OSError) as exc:
        print(f"Ошибка разбора сводки: {exc}", file=sys.stderr)
        return 1

    if mode == "quick":
        return _run_quick(summary, raw_dir, patient, args.edf, args.require_edf, target_sfreq)
    return _run_pipeline(summary, raw_dir, patient, None if mode == "all" else args.edf, config, target_sfreq)


def _mode(quick: bool, run_all: bool, edf: str | None) -> str:
    """Выбрать режим: ``quick``, ``all`` или ``one``."""
    if run_all and (quick or edf):
        raise ValueError("Флаг --all не совмещается с --quick и --edf.")
    if run_all:
        return "all"
    if quick:
        return "quick"
    if edf:
        return "one"
    raise ValueError("Укажите режим: --quick, --edf FILE или --all.")


def _run_quick(
    summary: PatientSummary,
    raw_dir: Path,
    patient: str,
    edf_argument: str | None,
    require_edf: bool,
    target_sfreq: float,
) -> int:
    print(_format_summary(summary))
    record = _select_record(summary, edf_argument)
    edf_path = _locate_edf(raw_dir, patient, edf_argument, record)
    if edf_path is None:
        expected = record.file_name if edf_argument is None else edf_argument
        message = (
            f"EDF не найден: {expected}. Сводка прочитана успешно. "
            f"Положите файл в {raw_dir / patient} и повторите команду."
        )
        if require_edf:
            print(message, file=sys.stderr)
            return 1
        print(message)
        return 0
    try:
        recording = load_edf(edf_path, target_sfreq)
    except (EdfLoadError, OSError, ValueError) as exc:
        print(f"Ошибка чтения EDF: {exc}", file=sys.stderr)
        return 1
    print(_format_recording(recording, record))
    return 0


def _run_pipeline(
    summary: PatientSummary,
    raw_dir: Path,
    patient: str,
    edf_argument: str | None,
    config: dict,
    target_sfreq: float,
) -> int:
    jobs = _jobs(summary, raw_dir, patient, edf_argument)
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


def _select_record(summary: PatientSummary, edf_argument: str | None) -> EdfRecord:
    if edf_argument:
        wanted = Path(edf_argument).name
        for record in summary.files:
            if record.file_name == wanted:
                return record
    with_seizures = summary.seizure_records()
    if with_seizures:
        return with_seizures[0]
    return summary.files[0]


def _locate_edf(
    raw_dir: Path,
    patient: str,
    edf_argument: str | None,
    record: EdfRecord,
) -> Path | None:
    if edf_argument:
        given = Path(edf_argument)
        candidates = (
            given if given.is_absolute() else Path.cwd() / given,
            raw_dir / patient / given.name,
            raw_dir / given.name,
        )
    else:
        candidates = (
            raw_dir / patient / record.file_name,
            raw_dir / record.file_name,
        )
    for candidate in candidates:
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


def _format_summary(summary: PatientSummary) -> str:
    rate = "не указана" if summary.sampling_rate_hz is None else f"{summary.sampling_rate_hz:g} Гц"
    lines = [
        f"Пациент: {summary.patient_id}",
        f"Сводка: {summary.source}",
        f"Частота в сводке: {rate}",
        f"Файлов: {summary.n_files}",
        f"Файлов с приступами: {summary.n_seizure_files}",
        f"Приступов всего: {summary.n_seizures}",
        "",
        f"{'Файл':<20} {'Приступов':>9}  Интервалы, с",
    ]
    for record in summary.seizure_records():
        intervals = "; ".join(f"{seizure.start_sec:g}–{seizure.end_sec:g}" for seizure in record.seizures)
        lines.append(f"{record.file_name:<20} {record.n_seizures:>9}  {intervals}")
    return "\n".join(lines)


def _format_recording(recording: EdfRecording, record: EdfRecord) -> str:
    lines = [
        "",
        f"EDF: {recording.path}",
        (
            f"Форма тензора [C, T]: [{recording.n_channels}, {recording.n_samples}], "
            f"исходная частота {recording.source_sfreq:g} Гц, "
            f"после приведения {recording.sfreq:g} Гц, "
            f"длительность {recording.duration_sec:.3f} с"
        ),
        "Каналы EDF:",
    ]
    for index, name in enumerate(recording.channel_names, start=1):
        lines.append(f"  {index:>2}. {name}")
    if record.channels and tuple(record.channels) != recording.channel_names:
        lines.append(
            "Внимание: список каналов в сводке и подписи в EDF различаются. "
            "Сопоставление каналов выполняется отдельным шагом."
        )
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
