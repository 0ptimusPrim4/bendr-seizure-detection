"""Проверка чтения одного пациента CHB-MIT.

Скрипт разбирает ``chbXX-summary.txt`` и, если рядом лежит EDF, читает одну
запись через MNE. Это первый сквозной контроль конвейера данных: без обучения
и без скачивания всей базы.

Запуск из корня репозитория:

    python scripts/check_patient.py --patient chb01 --data-root data/raw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.annotations import AnnotationParseError, EdfRecord, PatientSummary, parse_summary
from src.preprocessing.edf_loader import EdfLoadError, EdfRecording, load_edf


def main(argv: list[str] | None = None) -> int:
    """Разобрать сводку пациента и при наличии файла прочитать один EDF.

    Returns:
        0, если сводка прочитана. 1, если сводка или EDF повреждены либо
        запрошенный EDF обязателен и не найден. 2, если сводки нет на диске.
    """
    parser = argparse.ArgumentParser(
        description="Проверка чтения сводки приступов и одного EDF-файла CHB-MIT."
    )
    parser.add_argument("--patient", default="chb01", help="Идентификатор пациента, например chb01.")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Каталог с пациентами. По умолчанию paths.raw_dir из конфига.",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "default.yaml"),
        help="Путь к YAML-конфигу с целевой частотой дискретизации.",
    )
    parser.add_argument(
        "--edf",
        default=None,
        help="Какой EDF читать. По умолчанию — первый файл с приступом.",
    )
    parser.add_argument(
        "--require-edf",
        action="store_true",
        help="Завершаться с ошибкой, если EDF-файл не найден.",
    )
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
            f"{raw_dir / patient / (patient + '-summary.txt')}.\n"
            "Скачайте CHB-MIT с https://physionet.org/content/chbmit/1.0.0/ "
            "и распакуйте каталог пациента в data/raw/.",
            file=sys.stderr,
        )
        return 2

    try:
        summary = parse_summary(summary_path, patient_id=patient)
    except (AnnotationParseError, FileNotFoundError, OSError) as exc:
        print(f"Ошибка разбора сводки: {exc}", file=sys.stderr)
        return 1

    print(_format_summary(summary))

    record = _select_record(summary, args.edf)
    edf_path = _locate_edf(raw_dir, patient, args.edf, record)
    if edf_path is None:
        expected = record.file_name if args.edf is None else args.edf
        message = (
            f"EDF не найден: {expected}. Сводка прочитана успешно. "
            f"Положите файл в {raw_dir / patient} и повторите команду."
        )
        if args.require_edf:
            print(message, file=sys.stderr)
            return 1
        print(message)
        return 0

    try:
        recording = load_edf(edf_path, target_sfreq)
    except (EdfLoadError, FileNotFoundError, ValueError, OSError) as exc:
        print(f"Ошибка чтения EDF: {exc}", file=sys.stderr)
        return 1

    print(_format_recording(recording, record))
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
    candidates = (
        raw_dir / patient / f"{patient}-summary.txt",
        raw_dir / f"{patient}-summary.txt",
    )
    for candidate in candidates:
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
        intervals = "; ".join(
            f"{seizure.start_sec:g}–{seizure.end_sec:g}" for seizure in record.seizures
        )
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
