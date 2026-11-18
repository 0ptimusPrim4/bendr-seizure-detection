"""Разбор сводок приступов CHB-MIT (``chbXX-summary.txt``).

Из сводки извлекаются имена EDF-файлов, число приступов в каждом файле
и интервалы приступов в секундах от начала записи. Дополнительно сохраняются
частота дискретизации (sampling rate) и монтаж каналов, если они указаны:
у части пациентов монтаж меняется между файлами.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_SAMPLING_RE = re.compile(
    r"^Data Sampling Rate:\s*([+-]?\d+(?:\.\d+)?)\s*Hz\s*$",
    re.IGNORECASE,
)
_CHANNEL_RE = re.compile(r"^Channel\s+(\d+)\s*:\s*(.*?)\s*$", re.IGNORECASE)
_FILE_NAME_RE = re.compile(r"^File Name:\s*(.+?)\s*$", re.IGNORECASE)
_START_CLOCK_RE = re.compile(r"^File Start Time:\s*(.+?)\s*$", re.IGNORECASE)
_END_CLOCK_RE = re.compile(r"^File End Time:\s*(.+?)\s*$", re.IGNORECASE)
_N_SEIZURES_RE = re.compile(
    r"^Number of Seizures in File:\s*(\d+)\s*$",
    re.IGNORECASE,
)
_SEIZURE_START_RE = re.compile(
    r"^Seizure(?:\s+(\d+))?\s+Start Time:\s*([+-]?\d+(?:\.\d+)?)\s+seconds?\s*$",
    re.IGNORECASE,
)
_SEIZURE_END_RE = re.compile(
    r"^Seizure(?:\s+(\d+))?\s+End Time:\s*([+-]?\d+(?:\.\d+)?)\s+seconds?\s*$",
    re.IGNORECASE,
)
_SUMMARY_NAME_RE = re.compile(r"^(chb\d+)-summary$", re.IGNORECASE)


class AnnotationParseError(ValueError):
    """Сводка CHB-MIT не соответствует ожидаемому формату PhysioNet."""


@dataclass(frozen=True)
class SeizureInterval:
    """Один приступ внутри EDF-файла.

    Время считается в секундах от начала этого файла, как в сводке,
    а не по настенным часам.
    """

    index: int
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        """Длительность приступа в секундах."""
        return self.end_sec - self.start_sec


@dataclass(frozen=True)
class EdfRecord:
    """Описание одного EDF-файла из сводки пациента."""

    file_name: str
    start_time: str | None
    end_time: str | None
    n_seizures: int
    seizures: tuple[SeizureInterval, ...]
    channels: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.n_seizures != len(self.seizures):
            raise AnnotationParseError(
                f"В файле {self.file_name} число приступов ({self.n_seizures}) "
                f"не совпадает с числом интервалов ({len(self.seizures)})."
            )
        if self.n_seizures < 0:
            raise AnnotationParseError(
                f"В файле {self.file_name} число приступов отрицательное."
            )


@dataclass(frozen=True)
class PatientSummary:
    """Разобранная сводка одного пациента."""

    patient_id: str
    sampling_rate_hz: float | None
    files: tuple[EdfRecord, ...]
    source: str

    @property
    def n_files(self) -> int:
        """Число EDF-файлов в сводке."""
        return len(self.files)

    @property
    def n_seizure_files(self) -> int:
        """Число файлов, в которых есть хотя бы один приступ."""
        return sum(1 for record in self.files if record.n_seizures > 0)

    @property
    def n_seizures(self) -> int:
        """Суммарное число приступов по всем файлам."""
        return sum(record.n_seizures for record in self.files)

    def seizure_records(self) -> tuple[EdfRecord, ...]:
        """Файлы, содержащие приступы, в порядке сводки."""
        return tuple(record for record in self.files if record.n_seizures > 0)


def parse_summary(path: str | Path, patient_id: str | None = None) -> PatientSummary:
    """Прочитать ``chbXX-summary.txt`` с диска.

    Args:
        path: Путь к сводке. По имени ``chbXX-summary.txt`` определяется пациент,
            если ``patient_id`` не передан.
        patient_id: Идентификатор пациента, например ``chb01``. Если не задан,
            берётся из имени файла.

    Returns:
        Разобранная сводка.

    Raises:
        FileNotFoundError: Файл сводки не существует.
        AnnotationParseError: Файл повреждён или формат не распознан.
    """
    summary_path = Path(path)
    if not summary_path.exists():
        raise FileNotFoundError(f"Файл сводки не найден: {summary_path}")
    if not summary_path.is_file():
        raise AnnotationParseError(f"Ожидался файл сводки, а не каталог: {summary_path}")

    try:
        text = summary_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise AnnotationParseError(
            f"Сводка {summary_path} не в кодировке UTF-8: {exc.reason}"
        ) from exc

    resolved_id = patient_id if patient_id is not None else _patient_id_from_name(summary_path)
    return parse_summary_text(text, patient_id=resolved_id, source_name=str(summary_path))


def parse_summary_text(
    text: str,
    patient_id: str,
    source_name: str = "<memory>",
) -> PatientSummary:
    """Разобрать уже прочитанный текст сводки.

    Поддерживаются оба варианта разметки PhysioNet: одна пара
    ``Seizure Start/End Time`` и нумерованные ``Seizure N Start/End Time``.
    Блоки ``Channels in EDF Files`` и ``Channels changed`` привязываются
    к последующим файлам. Настенные часы начала и конца файла необязательны
    (в сводке chb24 их нет).

    Args:
        text: Содержимое сводки.
        patient_id: Идентификатор пациента, например ``chb17``.
        source_name: Имя источника для сообщений об ошибках.

    Returns:
        Разобранная сводка.

    Raises:
        AnnotationParseError: Текст не является корректной сводкой.
    """
    normalized_id = patient_id.strip().lower()
    if not normalized_id:
        raise AnnotationParseError(f"{source_name}: не задан идентификатор пациента.")

    sampling_rate_hz: float | None = None
    current_channels: list[str] = []
    collecting_channels = False
    builder: dict[str, object] | None = None
    files: list[EdfRecord] = []
    seen_names: set[str] = set()

    def fail(message: str) -> None:
        raise AnnotationParseError(f"{source_name}: {message}")

    def finish_open_file() -> None:
        nonlocal builder
        if builder is None:
            return
        record = _finish_file(builder, source_name)
        if record.file_name in seen_names:
            fail(f"файл {record.file_name} встречается в сводке больше одного раза.")
        seen_names.add(record.file_name)
        files.append(record)
        builder = None

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or set(stripped) <= {"*"}:
            continue

        lower = stripped.lower()
        if lower.startswith("channels in edf") or lower.startswith("channels changed"):
            finish_open_file()
            collecting_channels = True
            current_channels = []
            continue

        if collecting_channels:
            channel_match = _CHANNEL_RE.match(stripped)
            if channel_match is not None:
                channel_name = channel_match.group(2).strip()
                if not channel_name:
                    fail(f"строка {line_no}: пустое имя канала.")
                current_channels.append(channel_name)
                continue
            collecting_channels = False

        sampling_match = _SAMPLING_RE.match(stripped)
        if sampling_match is not None:
            value = float(sampling_match.group(1))
            if value <= 0:
                fail(f"строка {line_no}: частота дискретизации должна быть положительной.")
            if sampling_rate_hz is not None and sampling_rate_hz != value:
                fail(
                    "в сводке указаны разные частоты дискретизации: "
                    f"{sampling_rate_hz} Гц и {value} Гц."
                )
            sampling_rate_hz = value
            continue

        file_match = _FILE_NAME_RE.match(stripped)
        if file_match is not None:
            finish_open_file()
            file_name = file_match.group(1).strip()
            if not file_name:
                fail(f"строка {line_no}: пустое имя файла.")
            builder = {
                "file_name": file_name,
                "line_no": line_no,
                "start_time": None,
                "end_time": None,
                "n_seizures": None,
                "channels": tuple(current_channels),
                "starts": [],
                "ends": [],
            }
            continue

        if builder is None:
            if lower.startswith("seizure") or lower.startswith("number of seizures"):
                fail(f"строка {line_no}: запись о приступе встретилась до имени файла.")
            continue

        start_clock = _START_CLOCK_RE.match(stripped)
        if start_clock is not None:
            _set_once(builder, "start_time", start_clock.group(1).strip(), "время начала файла", line_no, fail)
            continue

        end_clock = _END_CLOCK_RE.match(stripped)
        if end_clock is not None:
            _set_once(builder, "end_time", end_clock.group(1).strip(), "время конца файла", line_no, fail)
            continue

        count_match = _N_SEIZURES_RE.match(stripped)
        if count_match is not None:
            if builder["n_seizures"] is not None:
                fail(f"строка {line_no}: число приступов указано повторно для {builder['file_name']}.")
            builder["n_seizures"] = int(count_match.group(1))
            continue

        seizure_start = _SEIZURE_START_RE.match(stripped)
        if seizure_start is not None:
            _append_endpoint(builder, "starts", seizure_start, line_no, fail)
            continue

        seizure_end = _SEIZURE_END_RE.match(stripped)
        if seizure_end is not None:
            _append_endpoint(builder, "ends", seizure_end, line_no, fail)
            continue

        if lower.startswith("seizure") or lower.startswith("file ") or lower.startswith("number of seizures"):
            fail(f"строка {line_no}: не разобрана строка сводки: {stripped}")

    finish_open_file()
    if not files:
        fail("в сводке нет ни одного EDF-файла.")

    return PatientSummary(
        patient_id=normalized_id,
        sampling_rate_hz=sampling_rate_hz,
        files=tuple(files),
        source=source_name,
    )


def _patient_id_from_name(path: Path) -> str:
    match = _SUMMARY_NAME_RE.match(path.stem)
    if match is None:
        raise AnnotationParseError(
            f"Не удалось определить пациента по имени «{path.name}». "
            "Ожидается файл вида chb01-summary.txt, либо передайте patient_id явно."
        )
    return match.group(1).lower()


def _set_once(
    builder: dict[str, object],
    key: str,
    value: str,
    title: str,
    line_no: int,
    fail: callable,
) -> None:
    if not value:
        fail(f"строка {line_no}: пустое поле «{title}».")
    if builder[key] is not None:
        fail(f"строка {line_no}: {title} указано повторно для {builder['file_name']}.")
    builder[key] = value


def _append_endpoint(
    builder: dict[str, object],
    key: str,
    match: re.Match[str],
    line_no: int,
    fail: callable,
) -> None:
    index_text = match.group(1)
    seconds = float(match.group(2))
    if seconds < 0:
        fail(f"строка {line_no}: время приступа не может быть отрицательным ({seconds}).")
    index = int(index_text) if index_text is not None else None
    if index is not None and index < 1:
        fail(f"строка {line_no}: номер приступа должен начинаться с 1.")
    endpoints: list[tuple[int | None, float, int]] = builder[key]  # type: ignore[assignment]
    endpoints.append((index, seconds, line_no))


def _finish_file(builder: dict[str, object], source_name: str) -> EdfRecord:
    file_name = str(builder["file_name"])
    n_seizures = builder["n_seizures"]
    if n_seizures is None:
        raise AnnotationParseError(
            f"{source_name}: для файла {file_name} не указано число приступов "
            f"(строка {builder['line_no']})."
        )
    seizures = _pair_seizures(
        file_name=file_name,
        n_seizures=int(n_seizures),
        starts=builder["starts"],  # type: ignore[arg-type]
        ends=builder["ends"],  # type: ignore[arg-type]
        source_name=source_name,
    )
    return EdfRecord(
        file_name=file_name,
        start_time=builder["start_time"],  # type: ignore[arg-type]
        end_time=builder["end_time"],  # type: ignore[arg-type]
        n_seizures=int(n_seizures),
        seizures=seizures,
        channels=builder["channels"],  # type: ignore[arg-type]
    )


def _pair_seizures(
    file_name: str,
    n_seizures: int,
    starts: list[tuple[int | None, float, int]],
    ends: list[tuple[int | None, float, int]],
    source_name: str,
) -> tuple[SeizureInterval, ...]:
    def fail(message: str) -> None:
        raise AnnotationParseError(f"{source_name}: файл {file_name}: {message}")

    if len(starts) != n_seizures or len(ends) != n_seizures:
        fail(
            f"заявлено приступов: {n_seizures}, "
            f"найдено начал: {len(starts)}, концов: {len(ends)}."
        )
    if n_seizures == 0:
        return ()

    indexes = [item[0] for item in starts] + [item[0] for item in ends]
    if all(index is None for index in indexes):
        intervals: list[SeizureInterval] = []
        for number, ((_, start_sec, _), (_, end_sec, _)) in enumerate(zip(starts, ends), start=1):
            _check_bounds(start_sec, end_sec, number, fail)
            intervals.append(SeizureInterval(index=number, start_sec=start_sec, end_sec=end_sec))
        return tuple(intervals)

    if any(index is None for index in indexes):
        fail("номера приступов указаны только у части интервалов.")

    start_map: dict[int, float] = {}
    for index, start_sec, line_no in starts:
        assert index is not None
        if index in start_map:
            fail(f"начало приступа {index} повторяется (строка {line_no}).")
        start_map[index] = start_sec

    end_map: dict[int, float] = {}
    for index, end_sec, line_no in ends:
        assert index is not None
        if index in end_map:
            fail(f"конец приступа {index} повторяется (строка {line_no}).")
        end_map[index] = end_sec

    expected = set(range(1, n_seizures + 1))
    found = set(start_map) | set(end_map)
    if set(start_map) != expected or set(end_map) != expected:
        fail(f"ожидались номера приступов 1..{n_seizures}, получены {sorted(found)}.")

    intervals = []
    for index in sorted(start_map):
        if index not in end_map:
            fail(f"для приступа {index} нет времени конца.")
        _check_bounds(start_map[index], end_map[index], index, fail)
        intervals.append(
            SeizureInterval(index=index, start_sec=start_map[index], end_sec=end_map[index])
        )
    missing_starts = sorted(set(end_map) - set(start_map))
    if missing_starts:
        fail(f"для приступов {missing_starts} нет времени начала.")
    return tuple(intervals)


def _check_bounds(start_sec: float, end_sec: float, index: int, fail: callable) -> None:
    if end_sec <= start_sec:
        fail(
            f"приступ {index} имеет неположительную длительность: "
            f"начало {start_sec} с, конец {end_sec} с."
        )
