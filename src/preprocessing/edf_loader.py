"""Чтение многоканальных записей ЭЭГ (EEG) в формате EDF через MNE.

Загрузчик возвращает сигнал в форме ``[C, T]`` (каналы, отсчёты) и приводит
частоту дискретизации (sampling rate) к значению из конфигурации. Имена каналов
читаются из заголовка EDF без переименования дубликатов: в CHB-MIT канал
``T8-P8`` встречается дважды, и это разные строки монтажа.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SFREQ_ATOL = 1e-3


class EdfLoadError(RuntimeError):
    """EDF-файл не удалось прочитать или привести к целевой частоте."""


@dataclass
class EdfRecording:
    """Сигнал одного EDF-файла после приведения частоты.

    Attributes:
        data: Массив ``float32`` формы ``[C, T]``.
        channel_names: Имена каналов в порядке строк ``data``. Дубликаты сохраняются.
        sfreq: Частота дискретизации после приведения, Гц.
        source_sfreq: Частота в исходном файле, Гц.
        path: Путь к прочитанному файлу.
    """

    data: np.ndarray
    channel_names: tuple[str, ...]
    sfreq: float
    source_sfreq: float
    path: Path

    @property
    def n_channels(self) -> int:
        """Число каналов."""
        return int(self.data.shape[0])

    @property
    def n_samples(self) -> int:
        """Число отсчётов на канал."""
        return int(self.data.shape[1])

    @property
    def duration_sec(self) -> float:
        """Длительность записи в секундах."""
        return self.n_samples / self.sfreq


def load_edf(path: str | Path, target_sfreq: float) -> EdfRecording:
    """Прочитать EDF и привести частоту дискретизации к ``target_sfreq``.

    Если файл уже записан на целевой частоте, передискретизация (resampling)
    не выполняется, чтобы не менять отсчёты. Иначе используется передискретизация
    MNE. Целевая частота задаётся вызывающим кодом из ``configs/default.yaml``
    и здесь не зафиксирована числом по умолчанию.

    Args:
        path: Путь к ``.edf``.
        target_sfreq: Желаемая частота дискретизации, Гц. Для CHB-MIT это 256.

    Returns:
        Запись с сигналом ``[C, T]`` и исходными именами каналов.

    Raises:
        FileNotFoundError: Файл не существует.
        ValueError: Целевая частота не является положительным числом.
        EdfLoadError: Файл повреждён, пуст или не приводится к целевой частоте.
    """
    edf_path = Path(path)
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF-файл не найден: {edf_path}")
    if not edf_path.is_file():
        raise EdfLoadError(f"Ожидался EDF-файл, а не каталог: {edf_path}")
    if not math.isfinite(target_sfreq) or target_sfreq <= 0:
        raise ValueError(
            "Целевая частота дискретизации должна быть положительным конечным числом, "
            f"получено: {target_sfreq}."
        )

    raw, source_sfreq = _read_with_mne(edf_path)
    try:
        if abs(source_sfreq - target_sfreq) > _SFREQ_ATOL:
            raw.resample(target_sfreq, verbose="ERROR")
        achieved = float(raw.info["sfreq"])
        if abs(achieved - target_sfreq) > max(_SFREQ_ATOL, target_sfreq * 1e-6):
            raise EdfLoadError(
                f"После чтения {edf_path.name} частота {achieved} Гц "
                f"не совпала с целевой {target_sfreq} Гц."
            )
        data = np.ascontiguousarray(raw.get_data(), dtype=np.float32)
        mne_names = tuple(str(name).strip() for name in raw.ch_names)
    finally:
        raw.close()

    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        raise EdfLoadError(f"EDF-файл {edf_path.name} не содержит сигнала.")

    channel_names = _read_original_labels(edf_path, fallback=mne_names)
    if len(channel_names) != data.shape[0]:
        raise EdfLoadError(
            f"В {edf_path.name} число имён каналов ({len(channel_names)}) "
            f"не совпало с числом строк сигнала ({data.shape[0]})."
        )

    return EdfRecording(
        data=data,
        channel_names=channel_names,
        sfreq=achieved,
        source_sfreq=source_sfreq,
        path=edf_path,
    )


def _read_with_mne(path: Path) -> tuple[object, float]:
    try:
        import mne
    except ImportError as exc:  # pragma: no cover - зависимость объявлена в requirements
        raise EdfLoadError(
            "Для чтения EDF нужен пакет mne. Установите зависимости: pip install -r requirements.txt"
        ) from exc

    try:
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    except Exception as exc:
        raise EdfLoadError(f"Не удалось прочитать EDF {path}: {exc}") from exc

    source_sfreq = float(raw.info["sfreq"])
    if not math.isfinite(source_sfreq) or source_sfreq <= 0:
        raw.close()
        raise EdfLoadError(f"В файле {path.name} некорректная частота дискретизации: {source_sfreq}.")
    return raw, source_sfreq


def _read_original_labels(path: Path, fallback: tuple[str, ...]) -> tuple[str, ...]:
    """Вернуть подписи каналов из заголовка EDF, сохраняя дубликаты.

    MNE делает имена уникальными и может переименовать второй ``T8-P8``.
    Для сопоставления каналов нужны исходные подписи, поэтому заголовок
    читается отдельно через pyedflib. Если заголовок недоступен, остаются
    имена MNE.
    """
    try:
        import pyedflib
    except ImportError:
        return fallback

    try:
        reader = pyedflib.EdfReader(str(path))
    except Exception:
        return fallback

    try:
        labels = tuple(
            reader.getLabel(index).replace("\x00", "").strip()
            for index in range(reader.signals_in_file)
        )
    except Exception:
        return fallback
    finally:
        reader.close()

    if not labels or any(not label for label in labels):
        return fallback
    return labels
