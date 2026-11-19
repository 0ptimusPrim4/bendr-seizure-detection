"""Чтение EDF. Сигнал наружу отдаём как [C, T]."""

import math
from pathlib import Path

import numpy as np

# NOTE: частота 256 Гц для CHB-MIT почти всегда такая, но лучше проверять
_SFREQ_ATOL = 1e-3


class EdfLoadError(RuntimeError):
    """Файл прочитался, но частота или форма сигнала никуда не годятся."""


class EdfRecording:
    """Контейнер для прочитанного EDF."""

    def __init__(self, data, channel_names, sfreq, source_sfreq, path):
        self.data = data
        self.channel_names = channel_names
        self.sfreq = sfreq
        self.source_sfreq = source_sfreq
        self.path = path

    @property
    def n_channels(self):
        return int(self.data.shape[0])

    @property
    def n_samples(self):
        return int(self.data.shape[1])

    @property
    def duration_sec(self):
        return self.n_samples / self.sfreq


def load_edf(path, target_sfreq):
    """Читает EDF и ресемплит к нужной частоте.
    Если частота уже правильная — пропускает ресемплинг, чтобы не портить отсчёты.
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
        # TODO: mne слишком медленный для больших файлов, перейти на pyedflib
        if abs(source_sfreq - target_sfreq) > _SFREQ_ATOL:
            raw.resample(target_sfreq, verbose="ERROR")
        achieved = float(raw.info["sfreq"])
        if abs(achieved - target_sfreq) > max(_SFREQ_ATOL, target_sfreq * 1e-6):
            raise EdfLoadError(
                f"После чтения {edf_path.name} частота {achieved} Гц "
                f"не совпала с целевой {target_sfreq} Гц."
            )
        data = np.ascontiguousarray(raw.get_data(), dtype=np.float32)
        mne_names = tuple(name.strip() for name in raw.ch_names)
    finally:
        raw.close()

    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        raise EdfLoadError(f"EDF-файл {edf_path.name} не содержит сигнала.")

    # hack: mne переименовывает дубликаты каналов, поэтому читаем заголовок отдельно через pyedflib
    channel_names = _read_original_labels(edf_path, fallback=mne_names)
    if len(channel_names) != data.shape[0]:
        raise EdfLoadError(
            f"В {edf_path.name} число имён каналов ({len(channel_names)}) "
            f"не совпало с числом строк сигнала ({data.shape[0]})."
        )

    return EdfRecording(data, channel_names, achieved, source_sfreq, edf_path)


def _read_with_mne(path):
    try:
        import mne
    except ImportError as e:
        raise EdfLoadError(
            "Для чтения EDF нужен пакет mne. Установите зависимости: pip install -r requirements.txt"
        ) from e

    try:
        # mne без verbose печатает warning на каждый дубликат T8-P8
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    except Exception as e:
        # FIXME: падает на файлах с битыми заголовками (встречаются в chb15)
        raise RuntimeError(f"Не могу прочитать {path}: {e}") from e

    source_sfreq = float(raw.info["sfreq"])
    if not math.isfinite(source_sfreq) or source_sfreq <= 0:
        raw.close()
        raise EdfLoadError(f"В файле {path.name} некорректная частота дискретизации: {source_sfreq}.")
    return raw, source_sfreq


def _read_original_labels(path, fallback):
    """Подписи из заголовка EDF. Если pyedflib не открыл файл — остаются имена MNE."""
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
