# -*- coding: utf-8 -*-
"""
Назначение: предобработка записей CHB-MIT для дообучения BENDR.
Дата: 23.11.2026

Конвейер сохранён из репозитория пользователя: разбор сводки, чтение EDF,
сопоставление 19 отведений со слотами 10–20, полосовой Баттерворт 0,5–50 Гц
порядка 5, окна 4 с без перекрытия, исключение 15 мин постиктала,
z-score по обучающей выборке. Канал SCALE считается как относительная
амплитуда (среднее модулей 19 каналов), что закрывает TODO исходного кода.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
from scipy.signal import butter, filtfilt

_SFREQ_ATOL = 1e-3
_FILE_RE = re.compile(r"^File Name:\s*(.+?)\s*$", re.I)
_N_SEIZ_RE = re.compile(r"^Number of Seizures in File:\s*(\d+)\s*$", re.I)
_START_RE = re.compile(r"^Seizure(?:\s+(\d+))?\s+Start Time:\s*([+-]?\d+(?:\.\d+)?)\s+seconds?\s*$", re.I)
_END_RE = re.compile(r"^Seizure(?:\s+(\d+))?\s+End Time:\s*([+-]?\d+(?:\.\d+)?)\s+seconds?\s*$", re.I)
_CH_RE = re.compile(r"^Channel\s+(\d+)\s*:\s*(.*?)\s*$", re.I)
_RATE_RE = re.compile(r"^Data Sampling Rate:\s*([+-]?\d+(?:\.\d+)?)\s*Hz\s*$", re.I)


@dataclass
class SeizureInterval:
    start_sec: float
    end_sec: float


@dataclass
class EdfRecord:
    file_name: str
    seizures: tuple


def parse_summary(path, patient_id=None):
    """Разбирает chbXX-summary.txt, возвращает (sfreq, список EdfRecord)."""
    text = Path(path).read_text(encoding="utf-8-sig")
    sfreq = None
    files = []
    current = None
    pending_start = None
    seizures = []
    n_expected = 0

    def flush():
        nonlocal current, seizures, n_expected, pending_start
        if current is None:
            return
        files.append(EdfRecord(current, tuple(seizures)))
        current = None
        seizures = []
        n_expected = 0
        pending_start = None

    for raw in text.splitlines():
        line = raw.strip()
        m = _RATE_RE.match(line)
        if m:
            sfreq = float(m.group(1))
            continue
        m = _FILE_RE.match(line)
        if m:
            flush()
            current = m.group(1).strip()
            continue
        m = _N_SEIZ_RE.match(line)
        if m and current is not None:
            n_expected = int(m.group(1))
            continue
        m = _START_RE.match(line)
        if m and current is not None:
            pending_start = float(m.group(2))
            continue
        m = _END_RE.match(line)
        if m and current is not None and pending_start is not None:
            seizures.append(SeizureInterval(pending_start, float(m.group(2))))
            pending_start = None
            continue
    flush()
    if sfreq is None:
        sfreq = 256.0
    return sfreq, files


def _edf_labels(path):
    """Подписи каналов из двоичного заголовка EDF (без pyedflib)."""
    with open(path, "rb") as fh:
        header = fh.read(256)
        n_sig = int(header[252:256].decode("ascii", "ignore").strip() or "0")
        labels = fh.read(n_sig * 16)
    return tuple(labels[i * 16:(i + 1) * 16].decode("ascii", "replace").replace("\x00", "").strip()
                 for i in range(n_sig))


def load_edf(path, target_sfreq):
    """Читает EDF через mne, при необходимости ресемплирует, отдаёт (data[C,T], names, sfreq)."""
    import mne
    path = Path(path)
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    try:
        src = float(raw.info["sfreq"])
        if abs(src - target_sfreq) > _SFREQ_ATOL:
            raw.resample(target_sfreq, verbose="ERROR")
        data = np.ascontiguousarray(raw.get_data(), dtype=np.float32)
        sfreq = float(raw.info["sfreq"])
    finally:
        raw.close()
    names = _edf_labels(path)
    if len(names) != data.shape[0]:
        names = tuple(ch.strip() for ch in mne.io.read_raw_edf(path, preload=False, verbose="ERROR").ch_names)
    return data, names, sfreq


def _norm_name(name):
    return " ".join(str(name).strip().upper().split())


def map_channels(data, names, cfg):
    """Раскладывает 19 биполярных отведений по слотам 10–20, SCALE считает отдельно."""
    ch = cfg["data"]["channels"]
    signal = np.asarray(data)
    drop = {_norm_name(n) for n in ch.get("drop", [])}
    available = {}
    for i, name in enumerate(names):
        key = _norm_name(name)
        if not key or key in drop:
            continue
        if ch.get("drop_later_duplicates", False) and key in available:
            continue
        available[key] = i
    eeg_slots = [s for s in ch["bendr_order"] if s != "SCALE"]
    mapping = {str(k): str(v) for k, v in ch["mapping"].items()}
    rows = []
    absent = []
    for slot in eeg_slots:
        source = mapping[slot]
        idx = available.get(_norm_name(source))
        if idx is None:
            absent.append(f"{slot} ← {source}")
            rows.append(np.zeros(signal.shape[1], dtype=np.float32))
        else:
            rows.append(np.asarray(signal[idx], dtype=np.float32))
    if absent and str(ch.get("on_missing", "error")).lower() == "error":
        raise ValueError("Не найдены каналы: " + "; ".join(absent))
    eeg = np.stack(rows, axis=0)
    # Относительная амплитуда: среднее модулей 19 каналов в каждый момент времени.
    scale = np.mean(np.abs(eeg), axis=0, keepdims=True).astype(np.float32)
    return np.concatenate([eeg, scale], axis=0)


def bandpass_filter(data, sfreq, cfg):
    """Прямой–обратный Баттерворт, чтобы не сдвигать метки по времени."""
    p = cfg["data"]["filter"]
    signal = np.asarray(data, dtype=np.float64)
    order, low, high = int(p["order"]), float(p["low_hz"]), float(p["high_hz"])
    b, a = butter(order, [low, high], btype="bandpass", fs=float(sfreq))
    padlen = 3 * max(len(a), len(b))
    if signal.shape[-1] <= padlen:
        raise ValueError(f"Слишком короткая запись для filtfilt: {signal.shape[-1]}.")
    return np.ascontiguousarray(filtfilt(b, a, signal, axis=-1), dtype=np.float32)


def _overlaps(start, stop, ranges):
    return any(start < e and stop > b for b, e in ranges)


def make_windows(data, sfreq, seizures, cfg, max_background=None, rng=None):
    """Нарезка без перекрытия. Постиктал отбрасывается, пересечение с приступом даёт метку 1."""
    section = cfg["data"]
    signal = np.asarray(data, dtype=np.float32)
    window = int(round(float(section["window_sec"]) * float(sfreq)))
    exclusion = int(round(float(section["postictal_exclusion_min"]) * 60.0 * float(sfreq)))
    n_samples = signal.shape[1]
    seiz_ranges = []
    for sz in seizures:
        a = min(max(int(np.floor(sz.start_sec * sfreq)), 0), n_samples)
        b = min(max(int(np.ceil(sz.end_sec * sfreq)), 0), n_samples)
        if b > a:
            seiz_ranges.append((a, b))
    excl = []
    for _, end in seiz_ranges:
        e0, e1 = end, min(n_samples, end + exclusion)
        if e1 > e0:
            excl.append((e0, e1))
    # Куски постиктала, совпадающие с другим приступом, не выкидываем как фон.
    windows, labels, starts = [], [], []
    bg_idx = []
    for start in range(0, n_samples - window + 1, window):
        stop = start + window
        if _overlaps(start, stop, seiz_ranges):
            labels.append(1)
            windows.append(signal[:, start:stop])
            starts.append(start / float(sfreq))
        elif _overlaps(start, stop, excl):
            continue
        else:
            bg_idx.append(len(windows))
            labels.append(0)
            windows.append(signal[:, start:stop])
            starts.append(start / float(sfreq))
    if not windows:
        c = signal.shape[0]
        return (np.empty((0, c, window), np.float32),
                np.empty((0,), np.int64),
                np.empty((0,), np.float64))
    windows = np.stack(windows, 0)
    labels = np.asarray(labels, dtype=np.int64)
    starts = np.asarray(starts, dtype=np.float64)
    if max_background is not None:
        bg = np.where(labels == 0)[0]
        if len(bg) > int(max_background):
            rng = rng or np.random.default_rng(0)
            keep_bg = np.sort(rng.choice(bg, size=int(max_background), replace=False))
            keep = np.sort(np.concatenate([np.where(labels == 1)[0], keep_bg]))
            windows, labels, starts = windows[keep], labels[keep], starts[keep]
    return windows, labels, starts


class Normalizer:
    """Z-score по каналам, статистики только с обучения."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, x_train):
        x = np.asarray(x_train, dtype=np.float64)
        if x.ndim != 3:
            raise ValueError("Ожидалась форма [N, C, T].")
        flat = np.moveaxis(x, 1, 0).reshape(x.shape[1], -1)
        self.mean_ = flat.mean(axis=1)
        std = flat.std(axis=1)
        std[std == 0.0] = 1.0
        self.std_ = std
        return self

    def transform(self, x):
        if self.mean_ is None:
            raise RuntimeError("Сначала вызовите fit.")
        values = np.asarray(x, dtype=np.float64)
        m = self.mean_[None, :, None]
        s = self.std_[None, :, None]
        return np.ascontiguousarray((values - m) / s, dtype=np.float32)


def locate_summary(raw_dir, patient):
    raw_dir = Path(raw_dir)
    for cand in (raw_dir / patient / f"{patient}-summary.txt",
                 raw_dir / f"{patient}-summary.txt"):
        if cand.is_file():
            return cand
    return None


def locate_edf(raw_dir, patient, file_name):
    raw_dir = Path(raw_dir)
    name = Path(file_name).name
    for cand in (raw_dir / patient / name, raw_dir / name):
        if cand.is_file():
            return cand
    return None


def process_patient(patient, cfg, raw_dir=None, rng=None, apply_caps=True):
    """Полный конвейер одного пациента: окна, метки, начала окон в секундах, список приступов.

    apply_caps=False — не режет фоновые окна (нужно для теста, чтобы FP/ч
    считался по реальной длительности межприступных интервалов в выбранных файлах).
    """
    raw_dir = Path(raw_dir or cfg["paths"]["raw_dir"])
    summary = locate_summary(raw_dir, patient)
    if summary is None:
        raise FileNotFoundError(f"Нет сводки для {patient} в {raw_dir}.")
    _, records = parse_summary(summary, patient)
    target = float(cfg["data"]["sampling_rate_hz"])
    data_cfg = cfg["data"]
    if apply_caps:
        max_bg = data_cfg.get("max_background_per_file")
        max_bg_files = data_cfg.get("max_background_files")
    else:
        max_bg = data_cfg.get("test_max_background_per_file")  # обычно None
        max_bg_files = data_cfg.get("test_max_background_files")
    rng = rng or np.random.default_rng(int(cfg.get("seed", 0)))
    parts_w, parts_y, parts_s = [], [], []
    file_events = []
    bg_used = 0
    for rec in records:
        path = locate_edf(raw_dir, patient, rec.file_name)
        if path is None:
            continue
        if not rec.seizures:
            if max_bg_files is not None and bg_used >= int(max_bg_files):
                continue
            bg_used += 1
        data, names, sfreq = load_edf(path, target)
        mapped = map_channels(data, names, cfg)
        filtered = bandpass_filter(mapped, sfreq, cfg)
        w, y, st = make_windows(filtered, sfreq, rec.seizures, cfg, max_background=max_bg, rng=rng)
        if len(y) == 0:
            continue
        parts_w.append(w)
        parts_y.append(y)
        parts_s.append(st)
        events = [(sz.start_sec, sz.end_sec) for sz in rec.seizures]
        file_events.append((path.name, st, y, events))
    if not parts_y:
        raise RuntimeError(f"У {patient} не осталось окон.")
    return {
        "windows": np.concatenate(parts_w, 0),
        "labels": np.concatenate(parts_y, 0),
        "starts": np.concatenate(parts_s, 0),
        "file_events": file_events,
        "patient": patient,
    }
