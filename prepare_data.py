# -*- coding: utf-8 -*-
"""
Подготовка CHB-MIT для дообучения BENDR: скачивание записей и весов, предобработка, кэш окон.

  python prepare_data.py                      # пациенты chb01–chb23, всё по умолчанию
  python prepare_data.py --patients chb01 chb02

Конвейер: summary → EDF → 20 каналов 10-20 → 256 Гц → Баттерворт 0,5–50 Гц (порядок 5)
→ окна 4 с / 1024 отсчёта без перекрытия, 15 мин постиктала исключаются → data/cache/<пациент>/.
z-score здесь не делается: в кэш пишутся суммы по каналам, нормализует train_eval.py
по обучающим пациентам фолда.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PHYSIONET = "https://physionet.org/files/chbmit/1.0.0/"
RELEASE = "https://github.com/SPOClab-ca/BENDR/releases/download/v0.1-alpha"
WEIGHTS = {  # размер и SHA-256 файлов релиза v0.1-alpha
    "encoder.pt": (15_893_001, "32c913b1778d8bdf7f08bbe8ef5e6bef2954087b482851816deefc4d0fcf84cb"),
    "contextualizer.pt": (612_693_181, "b8ee0c8c8403809d65114ae88485cdf841dd1860a88ebebb4cdee2bf26f91526"),
}
ALL_PATIENTS = [f"chb{i:02d}" for i in range(1, 24)]  # 23 случая из SUBJECT-INFO

# Параметры предобработки (диссертация Benfenati / исходный configs/default.yaml)
SFREQ = 256.0
WINDOW_SEC = 4.0
POSTICTAL_MIN = 15.0
FILTER = (5, 0.5, 50.0)            # порядок, нижняя и верхняя частоты
DROP = ["FT9-FT10", "FT10-T8", "T7-FT9"]
# Слоты BENDR (19 электродов 10-20 + SCALE) ← биполярные отведения CHB-MIT в порядке сводки chb01.
# Это порядок слотов, а не геометрия: таблицы пар в диссертации нет.
MAPPING = {
    "FP1": "FP1-F7", "FP2": "F7-T7", "F7": "T7-P7", "F3": "P7-O1", "FZ": "FP1-F3",
    "F4": "F3-C3", "F8": "C3-P3", "T7": "P3-O1", "C3": "FP2-F4", "CZ": "F4-C4",
    "C4": "C4-P4", "T8": "P4-O2", "T5": "FP2-F8", "P3": "F8-T8", "PZ": "T8-P8",
    "P4": "P8-O2", "T6": "FZ-CZ", "O1": "CZ-PZ", "O2": "P7-T7",
}
SIGNATURE = hashlib.sha1(json.dumps([SFREQ, WINDOW_SEC, POSTICTAL_MIN, FILTER, DROP, MAPPING]).encode()).hexdigest()[:12]


def require(*modules):
    """Понятная ошибка, если не установлены зависимости."""
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if missing:
        sys.exit(f"Не установлены пакеты: {', '.join(missing)}.\n"
                 f"Выполните: pip install -r requirements.txt (torch — с сайта pytorch.org под вашу CUDA).")


require("numpy", "scipy", "mne")
import numpy as np  # noqa: E402


# ------------------------------------------------------------------ загрузка

def fetch(url, dest, missing_ok=False):
    dest = Path(dest)
    if dest.is_file() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "bendr-seizure-nir3"})
    try:
        with urllib.request.urlopen(req) as r, tmp.open("wb") as out:
            total, done = int(r.headers.get("Content-Length") or 0), 0
            while chunk := r.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if total > 50 << 20:
                    print(f"\r  {dest.name}: {done / total:6.1%}", end="", flush=True)
        if total > 50 << 20:
            print()
    except urllib.error.HTTPError as e:
        tmp.unlink(missing_ok=True)
        if missing_ok and e.code == 404:
            return False
        raise RuntimeError(f"Не удалось скачать {url}: {e}") from e
    tmp.replace(dest)
    print(f"  скачан {dest}")
    return True


def ensure_weights(weights_dir):
    """Скачивает веса BENDR, если их нет, и сверяет размер."""
    weights_dir = Path(weights_dir)
    for name, (size, digest) in WEIGHTS.items():
        dest = weights_dir / name
        if not dest.is_file():
            print(f"Скачиваю веса BENDR {name} ...")
            fetch(f"{RELEASE}/{name}", dest)
            h = hashlib.sha256(dest.read_bytes()).hexdigest()
            if h != digest:
                sys.exit(f"SHA-256 {dest} не совпал — удалите файл и запустите снова.")
        if dest.stat().st_size != size:
            sys.exit(f"Размер {dest} не совпал ({dest.stat().st_size} вместо {size}) — удалите файл.")


def ensure_raw(raw_dir, patient):
    """Докачивает недостающие записи пациента с PhysioNet (около 1–3 ГБ на пациента)."""
    raw_dir = Path(raw_dir)
    fetch(PHYSIONET + "RECORDS", raw_dir / "RECORDS")
    records = [r.strip() for r in (raw_dir / "RECORDS").read_text().splitlines() if r.strip().startswith(patient + "/")]
    fetch(f"{PHYSIONET}{patient}/{patient}-summary.txt", raw_dir / patient / f"{patient}-summary.txt")
    for rel in records:
        if not (raw_dir / rel).is_file():
            fetch(PHYSIONET + rel, raw_dir / rel)
            fetch(PHYSIONET + rel + ".seizures", raw_dir / (rel + ".seizures"), missing_ok=True)


# ------------------------------------------------------------ предобработка

def parse_summary(path):
    """chbXX-summary.txt → список (имя файла, [(начало, конец), ...])."""
    out, cur, start = [], None, None
    for line in Path(path).read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = line.strip()
        if m := re.match(r"^File Name:\s*(.+?)\s*$", line, re.I):
            cur = (m.group(1), [])
            out.append(cur)
        elif cur and (m := re.match(r"^Seizure(?:\s+\d+)?\s+Start Time:\s*([\d.]+)", line, re.I)):
            start = float(m.group(1))
        elif cur and start is not None and (m := re.match(r"^Seizure(?:\s+\d+)?\s+End Time:\s*([\d.]+)", line, re.I)):
            cur[1].append((start, float(m.group(1))))
            start = None
    return out


def edf_labels(path):
    # подписи из заголовка EDF: mne переименовывает дубликаты (T8-P8-0, T8-P8-1)
    with open(path, "rb") as fh:
        n = int(fh.read(256)[252:256].decode("ascii", "ignore").strip() or 0)
        raw = fh.read(n * 16)
    return [raw[i * 16:(i + 1) * 16].decode("ascii", "replace").strip().upper() for i in range(n)]


def load_file(path, seizures, on_missing="skip"):
    """Один EDF → окна [N, 20, 1024], метки, начала окон (с). None — файл пропущен."""
    import mne
    from scipy.signal import butter, sosfiltfilt
    raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
    if abs(raw.info["sfreq"] - SFREQ) > 1e-3:
        raw.resample(SFREQ, verbose="ERROR")
    data = raw.get_data().astype(np.float32)
    names = edf_labels(path)
    if len(names) != len(data):
        names = [c.upper() for c in raw.ch_names]
    idx = {}
    for i, n in enumerate(names):
        if n and n not in DROP and n not in idx:   # второй T8-P8 отбрасывается
            idx[n] = i
    absent = [src for src in MAPPING.values() if src not in idx]
    if absent and on_missing != "zero":
        return None, absent
    eeg = np.stack([data[idx[s]] if s in idx else np.zeros(data.shape[1], np.float32) for s in MAPPING.values()])
    scale = np.abs(eeg).mean(0, keepdims=True)  # SCALE — относительная амплитуда (выбрано автором)
    x = np.concatenate([eeg, scale])
    order, lo, hi = FILTER
    sos = butter(order, [lo, hi], btype="bandpass", fs=SFREQ, output="sos")
    x = sosfiltfilt(sos, x.astype(np.float64), axis=-1).astype(np.float32)

    win, n = int(WINDOW_SEC * SFREQ), x.shape[1]
    sz = [(int(a * SFREQ), min(n, int(np.ceil(b * SFREQ)))) for a, b in seizures]
    post = [(b, min(n, b + int(POSTICTAL_MIN * 60 * SFREQ))) for _, b in sz]
    hit = lambda s, e, rr: any(s < b and e > a for a, b in rr)
    starts, labels = [], []
    for s in range(0, n - win + 1, win):
        if hit(s, s + win, sz):
            labels.append(1)
        elif hit(s, s + win, post):
            continue
        else:
            labels.append(0)
        starts.append(s)
    if not starts:
        return None, []
    w = np.stack([x[:, s:s + win] for s in starts])
    return (w, np.array(labels, np.int8), np.array(starts) / SFREQ), []


def prepare_patient(patient, raw_dir, cache_dir, download=True, on_missing="skip"):
    """Строит кэш пациента; повторный запуск ничего не пересчитывает."""
    out = Path(cache_dir) / patient
    meta_path = out / "meta.json"
    if meta_path.is_file() and json.loads(meta_path.read_text())["signature"] == SIGNATURE:
        return f"{patient}: кэш готов"
    raw_dir = Path(raw_dir)
    if download:
        ensure_raw(raw_dir, patient)
    summary = raw_dir / patient / f"{patient}-summary.txt"
    if not summary.is_file():
        return f"{patient}: нет {summary}, пропуск"
    out.mkdir(parents=True, exist_ok=True)
    files, skipped, t0 = [], [], time.time()
    for name, seizures in parse_summary(summary):
        edf = raw_dir / patient / name
        if not edf.is_file():
            continue
        res, absent = load_file(edf, seizures, on_missing)
        if res is None:
            if absent:
                skipped.append({"file": name, "missing": absent})
            continue
        w, y, st = res
        stem = edf.stem
        np.save(out / f"{stem}_windows.npy", w)
        np.save(out / f"{stem}_labels.npy", y)
        np.save(out / f"{stem}_starts.npy", st)
        w64 = w.astype(np.float64)
        files.append({"file": name, "stem": stem, "n_windows": len(y), "n_seizure": int(y.sum()),
                      "seizures": seizures, "ch_sum": w64.sum((0, 2)).tolist(),
                      "ch_sumsq": (w64 ** 2).sum((0, 2)).tolist(), "ch_count": int(w.shape[0] * w.shape[2])})
    meta = {"patient": patient, "signature": SIGNATURE, "files": files, "skipped_files": skipped,
            "n_windows": sum(f["n_windows"] for f in files), "n_seizure": sum(f["n_seizure"] for f in files)}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    return (f"{patient}: {len(files)} файлов, {meta['n_windows']} окон (приступ {meta['n_seizure']}), "
            f"пропущено {len(skipped)} файлов с неполным монтажом, {time.time() - t0:.0f} с")


class PatientCache:
    """Окна пациента из кэша; сами окна читаются через memmap по требованию."""

    def __init__(self, cache_dir, patient):
        self.dir = Path(cache_dir) / patient
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.files = self.meta["files"]
        self._mm = {}
        self.labels = np.concatenate([np.load(self.dir / f"{f['stem']}_labels.npy") for f in self.files]).astype(np.int64)
        self.starts = np.concatenate([np.load(self.dir / f"{f['stem']}_starts.npy") for f in self.files])
        self.file_idx = np.concatenate([np.full(f["n_windows"], i) for i, f in enumerate(self.files)])
        self.local = np.concatenate([np.arange(f["n_windows"]) for f in self.files])

    def __len__(self):
        return len(self.labels)

    def __getstate__(self):  # в процессы DataLoader memmap не передаём
        return {**self.__dict__, "_mm": {}}

    def window(self, i):
        f = int(self.file_idx[i])
        if f not in self._mm:
            self._mm[f] = np.load(self.dir / f"{self.files[f]['stem']}_windows.npy", mmap_mode="r")
        return np.asarray(self._mm[f][self.local[i]], np.float32)

    def channel_stats(self, file_ids=None):
        ids = range(len(self.files)) if file_ids is None else file_ids
        return (sum(np.array(self.files[i]["ch_sum"]) for i in ids),
                sum(np.array(self.files[i]["ch_sumsq"]) for i in ids),
                sum(self.files[i]["ch_count"] for i in ids))


def prepare_all(patients, raw_dir, cache_dir, weights_dir, workers=1, download=True, on_missing="skip"):
    for d in (raw_dir, cache_dir, weights_dir):
        Path(d).mkdir(parents=True, exist_ok=True)
    ensure_weights(weights_dir)
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            futs = [ex.submit(prepare_patient, p, raw_dir, cache_dir, download, on_missing) for p in patients]
            for f in futs:
                print(f.result(), flush=True)
    else:
        for p in patients:
            print(prepare_patient(p, raw_dir, cache_dir, download, on_missing), flush=True)


def add_path_args(ap):
    ap.add_argument("--patients", nargs="+", default=ALL_PATIENTS, help="по умолчанию chb01–chb23")
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "chbmit"), help="записи CHB-MIT (EDF + summary)")
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"), help="кэш окон")
    ap.add_argument("--weights-dir", default=str(ROOT / "weights"), help="encoder.pt, contextualizer.pt")
    ap.add_argument("--no-download", action="store_true", help="не качать недостающие записи")
    ap.add_argument("--on-missing", choices=["skip", "zero"], default="skip",
                    help="файл с неполным монтажом: пропустить или заполнить нулями")
    ap.add_argument("--prep-workers", type=int, default=2, help="процессов предобработки")


def main():
    ap = argparse.ArgumentParser(description="Скачивание и предобработка CHB-MIT, загрузка весов BENDR.")
    add_path_args(ap)
    a = ap.parse_args()
    prepare_all(a.patients, a.raw_dir, a.cache_dir, a.weights_dir, a.prep_workers, not a.no_download, a.on_missing)


if __name__ == "__main__":
    main()
