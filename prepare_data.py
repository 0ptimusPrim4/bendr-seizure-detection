# -*- coding: utf-8 -*-
"""
Шаг 1. Подготовка данных: загрузка CHB-MIT и весов BENDR (по желанию), проверка,
предобработка и кэш окон.

Что делает:
  1) --download    скачивает индекс PhysioNet и записи выбранных пациентов в data/chbmit/;
  2) --weights     скачивает encoder.pt и contextualizer.pt (BENDR v0.1-alpha) и сверяет SHA-256;
  3) по умолчанию  для каждого пациента: summary → EDF → маппинг 20 каналов →
                   ресемплинг 256 Гц → Баттерворт 0,5–50 Гц → окна 4 с / 1024 отсчёта →
                   метки приступов → кэш data/cache/<пациент>/ (по .npy на файл + meta.json).
Нормализация здесь не применяется: в кэш пишутся статистики каналов, а z-score
делает train.py (по обучающим пациентам фолда, чтобы не было утечки из теста).

Примеры:
  python prepare_data.py --config configs/gpu.yaml --download --weights
  python prepare_data.py --config configs/gpu.yaml --patients chb01 chb02 --workers 4
  python prepare_data.py --config configs/gpu.yaml --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import common as C

PHYSIONET = "https://physionet.org/files/chbmit/1.0.0/"
RELEASE = "https://github.com/SPOClab-ca/BENDR/releases/download/v0.1-alpha"
WEIGHTS = {  # размеры и SHA-256 копий, использованных в НИР-3
    "encoder.pt": (15_893_001, "32c913b1778d8bdf7f08bbe8ef5e6bef2954087b482851816deefc4d0fcf84cb"),
    "contextualizer.pt": (612_693_181, "b8ee0c8c8403809d65114ae88485cdf841dd1860a88ebebb4cdee2bf26f91526"),
}


# ---------------------------- загрузка ---------------------------------

def fetch(url, dest, force=False, missing_ok=False):
    """Скачивание через временный .part, чтобы оборванный файл не считался готовым."""
    dest = Path(dest)
    if dest.is_file() and dest.stat().st_size > 0 and not force:
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


def download_chbmit(raw_dir, patients, force=False):
    """Индекс PhysioNet + все записи выбранных пациентов (полный набор ~43 ГБ)."""
    for name in ("RECORDS", "RECORDS-WITH-SEIZURES", "SUBJECT-INFO", "SHA256SUMS.txt"):
        fetch(PHYSIONET + name, raw_dir / name, force)
    records = [r.strip() for r in (raw_dir / "RECORDS").read_text().splitlines() if r.strip()]
    for p in patients:
        print(f"Пациент {p}")
        fetch(f"{PHYSIONET}{p}/{p}-summary.txt", raw_dir / p / f"{p}-summary.txt", force)
        for rel in (r for r in records if r.startswith(p + "/")):
            fetch(PHYSIONET + rel, raw_dir / rel, force)
            fetch(PHYSIONET + rel + ".seizures", raw_dir / (rel + ".seizures"), force, missing_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def download_weights(weights_dir, force=False):
    ok = True
    for name, (size, digest) in WEIGHTS.items():
        dest = weights_dir / name
        fetch(f"{RELEASE}/{name}", dest, force)
        real = dest.stat().st_size
        if real != size or sha256(dest) != digest:
            print(f"  ОШИБКА: {name} не совпал по размеру/SHA-256 ({real} байт)", file=sys.stderr)
            ok = False
        else:
            print(f"  {name}: размер и SHA-256 совпали")
    return ok


# ---------------------------- предобработка ---------------------------

def process_patient(patient, cfg, raw_dir, cache_dir, force=False):
    """Кэширует окна одного пациента. Повторный запуск пропускает готовые файлы."""
    sig = C.preprocessing_signature(cfg)
    out = cache_dir / patient
    meta_path = out / "meta.json"
    if meta_path.is_file() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("signature") == sig:
            return f"{patient}: кэш уже готов ({meta['n_windows']} окон), пропуск"
    summary = raw_dir / patient / f"{patient}-summary.txt"
    if not summary.is_file():
        return f"{patient}: нет {summary}, пропуск"
    out.mkdir(parents=True, exist_ok=True)
    sfreq = float(cfg["data"]["sampling_rate_hz"])
    files, skipped, t0 = [], [], time.time()
    for rec in C.parse_summary(summary):
        edf = raw_dir / patient / Path(rec.file_name).name
        if not edf.is_file():
            continue          # файл не скачан — работаем с тем, что есть
        stem = edf.stem
        try:
            data, names, fs = C.load_edf(edf, sfreq)
            x = C.bandpass(C.map_channels(data, names, cfg), fs, cfg)
        except C.MissingChannels as e:
            if str(cfg["data"]["channels"].get("on_missing")).lower() == "error":
                raise RuntimeError(f"{edf.name}: {e}") from e
            skipped.append({"file": edf.name, "reason": str(e)})
            continue
        w, y, st = C.make_windows(x, fs, rec.seizures, cfg)
        if len(y) == 0:
            continue
        np.save(out / f"{stem}_windows.npy", w)
        np.save(out / f"{stem}_labels.npy", y)
        np.save(out / f"{stem}_starts.npy", st)
        w64 = w.astype(np.float64)
        files.append({
            "file": edf.name, "windows": f"{stem}_windows.npy", "labels": f"{stem}_labels.npy",
            "starts": f"{stem}_starts.npy", "n_windows": int(len(y)), "n_seizure": int(y.sum()),
            "seizures": [list(s) for s in rec.seizures],
            # статистики для глобального z-score (по каналам)
            "ch_sum": w64.sum(axis=(0, 2)).tolist(), "ch_sumsq": (w64 ** 2).sum(axis=(0, 2)).tolist(),
            "ch_count": int(w.shape[0] * w.shape[2]),
        })
    meta = {"patient": patient, "signature": sig, "sfreq": sfreq,
            "window_sec": float(cfg["data"]["window_sec"]),
            "n_windows": sum(f["n_windows"] for f in files),
            "n_seizure": sum(f["n_seizure"] for f in files),
            "files": files, "skipped_files": skipped}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return (f"{patient}: {len(files)} файлов, {meta['n_windows']} окон, из них приступ "
            f"{meta['n_seizure']}; пропущено файлов {len(skipped)}; {time.time() - t0:.0f} с")


def check(cfg, raw_dir, cache_dir, weights_dir, patients):
    """Проверка наличия весов, записей и кэша без пересчёта."""
    for name in WEIGHTS:
        p = weights_dir / name
        print(f"веса {name}: {'есть' if p.is_file() else 'НЕТ'}")
    for p in patients:
        n_edf = len(list((raw_dir / p).glob("*.edf"))) if (raw_dir / p).is_dir() else 0
        meta = cache_dir / p / "meta.json"
        info = "нет кэша"
        if meta.is_file():
            m = json.loads(meta.read_text(encoding="utf-8"))
            info = f"кэш: {m['n_windows']} окон, приступ {m['n_seizure']}, пропущено файлов {len(m['skipped_files'])}"
        print(f"{p}: EDF {n_edf}; {info}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Подготовка CHB-MIT и весов BENDR (шаг 1 из 3).")
    ap.add_argument("--config", default="configs/gpu.yaml", help="YAML-конфиг (по умолчанию configs/gpu.yaml)")
    ap.add_argument("--patients", nargs="+", help="пациенты, по умолчанию data.patients из конфига")
    ap.add_argument("--download", action="store_true", help="скачать записи пациентов с PhysioNet")
    ap.add_argument("--weights", action="store_true", help="скачать и проверить веса BENDR v0.1-alpha")
    ap.add_argument("--check", action="store_true", help="только проверить наличие данных, весов и кэша")
    ap.add_argument("--no-preprocess", action="store_true", help="не строить кэш окон")
    ap.add_argument("--workers", type=int, default=1, help="процессов для предобработки (по пациентам)")
    ap.add_argument("--force", action="store_true", help="перекачать / пересчитать заново")
    ap.add_argument("--set", nargs="*", default=[], metavar="КЛЮЧ=ЗНАЧЕНИЕ",
                    help="переопределить ключи конфига, например data.channels.on_missing=zero")
    args = ap.parse_args(argv)

    cfg = C.apply_overrides(C.load_config(args.config), args.set)
    paths = cfg["paths"]
    raw_dir, cache_dir = C.resolve_path(paths["raw_dir"]), C.resolve_path(paths["cache_dir"])
    weights_dir = C.resolve_path(paths["weights_dir"])
    patients = args.patients or cfg["data"]["patients"]

    if args.check:
        check(cfg, raw_dir, cache_dir, weights_dir, patients)
        return 0
    if args.weights and not download_weights(weights_dir, args.force):
        return 1
    if args.download:
        download_chbmit(raw_dir, patients, args.force)
    if args.no_preprocess:
        return 0
    cache_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(p, cfg, raw_dir, cache_dir, args.force) for p in patients]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            for msg in ex.map(process_patient, *zip(*jobs)):
                print(msg, flush=True)
    else:
        for job in jobs:
            print(process_patient(*job), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
