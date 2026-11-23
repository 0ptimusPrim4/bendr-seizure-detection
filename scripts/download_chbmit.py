# -*- coding: utf-8 -*-
"""
Назначение: скачать выбранных пациентов CHB-MIT (или индексные файлы) с PhysioNet.
Дата: 23.11.2026

Полный набор — около 42,6 ГБ (карточка PhysioNet 1.0.0).
Источник: https://physionet.org/content/chbmit/1.0.0/
Файлы:    https://physionet.org/files/chbmit/1.0.0/
"""

from __future__ import annotations
import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://physionet.org/files/chbmit/1.0.0/"
INDEX_FILES = [
    "RECORDS",
    "RECORDS-WITH-SEIZURES",
    "SUBJECT-INFO",
    "SHA256SUMS.txt",
    "ANNOTATORS",
]


def fetch(url: str, dest: Path, skip_existing=True) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if skip_existing and dest.is_file() and dest.stat().st_size > 0:
        print(f"  есть: {dest}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  качаю: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "bendr-seizure-nir3"})
    try:
        with urllib.request.urlopen(req) as resp, tmp.open("wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Не удалось скачать {url}: {exc}") from exc
    tmp.replace(dest)
    print(f"  готово: {dest} ({dest.stat().st_size} байт)")


def read_records(raw_dir: Path) -> list[str]:
    rec = raw_dir / "RECORDS"
    if not rec.is_file():
        return []
    return [line.strip() for line in rec.read_text(encoding="utf-8").splitlines() if line.strip()]


def files_for_patient(records: list[str], patient: str) -> list[str]:
    prefix = f"{patient}/"
    out = []
    for rel in records:
        if rel.startswith(prefix):
            out.append(rel)
            # Рядом с записью, содержащей приступ, лежит файл *.seizures.
            # Его нет в RECORDS, поэтому пробуем скачать всегда: 404 просто пропускаем.
            if rel.endswith(".edf"):
                out.append(rel + ".seizures")
    # Сводка всегда нужна.
    out.append(f"{patient}/{patient}-summary.txt")
    # уникальные, с сохранением порядка
    seen = set()
    uniq = []
    for item in out:
        if item not in seen:
            seen.add(item)
            uniq.append(item)
    return uniq


def try_fetch(url: str, dest: Path, skip_existing=True) -> bool:
    try:
        fetch(url, dest, skip_existing=skip_existing)
        return True
    except RuntimeError as exc:
        msg = str(exc)
        if "404" in msg or "Not Found" in msg:
            if dest.with_suffix(dest.suffix + ".part").exists():
                dest.with_suffix(dest.suffix + ".part").unlink()
            return False
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Скачать CHB-MIT с PhysioNet в data/chbmit/."
    )
    parser.add_argument(
        "--dir",
        default=str(ROOT / "data" / "chbmit"),
        help="Каталог назначения (по умолчанию data/chbmit).",
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        default=None,
        help="Список пациентов, например: chb01 chb02 chb03. По умолчанию — только индексные файлы.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Скачать все записи из RECORDS (около 43 ГБ). Сначала качается индекс.",
    )
    parser.add_argument("--force", action="store_true", help="Перезаписывать существующие файлы.")
    args = parser.parse_args(argv)
    raw_dir = Path(args.dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Индексные файлы → {raw_dir}")
    for name in INDEX_FILES:
        fetch(BASE + name, raw_dir / name, skip_existing=not args.force)

    records = read_records(raw_dir)
    if not records:
        print("RECORDS пуст или не скачался.", file=sys.stderr)
        return 1

    if args.all:
        patients = sorted({rel.split("/")[0] for rel in records if "/" in rel})
    elif args.patients:
        patients = args.patients
    else:
        print("Индекс готов. Чтобы скачать записи, укажите --patients chb01 chb02 ... или --all.")
        return 0

    for patient in patients:
        print(f"\nПациент {patient}")
        rels = files_for_patient(records, patient)
        if not rels:
            print(f"  В RECORDS нет записей для {patient}.", file=sys.stderr)
            continue
        for rel in rels:
            dest = raw_dir / rel
            url = BASE + rel
            if rel.endswith(".seizures"):
                try_fetch(url, dest, skip_existing=not args.force)
            else:
                fetch(url, dest, skip_existing=not args.force)
    print("\nГотово.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
