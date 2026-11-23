# -*- coding: utf-8 -*-
"""
Назначение: скачать официальные веса BENDR (релиз v0.1-alpha) в каталог weights/.
Дата: 23.11.2026

Проверено 23.11.2026 (HTTP 302 на githubusercontent, размеры из GitHub API):
  encoder.pt         15 893 001 байт
  contextualizer.pt 612 693 181 байт
SHA-256 локальных копий, использованных в НИР-3:
  encoder.pt         32c913b1778d8bdf7f08bbe8ef5e6bef2954087b482851816deefc4d0fcf84cb
  contextualizer.pt  b8ee0c8c8403809d65114ae88485cdf841dd1860a88ebebb4cdee2bf26f91526
"""

from __future__ import annotations
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "https://github.com/SPOClab-ca/BENDR/releases/download/v0.1-alpha"

FILES = {
    "encoder.pt": {
        "url": f"{RELEASE}/encoder.pt",
        "size": 15_893_001,
        "sha256": "32c913b1778d8bdf7f08bbe8ef5e6bef2954087b482851816deefc4d0fcf84cb",
    },
    "contextualizer.pt": {
        "url": f"{RELEASE}/contextualizer.pt",
        "size": 612_693_181,
        "sha256": "b8ee0c8c8403809d65114ae88485cdf841dd1860a88ebebb4cdee2bf26f91526",
    },
}


def sha256_of(path: Path, chunk=1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Скачиваю {url}")
    print(f"  → {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "bendr-seizure-nir3"})
    with urllib.request.urlopen(req) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / total:6.1%} ({done / 1e6:.1f} / {total / 1e6:.1f} МБ)", end="", flush=True)
            else:
                print(f"\r  {done / 1e6:.1f} МБ", end="", flush=True)
    print()
    tmp.replace(dest)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Скачать веса BENDR v0.1-alpha.")
    parser.add_argument("--dir", default=str(ROOT / "weights"), help="Куда сохранить файлы.")
    parser.add_argument("--force", action="store_true", help="Скачать заново, даже если файл уже есть.")
    args = parser.parse_args(argv)
    out_dir = Path(args.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, meta in FILES.items():
        dest = out_dir / name
        if dest.is_file() and not args.force:
            size = dest.stat().st_size
            print(f"{name} уже есть ({size} байт), проверяю SHA-256...")
        else:
            download(meta["url"], dest)
            size = dest.stat().st_size
        if size != meta["size"]:
            print(f"  ОШИБКА: ожидался размер {meta['size']}, получено {size}.", file=sys.stderr)
            ok = False
            continue
        digest = sha256_of(dest)
        if digest != meta["sha256"]:
            print(f"  ОШИБКА: SHA-256 не совпал: {digest}", file=sys.stderr)
            ok = False
        else:
            print(f"  OK: размер и SHA-256 совпали.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
