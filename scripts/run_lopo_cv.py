# -*- coding: utf-8 -*-
"""Обёртка: python scripts/run_lopo_cv.py — то же, что python -m bendr_seizure train."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bendr_seizure.cli import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv or argv[0].startswith("-"):
        argv = ["train", *argv]
    sys.exit(main(argv))
