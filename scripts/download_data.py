# -*- coding: utf-8 -*-
"""Совместимая обёртка над scripts/download_chbmit.py.

Раньше планировалось класть записи в data/raw/. Актуальный каталог
по умолчанию — data/chbmit/ (см. README).
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from download_chbmit import main


if __name__ == "__main__":
    sys.exit(main())
