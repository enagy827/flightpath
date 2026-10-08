"""Where data lives. Large data stays OUTSIDE OneDrive.

Default: ~/flightpath_data (C:\\Users\\enagy\\flightpath_data on Emma's laptop).
Override with the FLIGHTPATH_DATA environment variable (e.g. in .env).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA = Path(os.getenv("FLIGHTPATH_DATA", Path.home() / "flightpath_data"))
PROCESSED = DATA / "processed"          # cleaned, analysis-ready tables (small)
STORE_PATH = PROCESSED / "store.parquet"


def ensure_dirs() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
