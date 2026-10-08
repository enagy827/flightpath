"""Load global settings and carrier-specific parameters."""
from __future__ import annotations

from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load_settings(path: Path | None = None) -> dict[str, Any]:
    """Load config/settings.yaml.

    TODO: read YAML, resolve relative paths against ROOT, load .env via python-dotenv.
    """
    raise NotImplementedError


def load_carrier(ticker: str) -> dict[str, Any]:
    """Load config/carriers/<ticker>.yaml.

    TODO: read YAML; validate required keys (bts_carrier_code, hubs, reported_metrics).
    """
    raise NotImplementedError
