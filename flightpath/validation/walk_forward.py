"""Expanding-window (and rolling) walk-forward backtest.

For each quarter q and each as_of date within q (e.g. weekly):
    1. engine.fit(as_of)                          - only data available at as_of
    2. state = engine.run(carrier, q, as_of)
    3. store forecast distribution
After loop: compare to realized values from SEC (available after quarter).

Tip: validate operating modules on MONTHLY BTS data too - ~3x more test points
than quarterly financials.
Never touch settings.validation.holdout_start onward during development.
"""
from __future__ import annotations

from datetime import date

import pandas as pd


def walk_forward(engine, carrier: str, start_quarter: str, end_quarter: str,
                 scheme: str = "expanding", as_of_freq: str = "W",
                 exclude_holdout: bool = True) -> pd.DataFrame:
    """Return long DataFrame[quarter, as_of, target, mean, p05..p95, realized]. TODO."""
    raise NotImplementedError


def realized_outcomes(store, carrier: str, targets: list[str]) -> pd.DataFrame:
    """Reported values per quarter from SEC loader. TODO."""
    raise NotImplementedError
