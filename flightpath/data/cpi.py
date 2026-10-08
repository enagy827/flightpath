"""Airline fares in the consumer price index (BLS series CUUR0000SETG01, not seasonally adjusted), notebook 07.

Monthly index, downloaded by scripts/fetch_public_data.py (step fred: BLS API). BLS publishes each month's CPI
around the 10th-15th of the next month; government shutdowns delayed some (September 2025's came out on 24 October
2025, October 2025's was never published). Rule used here, so nothing is used before it was public: a month counts
as known from the 25th of the following month.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SERIES = "CUUR0000SETG01"
PUBLISH_DAY = 25


def load_airfare_cpi(data_dir=None) -> pd.Series:
    from ..paths import DATA
    p = (data_dir or DATA) / "fred" / f"{SERIES}.csv"
    if not p.exists():
        raise FileNotFoundError("run: python scripts/fetch_public_data.py fred")
    d = pd.read_csv(p, parse_dates=["observation_date"])
    return d.set_index("observation_date")[SERIES].astype(float).sort_index()


def known_from(month: pd.Timestamp) -> pd.Timestamp:
    return (pd.Timestamp(month) + pd.offsets.MonthBegin(1)) + pd.Timedelta(days=PUBLISH_DAY - 1)


def quarterly_yoy(cpi: pd.Series) -> pd.Series:
    """Quarter-average index vs the same quarter a year earlier (%), quarters with all three months."""
    q = cpi.groupby(cpi.index.to_period("Q")).agg(["mean", "count"])
    m = q["mean"].where(q["count"] == 3)
    return ((m / m.shift(4) - 1) * 100).dropna()


def growth_as_of(cpi: pd.Series, q, as_of) -> dict:
    """Airfare CPI growth for quarter q as known on `as_of`: the quarter's months already published vs the same
    months a year earlier; none published yet: the latest published month's year-over-year growth."""
    q, as_of = pd.Period(q, "Q"), pd.Timestamp(as_of)
    pub = [m for m in cpi.index if known_from(m) <= as_of]
    months = [m for m in pd.date_range(q.start_time, periods=3, freq="MS") if m in pub]
    ly = lambda ms: [m - pd.DateOffset(years=1) for m in ms]
    use, how = (months, "months of the quarter published") if months else (pub[-1:], "latest published month")
    if not use:
        return {"growth_pct": np.nan, "months": [], "rule": how}
    g = (cpi.reindex(use).sum() / cpi.reindex(ly(use)).sum() - 1) * 100
    return {"growth_pct": float(g), "months": [f"{m:%Y-%m}" for m in use], "rule": how}
