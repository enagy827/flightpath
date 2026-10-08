"""Helpers shared by the revenue-chain modules: quarterly reported series + TSA quarter-to-date."""
from __future__ import annotations

import numpy as np
import pandas as pd

COVID_YOY = (pd.Period("2020Q1", "Q"), pd.Period("2022Q1", "Q"))   # YoY comparisons distorted by COVID


def reported_q(store, metric: str, as_of) -> pd.Series:
    x = store.series_as_of(f"delta.reported.{metric}", as_of)
    x.index = x.index.to_period("Q")
    return x.astype(float)


def tsa_quarter_yoy(store, as_of, quarter: str) -> dict:
    """TSA throughput YoY for `quarter`, using only days observable at as_of.
    Compares the same number of days against the same weekdays a year earlier (364-day shift).
    Returns yoy (log), share of the quarter observed, and the full-quarter YoY history for fitting."""
    tsa = store.series_as_of("tsa.throughput", as_of)
    q = pd.Period(quarter, freq="Q")
    days = pd.date_range(q.start_time, q.end_time.normalize(), freq="D")
    seen = tsa.reindex(days).dropna()
    ly = tsa.reindex(seen.index - pd.Timedelta(days=364)).dropna()
    n = min(len(seen), len(ly))
    yoy = float(np.log(seen.iloc[:n].sum() / ly.iloc[:n].sum())) if n else np.nan
    return {"yoy": yoy, "share_observed": n / len(days)}


def tsa_full_quarter_yoy(store, as_of) -> pd.Series:
    tsa = store.series_as_of("tsa.throughput", as_of)
    qs = tsa.groupby(tsa.index.to_period("Q")).agg(["sum", "count"])
    qs = qs[qs["count"] >= 89]["sum"]
    return np.log(qs / qs.shift(4)).dropna()


def fit_window(idx: pd.PeriodIndex) -> np.ndarray:
    return idx > COVID_YOY[1]
