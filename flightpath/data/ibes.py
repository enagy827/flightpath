"""Wall Street consensus from LSEG I/B/E/S (WRDS, IBES Academic; I/B/E/S ticker DAL), point in time (notebook 07).

Files (downloaded by hand from WRDS into flightpath_data/ibes/): ibes_details.csv - every analyst estimate (EPS,
SAL = revenue $M, OPR = operating profit $M) for the next 1-3 quarters (FPI 6-8), with the date it was announced
(ANNDATS) and the date I/B/E/S recorded it (ACTDATS); ibes_summaries.csv - I/B/E/S's own monthly consensus.

Consensus on day t for a quarter: each analyst's (ESTIMATOR + ANALYS) latest estimate recorded by t (ACTDATS <= t,
so nothing later is used), dropped if older than `max_age` days; the median across analysts (I/B/E/S reports both
mean and median; the median is less sensitive to one stale or odd estimate).

LICENSE: WRDS data is under Georgia Tech's academic license - fine for research and summary results; check before
showing raw I/B/E/S figures in an app shared outside the university.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_detail(data_dir=None) -> pd.DataFrame:
    from ..paths import DATA
    d = pd.read_csv((data_dir or DATA) / "ibes" / "ibes_details.csv",
                    parse_dates=["ACTDATS", "ANNDATS", "REVDATS", "FPEDATS", "ANNDATS_ACT"])
    d["q"] = d.FPEDATS.dt.to_period("Q")
    d["who"] = d.ESTIMATOR.astype(str) + "/" + d.ANALYS.astype(str)
    return d


def load_summary(data_dir=None) -> pd.DataFrame:
    from ..paths import DATA
    s = pd.read_csv((data_dir or DATA) / "ibes" / "ibes_summaries.csv", parse_dates=["STATPERS", "FPEDATS", "ANNDATS_ACT"])
    s["q"] = s.FPEDATS.dt.to_period("Q")
    return s


def consensus(d: pd.DataFrame, measure: str, q, as_of, max_age: int = 120, stat: str = "median") -> dict:
    """Consensus for quarter q on date as_of from the detail file (see module docstring)."""
    q, as_of = pd.Period(q, "Q"), pd.Timestamp(as_of)
    x = d[(d.MEASURE == measure) & (d.q == q) & (d.ACTDATS <= as_of) & (d.ACTDATS >= as_of - pd.Timedelta(days=max_age))]
    if x.empty:
        return {"value": np.nan, "n": 0}
    last = x.sort_values(["ACTDATS", "ANNDATS"]).groupby("who").tail(1)
    v = float(last.VALUE.median() if stat == "median" else last.VALUE.mean())
    return {"value": v, "n": int(len(last)), "low": float(last.VALUE.min()), "high": float(last.VALUE.max())}


def actuals(d: pd.DataFrame) -> pd.DataFrame:
    """I/B/E/S actuals by quarter and measure, with the report date."""
    a = d.dropna(subset=["ACTUAL"]).groupby(["q", "MEASURE"]).agg(actual=("ACTUAL", "last"), reported=("ANNDATS_ACT", "last"))
    return a.unstack("MEASURE")
