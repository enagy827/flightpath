"""Market baseline B to compare FlightPath against.

Longaeva's dataset list has no consensus estimates, so the baseline is **Delta's own guidance**
(config/guidance.yaml, point-in-time by release date). If no guidance is public at as_of, fall back to
the naive benchmark (prior year x recent growth). Outputs are labeled with the source used.
Passenger revenue growth = guided (adjusted) total revenue growth + a per-quarter adjustment
(`passenger_vs_total_yoy_adj`, from the latest quarter's gap between the two).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from ..paths import ROOT

GUIDANCE_PATH = ROOT / "config" / "guidance.yaml"


def load_guidance(path: Path = GUIDANCE_PATH) -> list[dict]:
    if not Path(path).exists():
        return []
    return yaml.safe_load(Path(path).read_text()) or []


def guidance_for(quarter: str, as_of) -> dict | None:
    t = pd.Timestamp(as_of)
    for g in load_guidance():
        if str(g["quarter"]) == quarter and pd.Timestamp(g["released"]) <= t:
            return g
    return None


def _reported(store, metric, as_of) -> pd.Series:
    x = store.series_as_of(f"delta.reported.{metric}", as_of)
    x.index = x.index.to_period("Q")
    return x


def baseline(store, quarter: str, as_of) -> dict:
    """Baseline passenger revenue ($), fuel price ($/gal) and fuel expense ($) for `quarter`."""
    q = pd.Period(quarter, freq="Q")
    rev = _reported(store, "passenger_revenue_m", as_of)
    out = {"quarter": quarter}
    g = guidance_for(quarter, as_of)
    if g is not None and g.get("total_revenue_yoy"):
        adj = g.get("passenger_vs_total_yoy_adj", 0.0)
        lo, hi = (x + adj for x in g["total_revenue_yoy"])
        out["passenger_revenue"] = float(rev.loc[q - 4]) * 1e6 * (1 + (lo + hi) / 2)
        out["passenger_revenue_range"] = (float(rev.loc[q - 4]) * 1e6 * (1 + lo), float(rev.loc[q - 4]) * 1e6 * (1 + hi))
        out["source"] = f"Delta guidance ({g['released']})"
        out["fuel_price"] = g.get("fuel_price_gal")
        out["guidance"] = g
    else:
        r = rev.reindex(pd.period_range(rev.index.min(), rev.index.max(), freq="Q"))
        yoy = (r / r.shift(4) - 1).dropna().tail(2)
        out["passenger_revenue"] = float(rev.loc[q - 4] * (1 + yoy.mean())) * 1e6
        out["source"] = "naive (prior year x recent growth) - no guidance public at as_of"
        price = _reported(store, "fuel_price_gal", as_of)
        out["fuel_price"] = float(price.iloc[-1])
    return out
