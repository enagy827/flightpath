"""Attribution: which signals moved the forecast, and by how much.

Approach: between two as_of dates (t0 -> t1), update one input group at a time
(TSA, fuel price, capacity guidance, evidence) holding others at t0, and record the
change in expected outcome (sequential or Shapley-style to avoid order dependence).
Output drives the waterfall chart in the app.
"""
from __future__ import annotations

from datetime import date

import pandas as pd


def attribute_change(engine, carrier: str, quarter: str, t0: date, t1: date,
                     target: str = "passenger_revenue",
                     groups: tuple[str, ...] = ("tsa", "eia", "capacity_guidance", "evidence"),
                     method: str = "shapley") -> pd.DataFrame:
    """Return DataFrame[group, delta_mean, delta_p10, delta_p90]. TODO."""
    raise NotImplementedError
