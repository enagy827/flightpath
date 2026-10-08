"""Point-in-time backtest of the full chain: capacity -> fuel -> load factor -> yield -> passenger revenue."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..simulation.engine import SimulationEngine

TARGETS = {"asm": ("asm_m", 1e6), "load_factor": (None, 1), "yield_": ("yield_cents", 1),
           "passenger_revenue": ("passenger_revenue_m", 1e6), "fuel_expense": ("fuel_expense_m", 1e6)}


def _actuals(store, q):
    final = "2100-01-01"
    g = lambda m: store.series_as_of(f"delta.reported.{m}", final).get(q.start_time, np.nan)
    a = {"asm": g("asm_m") * 1e6, "yield_": g("yield_cents"), "passenger_revenue": g("passenger_revenue_m") * 1e6,
         "fuel_expense": g("fuel_expense_m") * 1e6}
    a["load_factor"] = g("rpm_m") / g("asm_m")
    return a


def naive_revenue(store, q, as_of) -> float:
    """Benchmark: prior-year passenger revenue x (1 + average YoY growth of the last 2 reported quarters)."""
    r = store.series_as_of("delta.reported.passenger_revenue_m", as_of)
    r.index = r.index.to_period("Q")
    yoy = (r / r.shift(4) - 1).dropna().tail(2)
    return float(r.loc[q - 4] * (1 + yoy.mean())) * 1e6


def backtest_chain(store, quarters, n_paths: int = 4000) -> pd.DataFrame:
    rows = []
    for q in pd.period_range(quarters[0], quarters[1], freq="Q"):
        act = _actuals(store, q)
        if np.isnan(act["passenger_revenue"]):
            continue
        for label, as_of in [("mid-quarter (day 45)", q.start_time + pd.Timedelta(days=45)),
                             ("quarter end + 8 days", q.end_time.normalize() + pd.Timedelta(days=8))]:
            eng = SimulationEngine(store, n_paths=n_paths, seed=0)
            st = eng.run(str(q), as_of)
            row = {"quarter": str(q), "when": label}
            for k in TARGETS:
                x = getattr(st, k)
                lo, hi = np.quantile(x, [0.1, 0.9])
                row[f"{k}_pred"], row[f"{k}_actual"] = float(np.median(x)), act[k]
                row[f"{k}_err_pct"] = (np.median(x) / act[k] - 1) * 100
                row[f"{k}_in80"] = bool(lo <= act[k] <= hi)
            nv = naive_revenue(store, q, as_of)
            row["revenue_naive_err_pct"] = (nv / act["passenger_revenue"] - 1) * 100
            row["p_rev_above_naive"] = float((st.passenger_revenue > nv).mean())
            rows.append(row)
    return pd.DataFrame(rows)
