"""Point-in-time backtest of the fuel module against Delta's reported quarters."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..modules.fuel import FuelModule
from ..simulation.state import BusinessState


def backtest_fuel(store, quarters, n_paths: int = 4000, seed: int = 0) -> pd.DataFrame:
    """For each quarter: forecast at day 45 (mid-quarter) and day 8 after quarter end.
    ASM is set to the actual reported value so the fuel module is tested on its own."""
    final = "2100-01-01"
    rep = {k: store.series_as_of(f"delta.reported.{k}", final) for k in
           ("fuel_price_gal", "fuel_expense_m", "asm_m", "fuel_gallons_m")}
    rows = []
    for q in pd.period_range(quarters[0], quarters[1], freq="Q"):
        t = q.start_time
        if t not in rep["fuel_expense_m"].index:
            continue
        act = {k: v.loc[t] for k, v in rep.items()}
        naive = store.series_as_of("delta.reported.fuel_price_gal", q.end_time).iloc[-1]
        for label, as_of in [("mid-quarter (day 45)", q.start_time + pd.Timedelta(days=45)),
                             ("quarter end + 8 days", q.end_time.normalize() + pd.Timedelta(days=8))]:
            fm = FuelModule(); fm.fit(store, as_of)
            st = BusinessState("DAL", str(q), as_of, n_paths)
            st.meta["nowcast"] = {"fuel": fm.nowcast(store, as_of, str(q))}
            st.asm = np.full(n_paths, act["asm_m"] * 1e6)
            st = fm.simulate(st, {}, np.random.default_rng(seed))
            lo, hi = np.quantile(st.fuel_expense, [0.1, 0.9])
            rows.append({
                "quarter": str(q), "when": label,
                "price_pred": np.median(st.fuel_price), "price_actual": act["fuel_price_gal"],
                "price_naive_last_q": naive,
                "gallons_pred_m": np.median(st.fuel_gallons) / 1e6, "gallons_actual_m": act["fuel_gallons_m"],
                "expense_pred_m": np.median(st.fuel_expense) / 1e6, "expense_actual_m": act["fuel_expense_m"],
                "in_80pct_interval": bool(lo / 1e6 <= act["fuel_expense_m"] <= hi / 1e6),
            })
    r = pd.DataFrame(rows)
    r["price_err_pct"] = (r.price_pred / r.price_actual - 1) * 100
    r["naive_err_pct"] = (r.price_naive_last_q / r.price_actual - 1) * 100
    r["gallons_err_pct"] = (r.gallons_pred_m / r.gallons_actual_m - 1) * 100
    r["expense_err_pct"] = (r.expense_pred_m / r.expense_actual_m - 1) * 100
    return r
