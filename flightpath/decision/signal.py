"""Fundamental Surprise Score and P(beat).

    FSS = (E[Y | info_t] - B) / sd(Y | info_t)      (proposal eq. 5)

Reported for each metric, plus an operating-income impact combining revenue and fuel:
    impact = (passenger revenue - baseline revenue) - (fuel expense - baseline fuel expense)
where baseline fuel expense = model gallons x guided fuel price (isolates the price view).
Other cost lines (labor, maintenance...) are not modeled and assumed in line with guidance.
"""
from __future__ import annotations

import numpy as np


def surprise(paths: np.ndarray, baseline: float, higher_is_better: bool = True) -> dict:
    x = np.asarray(paths, dtype=float)
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    fss = (mean - baseline) / sd if sd > 0 else 0.0
    p_above = float((x > baseline).mean())
    return {"model_mean": mean, "model_sd": sd, "baseline": float(baseline), "fss": fss,
            "pct_vs_baseline": (mean / baseline - 1) if baseline else float("nan"), "p_above": p_above,
            "p_better": p_above if higher_is_better else 1 - p_above}


def operating_impact(state, base: dict) -> dict:
    rev_gap = state.passenger_revenue - base["passenger_revenue"]
    fuel_gap = state.fuel_expense - state.fuel_gallons * base["fuel_price"]
    extra = state.disruption_cost if state.disruption_cost is not None else 0.0
    impact = rev_gap - fuel_gap - extra
    s = surprise(impact, 0.0)
    s.update(revenue_gap_mean=float(rev_gap.mean()), fuel_gap_mean=float(fuel_gap.mean()))
    return s
