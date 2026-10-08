"""Backtest of the whole simulator (notebook 06): for every past quarter and several dates, build the base with only
what was public that day (simulation.general), simulate, and keep what is needed to score it against Delta's report."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..simulation.general import Context, actual, build_base
from ..forecast.update import counterfactual
from ..simulation.quarter import draw, evaluate

# dates relative to each quarter: days from its start (negative = before it starts, i.e. forecasting next quarter),
# or "end+7" = 7 days after it ends (Delta has not reported yet)
OFFSETS = [-20, 20, 45, 70, "end+7"]
METRICS = {"eps": "eps", "operating_margin": "operating_margin", "revenue_pct": "total_revenue_yoy"}


def as_of_date(q: pd.Period, off) -> pd.Timestamp:
    return q.end_time.normalize() + pd.Timedelta(days=7) if off == "end+7" else q.start_time + pd.Timedelta(days=off)


def run_one(ctx: Context, q, off, n=50_000, seed=0) -> dict:
    q = pd.Period(q, "Q")
    d = as_of_date(q, off)
    B = build_base(ctx, q, d)
    act = actual(ctx, q)
    if act["filed"] <= d:
        raise ValueError("forecast date after Delta's report")
    row = {"quarter": str(q), "offset": str(off), "as_of": d, "days_from_start": (d - q.start_time).days,
           "last_quarter": B["last_quarter"]["source"].split(" ")[0]}
    for mode in ("joint", "independent"):
        S = evaluate(B, draw(B, n, seed, mode))
        for m, gk in METRICS.items():
            x = S[m].values
            row[f"{mode} {m} median"] = float(np.median(x))
            row[f"{mode} {m} p10"], row[f"{mode} {m} p90"] = (float(v) for v in np.percentile(x, [10, 90]))
            row[f"{mode} {m} actual pct rank"] = float((x < act[m]).mean() * 100)   # where the actual fell in the runs
            g = B["guidance"].get(gk)
            if g:
                row[f"{mode} {m} share below guidance"] = float((x < g["low"]).mean() * 100)
                row[f"{mode} {m} share above guidance"] = float((x > g["high"]).mean() * 100)
        if mode == "joint":       # pieces, for "where does it miss"
            row.update({"pax revenue % median": float(S.passenger_revenue_pct.median()), "fuel price median": float(S.fuel_price.median()),
                        "nonfuel cost median": float(S.nonfuel_cost.median()), "other revenue median": float(S.other_revenue.median()),
                        "load factor chg median": float(S.load_factor_chg_pts.median()), "yield % median": float(S.yield_pct.median())})
    # candidate F: Delta's guidance midpoint + the fuel news since then (flightpath.forecast, notebook 07 tests the
    # other news channels); its range here is the simulator's spread around it, as first tested
    RJ = draw(B, n, seed, "joint")
    S = evaluate(B, RJ)
    for m, gk in METRICS.items():
        g = B["guidance"].get(gk)
        if not g:
            continue
        gd = pd.Timestamp(g["filed"]) + pd.Timedelta(days=1)
        Sf = evaluate(counterfactual(B, B, gd, ("fuel",)), RJ)
        xf = (g["low"] + g["high"]) / 2 + (S[m].values - np.median(Sf[m].values))
        row[f"F {m} median"] = float(np.median(xf))
        row[f"F {m} p10"], row[f"F {m} p90"] = (float(v) for v in np.percentile(xf, [10, 90]))
        row[f"F {m} share below guidance"] = float((xf < g["low"]).mean() * 100)
        row[f"F {m} share above guidance"] = float((xf > g["high"]).mean() * 100)
        row[f"F {m} fuel news"] = float(np.median(S[m].values) - np.median(Sf[m].values))
        row[f"guidance filed {m}"] = g["filed"]
    for m, gk in METRICS.items():
        row[f"actual {m}"] = float(act[m])
        g = B["guidance"].get(gk)
        row[f"guidance {m} low"] = g["low"] if g else np.nan
        row[f"guidance {m} high"] = g["high"] if g else np.nan
    bi = B["base_inputs"]
    row.update({"capacity est %": bi["capacity"], "demand est %": bi["demand"], "fuel market chg %": bi["fuel"],
                "days known (OpenSky)": B["cap"]["days_known"], "fuel days still to come": B["market_days_left"],
                "unit cost guided": B["unit_cost_guided_pct"] is not None,
                "n capacity errors": len(B["cap"]["measurement_errors_pct"]), "n demand errors": len(B["rev"]["demand_conversion_errors"]),
                "n joint quarters": len(B["ce"]["joint_surprises"]["quarters"]), "n eps bridge errors": len(B["ce"]["eps_bridge"]["errors_usd"]),
                "mro definition": B["mro_definition"]})
    return row


def run(ctx: Context, quarters, offsets=OFFSETS, n=50_000, log=print) -> pd.DataFrame:
    rows = []
    for q in quarters:
        for off in offsets:
            try:
                rows.append(run_one(ctx, q, off, n))
            except Exception as e:                      # e.g. a date before last year's quarter was reported
                log(f"{q} {off}: skipped ({e})")
    return pd.DataFrame(rows)
