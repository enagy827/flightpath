"""Backtest of the forecaster (notebook 07): Delta's guidance + news channels, at several dates in past quarters."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from ..forecast.update import CHANNELS, GUIDED, forecast
from ..simulation.general import Context, actual, build_base
from ..simulation.quarter import draw, evaluate
from .sim_backtest import as_of_date

OFFSETS = [20, 45, 70, "end+7"]          # the forecaster needs guidance, given ~10 days into the quarter
CHANNEL_SETS = [c for k in range(1, len(CHANNELS) + 1) for c in combinations(CHANNELS, k)]


def run(ctx: Context, quarters, offsets=OFFSETS, n=50_000, seed=0, log=print, channel_sets=None) -> pd.DataFrame:
    rows, gbase = [], {}
    for q in quarters:
        q = pd.Period(q, "Q")
        act = actual(ctx, q)
        for off in offsets:
            d = as_of_date(q, off)
            B = build_base(ctx, q, d)
            if "eps" not in B["guidance"]:
                log(f"{q} {off}: no EPS guidance in force"); continue
            gd = pd.Timestamp(B["guidance"]["eps"]["filed"]) + pd.Timedelta(days=1)
            if (q, gd) not in gbase:
                gbase[(q, gd)] = build_base(ctx, q, gd)
            R = draw(B, n, seed, "joint")
            S = evaluate(B, R)
            for chans in (channel_sets or CHANNEL_SETS):
                F = forecast(ctx, B, R, chans, Bg=gbase[(q, gd)], S=S, seed=seed)
                for m, f in F.items():
                    x = f["runs"]
                    rows.append({"quarter": str(q), "offset": str(off), "channels": "+".join(chans), "metric": m,
                                 "guidance_low": f["guidance_low"], "guidance_high": f["guidance_high"],
                                 "guidance_mid": f["guidance_mid"], "news": f["news"], "forecast": f["forecast"],
                                 "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)),
                                 "prob_below": float((x < f["guidance_low"]).mean() * 100),
                                 "prob_above": float((x > f["guidance_high"]).mean() * 100),
                                 "actual": float(act[m]), "n_guidance_misses": f["n_guidance_misses"],
                                 "simulator": float(S[m].median()),
                                 **{f"{c}_{k}": v for c, i in (f.get("fares_news") or {}).items()
                                    for k, v in i.items() if k in ("growth_now_pct", "growth_at_guidance_pct", "link", "fare_news_pct")}})
    return pd.DataFrame(rows)


# Delta's mid-quarter guidance updates in the backtest window (non-earnings 8-Ks, notebook 04's extraction), with the
# direction each one moved a metric. Direction from the filing's own words; "reaffirmed" = no change (0).
UPDATES = [
    ("2023Q3", "2023-09-14", "eps", -1, "Expect September quarter earnings of $1.85 to $2.05 per share (from $2.20-2.50)"),
    ("2023Q3", "2023-09-14", "operating_margin", -1, "operating margin of ~13% (from 14-16%)"),
    ("2024Q1", "2024-03-12", "revenue_pct", 1, "total revenue growth in the top half of the initial guidance range"),
    ("2024Q3", "2024-08-08", "revenue_pct", -1, "the direct revenue impact of the incident is estimated to be $380 million"),
    ("2024Q3", "2024-09-12", "eps", 1, "earnings per share is expected to be at the high-end of initial guidance"),
    ("2024Q3", "2024-09-12", "revenue_pct", 0, "Up 2% - 4% (reaffirmed)"),
    ("2025Q1", "2025-03-10", "eps", -1, "Earnings Per Share $0.30 - $0.50 (from $0.70-1.00)"),
    ("2025Q1", "2025-03-10", "operating_margin", -1, "Operating Margin 4% - 5% (from 6-8%)"),
    ("2025Q1", "2025-03-10", "revenue_pct", -1, "total revenue growth for the March quarter of 3 to 4 percent (from 7-9%)"),
    ("2025Q3", "2025-09-11", "revenue_pct", 1, "up 2 to 4 percent ... in the upper half of Delta's initial guidance range"),
    ("2026Q1", "2026-03-17", "eps", 0, "Expect March Quarter Earnings Within Initial Guidance Range (reaffirmed)"),
    ("2026Q1", "2026-03-17", "revenue_pct", 1, "1Q26 Updated Guide +HSD (from +5-7%)"),
]
NO_UPDATE = ["2023Q4", "2024Q2", "2024Q4", "2025Q2", "2025Q4", "2026Q2"]     # quarters with no mid-quarter update


def update_test(ctx: Context, days_before=(45, 30, 14, 1), control_days=(40, 55, 70), n=30_000, seed=0) -> pd.DataFrame:
    """Forecasts made shortly before each mid-quarter update (with the guidance then in force), and on comparable days
    of quarters with no update (to count false alarms). Forecaster = guidance + fuel news; simulator alone for comparison."""
    rows = []

    def one(q, d, tag, upd=None):
        B = build_base(ctx, q, d)
        if "eps" not in B["guidance"]:
            return
        R = draw(B, n, seed, "joint")
        S = evaluate(B, R)
        for m, f in forecast(ctx, B, R, ("fuel",), S=S, seed=seed).items():
            rows.append({"tag": tag, "quarter": q, "update_date": upd, "date": d, "metric": m,
                         "days_before": None if upd is None else (pd.Timestamp(upd) - d).days,
                         "day_of_quarter": (d - pd.Period(q, "Q").start_time).days,
                         "low": f["guidance_low"], "high": f["guidance_high"], "mid": f["guidance_mid"],
                         "forecaster": f["forecast"], "simulator": float(S[m].median())})
    for q, u in sorted({(q, u) for q, u, *_ in UPDATES}):
        for k in days_before:
            d = pd.Timestamp(u) - pd.Timedelta(days=k)
            if d >= pd.Period(q, "Q").start_time + pd.Timedelta(days=12):     # after the quarter's initial guidance
                one(q, d, "before update", u)
    for q in NO_UPDATE:
        for k in control_days:
            one(q, pd.Period(q, "Q").start_time + pd.Timedelta(days=k), "no update")
    return pd.DataFrame(rows)
