"""The forecaster: Delta's guidance, updated with the live news since it was given (notebook 07).

    forecast = middle of Delta's guidance range  +  what the simulator says the news since then is worth

"News" is measured channel by channel, with the simulator run twice on the SAME base and the SAME random draws:
once with the channel's input as it stands today, once as it stood the day after Delta gave the guidance (the
counterfactual: "what Delta knew"). Everything else is identical, so the simulator's own level errors cancel and
only the effect of the news remains (notebook 06, section 4).

Channels:
  fuel      market jet fuel known up to today vs only up to the guidance date (rest at that day's price) - moves fuel
            cost and, through the measured link, fares;
  capacity  our capacity estimate today vs on the guidance date (OpenSky; with Delta's capacity plan when it gives one);
  demand    our industry demand estimate (TSA, converted) today vs on the guidance date;
  rivals    rivals' flying (OpenSky, data.rivals.LiveRivals) today vs on the guidance date, turned into Delta's fares
            with the measured link (+1% rival seats -> about -0.44% yield, refitted on what was public each day),
            and run through the simulator as extra fares (Levers.fare_pct);
  cpi       airline fares in the consumer price index (BLS), the quarter's months published by today vs on the
            guidance date, turned into Delta's fares with the measured link (refitted each day), as extra fares.

Range: the paired run-by-run difference (the uncertainty of the news effect itself) plus Delta's own guidance misses
at the same point of past quarters (only quarters reported by then), centered - no direction, as everywhere else.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..simulation.quarter import Levers, evaluate

CHANNELS = ("fuel", "capacity", "demand")          # all combinations are tested in notebook 07, section 2
EXTRA_CHANNELS = ("rivals", "cpi")                       # tested next to fuel (section 2b)
GUIDED = {"eps": "eps", "operating_margin": "operating_margin", "revenue_pct": "total_revenue_yoy"}


def guidance_date(B: dict, metric: str = "eps") -> pd.Timestamp | None:
    g = B["guidance"].get(GUIDED[metric])
    return None if g is None else pd.Timestamp(g["filed"]) + pd.Timedelta(days=1)


def counterfactual(B: dict, Bg: dict, gd: pd.Timestamp, channels) -> dict:
    """The base B with each channel's input put back to what it was on the guidance date (Bg = base as of then)."""
    C = dict(B)
    if "fuel" in channels:
        C["as_of"] = gd
    if "capacity" in channels:
        C["cap"] = dict(B["cap"], consolidated_seat_miles=Bg["cap"]["consolidated_seat_miles"])
    if "demand" in channels:
        C["base_inputs"] = dict(B["base_inputs"], demand=Bg["base_inputs"]["demand"])
    return C


def rivals_news(ctx, q, as_of, gd) -> dict:
    """The rivals channel's input: rival growth now and on the guidance date, the link, and the fares news (%)."""
    L = ctx.live_rivals()
    now, then = L.growth(q, as_of), L.growth(q, gd)
    beta, se, n = ctx.rivals_link(as_of)
    news = beta * (now["growth_pct"] - then["growth_pct"])
    return {"growth_now_pct": now["growth_pct"], "growth_at_guidance_pct": then["growth_pct"], "days_now": now["days"],
            "link": beta, "link_se": se, "link_quarters": n, "fare_news_pct": 0.0 if np.isnan(news) else float(news)}


def cpi_news(ctx, q, as_of, gd) -> dict:
    from ..data.cpi import growth_as_of
    beta, se, n = ctx.cpi_link(as_of)
    now, then = growth_as_of(ctx.cpi, q, as_of), growth_as_of(ctx.cpi, q, gd)
    news = beta * (now["growth_pct"] - then["growth_pct"])
    return {"growth_now_pct": now["growth_pct"], "growth_at_guidance_pct": then["growth_pct"], "months_now": now["months"],
            "link": beta, "link_se": se, "link_quarters": n, "fare_news_pct": 0.0 if np.isnan(news) else float(news)}


def guidance_misses(ctx, q, as_of, metric: str) -> np.ndarray:
    """Actual minus the middle of Delta's guidance in force at the same point of past quarters (reported by as_of)."""
    from ..simulation.general import actual
    q, as_of = pd.Period(q, "Q"), pd.Timestamp(as_of)
    off = as_of - q.start_time
    out = []
    for p in ctx.Dc.index:
        if p >= q or ctx.filed(p) > as_of or p < pd.Period("2022Q2", "Q"):
            continue
        g = ctx.guided_mid(GUIDED[metric], p, min(p.start_time + off, ctx.filed(p) - pd.Timedelta(days=1)))
        if pd.notna(g):
            a = actual(ctx, p)[metric]
            if pd.notna(a):
                out.append(a - g)
    return np.asarray(out, float)


def forecast(ctx, B: dict, R: dict, channels=CHANNELS, Bg: dict | None = None, metrics=tuple(GUIDED), seed: int = 0,
             S: pd.DataFrame | None = None) -> dict:
    """Guidance + news for each metric. Returns {metric: {...}} with the middle of guidance, the news effect
    (median of the paired run-by-run differences), the forecast, and its runs (for ranges and probabilities).
    Bg (the base as of the day after the guidance) and S (evaluate(B, R)) can be passed in to save time."""
    from ..simulation.general import build_base
    q, as_of = pd.Period(B["quarter"], "Q"), pd.Timestamp(B["as_of"])
    S = evaluate(B, R) if S is None else S
    rng = np.random.default_rng(seed)
    out, cache = {}, {}
    for m in metrics:
        g = B["guidance"].get(GUIDED[m])
        if g is None:
            continue
        gd = pd.Timestamp(g["filed"]) + pd.Timedelta(days=1)
        if gd not in cache:
            if "capacity" in channels or "demand" in channels:
                Bgd = Bg if (Bg is not None and pd.Timestamp(Bg["as_of"]) == gd) else build_base(ctx, q, gd)
            else:
                Bgd = B
            cache[gd] = evaluate(counterfactual(B, Bgd, gd, channels), R)
        Sc = cache[gd]
        Sn = S
        fare_ch = [c for c in ("rivals", "cpi") if c in channels]        # today's run with the fares news
        if fare_ch:
            k = (tuple(fare_ch), gd)
            if k not in cache:
                info = {c: (rivals_news if c == "rivals" else cpi_news)(ctx, q, as_of, gd) for c in fare_ch}
                fp = (np.prod([1 + i["fare_news_pct"] / 100 for i in info.values()]) - 1) * 100
                cache[k] = (info, evaluate(B, R, Levers(fare_pct=fp)) if fp else S)
            Sn = cache[k][1]
        diff = Sn[m].values - Sc[m].values                               # paired: same draws, only the news differs
        mid = (g["low"] + g["high"]) / 2
        miss = guidance_misses(ctx, q, as_of, m)
        n_miss = len(miss)
        miss = miss - miss.mean() if n_miss else np.zeros(1)
        runs = mid + diff + rng.choice(miss, len(diff))
        out[m] = {"guidance_low": g["low"], "guidance_high": g["high"], "guidance_mid": mid, "guidance_filed": g["filed"],
                  "news": float(np.median(diff)), "forecast": float(mid + np.median(diff)), "runs": runs,
                  "n_guidance_misses": n_miss,
                  "fares_news": cache[(tuple(fare_ch), gd)][0] if fare_ch else None}
    return out
