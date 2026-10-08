"""Capacity module: Delta available seat miles (ASM) for a quarter.

Capacity is set in advance by schedules, so it is a supply-side state, not a response to
same-quarter demand. Two estimators, combined when both exist:

  1. OpenSky nowcast (edge): OpenSky seat-hours (seats by aircraft type x hours tracked) scaled by
     ratio = BTS ASM / OpenSky seat-hours, learned on the last few months BTS has published (as_of!).
     Validation 2019-2021 (notebook 01): flight coverage 98-100% since 2020; seat-hours with a
     3-month-lagged ratio -> ~3.5% monthly / ~3% quarterly error in normal periods, worse when the
     domestic/international mix shifts fast. So OpenSky is weighted by its (larger) error and mainly
     catches DEVIATIONS from guidance (outages, cancellations, cuts).
  2. Guidance / trend: prior-year quarter x (1 + expected YoY growth). Growth comes from Delta's
     capacity guidance when provided, else the trailing YoY trend of reported ASM.

Reported (consolidated) ASM > T-100 mainline ASM (regional partners). All BTS-based estimates are
converted to the reported basis with a mainline->consolidated ratio learned from past quarters.

Also: BTS air carrier employment (P1) and competitor capacity events (stretch) - not yet wired.
Output: asm (paths)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Module


def _quarter_months(quarter: str) -> pd.DatetimeIndex:
    q = pd.Period(quarter, freq="Q")
    return pd.date_range(q.start_time, periods=3, freq="MS")


class CapacityModule(Module):
    name = "capacity"
    inputs = ()
    outputs = ("asm",)

    def __init__(self, lookback_months: int = 3, guidance_yoy: float | None = None,
                 guidance_sd: float = 0.01, proxy: str = "opensky.dl.seat_hours",
                 opensky_min_cv: float = 0.04):
        self.lookback = lookback_months
        self.guidance_yoy = guidance_yoy      # e.g. 0.03 for "capacity up ~3%"
        self.guidance_sd = guidance_sd
        self.proxy = proxy
        self.opensky_min_cv = opensky_min_cv     # floor from the validation (~4% quarterly error)
        self.params: dict = {}

    # ---------------------------------------------------------------- fit
    def fit(self, store, as_of) -> None:
        p = {}
        bts = store.series_as_of("bts.t100.dl.asm", as_of)
        osk = store.series_as_of(self.proxy, as_of)
        both = pd.concat([bts, osk], axis=1, join="inner").dropna().tail(self.lookback)
        if len(both) >= 3:
            ratio = both.iloc[:, 0] / both.iloc[:, 1]
            p["coverage_ratio"] = float(ratio.median())
            p["coverage_ratio_cv"] = float(ratio.std(ddof=1) / ratio.mean())
        # mainline (T-100) -> consolidated (reported) ratio, by quarter
        rep = store.series_as_of("delta.reported.asm_m", as_of) * 1e6
        if len(bts) and len(rep):
            bq = bts.groupby(bts.index.to_period("Q")).agg(["sum", "count"])
            bq = bq[bq["count"] == 3]["sum"]
            rq = rep.groupby(rep.index.to_period("Q")).sum()
            # last 4 quarters: the mainline share drifts (90% in 2019 -> ~93% since 2022, notebook 00)
            j = pd.concat([rq, bq], axis=1, join="inner").dropna().tail(4)
            if len(j):
                p["consolidated_ratio"] = float((j.iloc[:, 0] / j.iloc[:, 1]).median())
        p["reported_asm"] = rep
        self.params = p

    # ---------------------------------------------------------------- nowcast
    def nowcast(self, store, as_of, quarter: str) -> dict:
        """Mean and sd of the quarter's reported-basis ASM, with the method used."""
        p = self.params
        estimates = []
        months = _quarter_months(quarter)

        # 1. OpenSky-based, if every month of the quarter has OpenSky data by as_of
        osk = store.series_as_of(self.proxy, as_of)
        if "coverage_ratio" in p and all(m in osk.index for m in months):
            mainline = osk.loc[months].sum() * p["coverage_ratio"]
            cons = mainline * p.get("consolidated_ratio", 1.0)
            estimates.append(("opensky", cons, cons * max(p["coverage_ratio_cv"], self.opensky_min_cv)))

        # 2. prior-year quarter x (1 + growth)
        rep = p.get("reported_asm")
        q = pd.Period(quarter, freq="Q")
        if rep is not None and len(rep):
            rq = rep.groupby(rep.index.to_period("Q")).sum()
            if (q - 4) in rq.index:
                if self.guidance_yoy is not None:
                    g, sd = self.guidance_yoy, self.guidance_sd
                else:
                    # momentum: latest reported YoY growth (capacity plans are persistent);
                    # sd from how much YoY growth moves quarter to quarter
                    rq_full = rq.reindex(pd.period_range(rq.index.min(), rq.index.max(), freq="Q"))
                    yoy = (rq_full / rq_full.shift(4) - 1).dropna()
                    yoy = yoy[yoy.index > pd.Period("2022Q1", "Q")]
                    g = float(yoy.iloc[-1]) if len(yoy) else 0.0
                    sd = float(max(yoy.diff().dropna().abs().mean(), 0.01)) if len(yoy) > 2 else 0.03
                base = rq.loc[q - 4]
                estimates.append(("trend" if self.guidance_yoy is None else "guidance", base * (1 + g), base * sd))

        if not estimates:
            raise ValueError(f"no capacity estimate possible for {quarter} as of {as_of}")
        # inverse-variance combination
        w = np.array([1 / e[2] ** 2 for e in estimates])
        mean = float(np.dot(w, [e[1] for e in estimates]) / w.sum())
        sd = float(np.sqrt(1 / w.sum()))
        return {"mean": mean, "sd": sd, "components": {e[0]: (e[1], e[2]) for e in estimates}}

    # ---------------------------------------------------------------- simulate
    def simulate(self, state, shocks, rng):
        nc = state.meta["nowcast"]["capacity"]
        z = shocks.get("capacity", rng.standard_normal(state.n_paths))
        state.asm = nc["mean"] + nc["sd"] * z
        return state
