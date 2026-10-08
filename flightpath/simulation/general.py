"""The general engine: the simulator's base for ANY quarter, standing on ANY date, using only what was public that day
(notebook 06).

Notebook 05 ran the simulator for one worked example (Q3 2026 as of 7 October) from the results notebooks 01-04 saved.
This module rebuilds every one of those inputs from the raw data, as of a date `as_of`, with the same rules:

  capacity  (01) OpenSky seat-hours for the days of the quarter already flown (public the next day), outage days
                 filled, divided by the ratio to BTS of the 3 latest months BTS has published; the rest of the quarter
                 = BTS for the same dates a year earlier x the growth so far (the last 28 days if fewer than 7 days are
                 known; rule chosen in notebook 06); scaled to consolidated with the share rule "same quarter last
                 year + recent drift", from shares public that day.
  demand    (03) TSA growth so far this quarter vs the same dates a year earlier (the last 28 days if fewer than 7 days
                 are known), minus the average TSA - T-100 gap of the last 4 published quarters (from Q3 2022).
  fuel      (02) priced days known; days still to come at the last known price; translator refitted on the last 8
                 normal quarters filed by then; gallons = same quarter last year x the latest known yearly change.
  links     (03, 04) every model refitted on the quarters public that day.
  errors    every error history is the one that would have been known that day; capacity and demand errors are
            measured at the SAME point in the quarter (the same number of days from its start), because an estimate
            made on day 20 is less certain than one made after the quarter ends.
  guidance  Delta's guidance for the quarter in force that day (notebook 04).

If last quarter's results are not out yet (e.g. forecasting next quarter), last quarter is simulated first, as of the
same date, and its median results stand in for "last quarter" in the links.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from ..data.bts import T100_LAG_DAYS
from ..data.capacity_measure import (apply_seats, by_type, correct_for_outages, fill_outage_days, fill_with_flags,
                                     opensky_outages, published_months,
                                     seat_table_as_of)
from ..models import cost_links as CL
from ..models import revenue_links as RL
from .quarter import COVID_FUEL, Levers, draw, evaluate, market_quarter

SHARE_RULE = "same quarter last year + recent drift"
INCL_PS = lambda q: q < pd.Period("2018Q1", "Q") or q == pd.Period("2018Q3", "Q")   # CASM-Ex incl. profit sharing
COVID_SHARE = lambda q: pd.Period("2020Q1", "Q") <= q <= pd.Period("2022Q2", "Q")


class Context:
    """All raw data, loaded once."""
    OUTAGES = "rivals"        # OpenSky outage rule: "rivals" (notebook 07 detector) or "delta" (notebook 01's rule)
    RATIO_SKIP_OUTAGES = True # leave months with an outage day out of the OpenSky/BTS correction ratio

    def __init__(self, data_dir=None):
        from ..paths import DATA
        from ..data.bts import load_t100, load_t100_industry
        from ..data.eia import load_eia
        from ..data.fleet import load_aircraft_db
        from ..data.guidance import load_extracted
        from ..data.opensky import load_trino_flights
        from ..data.sec import reported_quarters
        from ..data.sec_adjusted import adjusted_quarters
        from ..data.tsa import load_tsa
        from ..validation.reconcile import quarterly_totals
        self.data = DATA if data_dir is None else data_dir
        # capacity (01)
        self.mainline = load_t100(carrier="DL")
        self.truth = self.mainline.groupby("period").asm.sum()                         # BTS mainline ASM, monthly
        self.q_true = self.truth.groupby(self.truth.index.to_period("Q")).sum(min_count=3)
        osk = load_trino_flights(aircraft_db=load_aircraft_db())
        day = osk.day.dt.tz_localize(None).dt.normalize()
        flights = osk.groupby(day).size().rename("flights").to_frame()
        _, self.flagged = fill_outage_days(flights)
        hours = ((osk.lastseen - osk.firstseen).dt.total_seconds() / 3600).clip(0, 20)
        H_raw = by_type(hours, day, osk.typecode)
        self.H_day_delta_rule = fill_with_flags(H_raw, self.flagged)          # notebook 01's rule (Delta's own counts)
        try:                                                                     # rivals-based detector (notebook 07)
            from ..data.rivals import opensky_daily_counts
            self.outages = opensky_outages(opensky_daily_counts(self.data))
            start = self.outages.index[0] + pd.Timedelta(days=21)                # detector needs 3 weeks of history
            before = self.H_day_delta_rule.loc[:start - pd.Timedelta(days=1)]
            after = correct_for_outages(H_raw.reindex(pd.date_range(H_raw.index[0], H_raw.index[-1])).fillna(0),
                                        self.outages).loc[start:]
            self.H_day_rivals_rule = pd.concat([before, after])
        except FileNotFoundError:
            self.outages, self.H_day_rivals_rule = None, None
        self.H_day = self.H_day_rivals_rule if (self.OUTAGES == "rivals" and self.H_day_rivals_rule is not None) \
            else self.H_day_delta_rule
        self.osk_last_day = day.max()
        del osk
        rq = reported_quarters()
        self.rep = rq[rq.quarter >= "2009Q1"].assign(q=lambda d: pd.PeriodIndex(d.quarter, freq="Q")).set_index("q")
        self.sh = (quarterly_totals(self.mainline).asm / 1e6 / self.rep.asm_m).dropna()
        self.share_known = pd.Series({k: max(k.end_time.normalize() + pd.Timedelta(days=T100_LAG_DAYS), self.rep.loc[k, "filed"])
                                      for k in self.sh.index})
        # demand, fuel, links (02, 03)
        self.industry = load_t100_industry()
        self.tsa = load_tsa().set_index("date").throughput.sort_index()
        self.jet = load_eia()["jet_gulf_coast"].set_index("date")["value"].sort_index()
        self.Dr = RL.quarterly_frame(self.rep, self.industry, self.jet)
        full_q = self.tsa.groupby(self.tsa.index.to_period("Q")).agg(["sum", "count"])
        full_q = full_q[full_q["count"] == [len(pd.date_range(q.start_time, q.end_time.normalize())) for q in full_q.index]]["sum"]
        tsa_yoy = ((full_q / full_q.shift(4) - 1) * 100).dropna()
        self.gap = (tsa_yoy - self.Dr.demand).dropna()                                 # TSA - T-100 growth, pts
        self.fuel_model = json.loads((self.data / "derived" / "fuel_model.json").read_text())
        # costs, EPS (04)
        self.A = adjusted_quarters(self.data)
        self.Dc = CL.quarterly_frame(self.A)
        self.G = load_extracted(self.data)
        self.price = self.A["fuel_price_gal_adj"].fillna(self.A["fuel_price_gal"]).astype(float)
        self.gpa = (self.A.fuel_gallons_m / self.A.asm_m).astype(float)
        self._cap_cache, self._dem_cache = {}, {}

    def set_outage_rule(self, rule: str = "rivals", skip_months: bool = True):
        """Switch the OpenSky outage rule ("rivals" = notebook 07's detector, "delta" = notebook 01's rule) and
        clear the cached estimates that depend on it."""
        self.OUTAGES, self.RATIO_SKIP_OUTAGES = rule, skip_months
        self.H_day = self.H_day_rivals_rule if (rule == "rivals" and self.H_day_rivals_rule is not None) else self.H_day_delta_rule
        self._cap_cache, self._dem_cache = {}, {}

    # ------------------------------------------------------------------------------------------- helpers
    def filed(self, q) -> pd.Timestamp:
        return self.A.loc[q, "filed"] if q in self.A.index else pd.Timestamp.max

    def reported_by(self, as_of) -> list:
        return [p for p in self.A.index if self.A.loc[p, "filed"] <= pd.Timestamp(as_of)]

    # ------------------------------------------------------------------------------------------- capacity (01)
    FILL = "bts_ly"     # chosen in notebook 06, section 2

    def mainline_estimate(self, q, as_of, fill: str | None = None) -> dict:
        """Delta mainline seat-miles for quarter q, BTS-equivalent, as of `as_of`: notebook 01's method for the days
        already flown, plus a rule for the days still to come (`fill`):
          "opensky_ly": the same days a year earlier in OpenSky (same weekday, 364 days back) x OpenSky's growth so far;
          "bts_ly":     the same dates a year earlier in BTS (each month spread evenly over its days) x the growth so
                        far of our BTS-equivalent measure over BTS on those dates.
        Growth so far uses the quarter's known days, or the last 28 days if fewer than 7 are known."""
        fill = fill or self.FILL
        q, as_of = pd.Period(q, "Q"), pd.Timestamp(as_of)
        key = (q, as_of, fill)
        if key in self._cap_cache:
            return self._cap_cache[key]
        last = min(as_of - pd.Timedelta(days=1), self.osk_last_day)                  # OpenSky: public the next day
        seats, avg = seat_table_as_of(self.mainline, as_of)
        S = apply_seats(self.H_day.loc[:last], seats, avg)
        months = S.groupby(S.index.to_period("M").start_time).sum()
        pub = published_months(self.truth, as_of)
        pub = pub[pub < q.start_time]
        if self.RATIO_SKIP_OUTAGES and self.outages is not None and self.OUTAGES == "rivals":
            o = self.outages[(self.outages.full | self.outages.partial)].index
            bad = set(o.to_period("M").start_time)
            pub = pd.DatetimeIndex([m for m in pub if m not in bad])
        pub = pub[-3:]
        ratio = (months.reindex(pub) / self.truth.reindex(pub)).mean()
        days = pd.date_range(q.start_time, q.end_time.normalize())
        obs, rest = days[days <= last], days[days > last]
        w, how = (obs, "growth so far this quarter") if len(obs) >= 7 else \
            (pd.date_range(last - pd.Timedelta(days=27), last), "growth over the last 28 days")
        if fill == "opensky_ly":
            ly = lambda d: S.reindex(d - pd.Timedelta(days=364))
            g = S.reindex(w).sum() / ly(w).sum()
            est = (S.reindex(obs).sum() + ly(rest).sum() * g) / ratio
        else:
            pub_all = published_months(self.truth, as_of)
            def bts_ly(d):            # BTS a year earlier, each month spread evenly over its days (published months only)
                d1 = d - pd.DateOffset(years=1)
                m = d1.to_period("M").start_time
                if not pd.Index(m.unique()).isin(pub_all).all():
                    return np.nan
                return float((self.truth.reindex(m).values / m.days_in_month.values).sum())
            g = (S.reindex(w).sum() / ratio) / bts_ly(w)
            est = S.reindex(obs).sum() / ratio + (bts_ly(rest) * g if len(rest) else 0.0)
        out = {"mainline": float(est), "days_known": len(obs), "days_filled": len(rest), "fill_growth_pct": (g - 1) * 100,
               "fill_rule": f"{fill}: {how}", "ratio_months": [f"{m:%Y-%m}" for m in pub]}
        self._cap_cache[key] = out
        return out

    def share(self, q, as_of, rule=SHARE_RULE) -> float:
        known = self.share_known[self.share_known <= pd.Timestamp(as_of)].index
        k = max(p for p in known if p <= q - 1)
        if rule == "previous known quarter":
            return self.sh[k]
        return self.sh[q - 4] + (self.sh[k] - self.sh[k - 4])

    # ------------------------------------------------------------------------------------------- rivals (07)
    def live_rivals(self):
        """Rivals' flying from OpenSky (data.rivals.LiveRivals), built on first use (~2 minutes)."""
        if getattr(self, "_live_rivals", None) is None:
            from ..data.opensky import load_trino_flights
            from ..data.rivals import LiveRivals
            dl = load_trino_flights(self.data, aircraft_db=pd.DataFrame({"icao24": [], "typecode": []}))
            self._live_rivals = LiveRivals(dl[["day", "origin", "destination"]], self.outages, self.data)
        return self._live_rivals

    def rivals_history(self) -> pd.DataFrame:
        """Quarterly T-100 rival growth (data.rivals.rival_growth) joined to the revenue-link frame."""
        if getattr(self, "_rivals_hist", None) is None:
            from ..data.rivals import market_capacity, rival_growth
            self._rivals_hist = self.Dr.join(rival_growth(market_capacity(self.data)))
        return self._rivals_hist

    RIVALS_FEATURES = ["yield_last", "dlf", "fuel", "rivals_on_delta_markets"]

    def rivals_link(self, as_of) -> tuple[float, float, int]:
        """+1% rival seats on Delta's markets -> x% on Delta's yield: notebook 03's yield model + T-100 rival growth,
        fitted on the usable quarters public by `as_of` (as in notebook 07, section 2). Returns (coefficient, standard error, quarters)."""
        key = pd.Timestamp(as_of)
        cache = self.__dict__.setdefault("_rl_cache", {})
        if key not in cache:
            Dy, f = self.rivals_history(), self.RIVALS_FEATURES
            rows = RL.training_rows(Dy, key, f, "yield")
            m = RL.fit(rows, f, "yield")
            cache[key] = (float(m["beta"]["rivals_on_delta_markets"]),
                          float(np.sqrt(m["cov"].loc["rivals_on_delta_markets", "rivals_on_delta_markets"])), int(m["n"]))
        return cache[key]

    CPI_FEATURES = ["yield_last", "dlf", "fuel", "cpi_fares"]

    def cpi_link(self, as_of) -> tuple[float, float, int]:
        """+1% airline-fare CPI (same quarter, year over year) -> x% on Delta's yield: notebook 03's yield model + the
        CPI, fitted on the usable quarters public by `as_of`. Returns (coefficient, standard error, quarters)."""
        from ..data.cpi import load_airfare_cpi, quarterly_yoy
        if getattr(self, "cpi", None) is None:
            self.cpi = load_airfare_cpi(self.data)
            self._cpi_hist = self.Dr.join(quarterly_yoy(self.cpi).rename("cpi_fares"))
        key = ("cpi", pd.Timestamp(as_of))
        cache = self.__dict__.setdefault("_rl_cache", {})
        if key not in cache:
            f = self.CPI_FEATURES
            m = RL.fit(RL.training_rows(self._cpi_hist, key[1], f, "yield"), f, "yield")
            cache[key] = (float(m["beta"]["cpi_fares"]), float(np.sqrt(m["cov"].loc["cpi_fares", "cpi_fares"])), int(m["n"]))
        return cache[key]

    # ------------------------------------------------------------------------------------------- demand (03)
    GAP_FROM = pd.Period("2022Q3", "Q")     # earlier quarters' gaps are distorted by COVID (-6 to +0.5 pts)

    def gap_as_of(self, as_of, n=4) -> float:
        """Average TSA - T-100 gap of the last `n` quarters T-100 had published (from Q3 2022; on 7 Oct 2026 this is
        notebook 03's rule exactly - the window start only matters for dates before 2024)."""
        pub = [p for p in self.gap.index if p.end_time.normalize() + pd.Timedelta(days=T100_LAG_DAYS) <= pd.Timestamp(as_of)
               and p >= self.GAP_FROM]
        return self.gap[pub[-n:]].mean() if pub else np.nan

    def tsa_growth(self, q, as_of) -> tuple[float, int, str]:
        """TSA growth for quarter q: days already known vs the same dates a year earlier; the rest of the quarter at
        that growth (i.e. the growth so far). Fewer than 7 days known: the last 28 days."""
        last = min(pd.Timestamp(as_of) - pd.Timedelta(days=1), self.tsa.index.max())
        days = pd.date_range(q.start_time, q.end_time.normalize())
        obs = days[days <= last]
        if len(obs) < 7:
            obs, how = pd.date_range(last - pd.Timedelta(days=27), last), "last 28 days"
        else:
            how = "growth so far this quarter"
        g = (self.tsa.reindex(obs).sum() / self.tsa.reindex(obs - pd.DateOffset(years=1)).sum() - 1) * 100
        return float(g), int((days <= last).sum()), how

    def demand(self, q, as_of) -> float:
        key = (q, pd.Timestamp(as_of))
        if key not in self._dem_cache:
            self._dem_cache[key] = self.tsa_growth(q, as_of)[0] - self.gap_as_of(as_of)
        return self._dem_cache[key]

    # ------------------------------------------------------------------------------------------- error histories
    def error_histories(self, q, as_of) -> dict:
        """Capacity measurement, capacity scaling and demand errors, each measured at the same number of days from
        the start of past quarters as `as_of` is from the start of q, and only for quarters whose truth was public."""
        q, as_of = pd.Period(q, "Q"), pd.Timestamp(as_of)
        q_end = q.end_time.normalize()
        if as_of > q_end:                       # after the quarter: the same number of days after past quarters' ends
            off = as_of - q_end
            t = lambda p: p.end_time.normalize() + off
        else:                                   # during or before: the same number of days from their starts
            off = as_of - q.start_time
            t = lambda p: p.start_time + off
        pub_q = [p for p in self.q_true.dropna().index
                 if p.end_time.normalize() + pd.Timedelta(days=T100_LAG_DAYS) <= as_of and t(p) <= as_of]
        meas = {p: (self.mainline_estimate(p, t(p))["mainline"] / self.q_true[p] - 1) * 100
                for p in pub_q if p >= pd.Period("2022Q1", "Q")}
        scal = {}
        for p in self.sh.index:
            if p < pd.Period("2011Q1", "Q") or any(COVID_SHARE(p - i) for i in range(7)) or self.share_known[p] > as_of:
                continue
            try:
                scal[p] = (self.sh[p] / self.share(p, t(p)) - 1) * 100
            except (ValueError, KeyError):
                continue
        dem = {}
        for p in self.gap.index:
            if p.end_time.normalize() + pd.Timedelta(days=T100_LAG_DAYS) > as_of or t(p) > as_of or p < self.GAP_FROM:
                continue
            v = self.demand(p, t(p))
            if pd.notna(v) and pd.notna(self.Dr.loc[p, "demand"]):
                dem[p] = v - self.Dr.loc[p, "demand"]
        return {"measurement": pd.Series(meas, dtype=float), "scaling": pd.Series(scal, dtype=float), "demand": pd.Series(dem, dtype=float)}

    @lru_cache(maxsize=None)
    def eps_bridge_errors(self) -> pd.Series:
        """04's EPS bridge (non-operating and tax = average of last 4 known quarters, shares = last quarter), each
        quarter rebuilt from its actual operating income with what was public 7 days after it ended."""
        D = self.Dc
        out = {}
        for p in D.index[(D.index >= pd.Period("2023Q2", "Q")) & D.eps_adj.notna() & D.operating_income_adj_m.notna()]:
            after = p.end_time.normalize() + pd.Timedelta(days=7)
            known = [k for k in D.index if k < p and pd.notna(D.loc[k, "nonop_m"]) and pd.notna(D.loc[k, "tax_rate"])
                     and self.A.loc[k, "adjusted_public"] <= after][-4:]
            shares = D.loc[[k for k in D.index if k < p and self.A.loc[k, "filed"] <= after][-1], "diluted_shares_m"]
            out[p] = CL.eps(D.loc[p, "operating_income_adj_m"], D.loc[known, "nonop_m"].mean(), D.loc[known, "tax_rate"].mean(), shares) - D.loc[p, "eps_adj"]
        return pd.Series(out)

    def guided_mid(self, metric, q, as_of):
        from ..data.guidance import in_force
        it = in_force(self.G, metric, q, as_of)
        return np.nan if it is None else (it.range_low + it.range_high) / 2

    @lru_cache(maxsize=None)
    def guided_capacity_errors(self) -> pd.Series:
        """Delta's capacity guidance (in force at quarter end) minus the capacity growth it reported, by quarter."""
        D = self.Dr
        out = {}
        for p in D.index:
            if not D.loc[p, "usable"] or pd.isna(D.loc[p, "capacity"]):
                continue
            g = self.guided_mid("capacity_yoy", p, p.end_time.normalize())
            if pd.notna(g):
                out[p] = g - D.loc[p, "capacity"]
        return pd.Series(out, dtype=float)

    @lru_cache(maxsize=None)
    def guided_unit_cost_errors(self) -> pd.Series:
        """04's N3: Delta's guided unit cost growth (in force at quarter end), as non-fuel cost growth error."""
        D = self.Dc
        out = {}
        for p in D.index:
            if not D.loc[p, "usable"] or INCL_PS(p) or pd.isna(D.loc[p, "nonfuel"]):
                continue
            g = self.guided_mid("nonfuel_unit_cost_yoy", p, p.end_time.normalize())
            if pd.notna(g):
                out[p] = CL.nonfuel_growth(g, D.loc[p, "capacity"]) - D.loc[p, "nonfuel"]
        return pd.Series(out)


# ----------------------------------------------------------------------------------------------- the base
def _model(m, features):
    return {"features": features, "beta": m["beta"].to_dict(), "cov": m["cov"].values.tolist(),
            "resid": {str(k): float(v) for k, v in m["resid"].items()}}


def build_base(ctx: Context, quarter, as_of, lag_runs: int = 20_000) -> dict:
    """Everything evaluate() needs for `quarter`, as of `as_of`, from the raw data public that day."""
    q, as_of = pd.Period(quarter, "Q"), pd.Timestamp(as_of)
    A, Dr, Dc = ctx.A, ctx.Dr, ctx.Dc
    lag_days = ctx.fuel_model["purchase_delay_days"]
    known = ctx.reported_by(as_of)
    if q - 4 not in known:
        raise ValueError(f"{q - 4} (last year's quarter) not reported by {as_of:%Y-%m-%d}")
    # ---- last quarter: reported, or simulated as of the same date
    if q - 1 in known:
        last = {"dlf": Dr.loc[q - 1, "dlf"], "yield": Dr.loc[q - 1, "yield"], "unit_cost": Dc.loc[q - 1, "unit_cost"],
                "other": Dc.loc[q - 1, "other"], "source": "reported"}
    else:
        B1 = build_base(ctx, q - 1, as_of, lag_runs)
        S1 = evaluate(B1, draw(B1, lag_runs, seed=1, surprises="independent"))
        other_g = (S1.other_revenue / B1["other_ly"] - 1) * 100
        last = {"dlf": S1.load_factor_chg_pts.median(), "yield": S1.yield_pct.median(), "unit_cost": S1.unit_cost_pct.median(),
                "other": other_g.median(), "source": f"simulated as of {as_of:%Y-%m-%d} (not reported yet)"}
    # ---- capacity (01), combined with Delta's capacity guidance when it gives a number (06, section 2):
    # days already flown come from OpenSky; days still to come follow Delta's plan. Weight = share of the quarter known.
    me = ctx.mainline_estimate(q, as_of)
    share = ctx.share(q, as_of)
    rep_ly = ctx.rep.loc[q - 4, "asm_m"] * 1e6
    opensky_yoy = (me["mainline"] / share / rep_ly - 1) * 100
    errs = ctx.error_histories(q, as_of)
    guided_cap = ctx.guided_mid("capacity_yoy", q, as_of)
    w = me["days_known"] / (me["days_known"] + me["days_filled"])
    if pd.isna(guided_cap):
        cap_yoy, w_os, rule = opensky_yoy, 1.0, "OpenSky (no capacity number from Delta)"
    else:
        cap_yoy, w_os = w * opensky_yoy + (1 - w) * guided_cap, w
        rule = f"OpenSky for the {w:.0%} of the quarter flown + Delta's guided {guided_cap:+.1f}% for the rest"
    gce = ctx.guided_capacity_errors()
    cons = rep_ly * (1 + cap_yoy / 100)
    cap = {"mainline_seat_miles": me["mainline"], "share": share, "consolidated_seat_miles": cons,
           "consolidated_q3_2025_reported": rep_ly, "consolidated_yoy_pct": cap_yoy, "opensky_yoy_pct": opensky_yoy,
           "guided_yoy_pct": None if pd.isna(guided_cap) else float(guided_cap), "capacity_rule": rule,
           # each error history scaled by the weight of its piece
           "measurement_errors_pct": (w_os * errs["measurement"]).tolist(), "scaling_errors_pct": (w_os * errs["scaling"]).tolist(),
           "guidance_errors_pct": [] if w_os == 1.0 else [(1 - w_os) * v for p, v in gce.items() if ctx.filed(p) <= as_of],
           "days_known": me["days_known"], "days_filled": me["days_filled"], "fill_rule": me["fill_rule"],
           "fill_growth_pct": me["fill_growth_pct"], "ratio_months": me["ratio_months"]}
    # ---- demand (03)
    tsa_g, tsa_days, tsa_how = ctx.tsa_growth(q, as_of)
    gap = ctx.gap_as_of(as_of)
    demand = tsa_g - gap
    # ---- fuel (02)
    jet = ctx.jet[:as_of]
    hist = [p for p in A.index if A.loc[p, "filed"] <= as_of and pd.notna(ctx.price[p])
            and not (COVID_FUEL[0] <= p <= COVID_FUEL[1])][-ctx.fuel_model["translator_window_quarters"]:]
    mkt_hist = np.array([market_quarter(jet, p, lag_days)[0] for p in hist])
    b, a = np.polyfit(mkt_hist, ctx.price[hist].values, 1)
    k = max(p for p in known if pd.notna(ctx.gpa.get(p)) and pd.notna(ctx.gpa.get(p - 4)))
    gpa_q = ctx.gpa[q - 4] * ctx.gpa[k] / ctx.gpa[k - 4]
    mkt_ly = market_quarter(jet, q - 4, lag_days)[0]
    mkt_q, mkt_known, mkt_left = market_quarter(jet, q, lag_days, as_of)
    filed_by = lambda d: {p: v for p, v in d.items() if ctx.filed(pd.Period(p, "Q")) <= as_of}
    fuel = {"translator_errors_pct": filed_by(ctx.fuel_model["translator_errors_pct"]),
            "gallons_errors_pct": filed_by(ctx.fuel_model["gallons_errors_pct"])}
    # ---- revenue links (03), refitted as of the date
    lf_m = RL.fit(RL.training_rows(Dr, as_of, RL.LF_FEATURES, "dlf"), RL.LF_FEATURES, "dlf")
    y_m = RL.fit(RL.training_rows(Dr, as_of, RL.YIELD_FEATURES, "yield"), RL.YIELD_FEATURES, "yield")
    common = lf_m["resid"].index.intersection(y_m["resid"].index)
    base_inputs = {"capacity": cap["consolidated_yoy_pct"], "demand": demand, "fuel": (mkt_q / mkt_ly - 1) * 100,
                   "dlf_last": float(last["dlf"]), "yield_last": float(last["yield"])}
    rev = {"load_factor_model": _model(lf_m, RL.LF_FEATURES), "yield_model": _model(y_m, RL.YIELD_FEATURES),
           "paired_residuals": {"quarters": [str(x) for x in common], "load_factor": lf_m["resid"][common].tolist(),
                                "yield": y_m["resid"][common].tolist()},
           "demand_conversion_errors": errs["demand"].tolist(), "base_inputs": base_inputs,
           "load_factor_last_year": float(Dr.loc[q - 4, "lf"])}
    # ---- costs, profit sharing, EPS bridge (04), refitted as of the date
    or_m = RL.fit(CL.training_rows(Dc, as_of, ["other_last", "pax"], "other"), ["other_last", "pax"], "other")
    uc_m = RL.fit(CL.training_rows(Dc, as_of, ["unit_cost_last", "capacity"], "unit_cost"), ["unit_cost_last", "capacity"], "unit_cost")
    mro_rows = [p for p in known if pd.notna(A.loc[p, "mro_expense_m"]) and pd.notna(A.loc[p, "mro_expense_m_prior_year"])]
    new_def = q >= pd.Period("2026Q1", "Q") and len(mro_rows) > 0
    mro = ({"2026": [A.loc[p, "mro_expense_m"] / A.loc[p, "other_revenue_adj_m"] * 100 for p in mro_rows],
            "2025": [A.loc[p, "mro_expense_m_prior_year"] / A.loc[p, "other_revenue_adj_m_prior_year"] * 100 for p in mro_rows]}
           if new_def else {"2026": [0.0], "2025": [0.0]})
    ps_q = [p for p in known if p >= pd.Period("2023Q2", "Q") and pd.notna(Dc.loc[p, "ps_share"])]
    recent = [p for p in Dc.index if pd.notna(Dc.loc[p, "nonop_m"]) and pd.notna(Dc.loc[p, "tax_rate"])
              and A.loc[p, "adjusted_public"] <= as_of and p < q][-4:]
    shares = Dc.loc[[p for p in known if p < q][-1], "diluted_shares_m"]
    ebe = ctx.eps_bridge_errors()
    gue = ctx.guided_unit_cost_errors()
    uc_guided = np.nan if INCL_PS(q) else ctx.guided_mid("nonfuel_unit_cost_yoy", q, as_of)
    tr = pd.Series(fuel["translator_errors_pct"]).rename(lambda s: pd.Period(s, "Q"))
    J = pd.DataFrame({"load factor": lf_m["resid"], "yield": y_m["resid"], "fuel price (translator)": tr,
                      "non-fuel unit cost": uc_m["resid"], "other revenue": or_m["resid"]}).dropna()
    ce = {"other_revenue_model": _model(or_m, ["other_last", "pax"]), "unit_cost_model": _model(uc_m, ["unit_cost_last", "capacity"]),
          "mro_cost_share_pct": mro,
          "profit_sharing": {"share_pct": 20.0, "share_errors_pts": {str(p): float(Dc.loc[p, "ps_share"] - 20) for p in ps_q}},
          "eps_bridge": {"errors_usd": {str(p): float(v) for p, v in ebe.items() if A.loc[p, "filed"] <= as_of}},
          "unit_cost_guided": {"errors_pct": {str(p): float(v) for p, v in gue.items() if A.loc[p, "filed"] <= as_of} or {"_": 0.0}},
          "joint_surprises": {"quarters": [str(x) for x in J.index], "columns": list(J.columns), "values": J.values.tolist()},
          "base": {"unit_cost_last": float(last["unit_cost"]), "other_last": float(last["other"]),
                   "nonop_musd": float(Dc.loc[recent, "nonop_m"].mean()), "tax_rate_pct": float(Dc.loc[recent, "tax_rate"].mean()),
                   "diluted_shares_m": float(shares)}}
    # ---- guidance in force (04)
    from ..data.guidance import in_force
    guidance = {}
    for m in ["total_revenue_yoy", "operating_margin", "eps", "fuel_price_gal", "nonfuel_unit_cost_yoy", "capacity_yoy"]:
        it = in_force(ctx.G, m, q, as_of)
        if it is not None:
            guidance[m] = {"low": float(it.range_low), "high": float(it.range_high), "given_as": it.range_from,
                           "filed": f"{it.filed:%Y-%m-%d}", "quote": it.quote}
    return {"quarter": str(q), "as_of": as_of, "jet": jet, "lag": lag_days,
            "translator": (a, b), "translator_quarters": [str(p) for p in hist], "market_ly": mkt_ly,
            "market_days_known": mkt_known, "market_days_left": mkt_left, "gallons_per_asm": gpa_q,
            "cap": cap, "rev": rev, "fuel": fuel, "ce": ce, "base_inputs": base_inputs,
            "pax_ly": float(A.loc[q - 4, "passenger_revenue_m"]), "other_ly": float(A.loc[q - 4, "other_revenue_adj_m"]),
            "revenue_ly": float(A.loc[q - 4, "revenue_adj_m"]), "nonfuel_ly_old": float(A.loc[q - 4, "nonfuel_cost_m"]),
            "lf_ly": float(Dr.loc[q - 4, "lf"]), "guidance": guidance,
            "unit_cost_guided_pct": None if pd.isna(uc_guided) else float(uc_guided),
            "mro_definition": "2026 (MRO cost on its own line)" if new_def else "before 2026 (MRO cost inside non-fuel)",
            "last_quarter": last, "demand_detail": {"tsa_growth_pct": tsa_g, "tsa_days_known": tsa_days, "tsa_rule": tsa_how, "gap_pts": gap}}


def actual(ctx: Context, quarter) -> dict:
    """Delta's reported (adjusted) results for a quarter: what a forecast is scored against."""
    q = pd.Period(quarter, "Q")
    A = ctx.A
    return {"revenue_pct": (A.loc[q, "revenue_adj_m"] / A.loc[q, "revenue_adj_m_prior_year"] - 1) * 100,
            "operating_margin": A.loc[q, "operating_income_adj_m"] / A.loc[q, "revenue_adj_m"] * 100,
            "eps": A.loc[q, "eps_adj"], "filed": A.loc[q, "filed"]}
