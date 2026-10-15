"""Data and calculations behind the FlightPath app (app/streamlit_app.py).

The app does not need the raw data (2 minutes to load, several GB). `build_snapshot` runs the engine once - for every
quarter Q3 2023 - Q4 2026, on every day from day 14 of the quarter to 14 days after it ends (or, with --weekly, every
7 days plus the backtest dates) - and saves what the app needs to app/snapshot/
(a few MB): each date's base case (inputs as public that day, error histories, links; market fuel only up to that
day), the inputs for the "news since guidance" counterfactual, the track record, the live signals and the event
library. Scenarios then run live in the app in a fraction of a second (`simulate`, `forecast`).

Rebuild after new data:   python scripts/build_app_snapshot.py
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .simulation.quarter import Levers, draw, evaluate
from .simulation.timing import rest_shares

SNAP = Path(__file__).resolve().parents[1] / "app" / "snapshot"
QUARTERS = [str(q) for q in pd.period_range("2023Q3", "2026Q4", freq="Q")]
GUIDED = {"eps": "eps", "operating_margin": "operating_margin", "revenue_pct": "total_revenue_yoy"}
# Which news each forecast line uses (notebook 07): EPS and margin = guidance + fuel news (section 2);
# revenue = guidance + fuel + capacity news (section 7e - provisional, adopted 7 Oct 2026).
CHANNELS_BY_METRIC = {"eps": ("fuel",), "operating_margin": ("fuel",), "revenue_pct": ("fuel", "capacity")}
# Range widening per metric (notebook 07, section 7h, 8 Oct 2026): the revenue 80% range held the actual only 59% of
# the time in the backtest (26 of 44 quarter-dates), so its spread around the forecast is stretched 1.75x - the smallest
# factor that brings coverage to 80% (36 of 44 = 82%; 77% when each quarter is left out of the choice in turn).
# EPS (90%) and margin (81%) already cover about 80% and are not changed.
RANGE_SCALE = {"eps": 1.0, "operating_margin": 1.0, "revenue_pct": 1.75}
NICE = {"eps": "Adjusted EPS", "operating_margin": "Operating margin", "revenue_pct": "Total revenue growth"}
UNIT = {"eps": "$", "operating_margin": "%", "revenue_pct": "%"}


# ----------------------------------------------------------------------------------------------- building
def grid_dates(q: str, today, step: int = 1) -> list[pd.Timestamp]:
    """Every `step` days (1 = every day) from day 14 of the quarter to 14 days after it ends, plus the backtest dates
    (day 20, 45, 70, 7 days after the end); nothing after `today`."""
    p, today = pd.Period(q, "Q"), pd.Timestamp(today)
    end = p.end_time.normalize()
    d = set(pd.date_range(p.start_time + pd.Timedelta(days=14), end + pd.Timedelta(days=14), freq=f"{step}D"))
    d |= {p.start_time + pd.Timedelta(days=k) for k in (20, 45, 70)} | {end + pd.Timedelta(days=7)}
    return sorted(x for x in d if x <= today)


def trim_base(B: dict) -> dict:
    """The base case with market jet fuel cut to what the app needs and to what was public on the date."""
    q = pd.Period(B["quarter"], "Q")
    out = dict(B)
    out["jet"] = B["jet"][(B["jet"].index >= q.start_time - pd.Timedelta(days=90)) & (B["jet"].index <= B["as_of"])]
    return out


EVENTS = [
    {"id": "outage_jul2024", "name": "Delta IT outage (CrowdStrike), July 2024",
     "what": "About 7,000 flights cancelled over five days; Delta's own measured cost.",
     "levers": {"outage": 1.0},
     "measured": "Revenue -$380M, non-fuel costs +$170M, fuel -$50M (Delta's Q3 2024 earnings release). "
                 "BTS On-Time counts 7,004 cancelled Delta + Endeavor flights, 16-24 Jul 2024.",
     "scale_rule": "outage = cancelled flights / 7,004 (e.g. 3,500 cancellations -> 0.5)."},
    {"id": "fuel_sep2023", "name": "Jet fuel spike, summer 2023 (led to Delta's 14 Sep 2023 EPS cut)",
     "what": "Market jet fuel rose sharply between Delta's July guidance and mid-September.", "levers": {},
     "measured": "", "scale_rule": "Use `fuel_pct` = the expected change in market jet fuel for the rest of the quarter."},
    {"id": "demand_mar2025", "name": "Softer demand, March 2025 (Delta cut revenue growth guidance from 7-9% to 3-4%)",
     "what": "Delta cited softer close-in demand (consumer and corporate confidence).", "levers": {},
     "measured": "", "scale_rule": "Use `demand_pts`; calibration below gives the demand change that matches Delta's cut."},
]
# Every lever acts from today to the end of the quarter (the days already flown stay as they were; notebook 05b),
# so a lever value is the size of the change WHILE it lasts, not its average over the whole quarter.
LEVER_DOCS = {
    "fuel_pct": "market jet fuel x% higher (+) or lower (-) than today's price, from today to the end of the quarter",
    "capacity_pct": "Delta flies x% more (+) or fewer (-) seat-miles than the base case from today to the end of the "
                    "quarter (PLANNED schedule changes)",
    "demand_pts": "industry passenger growth from today to the end of the quarter, percentage points added to the base case",
    "fare_pct": "fares x% higher from today to the end of the quarter, for reasons other than fuel, load factor and "
                "persistence (e.g. a fare war: negative)",
    "nonfuel_pct": "non-fuel costs x% above the base case from today to the end of the quarter (e.g. a new labor contract)",
    "outage": "size of an operational meltdown happening in the rest of the quarter, relative to Delta's July 2024 "
              "outage (1 = the same)",
}


REFERENCE_CANCELLATIONS = 7004      # On-Time: Delta + Endeavor cancellations 16-24 Jul 2024 (Delta: "about 7,000")


def measured_disruptions(min_cancelled: int = 300, min_weather_share: float = 0.6) -> list[dict]:
    """Every large weather-led disruption at Delta in BTS On-Time (Delta + Endeavor, domestic): days with 150+
    cancellations, grouped into episodes (with a day either side); kept when 300+ flights were cancelled and 60%+
    of them were cancelled for weather (BTS cause code B). COVID (2020) left out. Each episode: dates, cancellations,
    weather share, most-affected airports, size relative to the July 2024 outage, and the date it was public in
    On-Time (month end + ONTIME_LAG_DAYS) - the event box only sees episodes public by its as-of date."""
    from .data.bts import ONTIME_LAG_DAYS, load_ontime
    ot = load_ontime()
    ot = ot.assign(FlightDate=pd.to_datetime(ot.FlightDate))
    c = ot[ot.Cancelled == 1]
    day = c.groupby("FlightDate").size().asfreq("D").fillna(0)
    big = day[day >= 150]
    grp = (big.index.to_series().diff() != pd.Timedelta(days=1)).cumsum()
    out = []
    for _, idx in big.groupby(grp.values):
        a, b = idx.index.min() - pd.Timedelta(days=1), idx.index.max() + pd.Timedelta(days=1)
        if pd.Timestamp("2020-03-01") <= a <= pd.Timestamp("2020-12-31"):
            continue
        w = c[(c.FlightDate >= a) & (c.FlightDate <= b)]
        wshare = float((w.CancellationCode == "B").mean())
        if len(w) < min_cancelled or wshare < min_weather_share:
            continue
        out.append({"dates": f"{a:%d %b %Y} - {b:%d %b %Y}", "cancelled_flights": int(len(w)),
                    "weather_share_pct": round(wshare * 100), "top_airports": w.Origin.value_counts().head(4).to_dict(),
                    "outage_equivalent": round(len(w) / REFERENCE_CANCELLATIONS, 2),
                    "public_from": f"{b + pd.offsets.MonthEnd(0) + pd.Timedelta(days=ONTIME_LAG_DAYS):%Y-%m-%d}"})
    return out


def _calibrate_events(ctx, log=print) -> list[dict]:
    """Fill in the measured sizes of the library's fuel and demand events from the data (point in time)."""
    from .simulation.general import build_base
    ev = [dict(e) for e in EVENTS]
    jet = ctx.jet
    wk = lambda d: float(jet[:pd.Timestamp(d)].iloc[-5:].mean())
    a, b = wk("2023-07-14"), wk("2023-09-13")
    ev[1]["levers"] = {"fuel_pct": round((b / a - 1) * 100, 1)}
    ev[1]["measured"] = (f"Market jet fuel (US Gulf Coast, EIA), 5-day average: ${a:.2f}/gal the week after the July "
                         f"2023 guidance -> ${b:.2f}/gal the day before the cut ({(b / a - 1) * 100:+.0f}%).")
    B = build_base(ctx, "2025Q1", "2025-03-09")
    R = draw(B, 20_000, 0)
    base = np.median(evaluate(B, R).revenue_pct)
    f = lambda x: np.median(evaluate(B, R, Levers(demand_pts=x)).revenue_pct) - base
    lo, hi = -20.0, 0.0
    for _ in range(40):                                    # demand change that cuts revenue growth by 4.5 points
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < -4.5 else (lo, mid)
    ev[2]["levers"] = {"demand_pts": round((lo + hi) / 2, 1)}
    ev[2]["measured"] = (f"Delta's revenue growth guidance fell 4.5 points (midpoints 8% -> 3.5%); in the simulator "
                         f"(Q1 2025, as of 9 Mar 2025) that takes demand {(lo + hi) / 2:+.1f} points.")
    log("events calibrated")
    return ev


def build_entry(ctx, q: str, d, bg: dict | None = None) -> dict:
    """One date's snapshot entry: the base case as public on `d` and the inputs of the "news since guidance"
    counterfactual. Used by build_snapshot and, for a date not in the snapshot, by the app (live, with the raw data)."""
    from .forecast.update import guidance_misses
    from .simulation.general import build_base
    bg = {} if bg is None else bg
    d = pd.Timestamp(d)
    B = build_base(ctx, q, d)
    news = {}
    for m, gk in GUIDED.items():
        g = B["guidance"].get(gk)
        if g is None:
            continue
        gd = pd.Timestamp(g["filed"]) + pd.Timedelta(days=1)
        if (q, gd) not in bg:
            bg[(q, gd)] = build_base(ctx, q, gd)["cap"]["consolidated_seat_miles"]
        news[m] = {"gd": gd, "bg_cap": bg[(q, gd)], "misses": guidance_misses(ctx, q, d, m)}
    return {"B": trim_base(B), "news": news}


def build_snapshot(ctx, today="2026-10-07", out: Path = SNAP, quarters=QUARTERS, log=print, step: int = 1,
                   resume: bool = False) -> dict:
    """Write app/snapshot/. step=1: every day; step=7: weekly (faster). Progress is saved after each quarter to
    bases_partial.pkl, so an interrupted build can continue with resume=True (same data, same command)."""
    from .data.rivals import opensky_daily_counts
    from .simulation.general import actual
    from .validation.forecast_backtest import NO_UPDATE, UPDATES, update_test
    out.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp(today)
    partial = out / "bases_partial.pkl"
    entries = pickle.loads(partial.read_bytes()) if (resume and partial.exists()) else {}
    if entries:
        log(f"resuming: {len(entries)} dates already built")
    bg, acts = {}, {}
    total = sum(len(grid_dates(q, today, step)) for q in quarters)
    for q in quarters:
        for d in grid_dates(q, today, step):
            k = f"{q}|{d:%Y-%m-%d}"
            if k not in entries:
                entries[k] = build_entry(ctx, q, d, bg)
                if len(entries) % 25 == 0:
                    log(f"{len(entries)} of {total} dates")
        partial.write_bytes(pickle.dumps(entries, protocol=4))
        p = pd.Period(q, "Q")
        if ctx.filed(p) <= today:
            a = actual(ctx, p)
            acts[q] = {k: (str(v) if k == "filed" else float(v)) for k, v in a.items()}
        log(f"{q}: {len(grid_dates(q, today, step))} dates")
    (out / "bases.pkl").write_bytes(pickle.dumps(entries, protocol=4))
    partial.unlink(missing_ok=True)
    (out / "actuals.json").write_text(json.dumps(acts, indent=1))
    # track record: notebook 07's backtest (after the lock: the outage detector), the lines the app uses
    FB = pd.read_csv(ctx.data / "derived" / "forecaster_backtest_after_lock.csv")
    keep = (((FB.metric != "revenue_pct") & (FB.channels == "fuel")) | ((FB.metric == "revenue_pct") & (FB.channels == "fuel+capacity")))
    FB[keep][["quarter", "offset", "metric", "channels", "guidance_low", "guidance_high", "guidance_mid", "news", "forecast",
              "p10", "p90", "prob_below", "prob_above", "actual", "simulator"]].to_csv(out / "track_record.csv", index=False)
    log("mid-quarter update test ...")
    UT = update_test(ctx)
    UT["direction"] = np.nan
    for q, u, m, dirn, quote in UPDATES:
        UT.loc[(UT.quarter == q) & (UT.update_date == u) & (UT.metric == m), "direction"] = dirn
    UT.to_csv(out / "update_test.csv", index=False)
    upd = pd.DataFrame(UPDATES, columns=["quarter", "update_date", "metric", "direction", "quote"])
    upd.to_csv(out / "updates.csv", index=False)
    # live signals, daily
    start = pd.Timestamp("2022-07-01")
    H = ctx.H_day.sum(axis=1)
    riv = opensky_daily_counts(ctx.data).sum(axis=1)
    yoy7 = lambda s: (s.rolling(7).sum() / s.rolling(7).sum().shift(364) - 1) * 100
    tsa = ctx.tsa.asfreq("D")
    sig = pd.DataFrame({"delta_hours": H, "delta_yoy7": yoy7(H.asfreq("D")), "rivals_flights": riv,
                        "rivals_yoy7": yoy7(riv.asfreq("D")),
                        "outage_full": ctx.outages.full, "outage_partial": ctx.outages.partial,
                        "jet_fuel": ctx.jet.asfreq("D").ffill(), "tsa": tsa,
                        "tsa_yoy7": (tsa.rolling(7).sum() / tsa.shift(364).rolling(7).sum() - 1) * 100})
    sig = sig[(sig.index >= start) & (sig.index <= today)]
    sig.to_csv(out / "signals.csv", index_label="date")
    # library, consensus, lock, meta
    (out / "events.json").write_text(json.dumps({"events": _calibrate_events(ctx, log), "levers": LEVER_DOCS,
                                                 "measured_disruptions": measured_disruptions()}, indent=1))
    root = Path(__file__).resolve().parents[1]
    pd.read_csv(root / "data" / "wallstreet_consensus.csv").to_csv(out / "consensus.csv", index=False)
    # locked calls (notebook 08), copied byte for byte so each file keeps its SHA-256 fingerprint
    import hashlib
    meta_lock = {}
    for lock in sorted((ctx.data / "derived").glob("lock_*.json")):
        body = lock.read_bytes()
        (out / lock.name).write_bytes(body)
        meta_lock[lock.stem.replace("lock_", "")] = hashlib.sha256(body).hexdigest()
    meta = {"built": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"), "today": f"{today:%Y-%m-%d}",
            "lock_sha256": meta_lock,
            "data_through": {"OpenSky flights (Delta + 5 rivals)": f"{ctx.osk_last_day:%Y-%m-%d}",
                             "TSA checkpoint counts": f"{ctx.tsa.index.max():%Y-%m-%d}",
                             "Market jet fuel (EIA)": f"{ctx.jet.index.max():%Y-%m-%d}",
                             "BTS T-100 (published)": f"{ctx.truth.dropna().index.max():%Y-%m}",
                             "Delta filings (SEC)": f"{max(ctx.A.filed):%Y-%m-%d}"}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    build_paths(out, log)
    log(f"snapshot written to {out}")
    return meta


# ----------------------------------------------------------------------------------------------- using (no raw data)
def widen_track(T: pd.DataFrame) -> pd.DataFrame:
    """Apply RANGE_SCALE to the backtest's 80% ranges, so the track record shows the ranges the app now uses."""
    T = T.copy()
    k = T.metric.map(RANGE_SCALE).fillna(1.0)
    T["p10"] = T.forecast - k * (T.forecast - T.p10)
    T["p90"] = T.forecast + k * (T.p90 - T.forecast)
    return T


class Snapshot:
    def __init__(self, path: Path = SNAP):
        self.path = Path(path)
        self.entries = pickle.loads((self.path / "bases.pkl").read_bytes())
        self.actuals = json.loads((self.path / "actuals.json").read_text())
        self.track = widen_track(pd.read_csv(self.path / "track_record.csv"))
        self.updates_test = pd.read_csv(self.path / "update_test.csv", parse_dates=["date"])
        self.updates = pd.read_csv(self.path / "updates.csv")
        self.signals = pd.read_csv(self.path / "signals.csv", parse_dates=["date"]).set_index("date")
        self.events = json.loads((self.path / "events.json").read_text())
        self.consensus = pd.read_csv(self.path / "consensus.csv")
        self.meta = json.loads((self.path / "meta.json").read_text())
        ch = self.path / "consensus_history.csv"
        self.cons_hist = pd.read_csv(ch, parse_dates=["date"]) if ch.exists() else None
        fp = self.path / "forecast_path.csv"           # the forecast on every saved date (build_paths)
        self.paths = pd.read_csv(fp, parse_dates=["date"]) if fp.exists() else None
        # locked calls, by quarter (lock_<quarter>.json, notebook 08), with each file's SHA-256 fingerprint
        import hashlib
        self.locks = {}
        for lk in sorted(self.path.glob("lock_*.json")):
            body = lk.read_bytes().replace(b"\r\n", b"\n")        # a Windows copy may have CRLF line endings
            self.locks[lk.stem.replace("lock_", "")] = {**json.loads(body), "sha256": hashlib.sha256(body).hexdigest()}
        self.lock = self.locks.get("2026Q3")
        self.live = None            # set by the app when the raw data is on this machine: (q, d) -> entry
        self.dates = {}
        for k in self.entries:
            q, d = k.split("|")
            self.dates.setdefault(q, []).append(pd.Timestamp(d))

    def saved(self, q, d) -> bool:
        return f"{q}|{pd.Timestamp(d):%Y-%m-%d}" in self.entries

    def entry(self, q, d) -> dict:
        """The date's entry: from the snapshot, or worked out live from the raw data (self.live) if not saved."""
        k = f"{q}|{pd.Timestamp(d):%Y-%m-%d}"
        if k not in self.entries:
            if self.live is None:
                raise KeyError(f"{k} is not in the snapshot")
            self.entries[k] = self.live(q, pd.Timestamp(d))
        e = self.entries[k]
        if "rest_share" not in e["B"]:                   # share of the quarter still to come (notebook 05b)
            e["B"]["rest_share"] = rest_shares(self.signals, q, d)
        return e

    def consensus_as_of(self, q, d) -> dict | None:
        """Wall Street consensus known on date d: I/B/E/S (point in time) up to its last update; after that, the final
        press figure only within 5 days of the report. None when there is no figure for that date."""
        d = pd.Timestamp(d)
        if self.cons_hist is not None:
            r = self.cons_hist[(self.cons_hist.quarter == q) & (self.cons_hist.date == d)]
            if len(r) and pd.notna(r.iloc[0].eps):
                r = r.iloc[0]
                return {"eps": float(r.eps), "revenue_pct": float(r.revenue_pct) if pd.notna(r.revenue_pct) else None,
                        "n": int(r.eps_n), "source": "ibes", "through": self.meta.get("consensus_through")}
        rr = self.consensus[self.consensus.quarter == q]
        if len(rr) and d >= pd.Timestamp(rr.iloc[0].report_date) - pd.Timedelta(days=5):
            v = self.consensus_eps(q)
            return {"eps": v, "revenue_pct": None, "n": None, "source": "press", "through": None} if v else None
        return None

    def consensus_eps(self, q) -> float | None:
        r = self.consensus[self.consensus.quarter == q]
        if r.empty:
            return None
        v = [float(x) for x in str(r.iloc[0].consensus_values).split(";") if x.strip()]
        return float(np.median(v)) if v else None


def simulate(e: dict, n: int = 20_000, lv: Levers = Levers(), seed: int = 0):
    """The simulator on this date's base case: (draws, today's runs with the levers, runs without levers)."""
    B = e["B"]
    R = draw(B, n, seed, "joint")
    S0 = evaluate(B, R)
    S = S0 if not lv.active() else evaluate(B, R, lv)
    return R, S, S0


def forecast(e: dict, R: dict, S: pd.DataFrame, seed: int = 0) -> dict:
    """Guidance midpoint + news since the guidance (+ any scenario levers already in S), per metric, as in
    flightpath/forecast/update.py: the counterfactual puts each channel's input back to the guidance date."""
    B = e["B"]
    rng = np.random.default_rng(seed)
    out, cache = {}, {}
    for m, ni in e["news"].items():
        ch = CHANNELS_BY_METRIC[m]
        key = (ni["gd"], ch)
        if key not in cache:
            C = dict(B)
            if "fuel" in ch:
                C["as_of"] = ni["gd"]
            if "capacity" in ch:
                C["cap"] = dict(B["cap"], consolidated_seat_miles=ni["bg_cap"])
            cache[key] = evaluate(C, R)
        Sc = cache[key]
        g = B["guidance"][GUIDED[m]]
        mid = (g["low"] + g["high"]) / 2
        diff = S[m].values - Sc[m].values
        miss = np.asarray(ni["misses"], float)
        miss = miss - miss.mean() if len(miss) else np.zeros(1)
        runs = mid + diff + rng.choice(miss, len(diff))
        fc = float(mid + np.median(diff))
        runs = fc + RANGE_SCALE[m] * (runs - fc)              # section 7h: revenue range widened
        out[m] = {"low": g["low"], "high": g["high"], "mid": mid, "filed": g["filed"], "quote": g.get("quote", ""),
                  "news": float(np.median(diff)), "forecast": float(mid + np.median(diff)), "runs": runs,
                  "p10": float(np.percentile(runs, 10)), "p90": float(np.percentile(runs, 90)),
                  "below": float((runs < g["low"]).mean() * 100), "above": float((runs > g["high"]).mean() * 100),
                  "simulator": float(np.median(S[m])), "Sc": Sc}
    return out


def signal(f: dict) -> tuple[str, str]:
    """Where the forecast points vs Delta's range: 'below', 'above' or 'within' (notebook 07, section 5: a forecast
    outside the range happened without Delta later changing its numbers only 5% of the time)."""
    if f["forecast"] < f["low"]:
        return "below", "Pointing below Delta's guidance"
    if f["forecast"] > f["high"]:
        return "above", "Pointing above Delta's guidance"
    return "within", "Within Delta's guidance"


def risk_light(f: dict) -> tuple[str, str]:
    """Notebook 07, section 5: the forecaster outside Delta's range = a clear signal (caught the Sept 2023 cut 45 days
    early; false alarms 5%); only the simulator outside = a weak signal (saw the Mar 2025 cut; false alarms 63%)."""
    lo, hi = f["low"], f["high"]
    side = lambda v: "below" if v < lo else "above" if v > hi else None
    if side(f["forecast"]):
        return "red", f"Forecast {side(f['forecast'])} Delta's range - a clear signal (5% false alarms in the backtest)"
    if side(f["simulator"]):
        return "amber", f"Only the simulator is {side(f['simulator'])} the range - a weak signal (63% false alarms)"
    return "green", "Forecast and simulator inside Delta's range"


def fuel_split(e: dict, R: dict, S: pd.DataFrame) -> dict:
    """Fuel news split into its cost part and its fares part (as notebook 08, section 3)."""
    B = e["B"]
    C = dict(B, as_of=e["news"]["eps"]["gd"])
    Sc = evaluate(C, R)
    after = (1 - 0.20) * (1 - B["ce"]["base"]["tax_rate_pct"] / 100) / B["ce"]["base"]["diluted_shares_m"]
    return {"fuel cost": -float(np.median(S.fuel_cost - Sc.fuel_cost)) * after,
            "fares (follow fuel)": float(np.median(S.revenue - Sc.revenue)) * after}


def eps_bridge(B: dict, S: pd.DataFrame, S0: pd.DataFrame) -> pd.Series:
    """How a scenario's EPS change flows through the income statement (median of the run-by-run changes, per share,
    after tax): revenue, fuel, non-fuel costs, maintenance-for-others cost, profit sharing."""
    k = (1 - B["ce"]["base"]["tax_rate_pct"] / 100) / B["ce"]["base"]["diluted_shares_m"]
    d = lambda c, sgn=1: sgn * float(np.median(S[c] - S0[c])) * k
    parts = pd.Series({"revenue": d("revenue"), "fuel cost": d("fuel_cost", -1), "non-fuel costs": d("nonfuel_cost", -1),
                       "MRO cost": d("mro_cost", -1), "profit sharing": d("profit_sharing", -1)})
    parts["total (EPS)"] = float(np.median(S.eps - S0.eps))
    return parts


CHAIN = [("capacity_pct", "Capacity (seat-miles, YoY %)"), ("demand_pct", "Industry demand (YoY %)"),
         ("load_factor_chg_pts", "Load factor (change, pts)"), ("yield_pct", "Fares / yield (YoY %)"),
         ("passenger_revenue_pct", "Passenger revenue (YoY %)"), ("revenue_pct", "Total revenue (YoY %)"),
         ("fuel_price", "Delta fuel price ($/gal)"), ("fuel_cost", "Fuel cost ($M)"), ("unit_cost_pct", "Non-fuel unit cost (YoY %)"),
         ("nonfuel_cost", "Non-fuel costs ($M)"), ("profit_sharing", "Profit sharing ($M)"),
         ("operating_margin", "Operating margin (%)"), ("eps", "Adjusted EPS ($)")]


def chain_table(S: pd.DataFrame, S0: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"base case": [np.median(S0[c]) for c, _ in CHAIN], "scenario": [np.median(S[c]) for c, _ in CHAIN]},
                        index=[n for _, n in CHAIN]).assign(change=lambda d: d.scenario - d["base case"])


# ----------------------------------------------------------------------------------------------- event box (OpenAI)
EVENT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["levers", "reasoning", "basis", "confidence", "caveats"],
    "properties": {
        "levers": {"type": "object", "additionalProperties": False,
                   "required": list(LEVER_DOCS), "properties": {k: {"type": "number"} for k in LEVER_DOCS}},
        "reasoning": {"type": "string"}, "basis": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]}, "caveats": {"type": "string"}}}


def event_prompt(headline: str, e: dict, events: dict) -> tuple[str, str]:
    B = e["B"]
    q = pd.Period(B["quarter"], "Q")
    as_of = pd.Timestamp(B["as_of"])
    days_left = max((q.end_time.normalize() - as_of).days, 0)
    past = [x for x in events.get("measured_disruptions", []) if pd.Timestamp(x["public_from"]) <= as_of]
    system = (
        "You translate a news headline about Delta Air Lines (or the airline industry) into scenario lever settings "
        "for a quarterly simulator. RULES:\n"
        "1. Use ONLY the levers listed. Leave a lever at 0 unless the news clearly moves it.\n"
        "2. Size levers ONLY from (a) numbers stated in the headline, (b) the documented events, or (c) the measured "
        "disruptions table. NEVER use statistics from general knowledge or estimates not in these sources. In `basis`, "
        "name the source and show the arithmetic.\n"
        "3. Storms, IT failures and other operational disruptions: use `outage` ONLY, never `capacity_pct` as well - "
        "`outage` already includes the lost flying. `outage` = cancelled flights / 7,004 (Delta + Endeavor "
        "cancellations in the July 2024 outage, BTS On-Time). `capacity_pct` is for PLANNED schedule changes only.\n"
        "4. If the headline gives no size (e.g. a storm forecast), size it from comparable episodes in the measured "
        "disruptions table: every episode of the same kind (e.g. winter weather) hitting the same airports or region. "
        "In `basis`, list EACH one you use with its dates and outage_equivalent, then use their MEDIAN (show it) as the "
        "lever value, and give the lowest and highest in `caveats`. Set confidence to 'low'. Confidence 'medium' or "
        "'high' only when the headline states the size.\n"
        "5. Weather disruptions: say in `caveats` that `outage` uses the July 2024 outage's costs per cancelled flight "
        "(customer reimbursements, crew), which are probably higher than for weather cancellations, so the effect is "
        "likely an upper bound.\n"
        "6. An event affects only the days of the quarter still to come.\n\n"
        "LEVERS (each acts from today to the end of the quarter):\n" + json.dumps(LEVER_DOCS, indent=1) +
        "\n\nDOCUMENTED EVENTS:\n" + json.dumps(events["events"], indent=1) +
        "\n\nMEASURED DISRUPTIONS (BTS On-Time, Delta + Endeavor, weather-led, public by today):\n" + json.dumps(past, indent=1))
    user = (f"Quarter: {q} (Delta's fiscal quarter). Today in the simulator: {as_of:%d %b %Y}; "
            f"{days_left} days of the quarter left.\nHeadline: {headline}")
    return system, user


def interpret_event(headline: str, e: dict, events: dict, api_key: str, model: str = "gpt-4.1") -> dict:
    from openai import OpenAI
    system, user = event_prompt(headline, e, events)
    r = OpenAI(api_key=api_key).chat.completions.create(
        model=model, temperature=0, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_schema", "json_schema": {"name": "event", "schema": EVENT_SCHEMA, "strict": True}})
    out = json.loads(r.choices[0].message.content)
    lim = {"fuel_pct": (-50, 100), "capacity_pct": (-30, 15), "demand_pts": (-30, 15), "fare_pct": (-30, 30),
           "nonfuel_pct": (-15, 25), "outage": (0, 5)}
    out["levers"] = {k: float(np.clip(v, *lim[k])) for k, v in out["levers"].items()}
    return out


# ----------------------------------------------------------------------------------------------- daily forecast path
def build_paths(snap_dir: Path = SNAP, log=print) -> pd.DataFrame:
    """The forecast (EPS, margin, revenue: middle, 80% range, Delta's guidance range, and the simulator on its own) on
    every date in the snapshot,
    computed exactly as the app computes it for that date (same runs, same seed). Saved to forecast_path.csv so the
    app can draw day-by-day charts without rerunning the simulator for every day. Needs only the snapshot."""
    S = Snapshot(snap_dir)
    rows = []
    for q in sorted(S.dates):
        for x in sorted(S.dates[q]):
            e = S.entry(q, x)
            if not e["news"]:
                continue
            R, Sm, _ = simulate(e, 50_000, Levers())
            for m, f in forecast(e, R, Sm).items():
                rows.append({"quarter": q, "date": f"{x:%Y-%m-%d}", "metric": m,
                             **{k: f[k] for k in ("forecast", "low", "high", "p10", "p90", "simulator")}})
        log(f"{q}: forecast path done")
    out = pd.DataFrame(rows)
    out.to_csv(Path(snap_dir) / "forecast_path.csv", index=False)
    return out


# ----------------------------------------------------------------------------------------------- Wall Street consensus
def build_consensus(snap_dir: Path = SNAP, data_dir=None, max_age: int = 90) -> pd.DataFrame:
    """Wall Street consensus (I/B/E/S, point in time) for every snapshot date and every track-record date, EPS and total
    revenue growth (I/B/E/S SAL = Delta's adjusted revenue, vs the same quarter a year earlier). Dates after the last
    I/B/E/S update are left empty (the press figure for the final days before a report is used instead, labeled).
    Written to its own file (consensus_history.csv) so it can be left out of a hosted copy if the license requires."""
    from .data.ibes import actuals, consensus, load_detail
    d = load_detail(data_dir)
    last = d.ACTDATS.max()
    A = actuals(d)
    ents = pickle.loads((snap_dir / "bases.pkl").read_bytes())
    keys = {tuple(k.split("|")) for k in ents}
    tr = pd.read_csv(snap_dir / "track_record.csv")
    for qq, off in tr[["quarter", "offset"]].drop_duplicates().itertuples(index=False):
        P = pd.Period(qq, "Q")
        dd = P.end_time.normalize() + pd.Timedelta(days=7) if str(off) == "end+7" else P.start_time + pd.Timedelta(days=int(off))
        keys.add((qq, f"{dd:%Y-%m-%d}"))
    rows = []
    for qq, ds in sorted(keys):
        P, dt = pd.Period(qq, "Q"), pd.Timestamp(ds)
        if dt > last:
            continue
        e, s_ = consensus(d, "EPS", P, dt, max_age), consensus(d, "SAL", P, dt, max_age)
        ly = A[("actual", "SAL")].get(P - 4, np.nan)
        rows.append({"quarter": qq, "date": ds, "eps": e["value"], "eps_n": e["n"], "eps_low": e.get("low"), "eps_high": e.get("high"),
                     "revenue_pct": (s_["value"] / ly - 1) * 100 if s_["n"] else np.nan, "revenue_n": s_["n"]})
    out = pd.DataFrame(rows)
    out.to_csv(snap_dir / "consensus_history.csv", index=False)
    meta = json.loads((snap_dir / "meta.json").read_text())
    meta["consensus_through"] = f"{last:%Y-%m-%d}"
    (snap_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return out
