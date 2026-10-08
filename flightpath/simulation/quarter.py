"""The FlightPath simulator: one quarter of Delta, as a living system (notebook 05).

Every run draws the uncertain pieces (input measurement errors, each link's coefficients, each model's surprise),
then flows the levers through the measured links:

  capacity (01) ─┬──────────────────────────────► non-fuel costs (04) ─────────────┐
  demand (03) ───┼► load factor ─► yield (03) ─► passenger revenue ─► other revenue (04) ─► revenue
  fuel (02) ─────┴────────────────────┘   └► fuel cost (02: Delta's price x gallons)       │
                                                                                         ▼
           revenue − fuel − non-fuel − MRO cost − profit sharing (04) = operating income ─► EPS (04)

All draws are made once (common random numbers), so two scenarios differ only by their levers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field, fields
from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd

FORECAST_DATE = pd.Timestamp("2026-10-07")
COVID_FUEL = (pd.Period("2020Q2", "Q"), pd.Period("2021Q2", "Q"))      # notebook 02's COVID quarters

# The documented disruption case: Delta's own disclosure of the CrowdStrike-caused outage (July 2024), from its
# September-quarter 2024 earnings release: about 7,000 flights cancelled over five days; direct revenue impact
# ~$380M (refunds, compensation); non-fuel expense +$170M (customer reimbursements, crew); fuel expense −$50M.
OUTAGE_JULY_2024 = {"revenue_musd": -380.0, "nonfuel_musd": 170.0, "fuel_musd": -50.0, "flights_cancelled": 7000}


@dataclass
class Levers:
    demand_pts: float = 0.0       # industry passenger growth, points added to the base
    capacity_pct: float = 0.0     # Delta flies x% more (+) or fewer (−) seat-miles than the base
    fuel_pct: float = 0.0         # market jet fuel x% higher on days not yet priced (notebook 02's rule)
    fare_pct: float = 0.0         # fares x% higher for reasons other than fuel, load factor and persistence
    nonfuel_pct: float = 0.0      # non-fuel costs x% above the base (e.g. a labor contract)
    outage: float = 0.0           # size of an outage relative to July 2024 (1 = the same, 0.5 = half)
    ps_rate: float = 20.0         # profit-sharing rate on profit above $2.5bn a year (Delta's formula: 20%)
    # modes
    fare_mode: str = "extra"      # "extra": fares x% on top of what the links do; "total": fares end x% above the base
                                  # case in total (the links' automatic move is taken out, and reported as the split)
    isolation: bool = False       # behavioural links off (fuel -> fares, demand/capacity -> load factor,
                                  # passenger revenue -> other revenue, capacity -> unit cost); accounting kept
    fuel_hindsight: bool = False  # fuel lever moves the WHOLE quarter, priced days too ("what if it had been")
    rest_of_quarter: bool = False # all levers act only on the part of the quarter after `as_of` (notebook 05b):
                                  # capacity / non-fuel costs on the share of flying still to come, demand / fares on
                                  # the share of passengers still to come, an outage only if days remain, and fuel
                                  # priced on the gallons actually still to be bought. Needs base["rest_share"].

    def active(self) -> dict:
        """Levers moved from their defaults (modes are not levers)."""
        modes = {"fare_mode", "isolation", "fuel_hindsight", "rest_of_quarter"}
        return {f.name: getattr(self, f.name) for f in fields(self)
                if f.name not in modes and getattr(self, f.name) != f.default}

    def modes(self) -> dict:
        return {"fare_mode": self.fare_mode, "isolation": self.isolation, "fuel_hindsight": self.fuel_hindsight,
                "rest_of_quarter": self.rest_of_quarter}


def _market_daily():
    from ..data.eia import load_eia
    return load_eia()["jet_gulf_coast"].set_index("date")["value"].sort_index()


def market_quarter(jet: pd.Series, q: pd.Period, lag_days: int, as_of=None, lever_pct: float = 0.0, hindsight: bool = False):
    """Quarter-average market price with the purchase delay. Days after `as_of` are unknown: they are set to the
    last known price, moved by the lever (notebook 02, section 6). With `hindsight`, the lever moves every day of the
    quarter, known ones too ("what if fuel had been x% higher"). Returns (price, known days, unknown days)."""
    lag = pd.Timedelta(days=lag_days)
    window = pd.bdate_range(q.start_time - lag, q.end_time.normalize() - lag)
    as_of = pd.Timestamp(as_of) if as_of is not None else window[-1]
    known = jet[(jet.index >= window[0]) & (jet.index <= window[-1]) & (jet.index <= as_of)]
    n_left = int((window > as_of).sum())
    last = jet[:as_of].iloc[-1]
    k = (1 + lever_pct / 100)
    avg = (known.sum() * (k if hindsight else 1) + n_left * last * k) / max(len(known) + n_left, 1)
    return avg, len(known), n_left


def market_quarter_rest(jet: pd.Series, q: pd.Period, lag_days: int, as_of, lever_pct: float, cap_change: float):
    """Quarter-average market price paid when, from `as_of` on, Delta flies `cap_change` (fraction, e.g. -0.05) more
    and the unpriced days cost `lever_pct` more (notebook 05b). Each purchase day weighs the gallons it buys: fuel
    bought for flying after `as_of` (the last `lag_days` of known purchases, and every unpriced day) is scaled by
    1 + cap_change, so a smaller schedule buys fewer gallons at the new price. With cap_change = 0 this is exactly
    market_quarter."""
    lag = pd.Timedelta(days=lag_days)
    window = pd.bdate_range(q.start_time - lag, q.end_time.normalize() - lag)
    as_of = pd.Timestamp(as_of)
    known = jet[(jet.index >= window[0]) & (jet.index <= window[-1]) & (jet.index <= as_of)]
    n_left = int((window > as_of).sum())
    last = jet[:as_of].iloc[-1]
    g = 1 + cap_change
    mid = known.index > as_of - lag                        # bought already, burned after as_of
    num = known[~mid].sum() + g * known[mid].sum() + g * n_left * last * (1 + lever_pct / 100)
    den = (~mid).sum() + g * mid.sum() + g * n_left
    return num / max(den, 1e-9), len(known), n_left


def load_base(data_dir=None, as_of=FORECAST_DATE, quarter="2026Q3") -> dict:
    """Everything the simulator needs for `quarter`, as of `as_of`, from notebooks 01-04's saved results and the data
    that was public on `as_of`."""
    from ..paths import DATA
    from ..data.sec_adjusted import adjusted_quarters
    data = DATA if data_dir is None else data_dir
    d = lambda name: json.loads((data / "derived" / name).read_text())
    cap, rev, fuel, ce = d("capacity_2026Q3.json"), d("revenue_links_2026Q3.json"), d("fuel_model.json"), d("costs_eps_2026Q3.json")
    q = pd.Period(quarter, "Q")
    A = adjusted_quarters(data)
    jet = _market_daily()
    lag = fuel["purchase_delay_days"]
    # fuel translator (notebook 02, candidate 2): Delta's adjusted price on the market price, last 8 normal quarters
    # whose results were public on as_of
    price = A["fuel_price_gal_adj"].fillna(A["fuel_price_gal"]).astype(float)
    hist = [p for p in A.index if A.loc[p, "filed"] <= pd.Timestamp(as_of) and pd.notna(price[p])
            and not (COVID_FUEL[0] <= p <= COVID_FUEL[1])][-fuel["translator_window_quarters"]:]
    mkt_hist = np.array([market_quarter(jet, p, lag)[0] for p in hist])
    b, a = np.polyfit(mkt_hist, price[hist].values, 1)
    gpa = (A.fuel_gallons_m / A.asm_m).astype(float)
    gpa_q = gpa[q - 4] * gpa[q - 1] / gpa[q - 5]             # notebook 02's gallons rule
    mkt_ly = market_quarter(jet, q - 4, lag)[0]
    cb = ce["base"]
    return {"quarter": str(q), "as_of": pd.Timestamp(as_of), "jet": jet, "lag": lag,
            "translator": (a, b), "translator_quarters": [str(p) for p in hist], "market_ly": mkt_ly,
            "gallons_per_asm": gpa_q,
            "cap": cap, "rev": rev, "fuel": fuel, "ce": ce, "base_inputs": rev["base_inputs"],
            "pax_ly": cb["passenger_revenue_q3_2025_musd"], "other_ly": cb["other_revenue_q3_2025_musd"],
            "revenue_ly": cb["revenue_adj_q3_2025_musd"], "nonfuel_ly_old": cb["nonfuel_cost_q3_2025_old_def_musd"],
            "lf_ly": rev["load_factor_last_year"], "guidance": ce["guidance_q3_2026"],
            "unit_cost_guided_pct": ce.get("unit_cost_guided", {}).get("q3_2026_guided_pct")}


def draw(base: dict, n: int = 50_000, seed: int = 0, surprises: str = "joint") -> dict:
    """All random pieces of `n` runs, drawn once. Error histories are centered (no direction), as agreed.

    surprises = "joint": each run takes the five model surprises (load factor, yield, fuel translator, unit cost, other
    revenue) from ONE past quarter, so any co-movement comes along (notebook 04, section 8); "independent": load factor
    and yield paired, the other three each from its own full history."""
    rng = np.random.default_rng(seed)
    cen = lambda x: np.asarray(x, float) - np.mean(x)
    pick = lambda x: rng.choice(cen(x), n) if len(x) else np.zeros(n)   # no history yet (early backtests): no spread
    rev, ce, cap, fuel = base["rev"], base["ce"], base["cap"], base["fuel"]
    lf, ym = rev["load_factor_model"], rev["yield_model"]
    om, um = ce["other_revenue_model"], ce["unit_cost_model"]
    k = rng.integers(0, len(rev["paired_residuals"]["quarters"]), n)       # load factor + yield: same past quarter
    return {
        "n": n,
        "cap_meas": pick(cap["measurement_errors_pct"]), "cap_scale": pick(cap["scaling_errors_pct"]),
        "cap_guid": pick(cap.get("guidance_errors_pct", [])),          # Delta's capacity guidance part (06), if any
        "demand_err": pick(rev["demand_conversion_errors"]),
        "B_lf": rng.multivariate_normal(list(lf["beta"].values()), lf["cov"], n),
        "B_y": rng.multivariate_normal(list(ym["beta"].values()), ym["cov"], n),
        "r_lf": np.asarray(rev["paired_residuals"]["load_factor"])[k], "r_y": np.asarray(rev["paired_residuals"]["yield"])[k],
        "B_o": rng.multivariate_normal(list(om["beta"].values()), om["cov"], n), "r_o": pick(list(om["resid"].values())),
        "B_u": rng.multivariate_normal(list(um["beta"].values()), um["cov"], n), "r_u": pick(list(um["resid"].values())),
        "fuel_err": pick(list(fuel["translator_errors_pct"].values())),
        "gal_err": pick(list(fuel["gallons_errors_pct"].values())),
        "mro_share": rng.choice(ce["mro_cost_share_pct"]["2026"], n),
        "mro_share_ly": rng.choice(ce["mro_cost_share_pct"]["2025"], n),
        "ps_err": pick(list(ce["profit_sharing"]["share_errors_pts"].values())),
        "eps_err": pick(list(ce["eps_bridge"]["errors_usd"].values())),
        "r_u_guided": pick(list(ce.get("unit_cost_guided", {}).get("errors_pct", {"_": 0.0}).values())),
        "surprises": surprises,
    } | (_joint(ce, rng, n) if surprises == "joint" else {})


def _joint(ce, rng, n):
    js = ce["joint_surprises"]
    J = np.asarray(js["values"], float)
    J = J - J.mean(axis=0)                                  # centered, column by column
    k = rng.integers(0, len(J), n)
    col = {c: J[k, i] for i, c in enumerate(js["columns"])}
    return {"r_lf": col["load factor"], "r_y": col["yield"], "fuel_err": col["fuel price (translator)"],
            "r_u": col["non-fuel unit cost"], "r_o": col["other revenue"]}


def evaluate(base: dict, R: dict, lv: Levers = Levers()) -> pd.DataFrame:
    """Flow one set of levers through the links, for every drawn run. $ in millions."""
    bi, cap, ce = base["base_inputs"], base["cap"], base["ce"]
    q = pd.Period(base["quarter"], "Q")
    # capacity: notebook 01's measurement, minus its drawn measurement and scaling errors, then the lever
    asm0 = cap["consolidated_seat_miles"] / (1 + R["cap_meas"] / 100) / (1 + R["cap_scale"] / 100) / (1 + R["cap_guid"] / 100)
    if lv.rest_of_quarter:           # notebook 05b: levers act only on the part of the quarter still to come
        sh = base["rest_share"]
        s_fly, s_pax = sh["flying"], sh["passengers"]
    else:
        s_fly = s_pax = 1.0
    cap_x = lv.capacity_pct / 100 * s_fly                       # change in the quarter's total flying
    fare_x = lv.fare_pct * s_pax                                # % change in the quarter's average fare (extra)
    outage = lv.outage if (not lv.rest_of_quarter or s_fly > 0) else 0.0
    asm = asm0 * (1 + cap_x)
    cap_g0 = (asm0 / cap["consolidated_q3_2025_reported"] - 1) * 100
    cap_g = (asm / cap["consolidated_q3_2025_reported"] - 1) * 100
    dem0 = bi["demand"] - R["demand_err"]
    dem = dem0 + lv.demand_pts * s_pax
    mkt0 = market_quarter(base["jet"], q, base["lag"], base["as_of"])[0]
    if lv.rest_of_quarter and not lv.fuel_hindsight:
        mkt = market_quarter_rest(base["jet"], q, base["lag"], base["as_of"], lv.fuel_pct, lv.capacity_pct / 100)[0]
    else:
        mkt = market_quarter(base["jet"], q, base["lag"], base["as_of"], lv.fuel_pct, lv.fuel_hindsight)[0]
    fuel_chg0, fuel_chg = (mkt0 / base["market_ly"] - 1) * 100, (mkt / base["market_ly"] - 1) * 100
    Bl, By = R["B_lf"], R["B_y"]

    def lf_yield(d, c, f):
        dlf_ = Bl[:, 0] + Bl[:, 1] * bi["dlf_last"] + Bl[:, 2] * d + Bl[:, 3] * c + R["r_lf"]
        return dlf_, By[:, 0] + By[:, 1] * bi["yield_last"] + By[:, 2] * dlf_ + By[:, 3] * f + R["r_y"]
    # isolation: the behavioural links see the base case's inputs, so only the moved lever's own accounting acts
    dlf0, y0 = lf_yield(dem0, cap_g0, fuel_chg0)
    dlf, y_auto = (dlf0, y0) if lv.isolation else lf_yield(dem, cap_g, fuel_chg)
    if lv.fare_mode == "total":      # fares end lv.fare_pct above the base case in total, whatever the links did
        yld = ((1 + y0 / 100) * (1 + fare_x / 100) - 1) * 100
    else:
        yld = ((1 + y_auto / 100) * (1 + fare_x / 100) - 1) * 100
    lf_ly = base["lf_ly"]
    pax_g0 = ((1 + cap_g0 / 100) * ((lf_ly + dlf0) / lf_ly) * (1 + y0 / 100) - 1) * 100
    pax_g = ((1 + cap_g / 100) * ((lf_ly + dlf) / lf_ly) * (1 + yld / 100) - 1) * 100
    pax = base["pax_ly"] * (1 + pax_g / 100)
    Bo = R["B_o"]
    oth_g = Bo[:, 0] + Bo[:, 1] * ce["base"]["other_last"] + Bo[:, 2] * (pax_g0 if lv.isolation else pax_g) + R["r_o"]
    oth = base["other_ly"] * (1 + oth_g / 100)
    out = OUTAGE_JULY_2024
    revenue = pax + oth + outage * out["revenue_musd"]
    a, b = base["translator"]
    price = (a + b * mkt) / (1 + R["fuel_err"] / 100)            # notebook 02: error = predicted / actual − 1
    gallons = asm / 1e6 * base["gallons_per_asm"] * (1 + R["gal_err"] / 100)
    fuel_cost = gallons * price + outage * out["fuel_musd"]
    Bu = R["B_u"]
    c_uc = cap_g0 if lv.isolation else cap_g
    guided = base.get("unit_cost_guided_pct")
    if guided is None:               # notebook 04: Delta's guided unit cost when it gives a number, N2 otherwise
        uc = Bu[:, 0] + Bu[:, 1] * ce["base"]["unit_cost_last"] + Bu[:, 2] * c_uc + R["r_u"]
    else:                            # guidance is for the planned flying; a capacity lever moves it by N2's link
        uc = guided + Bu[:, 2] * (c_uc - cap_g0) + R["r_u_guided"]
    nf_ly = base["nonfuel_ly_old"] - R["mro_share_ly"] / 100 * base["other_ly"]    # last year on the 2026 definition
    nonfuel = nf_ly * (1 + uc / 100) * (1 + cap_g / 100) * (1 + lv.nonfuel_pct / 100 * s_fly) + outage * out["nonfuel_musd"]
    mro = R["mro_share"] / 100 * oth
    op_before_ps = revenue - fuel_cost - nonfuel - mro
    nonop = ce["base"]["nonop_musd"]
    ps = np.maximum(op_before_ps + nonop, 0) * (lv.ps_rate + R["ps_err"]) / 100
    op = op_before_ps - ps
    pretax = op + nonop
    tax = ce["base"]["tax_rate_pct"]
    eps = pretax * (1 - tax / 100) / ce["base"]["diluted_shares_m"] + R["eps_err"]
    return pd.DataFrame({"capacity_pct": cap_g, "demand_pct": dem, "load_factor_chg_pts": dlf, "yield_pct": yld,
                         "yield_auto_pct": ((1 + y_auto / 100) / (1 + y0 / 100) - 1) * 100,
                         "yield_extra_pct": ((1 + yld / 100) / (1 + y_auto / 100) - 1) * 100,
                         "passenger_revenue": pax, "passenger_revenue_pct": pax_g, "other_revenue": oth,
                         "revenue": revenue, "revenue_pct": (revenue / base["revenue_ly"] - 1) * 100,
                         "fuel_price": price, "gallons": gallons, "fuel_cost": fuel_cost, "unit_cost_pct": uc,
                         "nonfuel_cost": nonfuel, "mro_cost": mro, "profit_sharing": ps, "operating_income": op,
                         "operating_margin": op / revenue * 100, "pretax_income": pretax, "eps": eps})


def summarize(sim: pd.DataFrame, guidance: dict, cols=("revenue_pct", "operating_margin", "eps")) -> pd.DataFrame:
    """Median, 10th-90th percentiles, and the share of runs below / inside / above Delta's guidance range."""
    gmap = {"revenue_pct": "total_revenue_yoy", "operating_margin": "operating_margin", "eps": "eps"}
    rows = {}
    for c in cols:
        x = sim[c].values
        g = guidance.get(gmap.get(c, ""), None)
        r = {"median": np.median(x), "10th percentile": np.percentile(x, 10), "90th percentile": np.percentile(x, 90)}
        if g:
            r.update({"guidance low": g["low"], "guidance high": g["high"], "below %": (x < g["low"]).mean() * 100,
                      "inside %": ((x >= g["low"]) & (x <= g["high"])).mean() * 100, "above %": (x > g["high"]).mean() * 100})
        rows[c] = r
    return pd.DataFrame(rows).T


def shapley(base: dict, R: dict, levers: Levers, stat=lambda s: s["eps"].median()) -> pd.Series:
    """What drove it: each lever's share of the change from the base case, averaged over every order in which the
    levers could be added (Shapley values), so the shares add up exactly to the total change."""
    act = levers.active()
    names = list(act)
    k = len(names)
    cache = {}

    def val(subset):
        key = frozenset(subset)
        if key not in cache:
            cache[key] = stat(evaluate(base, R, Levers(**{n_: act[n_] for n_ in subset}, **levers.modes())))
        return cache[key]
    contrib = {}
    for i in names:
        others = [n_ for n_ in names if n_ != i]
        tot = 0.0
        for r in range(len(others) + 1):
            w = factorial(r) * factorial(k - r - 1) / factorial(k)
            for S in combinations(others, r):
                tot += w * (val(S + (i,)) - val(S))
        contrib[i] = tot
    s = pd.Series(contrib)
    s["base case"] = val(())
    s["scenario"] = val(tuple(names))
    return s
