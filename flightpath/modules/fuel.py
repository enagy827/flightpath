"""Fuel module - the most mechanical piece, built first.

    Fuel Expense = Fuel Consumption x Effective Fuel Price       (proposal eq. 4)

Consumption: gallons = ASM x gallons-per-ASM. Gallons/ASM trends slowly down (newer aircraft);
estimated from reported quarters (Delta's 8-K: fuel gallons consumed / ASM), same quarter last
year and recent trend. Later: OpenSky block hours by aircraft type, BTS Form 41 monthly gallons.

Effective price: Delta's reported price per gallon = a + b x (EIA Gulf Coast jet fuel, averaged over
the quarter shifted back ~14 days - Delta buys ahead). a, b fitted by regression on the last 8 normal
quarters (COVID quarters 2020Q2-2021Q2 excluded). Captures taxes, into-plane costs and Delta's Trainer
refinery. Backtest 2022Q3-2025Q2: ~2.5% mean abs price error vs ~10% for "last quarter's price".
The 14-day lag is confirmed independently on monthly BTS Form 41 fuel cost; regional diesel blends (EIA has no
NY Harbor/LA jet series) did not fit better than Gulf Coast jet (notebook 02).
Within the quarter: days already priced are known; remaining days follow a random walk from the
latest price with the historical daily volatility.

Output: fuel_gallons, fuel_price, fuel_expense
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Module

JET_SERIES = ["eia.jet_gulf_coast", "eia.jet_ny_harbor"]
COVID = (pd.Period("2020Q2", "Q"), pd.Period("2021Q2", "Q"))
PURCHASE_LAG_DAYS = 14


def _not_covid(idx) -> "np.ndarray":
    return ~((idx >= COVID[0]) & (idx <= COVID[1]))


def _window(q: pd.Period) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Market-price window that drives quarter q's fuel cost (shifted back by the purchase lag)."""
    lag = pd.Timedelta(days=PURCHASE_LAG_DAYS)
    return q.start_time - lag, q.end_time.normalize() - lag


def jet_price_daily(store, as_of) -> pd.Series:
    parts = [store.series_as_of(s, as_of) for s in JET_SERIES]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts, axis=1).mean(axis=1).dropna()


class FuelModule(Module):
    name = "fuel"
    inputs = ("asm",)
    outputs = ("fuel_gallons", "fuel_price", "fuel_expense")

    def __init__(self, basis_lookback_q: int = 8):
        self.basis_lookback_q = basis_lookback_q
        self.params: dict = {}

    def fit(self, store, as_of) -> None:
        p = {}
        gal = store.series_as_of("delta.reported.fuel_gallons_m", as_of) * 1e6
        asm = store.series_as_of("delta.reported.asm_m", as_of) * 1e6
        price = store.series_as_of("delta.reported.fuel_price_gal", as_of)
        gpa = (gal / asm).dropna()
        gpa.index = gpa.index.to_period("Q")
        p["gpa_by_quarter"] = gpa
        yoy = (gpa / gpa.shift(4)).dropna()
        yoy = yoy[_not_covid(yoy.index) & _not_covid(yoy.index - 4)].tail(4)
        p["gpa_yoy_ratio"] = float(yoy.mean()) if len(yoy) else 0.99
        p["gpa_sd"] = float(max(yoy.std(ddof=1), 0.005)) if len(yoy) > 1 else 0.01

        jet = jet_price_daily(store, as_of)
        if len(jet) and len(price):
            pq = price.copy(); pq.index = pq.index.to_period("Q")
            pq = pq[_not_covid(pq.index)].tail(self.basis_lookback_q)
            mk = pd.Series({q: jet[(jet.index >= _window(q)[0]) & (jet.index <= _window(q)[1])].mean() for q in pq.index})
            d = pd.DataFrame({"rep": pq, "mkt": mk}).dropna()
            if len(d) >= 4:
                b, a = np.polyfit(d["mkt"], d["rep"], 1)
                resid = d["rep"] - (a + b * d["mkt"])
                p.update(a=float(a), b=float(b), price_resid_sd=float(resid.std(ddof=2)))
            else:
                p.update(a=0.0, b=float((d["rep"] / d["mkt"]).median()) if len(d) else 1.0, price_resid_sd=0.15)
            r = np.log(jet).diff().dropna().tail(250)
            p["daily_vol"] = float(r.std(ddof=1))
        self.params = p

    def nowcast(self, store, as_of, quarter: str) -> dict:
        p = self.params
        q = pd.Period(quarter, freq="Q")
        gpa = p["gpa_by_quarter"]
        if (q - 4) in gpa.index:
            gpa_mean = gpa.loc[q - 4] * p["gpa_yoy_ratio"]
        else:
            gpa_mean = gpa.iloc[-1]
        jet = jet_price_daily(store, as_of)
        w0, w1 = _window(q)
        days = pd.bdate_range(w0, w1)
        known = jet[(jet.index >= w0) & (jet.index <= w1)]
        return {
            "gpa_mean": float(gpa_mean), "gpa_sd": float(gpa_mean * p["gpa_sd"]),
            "known_prices": known.values, "last_price": float(jet.iloc[-1]) if len(jet) else np.nan,
            "n_remaining_days": int(max(len(days) - len(known), 0)),
            "a": p.get("a", 0.0), "b": p.get("b", 1.0), "price_resid_sd": p.get("price_resid_sd", 0.15),
            "daily_vol": p.get("daily_vol", 0.02),
        }

    def simulate(self, state, shocks, rng):
        nc = state.meta["nowcast"]["fuel"]
        n = state.n_paths
        z_burn = shocks.get("fuel_burn", rng.standard_normal(n))
        gpa = nc["gpa_mean"] + nc["gpa_sd"] * z_burn
        state.fuel_gallons = state.asm * gpa

        # market price path: known days + random walk for the rest of the quarter
        k, rem = len(nc["known_prices"]), nc["n_remaining_days"]
        known_sum = float(np.sum(nc["known_prices"]))
        if rem > 0:
            z = shocks.get("fuel_price_path", rng.standard_normal((n, rem)))
            logp = np.log(nc["last_price"]) + np.cumsum(nc["daily_vol"] * z, axis=1)
            future_sum = np.exp(logp).sum(axis=1)
        else:
            future_sum = np.zeros(n)
        market_avg = (known_sum + future_sum) / max(k + rem, 1)
        scale = state.meta.get("fuel_price_multiplier", 1.0)       # scenario hook (e.g. +20%)
        resid = nc["price_resid_sd"] * shocks.get("fuel_basis", rng.standard_normal(n))
        state.fuel_price = nc["a"] + nc["b"] * market_avg * scale + resid
        state.fuel_expense = state.fuel_gallons * state.fuel_price
        return state
