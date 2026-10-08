"""Pricing module: passenger yield (cents per revenue passenger mile), "momentum + mechanism".

    yield growth(q) = yield growth(L) + c1 x [fuel growth(q) - fuel growth(L)] + c2 x [dLF(q) - dLF(L)] + e

Start from the latest reported yield trend (pricing is persistent), then adjust for:
  - fuel pass-through: fuel cost increases are partly passed into fares (c1 > 0)
  - pricing power: fuller planes than last quarter's trend -> higher fares (c2 > 0)
Growth = YoY log change; L = latest reported quarter. Fit without intercept since 2022Q2.
Out-of-sample 2023Q4-2025Q2: ~2.5% mean abs yield error vs ~4% for a levels regression (notebook 03).
Output: yield_
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ._quarterly import fit_window, reported_q
from .base import Module


def _frame(store, as_of) -> pd.DataFrame:
    asm, rpm = reported_q(store, "asm_m", as_of), reported_q(store, "rpm_m", as_of)
    y, fuel = reported_q(store, "yield_cents", as_of), reported_q(store, "fuel_price_gal", as_of)
    lf = rpm / asm
    d = pd.DataFrame({"yield": y, "fuel": fuel, "y_yoy": np.log(y / y.shift(4)),
                      "f_yoy": np.log(fuel / fuel.shift(4)), "dlf": lf - lf.shift(4)})
    return d.reindex(pd.period_range(d.index.min(), d.index.max(), freq="Q"))


class PricingModule(Module):
    name = "pricing"
    inputs = ("load_factor", "fuel_price")
    outputs = ("yield_",)

    def fit(self, store, as_of) -> None:
        d = _frame(store, as_of)
        x = pd.DataFrame({"y": d["y_yoy"] - d["y_yoy"].shift(1),
                          "d_fuel": d["f_yoy"] - d["f_yoy"].shift(1),
                          "d_dlf": d["dlf"] - d["dlf"].shift(1)})
        x = x[fit_window(x.index) & fit_window(x.index - 1)].dropna()
        m = sm.OLS(x["y"], x[["d_fuel", "d_dlf"]]).fit()
        self.params = {"coef": m.params.to_dict(), "resid_sd": float(np.sqrt(np.mean(m.resid ** 2))),
                       "n": len(x), "frame": d}

    def nowcast(self, store, as_of, quarter: str) -> dict:
        q = pd.Period(quarter, freq="Q")
        d = self.params["frame"]
        last = d["y_yoy"].dropna().index.max()
        steps = q.ordinal - last.ordinal
        return {"yield_prior_year": float(d.loc[q - 4, "yield"]), "fuel_prior_year": float(d.loc[q - 4, "fuel"]),
                "y_yoy_last": float(d.loc[last, "y_yoy"]), "f_yoy_last": float(d.loc[last, "f_yoy"]),
                "dlf_last": float(d.loc[last, "dlf"]), "last_reported": str(last),
                "coef": self.params["coef"], "resid_sd": self.params["resid_sd"] * np.sqrt(steps)}

    def simulate(self, state, shocks, rng):
        nc = state.meta["nowcast"]["pricing"]
        dem = state.meta["nowcast"]["demand"]
        n = state.n_paths
        dlf = state.load_factor - dem["lf_prior_year"]
        f_yoy = np.log(state.fuel_price / nc["fuel_prior_year"])
        c = nc["coef"]
        g = nc["y_yoy_last"] + c["d_fuel"] * (f_yoy - nc["f_yoy_last"]) + c["d_dlf"] * (dlf - nc["dlf_last"]) \
            + nc["resid_sd"] * shocks.get("yield", rng.standard_normal(n))
        state.yield_ = nc["yield_prior_year"] * np.exp(g)
        return state
