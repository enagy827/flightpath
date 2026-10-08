"""Demand -> load factor module ("momentum + mechanism").

    dLF(q) = dLF(L) + b1 x [TSA growth(q) - TSA growth(L)] + b2 x [ASM growth(q) - ASM growth(L)] + e

dLF = load factor change vs the same quarter last year (points); growth = YoY log change;
L = the latest quarter Delta has reported at as_of. Start from the most recent observed load-factor
trend, then adjust for how industry demand (TSA, daily) and Delta's capacity have moved since.
Demand growing faster than capacity fills planes (b1 > 0); extra capacity empties them (b2 < 0).
b1, b2 fitted without intercept on quarters since 2022Q2 (COVID YoY excluded).
A pure-levels regression was tried first: it overfit the 2022-23 recovery (notebook 03).
Output: load_factor, rpm
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ._quarterly import fit_window, reported_q, tsa_full_quarter_yoy, tsa_quarter_yoy
from .base import Module


def _frame(store, as_of) -> pd.DataFrame:
    asm, rpm = reported_q(store, "asm_m", as_of), reported_q(store, "rpm_m", as_of)
    lf = rpm / asm
    d = pd.DataFrame({"lf": lf, "asm": asm, "dlf": lf - lf.shift(4), "asm_yoy": np.log(asm / asm.shift(4))})
    d = d.reindex(pd.period_range(d.index.min(), d.index.max(), freq="Q"))
    d["tsa_yoy"] = tsa_full_quarter_yoy(store, as_of)
    return d


class DemandModule(Module):
    name = "demand"
    inputs = ("asm",)
    outputs = ("load_factor", "rpm")

    def __init__(self, tsa_rest_of_quarter_sd: float = 0.03):
        self.tsa_rest_sd = tsa_rest_of_quarter_sd
        self.params: dict = {}

    def fit(self, store, as_of) -> None:
        d = _frame(store, as_of)
        x = pd.DataFrame({"y": d["dlf"] - d["dlf"].shift(1),
                          "d_tsa": d["tsa_yoy"] - d["tsa_yoy"].shift(1),
                          "d_asm": d["asm_yoy"] - d["asm_yoy"].shift(1)})
        x = x[fit_window(x.index) & fit_window(x.index - 1)].dropna()
        m = sm.OLS(x["y"], x[["d_tsa", "d_asm"]]).fit()
        self.params = {"coef": m.params.to_dict(), "resid_sd": float(np.sqrt(np.mean(m.resid ** 2))),
                       "n": len(x), "frame": d}

    def nowcast(self, store, as_of, quarter: str) -> dict:
        q = pd.Period(quarter, freq="Q")
        d = self.params["frame"]
        last = d["dlf"].dropna().index.max()          # latest reported quarter
        steps = q.ordinal - last.ordinal              # 1 normally; 2 if the last quarter isn't out yet
        t = tsa_quarter_yoy(store, as_of, quarter)
        return {"tsa_yoy": t["yoy"], "tsa_share_observed": t["share_observed"],
                "lf_prior_year": float(d.loc[q - 4, "lf"]), "asm_prior_year": float(d.loc[q - 4, "asm"]) * 1e6,
                "dlf_last": float(d.loc[last, "dlf"]), "tsa_yoy_last": float(d.loc[last, "tsa_yoy"]),
                "asm_yoy_last": float(d.loc[last, "asm_yoy"]), "last_reported": str(last),
                "coef": self.params["coef"], "resid_sd": self.params["resid_sd"] * np.sqrt(steps)}

    def simulate(self, state, shocks, rng):
        nc = state.meta["nowcast"]["demand"]
        n = state.n_paths
        rest = 1 - nc["tsa_share_observed"]
        tsa_yoy = nc["tsa_yoy"] + self.tsa_rest_sd * rest * shocks.get("demand", rng.standard_normal(n))
        tsa_yoy = tsa_yoy + state.meta.get("demand_shock", 0.0)          # scenario hook (e.g. -0.05)
        asm_yoy = np.log(state.asm / nc["asm_prior_year"])
        c = nc["coef"]
        dlf = nc["dlf_last"] + c["d_tsa"] * (tsa_yoy - nc["tsa_yoy_last"]) \
            + c["d_asm"] * (asm_yoy - nc["asm_yoy_last"]) \
            + nc["resid_sd"] * shocks.get("load_factor", rng.standard_normal(n))
        state.load_factor = np.clip(nc["lf_prior_year"] + dlf, 0.3, 0.95)
        state.rpm = state.load_factor * state.asm
        state.meta["tsa_yoy_paths"] = tsa_yoy
        return state
