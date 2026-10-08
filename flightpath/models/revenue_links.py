"""Demand, capacity and fuel -> load factor -> yield -> passenger revenue (notebook 03).

All quarterly, as year-over-year changes, like-for-like (each change from the prior-year column of the same Delta
release, notebook 00). Every row carries the date its inputs became public, so a model fitted "as of" a date only
uses quarters whose outcome AND inputs were public by then.

Links (chosen in notebook 03):
  load factor change (pts)  = a + r * last quarter's change + b * demand growth + c * capacity growth
  yield growth (%)          = a + r * last quarter's growth + b * load factor change + f * fuel price change
  passenger revenue growth  = (1 + capacity) x (load factor / last year's) x (1 + yield) - 1     (an identity)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COVID_YOY = (pd.Period("2020Q1", "Q"), pd.Period("2023Q2", "Q"))   # the quarter or its year-earlier base is COVID-hit

LF_FEATURES = ["dlf_last", "demand", "capacity"]
YIELD_FEATURES = ["yield_last", "dlf", "fuel"]


def quarterly_frame(reported: pd.DataFrame, industry_monthly: pd.DataFrame, jet_daily: pd.Series,
                    fuel_delay_days: int = 14, t100_lag_days: int = 80) -> pd.DataFrame:
    """One row per quarter: Delta's capacity growth, load factor change, yield growth, passenger revenue growth (all
    vs the prior-year column of the same release), industry demand growth (T-100 passengers departing US airports),
    market jet fuel price change (quarter average, delayed by the purchase delay of notebook 02), plus when each was
    public. `reported` is indexed by quarter (Period)."""
    r = reported
    d = pd.DataFrame(index=r.index)
    d["valid"] = r["comparison_quarter"].astype(str) == (r.index - 4).astype(str)   # base really is a year earlier
    d["capacity"] = (r.asm_m / r.asm_m_prior_year - 1) * 100
    d["lf"] = r.rpm_m / r.asm_m * 100
    d["dlf"] = d["lf"] - r.rpm_m_prior_year / r.asm_m_prior_year * 100
    d["yield"] = (r.yield_cents / r.yield_cents_prior_year - 1) * 100
    d["pax_revenue"] = (r.passenger_revenue_m / r.passenger_revenue_m_prior_year - 1) * 100
    d["delta_public"] = r["filed"]
    iq = industry_monthly["passengers_from_us"].groupby(industry_monthly.index.to_period("Q")).sum(min_count=3)
    d["demand"] = ((iq / iq.shift(4) - 1) * 100).reindex(d.index)
    d["demand_public"] = [q.end_time.normalize() + pd.Timedelta(days=t100_lag_days) for q in d.index]
    lag = pd.Timedelta(days=fuel_delay_days)
    mkt = pd.Series({q: jet_daily[q.start_time - lag: q.end_time.normalize() - lag].mean() for q in d.index})
    d["fuel"] = ((mkt / mkt.shift(4) - 1) * 100).reindex(d.index)
    d["fuel_market"] = mkt.reindex(d.index)
    bad = ~d["valid"]
    d.loc[bad, ["dlf", "yield"]] = np.nan                          # a non-like-for-like change can't be a lag either
    d["dlf_last"] = d["dlf"].shift(1)
    d["yield_last"] = d["yield"].shift(1)
    covid = [(COVID_YOY[0] <= q <= COVID_YOY[1]) for q in d.index]
    d["usable"] = d["valid"] & ~pd.Series(covid, index=d.index)
    return d


def training_rows(d: pd.DataFrame, as_of, features: list, target: str) -> pd.DataFrame:
    """Usable quarters whose outcome and every input were public on `as_of` (nothing later)."""
    as_of = pd.Timestamp(as_of)
    t = d[d["usable"] & (d["delta_public"] <= as_of) & (d["demand_public"] <= as_of)]
    return t.dropna(subset=features + [target])


def fit(t: pd.DataFrame, features: list, target: str) -> dict:
    """Ordinary least squares with an intercept; returns coefficients, their covariance and residuals."""
    X = np.column_stack([np.ones(len(t))] + [t[f].values for f in features])
    y = t[target].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = resid @ resid / (len(y) - X.shape[1])
    cov = s2 * np.linalg.inv(X.T @ X)
    names = ["intercept"] + features
    return {"beta": pd.Series(beta, index=names), "cov": pd.DataFrame(cov, index=names, columns=names),
            "resid": pd.Series(resid, index=t.index), "n": len(y)}


def predict(model: dict, row: dict | pd.Series) -> float:
    b = model["beta"]
    return float(b["intercept"] + sum(b[f] * row[f] for f in b.index if f != "intercept"))


def pax_revenue_growth(capacity_pct: float, lf_last_year: float, dlf_pts: float, yield_pct: float) -> float:
    """Passenger revenue growth from the identity revenue = seat-miles x load factor x yield."""
    return ((1 + capacity_pct / 100) * ((lf_last_year + dlf_pts) / lf_last_year) * (1 + yield_pct / 100) - 1) * 100
