"""Other revenue, non-fuel costs, profit sharing and the step from operating income to EPS (notebook 04).

All on Delta's ADJUSTED basis (the basis of its guidance), quarterly, year-over-year changes from the same release
(like-for-like). Every row carries the date its inputs became public (Delta's filing date), so a model fitted "as of"
a date only uses quarters that were public by then.

Links (chosen in notebook 04):
  non-fuel unit cost growth (%)  = a + r * last quarter's growth + c * capacity growth
  non-fuel cost growth (%)       = (1 + unit cost growth) x (1 + capacity growth) - 1           (identity)
  other revenue growth (%)       = a + r * last quarter's growth + b * passenger revenue growth
  MRO expense (2026 definition)  = share x other revenue                                        (since 2026)
  profit sharing                 = k x pre-tax income before profit sharing
  EPS                            = (operating income + non-operating) x (1 - tax rate) / diluted shares
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .revenue_links import COVID_YOY


def quarterly_frame(adj: pd.DataFrame) -> pd.DataFrame:
    """`adj` = flightpath.data.sec_adjusted.adjusted_quarters(). One row per quarter."""
    a = adj
    d = pd.DataFrame(index=a.index)
    d["valid"] = a["comparison_quarter"].astype(str) == (a.index - 4).astype(str)
    d["delta_public"] = a["filed"]
    d["capacity"] = (a.asm_m / a.asm_m_prior_year - 1) * 100
    d["unit_cost"] = (a.nonfuel_cost_m / a.asm_m) / (a.nonfuel_cost_m_prior_year / a.asm_m_prior_year) * 100 - 100
    d["nonfuel"] = (a.nonfuel_cost_m / a.nonfuel_cost_m_prior_year - 1) * 100
    d["other"] = (a.other_revenue_adj_m / a.other_revenue_adj_m_prior_year - 1) * 100
    d["pax"] = (a.passenger_revenue_m / a.passenger_revenue_m_prior_year - 1) * 100
    d["revenue_adj"] = (a.revenue_adj_m / a.revenue_adj_m_prior_year - 1) * 100
    bad = ~d["valid"]
    d.loc[bad, ["unit_cost", "nonfuel", "other", "pax", "revenue_adj"]] = np.nan
    d["unit_cost_last"] = d["unit_cost"].shift(1)
    d["other_last"] = d["other"].shift(1)
    covid = [(COVID_YOY[0] <= q <= COVID_YOY[1]) for q in d.index]
    d["usable"] = d["valid"] & ~pd.Series(covid, index=d.index)
    # levels ($M) for profit sharing and the EPS bridge
    for c in ("revenue_adj_m", "passenger_revenue_m", "other_revenue_adj_m", "nonfuel_cost_m", "profit_sharing_m",
              "mro_expense_m", "operating_income_adj_m", "pretax_income_adj_m", "net_income_adj_m", "eps_adj",
              "diluted_shares_m", "asm_m"):
        d[c] = a[c]
    d["fuel_adj_m"] = a.fuel_gallons_m * a.fuel_price_gal_adj
    d["pretax_before_ps_m"] = d.pretax_income_adj_m + d.profit_sharing_m
    d["ps_share"] = d.profit_sharing_m / d.pretax_before_ps_m * 100
    d["nonop_m"] = d.pretax_income_adj_m - d.operating_income_adj_m
    d["tax_rate"] = (1 - d.net_income_adj_m / d.pretax_income_adj_m) * 100
    return d


def training_rows(d: pd.DataFrame, as_of, features: list, target: str) -> pd.DataFrame:
    """Usable quarters whose figures were public on `as_of` (Delta's filing date), nothing later."""
    t = d[d["usable"] & (d["delta_public"] <= pd.Timestamp(as_of))]
    return t.dropna(subset=features + [target])


def nonfuel_growth(unit_cost_pct: float, capacity_pct: float) -> float:
    return ((1 + unit_cost_pct / 100) * (1 + capacity_pct / 100) - 1) * 100


def eps(operating_income_m: float, nonop_m: float, tax_rate_pct: float, shares_m: float) -> float:
    return (operating_income_m + nonop_m) * (1 - tax_rate_pct / 100) / shares_m
