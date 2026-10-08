"""Reconcile BTS T-100 against Delta's reported operating statistics (notebook 00).

Two checks:
  1. Totals: quarterly BTS capacity/traffic vs Delta's reported ASM/RPM -> coverage ratio (validation + calibration)
  2. Regions: BTS capacity growth by region vs Delta's reported regional capacity change (same comparison base)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def quarterly_totals(routes: pd.DataFrame) -> pd.DataFrame:
    q = routes.assign(q=routes["period"].dt.to_period("Q")).groupby("q")[["asm", "rpm"]].sum()
    q["months"] = routes.assign(q=routes["period"].dt.to_period("Q")).groupby("q")["period"].nunique()
    return q[q["months"] == 3].drop(columns="months")


def reconcile_totals(mainline: pd.DataFrame, with_endeavor: pd.DataFrame, reported: pd.DataFrame) -> pd.DataFrame:
    """reported: index = quarter Period, columns asm_m, rpm_m (Delta's releases)."""
    a, b = quarterly_totals(mainline), quarterly_totals(with_endeavor)
    out = pd.DataFrame({
        "reported_asm_bn": reported["asm_m"] / 1e3,
        "bts_mainline_asm_bn": a["asm"] / 1e9,
        "bts_mainline_plus_endeavor_asm_bn": b["asm"] / 1e9,
    }).dropna()
    out["mainline_share"] = out["bts_mainline_asm_bn"] / out["reported_asm_bn"]
    out["mainline_plus_endeavor_share"] = out["bts_mainline_plus_endeavor_asm_bn"] / out["reported_asm_bn"]
    out["reported_load_factor"] = (reported["rpm_m"] / reported["asm_m"]).reindex(out.index)
    out["bts_mainline_load_factor"] = (a["rpm"] / a["asm"]).reindex(out.index)
    return out


def reconcile_regions(routes: pd.DataFrame, reported_regions: pd.DataFrame) -> pd.DataFrame:
    """Compare capacity growth by region, using the same base year Delta used in each release."""
    names = {"Domestic": "domestic", "Atlantic": "atlantic", "Latin America": "latin", "Pacific": "pacific"}
    q = routes.assign(q=routes["period"].dt.to_period("Q")).groupby(["q", "region"])["asm"].sum().unstack()
    rows = []
    for _, r in reported_regions.iterrows():
        qq = pd.Period(r["quarter"], "Q")
        base = pd.Period(f"{int(r['comparison_base_year'])}Q{qq.quarter}", "Q")
        if qq in q.index and base in q.index:
            bts = (q.loc[qq, names[r["region"]]] / q.loc[base, names[r["region"]]] - 1) * 100
            rows.append({"quarter": r["quarter"], "region": r["region"], "base_year": int(r["comparison_base_year"]),
                         "delta_reported_capacity_chg_pct": r["capacity_chg_pct"], "bts_capacity_chg_pct": bts,
                         "difference_pts": bts - r["capacity_chg_pct"]})
    return pd.DataFrame(rows)
