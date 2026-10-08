"""BusinessState: one array of n_paths per quantity, for one carrier-quarter."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

FIELDS = ["demand_rpm", "asm", "effective_asm", "rpm", "load_factor", "yield_",
          "passenger_revenue", "fuel_gallons", "fuel_price", "fuel_expense",
          "cancel_rate", "disruption_cost"]


@dataclass
class BusinessState:
    carrier: str
    quarter: str               # e.g. "2026Q3"
    as_of: date
    n_paths: int
    demand_rpm: np.ndarray | None = None
    asm: np.ndarray | None = None
    effective_asm: np.ndarray | None = None
    rpm: np.ndarray | None = None
    load_factor: np.ndarray | None = None
    yield_: np.ndarray | None = None
    passenger_revenue: np.ndarray | None = None
    fuel_gallons: np.ndarray | None = None
    fuel_price: np.ndarray | None = None
    fuel_expense: np.ndarray | None = None
    cancel_rate: np.ndarray | None = None
    disruption_cost: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def populated(self) -> list[str]:
        return [f for f in FIELDS if getattr(self, f) is not None]

    def summary(self, levels=(0.5, 0.8, 0.95)) -> pd.DataFrame:
        """Mean, median, sd and central intervals for each populated field."""
        rows = []
        for f in self.populated():
            x = np.asarray(getattr(self, f), dtype=float)
            row = {"field": f, "mean": x.mean(), "median": np.median(x), "sd": x.std(ddof=1)}
            for lv in levels:
                lo, hi = np.quantile(x, [(1 - lv) / 2, 1 - (1 - lv) / 2])
                row[f"p{int(lv * 100)}_lo"], row[f"p{int(lv * 100)}_hi"] = lo, hi
            rows.append(row)
        return pd.DataFrame(rows).set_index("field")

    def prob_above(self, field_name: str, threshold: float) -> float:
        """P(field > threshold), e.g. P(passenger revenue beats the guidance-implied value)."""
        x = np.asarray(getattr(self, field_name), dtype=float)
        return float((x > threshold).mean())
