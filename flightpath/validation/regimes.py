"""COVID structural-break handling.

Options (config: regimes.covid.treatment):
  exclude         - drop 2020Q2-2021Q4 from estimation
  separate_regime - regime dummy / separate parameters
  stress_test     - exclude from fit, run model through it to show failure detection
"""
from __future__ import annotations

import pandas as pd


def apply_regime_treatment(df: pd.DataFrame, treatment: str) -> pd.DataFrame:
    raise NotImplementedError


def detect_break(residuals: pd.Series) -> pd.Series:
    """Flag periods where relationships stop holding (e.g. CUSUM). Nice-to-have. TODO."""
    raise NotImplementedError
