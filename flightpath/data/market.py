"""Market expectations - needed for the DECISION layer.

  - Consensus estimates for Delta line items (passenger revenue, fuel, EPS).
    Usually paid (Bloomberg/FactSet/Visible Alpha) -> ASK LONGAEVA.
    Fallback: Delta's own guidance midpoint as the baseline.
  - DAL daily prices (for returns test).
  - DAL options: implied earnings move from straddle price around the earnings date.
    TODO: find source with history (Longaeva? paid vendor?); yfinance has only current chains.
"""
from __future__ import annotations

from datetime import date

import pandas as pd


def load_consensus(ticker: str, metric: str) -> pd.DataFrame:
    """Columns: quarter, as_of, consensus_mean, consensus_std, n_estimates. TODO."""
    raise NotImplementedError


def load_prices(ticker: str, start: date, end: date) -> pd.DataFrame:
    raise NotImplementedError


def implied_earnings_move(ticker: str, earnings_date: date, as_of: date) -> float:
    """Straddle-implied absolute move over the earnings date, as_of a given date. TODO."""
    raise NotImplementedError
