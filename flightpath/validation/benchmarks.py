"""Simple benchmarks FlightPath must beat (tests the core hypothesis)."""
from __future__ import annotations

import pandas as pd


def seasonal_naive(history: pd.DataFrame, target: str) -> pd.DataFrame:
    """Same quarter last year x trailing YoY growth. TODO."""
    raise NotImplementedError


def guidance_implied(history: pd.DataFrame, target: str) -> pd.DataFrame:
    """Management guidance midpoint applied to last year's value. TODO."""
    raise NotImplementedError


def consensus(target: str) -> pd.DataFrame:
    """Analyst consensus, if Longaeva provides it (likely under NDA). TODO."""
    raise NotImplementedError
