"""Corporate travel index from OTHER companies' SEC filings (P1) - a differentiator.

Idea: Delta's premium / corporate revenue depends on business travel budgets. Every
public company discusses travel & entertainment spend in 10-Qs, 8-Ks and earnings
releases. Use SEC EDGAR full-text search (on Longaeva's list, free, history since 2001)
to pull passages mentioning "business travel", "travel and entertainment", "T&E",
then classify each with the OpenAI batch model: increasing / decreasing / flat, and
whether it is a cost-cutting mention.

Output: quarterly (and filing-date-stamped) corporate_travel_index, weighted toward
large companies / sectors that are heavy business travelers (consulting, banks, tech).
Point-in-time: stamp each passage with its filing date.
"""
from __future__ import annotations

import pandas as pd

SEARCH_TERMS = ['"business travel"', '"travel and entertainment"', '"travel expenses"']


def search_filings(term: str, start: str, end: str, forms=("10-Q", "10-K", "8-K")) -> pd.DataFrame:
    """EDGAR full-text search -> (cik, company, form, filed_at, snippet). TODO."""
    raise NotImplementedError


def classify_passages(passages: pd.DataFrame, llm) -> pd.DataFrame:
    """Batch-classify direction of travel spend. TODO."""
    raise NotImplementedError


def build_index(labeled: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to a point-in-time index for the PointInTimeStore. TODO."""
    raise NotImplementedError
