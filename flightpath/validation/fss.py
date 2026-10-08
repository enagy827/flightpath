"""Investment validation: does the decision signal predict returns?

Signal construction lives in flightpath/decision/signal.py; this module tests it.
Test FSS vs subsequent DAL returns (earnings-window abnormal return vs. market / JETS ETF)
before defining any trading rule. Report IC, hit rate, and t-stats; be honest about
small sample size (~1 observation per quarter -> also use daily as_of snapshots carefully,
they are highly overlapping).
"""
from __future__ import annotations

import pandas as pd


def test_vs_returns(signal: pd.DataFrame, returns: pd.DataFrame) -> dict:
    raise NotImplementedError


def compare_signal_variants(results: pd.DataFrame) -> pd.DataFrame:
    """Base data only vs. base + edge data: does the edge data add anything? TODO.
    This comparison is the key slide for 'why our data is an edge'."""
    raise NotImplementedError
