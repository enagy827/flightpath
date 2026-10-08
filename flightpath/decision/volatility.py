"""Model distribution vs. options-implied earnings move.

FlightPath produces a full distribution, so it can say something about UNCERTAINTY,
not just direction:
  - model much tighter than options imply  -> options overpricing the event (sell vol idea)
  - model much wider (e.g. unresolved disruption) -> options underpricing risk (buy vol idea)
Mapping fundamentals -> stock move needs an estimated sensitivity (historical
earnings-day return vs. surprise). TODO: estimate in notebooks/05_decision_backtest.ipynb.
"""
from __future__ import annotations


def model_implied_move(fundamental_surprise_dist, sensitivity: float) -> float:
    raise NotImplementedError


def compare_to_options(model_move: float, implied_move: float) -> dict:
    """Return {'ratio': ..., 'view': 'rich' | 'cheap' | 'fair'}. TODO."""
    raise NotImplementedError
