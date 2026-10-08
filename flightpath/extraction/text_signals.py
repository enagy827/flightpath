"""High-volume text -> daily indices (reviews, Reddit, FlyerTalk, job postings).

Classify each post with a small model via Batch API (cheap), then aggregate:
  - premium_demand_index: upgrade clearance / sold-out premium cabin mentions
  - ops_pain_index: delays, cancellations, lost bags complaints (per hub)
  - pricing_sentiment: "fares are crazy high" vs "found a cheap deal"
  - hiring_index: postings by role family / hub (from jobs.py)
Normalize by total post volume; compare YoY. Each index must be tested for lead over
reported metrics (notebooks/04_edge_signals.ipynb) before entering the model.
"""
from __future__ import annotations

import pandas as pd

POST_LABELS = {
    "topic": ["pricing", "premium_cabin", "operations", "loyalty", "service", "other"],
    "sentiment": ["positive", "neutral", "negative"],
}


def classify_posts(posts: pd.DataFrame, llm) -> pd.DataFrame:
    """Batch-classify posts into POST_LABELS (+ hub if mentioned). TODO."""
    raise NotImplementedError


def build_indices(labeled: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """Aggregate to long-format Observations for the PointInTimeStore. TODO."""
    raise NotImplementedError
