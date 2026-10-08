"""Google Trends - search interest as an early demand signal (P1).

History back to 2004 -> backtestable. Weekly for multi-year windows, daily for short
windows (stitch carefully: each query is normalized to 0-100).
Keyword ideas (config: text_sources.trends_keywords): "cheap flights", "Delta flights",
"flights to <top Delta leisure destinations>", "SkyMiles".
Test in notebooks/04_edge_signals.ipynb whether it leads Delta RPM beyond TSA.
TODO: pick client (e.g. pytrends is unofficial and rate-limited) and handle throttling.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from .base import DataSource


class GoogleTrendsSource(DataSource):
    name = "trends"
    default_lag = timedelta(days=1)

    def fetch(self, start: date, end: date) -> None:
        raise NotImplementedError

    def parse(self) -> pd.DataFrame:
        raise NotImplementedError
