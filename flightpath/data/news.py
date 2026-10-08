"""News and event feeds - unstructured, sent to extraction/events.py.

Sources (all have history -> backtestable):
  - GDELT 2.0 (free, 15-min updates, archive since 2015): articles mentioning Delta,
    hubs, FAA, jet fuel, airline labor. Use for event DETECTION + timestamps.
  - Delta newsroom + 8-K filings: mid-quarter guidance updates (Delta often updates at
    investor conferences), route/fleet announcements.
  - FAA: NAS status / ground delay programs, airworthiness directives
    (e.g. engine issues grounding part of a fleet), ATC staffing.
  - Labor: union communications (ALPA, AFA), NMB filings.
  - Competitor news: capacity cuts, bankruptcies, route exits on Delta routes.

Timestamp = first publication time (point-in-time rule applies to news too).
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from .base import DataSource


class GDELTSource(DataSource):
    name = "gdelt"
    default_lag = timedelta(minutes=15)

    def fetch(self, start: date, end: date) -> None:
        # TODO: GDELT DOC 2.0 API or GKG files filtered by keyword list in config
        raise NotImplementedError

    def parse(self) -> pd.DataFrame:
        # returns documents (url, title, text/snippet, published_at) for extraction
        raise NotImplementedError


class FAASource(DataSource):
    name = "faa"
    default_lag = timedelta(hours=1)

    def fetch(self, start: date, end: date) -> None:
        # TODO: ground delay programs / ground stops at Delta hubs; airworthiness directives
        raise NotImplementedError

    def parse(self) -> pd.DataFrame:
        raise NotImplementedError
