"""SkyMiles award seat availability - EDGE SOURCE (P1).

Why: award inventory shrinks as flights fill. Tracking availability on key routes
weeks ahead gives a forward-looking read on load factor that few funds watch.

TODO:
  - choose source (e.g. a paid award-search API vs. own scraper) and CHECK TERMS OF SERVICE
  - check whether any historical archive exists; if not, start collecting NOW
    (scripts/collect_daily.py) - no history = cannot backtest
  - route panel from config/carriers/dal.yaml: key_routes
Series: awards.dl.<route>.<cabin>.saver_availability (share of flights with saver seats),
        by days-to-departure bucket
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from .base import DataSource


class AwardAvailabilitySource(DataSource):
    name = "awards"
    default_lag = timedelta(days=0)

    def fetch(self, start: date, end: date) -> None:
        raise NotImplementedError

    def parse(self) -> pd.DataFrame:
        raise NotImplementedError
