"""NOAA weather around Delta hubs. DEFERRED - lowest Stage 2 priority.

Use: severe-weather indicators (precip, snow, thunderstorms, wind) per hub-day,
feeding the disruption module - never a direct financial predictor.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from .base import DataSource


class NOAASource(DataSource):
    name = "noaa"
    default_lag = timedelta(days=1)

    def fetch(self, start: date, end: date) -> None:
        # TODO: NOAA CDO API (NOAA_TOKEN); GHCND stations at each hub airport
        raise NotImplementedError

    def parse(self) -> pd.DataFrame:
        raise NotImplementedError
