"""Common interface for all data sources."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, timedelta

import pandas as pd


class DataSource(ABC):
    name: str
    # Default publication lag used when a true release date is unknown.
    # Prefer real release calendars where available.
    default_lag: timedelta

    @abstractmethod
    def fetch(self, start: date, end: date) -> None:
        """Download raw files into data/raw/<name>/. Must be idempotent / cached."""

    @abstractmethod
    def parse(self) -> pd.DataFrame:
        """Parse raw files into long format:
        columns = [series, period, available_at, value, source, vintage]
        """

    def assign_availability(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill available_at = period + default_lag where no release date is known.
        TODO: implement; log a warning whenever the fallback is used.
        """
        raise NotImplementedError
