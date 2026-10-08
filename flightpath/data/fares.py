"""Forward fares on Delta's key routes - EDGE SOURCE (P1).

Why: direct early read on passenger yield instead of inferring it from demand/capacity.
Collect lowest available fare by route x cabin x days-to-departure, daily.

TODO: pick source (fare API vs scraping - CHECK TERMS OF SERVICE); check history
availability; start daily collection ASAP. Build a fixed route panel so the index is
comparable over time (same routes, same advance-purchase windows).
Series: fares.dl.<route>.<cabin>.<dtd_bucket>, plus a weighted fare index vs. last year.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from .base import DataSource


class FareSource(DataSource):
    name = "fares"
    default_lag = timedelta(days=0)

    def fetch(self, start: date, end: date) -> None:
        raise NotImplementedError

    def parse(self) -> pd.DataFrame:
        raise NotImplementedError
