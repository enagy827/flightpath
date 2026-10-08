"""Delta job postings - DEMOTED TO P2: BTS air carrier employment (bts.py) covers the
capacity-intent signal with full history and no scraping. Keep only if time allows.

Why: pilot / flight attendant / mechanic hiring pace signals capacity plans 2-4
quarters out; hiring freezes signal caution before management says so.

TODO: collect from Delta careers site (check ToS); check history (web archives?).
Raw text -> extraction layer (LLM tags role, location/hub, seniority) -> counts.
Series: jobs.dl.postings.<role_family>, jobs.dl.postings.<hub>
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from .base import DataSource


class JobPostingsSource(DataSource):
    name = "jobs"
    default_lag = timedelta(days=0)

    def fetch(self, start: date, end: date) -> None:
        raise NotImplementedError

    def parse(self) -> pd.DataFrame:
        raise NotImplementedError
