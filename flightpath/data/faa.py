"""FAA operations data and aircraft registry (both on Longaeva's public dataset list).

FAA OPSNET / ASPM (P1): daily airport operations counts and delay statistics.
  Live disruption signal at Delta hubs - covers part of what OpenSky can't (delays).
  TODO: check access - some ASPM reports require registration.

FAA Aircraft Registry (P1): every U.S.-registered aircraft with owner and model.
  Filter registrant = Delta Air Lines to build the fleet table (type, seats via type,
  age). U.S. government data - no license issue; preferred over the OpenSky aircraft DB.
  Downloadable as a full database zip from the FAA site.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from .base import DataSource


class FAAOpsSource(DataSource):
    name = "faa_ops"
    default_lag = timedelta(days=1)

    def fetch(self, start: date, end: date) -> None:
        raise NotImplementedError

    def parse(self) -> pd.DataFrame:
        # TODO: series faa.<hub>.operations, faa.<hub>.pct_ontime, faa.<hub>.avg_delay
        raise NotImplementedError


def load_registry(path: str, registrant: str = "DELTA AIR LINES") -> pd.DataFrame:
    """Delta fleet table: n_number, mode_s_hex (icao24), aircraft model, year built.
    icao24 lets live OpenSky flights be matched to aircraft type. TODO."""
    raise NotImplementedError
