"""EIA daily spot prices: U.S. Gulf Coast jet fuel ($/gal), Brent ($/bbl).

Downloaded by `scripts/fetch_public_data.py eia` -> ~/flightpath_data/eia/<name>.xls
(needs `xlrd` to read .xls:  pip install xlrd). Available ~1 week after the price date
(conservative; EIA posts spot prices with a short lag).
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from .base import DataSource

EIA_LAG_DAYS = 7
# EIA publishes jet fuel spot prices for the Gulf Coast only; regional ULSD (diesel) is the closest public
# proxy for NY Harbor / LA jet fuel (the two products trade nearly in lockstep).
FILES = {"jet_gulf_coast": "eia.jet_gulf_coast", "brent": "eia.brent", "ulsd_ny_harbor": "eia.ulsd_ny_harbor",
         "ulsd_gulf_coast": "eia.ulsd_gulf_coast", "ulsd_los_angeles": "eia.ulsd_los_angeles"}


def read_eia_xls(path) -> pd.DataFrame:
    """EIA history workbook: sheet 'Data 1', 2 header rows, then (date, value)."""
    df = pd.read_excel(path, sheet_name="Data 1", skiprows=2)
    df = df.iloc[:, :2]
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna()


def load_eia(data_dir=None) -> dict[str, pd.DataFrame]:
    from ..paths import DATA
    folder = (data_dir or DATA) / "eia"
    return {name: read_eia_xls(folder / f"{name}.xls") for name in FILES if (folder / f"{name}.xls").exists()}


def eia_observations(series: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for name, df in series.items():
        frames.append(pd.DataFrame({
            "series": FILES[name], "period": df["date"],
            "available_at": df["date"] + pd.Timedelta(days=EIA_LAG_DAYS),
            "value": df["value"], "source": "EIA"}))
    return pd.concat(frames, ignore_index=True)


class EIASource(DataSource):
    name = "eia"
    default_lag = timedelta(days=EIA_LAG_DAYS)

    def fetch(self, start: date, end: date) -> None:
        raise NotImplementedError("Run scripts/fetch_public_data.py eia on the laptop.")

    def parse(self) -> pd.DataFrame:
        return eia_observations(load_eia())
