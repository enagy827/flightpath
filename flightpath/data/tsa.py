"""TSA checkpoint throughput - daily, industry-wide demand signal (~1 day lag).

Downloaded by `scripts/fetch_public_data.py tsa` -> ~/flightpath_data/tsa/tsa_throughput.csv
Not Delta-specific: the demand module converts it to Delta demand via Delta's traffic share.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from .base import DataSource


def load_tsa(data_dir=None) -> pd.DataFrame:
    from ..paths import DATA
    df = pd.read_csv((data_dir or DATA) / "tsa" / "tsa_throughput.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def tsa_yoy(df: pd.DataFrame) -> pd.DataFrame:
    """Same-weekday year-over-year growth (compare to 364 days earlier) and 7-day averages."""
    s = df.set_index("date")["throughput"].asfreq("D")
    out = pd.DataFrame({"throughput": s, "ma7": s.rolling(7).mean()})
    out["ma7_ly"] = out["ma7"].shift(364)
    out["yoy_ma7"] = out["ma7"] / out["ma7_ly"] - 1
    return out.reset_index()


def tsa_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Daily observations; available the next day."""
    return pd.DataFrame({
        "series": "tsa.throughput",
        "period": df["date"],
        "available_at": df["date"] + pd.Timedelta(days=1),
        "value": df["throughput"],
        "source": "TSA",
    })


class TSASource(DataSource):
    name = "tsa"
    default_lag = timedelta(days=1)

    def fetch(self, start: date, end: date) -> None:
        raise NotImplementedError("Run scripts/fetch_public_data.py tsa on the laptop.")

    def parse(self) -> pd.DataFrame:
        return tsa_observations(load_tsa())
