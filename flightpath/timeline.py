"""Point-in-time data store - the main guard against look-ahead bias.

Every observation carries two dates:
    period       - the time the value describes (e.g. July 2025 traffic)
    available_at - when an investor could first have seen it (publication date)

All model code reads data through `PointInTimeStore.as_of(date)`, which only returns rows
with available_at <= date. Lagged sources (BTS ~2-3 months) are therefore automatically
calibration-only mid-quarter, and never leak into a historical nowcast.

Revised series: store every vintage (vintage = its own available_at); `as_of` returns the
latest vintage available at the query date.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

COLUMNS = ["series", "period", "available_at", "value", "source"]


@dataclass(frozen=True)
class Observation:
    series: str            # e.g. "bts.t100.dl.asm"
    period: date           # what the value describes
    available_at: date     # when it became public
    value: float
    source: str            # "TSA", "BTS", "EIA", "SEC", "OpenSky", ...


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    out = df[COLUMNS].copy()
    out["series"] = out["series"].astype(str)
    out["source"] = out["source"].astype(str)
    out["period"] = pd.to_datetime(out["period"]).dt.normalize()
    out["available_at"] = pd.to_datetime(out["available_at"]).dt.normalize()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    if (out["available_at"] < out["period"]).any():
        bad = out[out["available_at"] < out["period"]].head()
        raise ValueError(f"available_at before period (impossible):\n{bad}")
    return out.dropna(subset=["value"])


class PointInTimeStore:
    """Long-format store of observations, persisted to a parquet file."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        if self.path is not None and self.path.exists():
            self.df = pd.read_parquet(self.path)
        else:
            self.df = pd.DataFrame(columns=COLUMNS)

    # ------------------------------------------------------------------ writing
    def add(self, df: pd.DataFrame) -> int:
        """Append observations. Re-adding the same (series, period, available_at) replaces it.
        Returns the number of rows added."""
        new = _normalize(df)
        combined = pd.concat([self.df, new], ignore_index=True) if len(self.df) else new
        combined = combined.drop_duplicates(["series", "period", "available_at"], keep="last")
        self.df = combined.sort_values(["series", "period", "available_at"]).reset_index(drop=True)
        return len(new)

    def replace_series(self, prefix: str, df: pd.DataFrame) -> int:
        """Drop every series starting with `prefix`, then add df (for full reloads of a source)."""
        if len(self.df):
            self.df = self.df[~self.df["series"].str.startswith(prefix)]
        return self.add(df)

    def save(self) -> None:
        if self.path is None:
            raise ValueError("store has no path")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_parquet(self.path, index=False)

    # ------------------------------------------------------------------ reading
    def as_of(self, as_of: date | str | pd.Timestamp,
              series: str | list[str] | None = None) -> pd.DataFrame:
        """Observations visible on `as_of`: available_at <= as_of, latest vintage per period."""
        t = pd.Timestamp(as_of).normalize()
        d = self.df
        if series is not None:
            names = [series] if isinstance(series, str) else list(series)
            d = d[d["series"].isin(names)]
        d = d[d["available_at"] <= t]
        return (d.sort_values("available_at")
                 .drop_duplicates(["series", "period"], keep="last")
                 .sort_values(["series", "period"])
                 .reset_index(drop=True))

    def series_as_of(self, series: str, as_of) -> pd.Series:
        """One series as a time-indexed pandas Series (period -> value), point-in-time."""
        d = self.as_of(as_of, series)
        return d.set_index("period")["value"].rename(series)

    def latest_period(self, series: str, as_of) -> pd.Timestamp | None:
        """Most recent period visible for a series at as_of (shows data lag in the app)."""
        d = self.as_of(as_of, series)
        return None if d.empty else d["period"].max()

    def list_series(self, prefix: str = "") -> list[str]:
        return sorted(s for s in self.df["series"].unique() if s.startswith(prefix))
