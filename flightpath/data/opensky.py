"""OpenSky Network ADS-B flight tracking - EDGE SOURCE (P0).

Why: closes the biggest data gap. BTS publishes Delta flights 2-3 months late;
ADS-B shows flights actually flown DAILY: departures, block hours, aircraft type.

NOT available from OpenSky: schedules, cancellations, delays, passenger counts.
Cancellations must come from BTS on-time (backtest) or be inferred live as a drop in
flights flown vs. a same-weekday baseline (see modules/disruption.py).

Sources:
  - Backtest 2019-2021: Zenodo flight list v21.12 (CC BY 4.0), local files in
    ~/flightpath_data/opensky/flightlist/ (callsign prefix "DAL"; aircraft ids anonymized)
  - v23.00 (2019-2022, real icao24) is NON-COMMERCIAL only
  - Live: REST API (use approved by Longaeva, 2026-10-02; collector: scripts/collect_opensky_flights.py), OAuth2 client credentials (token expires after 30 min; refresh on 401).
    /flights/* only has previous-day-or-earlier data; daily credit quota applies.
  - Aircraft metadata: aircraft-database-complete-2025-08.csv (no formal license, cite source)

Uses:
  - capacity module: flown ASM = sum(seats(aircraft type) x great-circle distance)
  - fuel module: burn = f(aircraft type, block time)  -> better than gallons/ASM
  - disruption module: drop in flights flown vs. a same-weekday baseline

Access: free for research (register; historical data via Trino/Impala interface).
TODO: verify SEATS table against Delta's fleet page; live API loader (see scripts/test_opensky.py).
Validate: monthly OpenSky-derived ASM vs BTS T-100 ASM (must track closely before trusting).
"""
from __future__ import annotations

from datetime import date, timedelta
from math import asin, cos, radians, sin, sqrt

import pandas as pd

from .base import DataSource

# DEPRECATED (kept only so old code runs): approximate seat counts. Notebook 01 and capacity_measure.py use BTS
# seats per aircraft type instead (flightpath/data/fleet.py).
# Absolute accuracy matters less than consistency: the BTS calibration ratio absorbs level errors.
SEATS = {
    "A319": 132, "A320": 157, "A321": 191, "A21N": 194, "BCS1": 109, "BCS3": 130,
    "B712": 110, "B738": 160, "B739": 180, "B752": 199, "B753": 234, "B763": 211, "B764": 238,
    "A332": 223, "A333": 282, "A339": 281, "A359": 306, "MD88": 149, "MD90": 160,
    "B77L": 288, "B772": 291, "B77W": 300,
}
DEFAULT_SEATS = 165


def _haversine_mi(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * asin(sqrt(a))


def load_delta_flights(data_dir=None) -> pd.DataFrame:
    """Delta flights extracted by scripts/extract_opensky_delta.py (2019-2021 Zenodo lists)."""
    from ..paths import DATA
    folder = (data_dir or DATA) / "opensky" / "derived"
    if not list(folder.glob("delta_flights_*.parquet")):
        raise FileNotFoundError("run scripts/extract_opensky_delta.py first")
    df = pd.concat([pd.read_parquet(p) for p in sorted(folder.glob("delta_flights_*.parquet"))],
                   ignore_index=True)
    for c in ("firstseen", "lastseen", "day"):
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    return df


def load_trino_flights(data_dir=None, aircraft_db: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every Delta mainline flight (callsign DAL + digits) in OpenSky's historical database, 2019 - present, from the
    weekly files written by scripts/fetch_opensky_trino.py. One row per tracked flight, in the same columns as the
    Zenodo flight lists: callsign, icao24, typecode (via the OpenSky aircraft database), origin, destination (ICAO
    airport codes, missing when OpenSky didn't see the take-off/landing), firstseen, lastseen (UTC) and day (UTC
    date of lastseen - OpenSky's partition day). Duplicate rows (same aircraft, same first-seen time) are dropped."""
    from ..paths import DATA
    from .fleet import load_aircraft_db
    files = sorted(((data_dir or DATA) / "opensky" / "trino").glob("dal_*.parquet"))
    if not files:
        raise FileNotFoundError("run: python scripts/fetch_opensky_trino.py plan")
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    raw = raw.drop_duplicates(["icao24", "firstseen"])
    db = aircraft_db if aircraft_db is not None else load_aircraft_db(data_dir)
    tc = db.dropna(subset=["icao24"]).drop_duplicates("icao24").set_index("icao24")["typecode"]
    icao = raw["icao24"].astype(str).str.lower()
    out = pd.DataFrame({
        "callsign": raw["callsign"].astype(str).str.strip(), "icao24": icao, "typecode": icao.map(tc),
        "origin": raw["estdepartureairport"], "destination": raw["estarrivalairport"],
        "firstseen": pd.to_datetime(raw["firstseen"], utc=True), "lastseen": pd.to_datetime(raw["lastseen"], utc=True),
        "day": pd.to_datetime(raw["day"], utc=True)})
    return out.reset_index(drop=True)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add seats (by type), great-circle miles (when both airports known), tracked hours, ASM estimate."""
    import airportsdata
    ap = airportsdata.load("ICAO")
    out = df.copy()
    out["seats"] = out["typecode"].map(SEATS).fillna(DEFAULT_SEATS)

    def miles(row):
        a, b = ap.get(row.origin), ap.get(row.destination)
        if a is None or b is None or row.origin == row.destination:
            return float("nan")
        return _haversine_mi(a["lat"], a["lon"], b["lat"], b["lon"])

    pairs = out[["origin", "destination"]].drop_duplicates()
    pairs["miles"] = pairs.apply(miles, axis=1)
    out = out.merge(pairs, on=["origin", "destination"], how="left")
    out["tracked_hours"] = ((out["lastseen"] - out["firstseen"]).dt.total_seconds() / 3600).clip(0, 20)
    out["seat_hours"] = out["seats"] * out["tracked_hours"]      # best ASM proxy (needs no route)
    out["asm_est"] = out["seats"] * out["miles"]
    out["has_route"] = out["miles"].notna()
    return out


def monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly OpenSky Delta stats. ASM is scaled up for flights with unknown routes
    (assumes they look like the known-route flights that month)."""
    d = df.assign(period=df["day"].dt.tz_localize(None).dt.to_period("M").dt.start_time)
    g = d.groupby("period")
    m = pd.DataFrame({
        "flights": g.size(),
        "flights_with_route": g["has_route"].sum(),
        "tracked_hours": g["tracked_hours"].sum(),
        "asm_known": g["asm_est"].sum(),
        "seat_hours": g["seat_hours"].sum(),
    })
    m["route_share"] = m["flights_with_route"] / m["flights"]
    m["asm_est"] = m["asm_known"] / m["route_share"]
    return m.reset_index()


def opensky_observations(m: pd.DataFrame, lag_days: int = 1) -> pd.DataFrame:
    """Month is observable (complete) the day after it ends."""
    frames = []
    for col in ("flights", "tracked_hours", "asm_est", "seat_hours"):
        frames.append(pd.DataFrame({
            "series": f"opensky.dl.{col}", "period": m["period"],
            "available_at": m["period"] + pd.offsets.MonthEnd(0) + pd.Timedelta(days=lag_days),
            "value": m[col], "source": "OpenSky"}))
    return pd.concat(frames, ignore_index=True)


class OpenSkySource(DataSource):
    name = "opensky"
    default_lag = timedelta(days=1)

    def fetch(self, start: date, end: date) -> None:
        # TODO: query flights by callsign prefix "DAL" (+ Delta Connection partners?)
        raise NotImplementedError

    def parse(self) -> pd.DataFrame:
        return opensky_observations(monthly(enrich(load_delta_flights())))
