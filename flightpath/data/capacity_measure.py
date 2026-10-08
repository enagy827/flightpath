"""Measuring Delta's flying before BTS publishes it (notebook 01).

Two flight-level sources, both turned into capacity (ASM = seats x miles) with BTS seats per aircraft type:
  A. BTS On-Time Performance: every Delta mainline DOMESTIC flight with its tail number, public ~52 days after the
     month (notebook 00, s.10). Tail number -> aircraft type via the OpenSky aircraft database.
  B. OpenSky: every tracked Delta mainline flight (callsign DAL...), worldwide, available the next day.
     2019 - present from OpenSky's historical database (scripts/fetch_opensky_trino.py; opensky.load_trino_flights).
"""
from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

import numpy as np
import pandas as pd

from .fleet import attach_seats


def great_circle_mi(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * asin(sqrt(a))


# ----------------------------------------------------------------------------------------- A. On-Time (domestic)
def ontime_flights(ot: pd.DataFrame, aircraft_db: pd.DataFrame) -> pd.DataFrame:
    """Delta mainline flights actually flown (not cancelled), with ICAO type from the tail number."""
    reg = aircraft_db.dropna(subset=["registration"]).drop_duplicates("registration").set_index("registration")["typecode"]
    f = ot[(ot["Reporting_Airline"] == "DL") & (ot["Cancelled"] == 0)].copy()
    f["typecode"] = f["Tail_Number"].map(reg)
    f["period"] = pd.to_datetime(f["FlightDate"]).dt.to_period("M").dt.start_time
    return f


def ontime_monthly_capacity(flights: pd.DataFrame, t100_domestic: pd.DataFrame) -> pd.DataFrame:
    """Monthly domestic capacity estimate: sum over flights of (BTS seats for its type) x (BTS route distance)."""
    f = attach_seats(flights, t100_domestic)
    f["asm"] = f["seats"] * f["Distance"]
    m = f.groupby("period").agg(flights=("asm", "size"), asm=("asm", "sum"),
                                typed_share=("seats_source", lambda s: (s == "type").mean()))
    return m


# ----------------------------------------------------------------------------------------- B. OpenSky (all regions)
def opensky_from_api(df: pd.DataFrame, aircraft_db: pd.DataFrame) -> pd.DataFrame:
    """Harmonize REST API rows (collect_opensky_flights.py) to the Zenodo flight-list columns."""
    tc = aircraft_db.dropna(subset=["icao24"]).drop_duplicates("icao24").set_index("icao24")["typecode"]
    out = pd.DataFrame({
        "callsign": df["callsign"].str.strip(), "typecode": df["icao24"].str.lower().map(tc),
        "origin": df["estDepartureAirport"], "destination": df["estArrivalAirport"],
        "firstseen": pd.to_datetime(df["firstSeen"], unit="s", utc=True),
        "lastseen": pd.to_datetime(df["lastSeen"], unit="s", utc=True)})
    out["day"] = out["lastseen"].dt.floor("D")
    return out


def opensky_capacity(flights: pd.DataFrame, t100_routes: pd.DataFrame, by: str = "period") -> pd.DataFrame:
    """Capacity measures from tracked flights, aggregated by month ('period') or by 'day':
    flights, seat-miles where both airports are known (scaled up to all flights), and seat-hours (tracked time)."""
    import airportsdata
    ap = airportsdata.load("ICAO")
    f = flights.copy()
    f["day"] = pd.to_datetime(f["day"], utc=True)
    f["period"] = f["day"].dt.tz_localize(None).dt.to_period("M").dt.start_time
    pairs = f[["origin", "destination"]].drop_duplicates()

    def _mi(o, d):
        a, b = ap.get(o), ap.get(d)
        if a is None or b is None or o == d:
            return np.nan
        return great_circle_mi(a["lat"], a["lon"], b["lat"], b["lon"])
    pairs["miles"] = [_mi(o, d) for o, d in zip(pairs["origin"], pairs["destination"])]
    f = f.merge(pairs, on=["origin", "destination"], how="left")
    f = attach_seats(f, t100_routes)
    f["hours"] = ((pd.to_datetime(f["lastseen"], utc=True) - pd.to_datetime(f["firstseen"], utc=True))
                  .dt.total_seconds() / 3600).clip(0, 20)
    f["asm_known"] = f["seats"] * f["miles"]
    f["seat_hours"] = f["seats"] * f["hours"]
    key = "period" if by == "period" else f["day"].dt.tz_localize(None).rename("day")
    g = f.groupby(key).agg(flights=("seats", "size"), miles_known_share=("miles", lambda x: x.notna().mean()),
                           asm_known=("asm_known", "sum"), seat_hours=("seat_hours", "sum"),
                           typed_share=("seats_source", lambda s: (s == "type").mean()))
    g["asm_est"] = g["asm_known"] / g["miles_known_share"]
    return g


def lagged_ratio_error(measure: pd.Series, truth: pd.Series, lag_months: int = 3, window: int = 3) -> pd.DataFrame:
    """Calibrate a measure to the truth with the average ratio of `window` months that were public `lag_months`
    earlier, then report the % error each month - exactly what is possible in real time."""
    d = pd.DataFrame({"measure": measure, "truth": truth})
    d = d[d["measure"].notna()]                 # keep months BTS hasn't published yet: they get an estimate, no error
    d["ratio"] = d["measure"] / d["truth"]
    d["ratio_used"] = d["ratio"].shift(lag_months).rolling(window).mean()
    d["estimate"] = d["measure"] / d["ratio_used"]
    d["error_pct"] = (d["estimate"] / d["truth"] - 1) * 100
    return d


def same_weekday_baseline(daily: pd.Series, weeks: int = 3) -> pd.Series:
    """For each day, the median of the same weekday in the previous `weeks` weeks (past only, so it can be computed
    on the day itself). Used to spot days when OpenSky's own data collection failed."""
    lagged = pd.concat([daily.shift(7 * k, freq="D") for k in range(1, weeks + 1)], axis=1).reindex(daily.index)
    return lagged.median(axis=1, skipna=True)


def fill_outage_days(daily: pd.DataFrame, flag_col: str = "flights", threshold: float = 0.5,
                     weeks: int = 3) -> tuple[pd.DataFrame, pd.Series]:
    """OpenSky occasionally loses whole days (e.g. 3-5 Dec 2023: no flights recorded). A day is flagged when its
    `flag_col` count is below `threshold` x the same-weekday baseline; every column of a flagged day is replaced by
    its own same-weekday baseline. The threshold is far below real disruptions (the July 2024 IT outage cut Delta's
    tracked flights to ~60% of normal, and is kept). Returns (filled daily frame, flags)."""
    d = daily.asfreq("D")
    flags = (d[flag_col].fillna(0) < threshold * same_weekday_baseline(d[flag_col].fillna(0), weeks))
    out = d.copy()
    for c in d.columns:
        base = same_weekday_baseline(d[c].where(~flags), weeks)
        out[c] = d[c].where(~flags, base)
    return out, flags


# ------------------------------------------------------------- OpenSky outage detector (uses rival airlines)
RIVALS = ("AAL", "ASA", "JBU", "SWA", "UAL")
FULL_OUTAGE = 0.5        # rivals' median below 50% of usual: OpenSky lost (almost) the whole day
PARTIAL_OUTAGE = 0.9     # rivals' median below 90% of usual ...
SAME_DROP = 0.2          # ... and every rival down by about the same amount (highest - lowest ratio below 0.2)


def holiday_window(days: pd.DatetimeIndex) -> pd.Series:
    """Days around Thanksgiving (Wed - Mon), Christmas / New Year (24 Dec - 3 Jan) and 4 July (3 - 6 Jul), when
    every airline really flies a thinner schedule. A uniform drop on these days is not treated as a partial outage."""
    d = pd.DatetimeIndex(days)
    nov1 = pd.to_datetime(dict(year=d.year, month=11, day=1))
    thanks = nov1 + pd.to_timedelta((3 - nov1.dt.weekday) % 7 + 21, unit="D")      # 4th Thursday of November
    off = (d - pd.DatetimeIndex(thanks)).days
    md_ = d.month * 100 + d.day
    return pd.Series(((off >= -1) & (off <= 4)) | (md_ >= 1224) | (md_ <= 103) | ((md_ >= 703) & (md_ <= 706)), index=d)


def yoy_normalized(daily: pd.Series) -> pd.Series:
    """Each day vs the same weekday 52 weeks earlier, divided by the median of that ratio over the previous 28 days
    (past days only). 1 = a normal day for the season; seasonality and growth cancel out."""
    x = (daily / daily.shift(364)).replace([np.inf, -np.inf], np.nan)
    return x / x.shift(1).rolling(28, min_periods=14).median()


def opensky_outages(counts: pd.DataFrame) -> pd.DataFrame:
    """OpenSky outage detector. `counts`: tracked flights per day, one column per airline (the RIVALS; other
    columns are ignored). OpenSky losing data hits every airline by the same proportion; a storm or an IT failure
    hits some airlines much more than others; a holiday hits all of them but is known in advance. Rule, per day:
      full outage    - the rivals' median is below FULL_OUTAGE x usual (usual = same weekday over the previous 3 weeks);
      partial outage - not full, not a holiday-window day, the rivals' median is below PARTIAL_OUTAGE x usual AND every
                       rival is within SAME_DROP of the others - checked against both the same-weekday baseline
                       and, when a year of history exists, the year-over-year comparison (which removes seasonal
                       swings like mid-January). `coverage` = the share of flights OpenSky still recorded.
    Uses past days only, so it can run on the day itself. Returns a frame: full, partial, coverage."""
    c = counts[[r for r in RIVALS if r in counts]].asfreq("D").fillna(0)
    w = pd.DataFrame({a: c[a] / same_weekday_baseline(c[a]) for a in c})
    n = pd.DataFrame({a: yoy_normalized(c[a]) for a in c})
    medw, sprw = w.median(axis=1), w.max(axis=1) - w.min(axis=1)
    medn, sprn = n.median(axis=1), n.max(axis=1) - n.min(axis=1)
    has_yoy = n.notna().all(axis=1)
    full = medw < FULL_OUTAGE
    partial = ((medw < PARTIAL_OUTAGE) & (sprw < SAME_DROP) & (~has_yoy | ((medn < PARTIAL_OUTAGE) & (sprn < SAME_DROP)))
               & ~full & ~holiday_window(c.index))
    coverage = pd.concat([medw, medn.where(has_yoy)], axis=1).max(axis=1).where(partial, 1.0)
    return pd.DataFrame({"full": full, "partial": partial, "coverage": coverage})


def correct_for_outages(daily: pd.DataFrame, out: pd.DataFrame, weeks: int = 3) -> pd.DataFrame:
    """Partial-outage days: divided by the share OpenSky recorded. Full-outage days: missing, replaced by the
    same-weekday baseline (as fill_with_flags). Days outside the detector's range are left as they are."""
    d = daily.copy()
    cov = out.coverage.reindex(d.index).fillna(1.0)
    d = d.div(cov, axis=0)
    full = out.full.reindex(d.index).fillna(False).astype(bool)
    return fill_with_flags(d, full, weeks)


def same_month_last_year_estimate(measure: pd.Series, truth: pd.Series) -> pd.Series:
    """Alternative real-time correction: truth a year earlier x the measure's change over that year."""
    m = measure.asfreq("MS")
    t = truth.reindex(m.index)
    return t.shift(12) * m / m.shift(12)


def published_months(truth: pd.Series, as_of, lag_days: int | None = None) -> pd.DatetimeIndex:
    """Months of a BTS monthly series that were public on `as_of` (month end + the T-100 publication delay)."""
    from .bts import T100_LAG_DAYS
    lag = T100_LAG_DAYS if lag_days is None else lag_days
    idx = truth.dropna().index
    return idx[(idx + pd.offsets.MonthEnd(0) + pd.Timedelta(days=lag)) <= pd.Timestamp(as_of)]


def quarter_estimate_as_of(measure: pd.Series, truth: pd.Series, quarter, as_of, window: int = 3) -> tuple[float, pd.DatetimeIndex]:
    """BTS-equivalent estimate of a quarter, made on `as_of` with everything public that day: the quarter's measure
    (all three months) divided by the average ratio measure / BTS of the `window` most recent months BTS had
    published by `as_of`. Returns (estimate, months used for the ratio). Nothing published after `as_of` is used."""
    q = pd.Period(quarter, "Q")
    months = pd.date_range(q.start_time, periods=3, freq="MS")
    pub = published_months(truth, as_of)
    pub = pub[pub < months[0]][-window:]           # never a month of the quarter itself
    ratio = (measure.reindex(pub) / truth.reindex(pub)).mean()
    return float(measure.reindex(months).sum() / ratio), pub


# ------------------------------------------------------------- estimates tied to a date ("as of"), no future data
def fill_with_flags(daily: pd.DataFrame, flags: pd.Series, weeks: int = 3) -> pd.DataFrame:
    """Replace flagged days (see fill_outage_days) in every column with that column's same-weekday baseline."""
    d = daily.reindex(flags.index).fillna(0)
    return pd.DataFrame({c: d[c].where(~flags, same_weekday_baseline(d[c].where(~flags), weeks)) for c in d.columns})


def seat_table_as_of(t100_routes: pd.DataFrame, as_of, window: int = 12) -> tuple[dict, float]:
    """Seats per departure for each BTS aircraft code, from the `window` most recent T-100 months PUBLISHED by
    `as_of` (month end + T100_LAG_DAYS). Returns ({bts_code: seats}, fleet average)."""
    from .bts import T100_LAG_DAYS
    months = pd.DatetimeIndex(sorted(t100_routes["period"].unique()))
    pub = months[(months + pd.offsets.MonthEnd(0) + pd.Timedelta(days=T100_LAG_DAYS)) <= pd.Timestamp(as_of)][-window:]
    g = t100_routes[t100_routes["period"].isin(pub)].groupby("aircraft_type")[["seats", "departures_performed"]].sum()
    g = g[g["departures_performed"] > 0]
    return (g["seats"] / g["departures_performed"]).to_dict(), float(g["seats"].sum() / g["departures_performed"].sum())


def by_type(values: pd.Series, period: pd.Series, typecode: pd.Series) -> pd.DataFrame:
    """Sum of `values` (hours flown, or miles) per period and BTS aircraft code; unknown types in column -1."""
    from .fleet import ICAO_TO_BTS
    code = typecode.map(ICAO_TO_BTS).fillna(-1).astype(int)
    return values.groupby([period, code]).sum().unstack(fill_value=0)


def apply_seats(per_type: pd.DataFrame, seats: dict, avg: float) -> pd.Series:
    """Seat-weighted total per period: each column (BTS code) x its seats; unknown or unseen types get the average."""
    w = pd.Series({c: seats.get(c, avg) if c != -1 else avg for c in per_type.columns})
    return (per_type * w).sum(axis=1)
