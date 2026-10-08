"""Bureau of Transportation Statistics - carrier-level operating data.

Datasets (TranStats):
  - T-100 Domestic/International Segment: departures, seats, passengers, RPM/ASM inputs
    (monthly, ~2-3 month lag) -> CALIBRATION, not intra-quarter signal
  - Reporting Carrier On-Time Performance: delays, cancellations, diversions
    (monthly, ~2 month lag)
  - Form 41 financials (P-12 fuel, etc.) - optional cross-check

Filter on carrier code from config (DL). Note: regional partners (Endeavor, SkyWest
operating as Delta Connection) report under their own codes - decide whether to include.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from .base import DataSource


# ----------------------------------------------------------------------------- T-100
# Publication delay (days after month end until the month is downloadable), from BTS sources:
#  - TranStats database release log (https://www.transtats.bts.gov/releaseinfo.asp, read 2026-10-01):
#      T-100: "9/14/2026: 6/2026 added 4/2026 updated"  -> June 2026 published after 76 days
#  - BTS airline traffic press releases (headline numbers, published a few days before the database update):
#      Oct-2025 data 2026-01-09 (70 days)  https://www.bts.gov/newsroom/october-2025-us-airline-traffic-data-16-same-month-last-year
#      May-2026 data 2026-08-13 (74 days)  https://www.bts.gov/newsroom/may-2026-us-airline-traffic-data-down-07-same-month-last-year
#      June-2026 data scheduled 2026-09-11 (73 days), same page
# Set to 80 days: longer than every observed delay, so the point-in-time store never uses a month early.
# Known limitation: BTS revises recent months (April 2026 was "updated" in the Sept release); we hold only the
# latest version of each month, so history uses revised values. Revisions are typically small.
T100_LAG_DAYS = 80


def _region(iata: str, airports: dict) -> str:
    """Delta-style reporting region from an airport code (via its time zone)."""
    a = airports.get(iata)
    if a is None:
        return "unknown"
    if a["country"] in ("US", "CA"):          # Canada counted as domestic: tested in notebook 00 (immaterial, ~0.4%)
        return "domestic"
    tz = a.get("tz") or ""
    if tz.startswith(("Europe/", "Africa/", "Atlantic/", "Indian/")) or a["country"] in ("IL", "AE", "IN", "SA", "QA"):
        return "atlantic"
    if tz.startswith(("Asia/", "Australia/", "Pacific/")):
        return "pacific"
    if tz.startswith("America/"):
        return "latin"
    return "unknown"


# Delta mainline = DL, plus Northwest (NW) in 2009: Northwest merged into Delta on 29 Oct 2008 but kept filing its
# own T-100 reports through Dec 2009 (verified in notebook 00: NW appears in every month of 2009 and never after).
# Delta's reported numbers include Northwest for all of 2009, so both codes count as Delta mainline.
CARRIER_FAMILY = {"DL": ("DL", "NW")}
CARRIER_FAMILY_OF = {code: fam for fam, codes in CARRIER_FAMILY.items() for code in codes}


def load_t100(data_dir=None, carrier: str | tuple = "DL", passenger_class: tuple = ("F",)) -> pd.DataFrame:
    """Route-month rows for one or more carriers from the T100_<year>.zip files.

    carrier: "DL" = Delta mainline; "9E" = Endeavor Air (wholly owned by Delta, flies only as Delta Connection).
    SkyWest (OO) and Republic (YX) also fly Delta Connection but for other airlines too, and T-100 does not say
    which flights were for Delta, so they can't be included.

    Adds: asm = seats x distance, rpm = passengers x distance, block_hours, air_hours,
    region of the flight (domestic / atlantic / latin / pacific) from the non-US endpoint.
    CLASS 'F' = scheduled passenger service. Charters (CLASS 'L') are excluded: ~0.2% of Delta mainline ASM.

    Region uses the non-U.S. end of each route; Canada counted as domestic. Validated in notebook 00: BTS regional
    capacity growth matches Delta's reported regional capacity growth within ~1 pt (Delta reports whole percents);
    Canada is 0.4% of ASM and assigning it elsewhere changes the match by <0.1 pt, so the choice is immaterial.
    """
    import zipfile
    from ..paths import DATA
    import airportsdata

    folder = (data_dir or DATA) / "bts" / "t100"
    frames = []
    for z in sorted(folder.glob("T100_*.zip")):
        with zipfile.ZipFile(z) as zf:
            name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
            df = pd.read_csv(zf.open(name), low_memory=False)
        carriers = (carrier,) if isinstance(carrier, str) else tuple(carrier)
        codes = [c for k in carriers for c in CARRIER_FAMILY.get(k, (k,))]
        df = df[(df["UNIQUE_CARRIER"].isin(codes)) & (df["CLASS"].isin(passenger_class))].copy()
        df["REPORTING_CODE"] = df["UNIQUE_CARRIER"]
        df["UNIQUE_CARRIER"] = df["UNIQUE_CARRIER"].map(CARRIER_FAMILY_OF).fillna(df["UNIQUE_CARRIER"])
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.lower() for c in df.columns]
    df["period"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    df["asm"] = df["seats"] * df["distance"]
    df["rpm"] = df["passengers"] * df["distance"]
    df["block_hours"] = df["ramp_to_ramp"] / 60.0
    df["air_hours"] = df["air_time"] / 60.0
    airports = airportsdata.load("IATA")
    reg_o = df["origin"].map(lambda c: _region(c, airports))
    reg_d = df["dest"].map(lambda c: _region(c, airports))
    df["region"] = reg_d.where(reg_d != "domestic", reg_o)
    return df


def load_t100_industry(data_dir=None, passenger_class: tuple = ("F",)) -> pd.DataFrame:
    """Industry-wide monthly passengers from the same T100_<year>.zip files (ALL carriers, US and foreign, scheduled
    passenger service): passengers on flight segments departing a US airport, split into domestic (both ends in the
    US) and international departures. Used as a long-history stand-in for TSA checkpoint counts (notebook 03).
    Note: a segment count - a connecting passenger counts once per flight, while TSA counts them once per screening."""
    import zipfile
    from ..paths import DATA
    import airportsdata
    ap = airportsdata.load("IATA")
    country = {k: v["country"] for k, v in ap.items()}
    folder = (data_dir or DATA) / "bts" / "t100"
    out = []
    for z in sorted(folder.glob("T100_*.zip")):
        with zipfile.ZipFile(z) as zf:
            name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
            df = pd.read_csv(zf.open(name), usecols=["PASSENGERS", "SEATS", "ORIGIN", "DEST", "YEAR", "MONTH", "CLASS"],
                             low_memory=False)
        df = df[df["CLASS"].isin(passenger_class)]
        o_us = df["ORIGIN"].map(country).isin(["US", "PR", "VI", "GU"])
        d_us = df["DEST"].map(country).isin(["US", "PR", "VI", "GU"])
        df = df[o_us].assign(domestic=d_us[o_us])
        g = df.groupby(["YEAR", "MONTH", "domestic"])["PASSENGERS"].sum().unstack(fill_value=0)
        out.append(g)
    g = pd.concat(out)
    g.index = pd.to_datetime([f"{y}-{m:02d}-01" for y, m in g.index])
    g = g.rename(columns={True: "domestic_passengers", False: "international_departing_passengers"})
    g["passengers_from_us"] = g.sum(axis=1)
    g.index.name = "period"
    return g.sort_index()


METRICS = ["departures_scheduled", "departures_performed", "seats", "passengers",
           "asm", "rpm", "block_hours", "air_hours"]


def t100_monthly(routes: pd.DataFrame, by_region: bool = False) -> pd.DataFrame:
    """Monthly totals (+ load factor, completion factor, stage length, seats per departure)."""
    keys = ["period", "region"] if by_region else ["period"]
    m = routes.groupby(keys)[METRICS].sum().reset_index()
    m["load_factor"] = m["rpm"] / m["asm"]
    m["completion_factor"] = m["departures_performed"] / m["departures_scheduled"]
    m["avg_stage_length"] = m["asm"] / m["seats"]
    m["seats_per_departure"] = m["seats"] / m["departures_performed"]
    return m


def t100_observations(monthly: pd.DataFrame, carrier: str = "dl") -> pd.DataFrame:
    """Long-format observations for the PointInTimeStore (series bts.t100.<carrier>.<metric>)."""
    cols = METRICS + ["load_factor", "completion_factor"]
    long = monthly.melt(id_vars="period", value_vars=cols, var_name="metric", value_name="value")
    long["series"] = f"bts.t100.{carrier}." + long["metric"]
    month_end = long["period"] + pd.offsets.MonthEnd(0)
    long["available_at"] = month_end + pd.Timedelta(days=T100_LAG_DAYS)
    long["source"] = "BTS"
    return long[["series", "period", "available_at", "value", "source"]]


def t100_all_observations(data_dir=None) -> pd.DataFrame:
    """Store series for Delta mainline ('dl'), Endeavor ('9e') and both combined ('dl9e')."""
    frames = []
    for code, carriers in (("dl", "DL"), ("9e", "9E"), ("dl9e", ("DL", "9E"))):
        frames.append(t100_observations(t100_monthly(load_t100(data_dir, carrier=carriers)), carrier=code))
    return pd.concat(frames, ignore_index=True)


class BTST100Source(DataSource):
    """Wrapper so T-100 fits the DataSource interface (files are downloaded manually)."""
    name = "bts_t100"
    default_lag = timedelta(days=T100_LAG_DAYS)

    def fetch(self, start: date, end: date) -> None:
        raise NotImplementedError("Download T100_<year>.zip manually from TranStats (see docs).")

    def parse(self) -> pd.DataFrame:
        return t100_all_observations()


class BTSOnTimeSource(DataSource):
    name = "bts_ontime"
    default_lag = timedelta(days=60)  # TODO: verify

    def fetch(self, start: date, end: date) -> None:
        raise NotImplementedError

    def parse(self) -> pd.DataFrame:
        # TODO: carrier-month and hub-month cancellation %, delay minutes, diversions
        raise NotImplementedError


class BTSFuelSource(DataSource):
    """Form 41 Schedule P-12(a): Airline Fuel Cost and Consumption - P0.

    Monthly gallons consumed and total fuel cost by carrier (domestic / international).
    The best calibration target for the fuel module: 12 points per year instead of 4
    from the 10-Q. Listed in Longaeva's public dataset PDF (BTS TranStats -> Form 41).
    """
    name = "bts_fuel"
    default_lag = timedelta(days=90)  # TODO: verify

    def fetch(self, start: date, end: date) -> None:
        raise NotImplementedError

    def parse(self) -> pd.DataFrame:
        # TODO: series bts.fuel.dl.gallons, bts.fuel.dl.cost, bts.fuel.dl.cost_per_gallon
        raise NotImplementedError


class BTSDB1BSource(DataSource):
    """DB1B Origin & Destination Survey: 10% sample of airline tickets - P1.

    Quarterly fares by route and carrier (~3-4 month lag). Gives the fare HISTORY
    that forward-fare scraping can't: calibrate the pricing module (yield response to
    demand/capacity) by route. Large files - filter to Delta (ticketing / operating
    carrier DL) and the key route panel early.
    """
    name = "bts_db1b"
    default_lag = timedelta(days=110)  # TODO: verify

    def fetch(self, start: date, end: date) -> None:
        raise NotImplementedError

    def parse(self) -> pd.DataFrame:
        # TODO: series db1b.dl.<route>.avg_fare, db1b.dl.<route>.yield (fare / miles)
        raise NotImplementedError


class BTSEmploymentSource(DataSource):
    """Air Carrier Employment: monthly full-time-equivalent headcount by carrier - P1.

    Replaces job-posting scraping as the capacity-intent signal: full history, public,
    no ToS risk. Pilots and flight attendants lead capacity by 2-4 quarters.
    """
    name = "bts_employment"
    default_lag = timedelta(days=60)  # TODO: verify

    def fetch(self, start: date, end: date) -> None:
        raise NotImplementedError

    def parse(self) -> pd.DataFrame:
        # TODO: series bts.employment.dl.fte (+ by category if available)
        raise NotImplementedError


# ----------------------------------------------------------------------------- Form 41 P-12(a) fuel
# TranStats release log (read 2026-10-01): Form 41 "9/15/2026: Q2/2026 added" -> June 2026 after 77 days. Use 85.
F41_LAG_DAYS = 85


def load_form41_fuel(data_dir=None, carrier: str = "DL") -> pd.DataFrame:
    """Monthly mainline fuel gallons and cost (scheduled + non-scheduled, domestic + international).
    File: ~/flightpath_data/bts/form41_fuel/P12a_fuel_all_years.zip (manual TranStats download)."""
    import zipfile
    from ..paths import DATA
    z = zipfile.ZipFile((data_dir or DATA) / "bts" / "form41_fuel" / "P12a_fuel_all_years.zip")
    name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
    df = pd.read_csv(z.open(name), low_memory=False)
    df = df[df["UNIQUE_CARRIER"] == carrier].copy()
    df["period"] = pd.to_datetime(dict(year=df["YEAR"], month=df["MONTH"], day=1))
    out = df.groupby("period")[["TOTAL_GALLONS", "TOTAL_COST", "TDOMT_GALLONS", "TINT_GALLONS"]].sum()
    out.columns = ["gallons", "cost", "gallons_domestic", "gallons_international"]
    out["price_per_gal"] = out["cost"] / out["gallons"]
    return out.reset_index()


def form41_observations(f41: pd.DataFrame, carrier: str = "dl") -> pd.DataFrame:
    cols = ["gallons", "cost", "price_per_gal"]
    long = f41.melt(id_vars="period", value_vars=cols, var_name="metric", value_name="value")
    return pd.DataFrame({"series": f"bts.f41.{carrier}." + long["metric"], "period": long["period"],
                         "available_at": long["period"] + pd.offsets.MonthEnd(0) + pd.Timedelta(days=F41_LAG_DAYS),
                         "value": long["value"], "source": "BTS"})


# ----------------------------------------------------------------------------- On-Time Performance
# Publication delay, from the TranStats release log (https://www.transtats.bts.gov/releaseinfo.asp, read 2026-10-01):
#   Reporting Carrier On-Time Performance: "9/21/2026: 7/2026 added" -> July 2026 published after 52 days.
# Set to 60 days (margin above the observed delay; only one release observed so far).
ONTIME_LAG_DAYS = 60


def load_ontime(data_dir=None) -> pd.DataFrame:
    """Every Delta (DL) and Endeavor (9E) flight from BTS On-Time Performance, one row per scheduled flight,
    with Cancelled (0/1) and Diverted flags. Files from `scripts/fetch_public_data.py ontime`."""
    from ..paths import DATA
    folder = (data_dir or DATA) / "bts" / "ontime"
    files = sorted(folder.glob("ontime_DL_9E_*.parquet"))
    if not files:
        raise FileNotFoundError("run: python scripts/fetch_public_data.py ontime")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    return df


def ontime_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly scheduled flights, cancellations and cancellation rate, per carrier."""
    g = df.assign(period=df["FlightDate"].dt.to_period("M").dt.start_time).groupby(["period", "Reporting_Airline"])
    m = g.agg(scheduled=("Cancelled", "size"), cancelled=("Cancelled", "sum"), diverted=("Diverted", "sum")).reset_index()
    m["cancel_rate"] = m["cancelled"] / m["scheduled"]
    return m


def ontime_observations(m: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for carrier, d in m.groupby("Reporting_Airline"):
        for col in ("scheduled", "cancelled", "cancel_rate"):
            frames.append(pd.DataFrame({
                "series": f"bts.ontime.{carrier.lower()}.{col}", "period": d["period"],
                "available_at": d["period"] + pd.offsets.MonthEnd(0) + pd.Timedelta(days=ONTIME_LAG_DAYS),
                "value": d[col], "source": "BTS"}))
    return pd.concat(frames, ignore_index=True)
