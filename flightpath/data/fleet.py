"""Aircraft types and seats: link flight-level sources (OpenSky, On-Time tail numbers) to BTS seat counts.

- OpenSky and On-Time identify aircraft by ICAO type designator (e.g. "A321"), via the OpenSky aircraft database
  (icao24 / registration -> typecode). BTS T-100 identifies them by numeric codes (e.g. 699) and reports the SEATS
  actually flown on each code. Seats therefore come from BTS, never from a hand-made table.
- ICAO_TO_BTS was derived from the data (notebook 01, section 3): for every Delta route-month 2023-2025, On-Time
  flights by ICAO type were matched with T-100 departures by BTS code; each pair below is a MUTUAL best match
  (the BTS code's most frequent ICAO type and vice versa). Pre-2020 types (MD88, MD90, B737, B772) were matched the
  same way on 2019 OpenSky flights. Names are checked against BTS's own aircraft-type lookup table when available.
"""
from __future__ import annotations

import pandas as pd

ICAO_TO_BTS = {
    "B712": 608, "B737": 612, "B738": 614, "B739": 888, "B752": 622, "B753": 623, "B763": 626, "B764": 624,
    "B772": 627, "B77L": 627, "A319": 698, "A320": 694, "A321": 699, "A21N": 721, "A332": 696, "A333": 687,
    "A339": 339, "A359": 359, "BCS1": 723, "BCS3": 724, "MD88": 655, "MD90": 656,
}

PUBLICATION_LAG_MONTHS = 3   # T-100 for month m is public ~80 days after month end (notebook 00, section 10)


def load_aircraft_db(data_dir=None) -> pd.DataFrame:
    """OpenSky aircraft database snapshot: icao24, registration, typecode (one row per airframe)."""
    from ..paths import DATA
    path = sorted(((data_dir or DATA) / "opensky" / "aircraft_db").glob("aircraft-database-complete-*.csv"))[-1]
    db = pd.read_csv(path, quotechar="'", usecols=["icao24", "registration", "typecode", "operatorIcao"], dtype=str)
    return db


def seats_by_type(t100_routes: pd.DataFrame, month: pd.Timestamp, window: int = 12) -> tuple[dict, float]:
    """Seats per departure for each BTS aircraft code, from the `window` months of T-100 that were PUBLIC when `month`
    is being measured (i.e. ending PUBLICATION_LAG_MONTHS before it). Returns ({bts_code: seats}, fleet average)."""
    end = pd.Timestamp(month) - pd.DateOffset(months=PUBLICATION_LAG_MONTHS)
    start = end - pd.DateOffset(months=window - 1)
    w = t100_routes[(t100_routes["period"] >= start) & (t100_routes["period"] <= end)]
    g = w.groupby("aircraft_type")[["seats", "departures_performed"]].sum()
    g = g[g["departures_performed"] > 0]
    per = (g["seats"] / g["departures_performed"]).to_dict()
    avg = float(g["seats"].sum() / g["departures_performed"].sum()) if len(g) else float("nan")
    return per, avg


def attach_seats(flights: pd.DataFrame, t100_routes: pd.DataFrame, month_col: str = "period") -> pd.DataFrame:
    """Add `seats` to flight-level rows with an ICAO `typecode`: BTS seats per departure for that type, measured as of
    each month. Unknown types get the fleet average (flagged in `seats_source`)."""
    out = flights.copy()
    out["bts_code"] = out["typecode"].map(ICAO_TO_BTS)
    out["seats"] = float("nan")
    out["seats_source"] = "type"
    for mth, idx in out.groupby(month_col).groups.items():
        per, avg = seats_by_type(t100_routes, mth)
        s = out.loc[idx, "bts_code"].map(per)
        out.loc[idx, "seats_source"] = s.notna().map({True: "type", False: "fleet average"})
        out.loc[idx, "seats"] = s.fillna(avg)
    return out
