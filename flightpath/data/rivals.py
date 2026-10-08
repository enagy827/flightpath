"""Rivals' capacity on Delta's markets, from BTS T-100 (all carriers), quarterly (notebook 07).

A market is an airport pair (either direction). For each quarter, the markets Delta (DL, plus Northwest in 2009 and
Endeavor 9E) flew a year earlier are fixed, and the seat-miles every other carrier flew on them are compared with a
year earlier. Note: regional airlines that fly for several majors (SkyWest, Republic) count as rivals here, since
T-100 doesn't say for whom a flight was flown.

Public like the rest of T-100: about 80 days after each month ends (bts.T100_LAG_DAYS).
"""
from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd

DELTA_CODES = ("DL", "NW", "9E")


def market_capacity(data_dir=None) -> pd.DataFrame:
    """Seat-miles by quarter and market, split Delta / rivals."""
    from ..paths import DATA
    frames = []
    for z in sorted(((data_dir or DATA) / "bts" / "t100").glob("T100_*.zip")):
        with zipfile.ZipFile(z) as zf:
            name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
            df = pd.read_csv(zf.open(name), usecols=["UNIQUE_CARRIER", "ORIGIN", "DEST", "YEAR", "MONTH", "CLASS", "SEATS", "DISTANCE"],
                             low_memory=False)
        df = df[df.CLASS == "F"]
        o, d = df.ORIGIN.values.astype(str), df.DEST.values.astype(str)
        df = df.assign(asm=df.SEATS * df.DISTANCE,
                       q=pd.PeriodIndex(pd.to_datetime(dict(year=df.YEAR, month=df.MONTH, day=1)), freq="Q"),
                       mkt=np.where(o < d, np.char.add(np.char.add(o, "-"), d), np.char.add(np.char.add(d, "-"), o)),
                       delta=df.UNIQUE_CARRIER.isin(DELTA_CODES))
        frames.append(df.groupby(["q", "mkt", "delta"]).asm.sum())
    m = pd.concat(frames).groupby(level=[0, 1, 2]).sum().unstack("delta", fill_value=0)
    m.columns = ["rivals" if not c else "delta" for c in m.columns]
    return m


def rival_growth(m: pd.DataFrame) -> pd.DataFrame:
    """Quarterly year-over-year growth (%): rivals on the markets Delta flew a year earlier; rivals everywhere;
    Delta itself (a check against Delta's reported capacity)."""
    qs = sorted(m.index.get_level_values(0).unique())
    rows = {}
    for q in qs:
        if q - 4 not in qs:
            continue
        ly, cur = m.loc[q - 4], m.loc[q]
        mk = ly.index[ly.delta > 0]
        rows[q] = {"rivals_on_delta_markets": (cur.reindex(mk).rivals.fillna(0).sum() / ly.loc[mk, "rivals"].sum() - 1) * 100,
                   "rivals_all_markets": (cur.rivals.sum() / ly.rivals.sum() - 1) * 100,
                   "delta_t100": (cur.delta.sum() / ly.delta.sum() - 1) * 100}
    return pd.DataFrame(rows).T


# ------------------------------------------------------------- OpenSky (live): rivals' daily flights
def opensky_daily_counts(data_dir=None, markets: bool = False):
    """Rival airlines' flights tracked by OpenSky per day (scripts/fetch_opensky_rivals.py), one column per airline
    (UAL, AAL, SWA, ASA, JBU). Day = OpenSky's partition day (UTC date the flight was last seen). Duplicates (same
    callsign, same first-seen time) dropped. With markets=True also returns the flights with their airport pair."""
    from ..paths import DATA
    files = sorted(((data_dir or DATA) / "opensky" / "trino_rivals").glob("rivals_*.parquet"))
    if not files:
        raise FileNotFoundError("run: python scripts/fetch_opensky_rivals.py plan")
    cols = ["callsign", "firstseen", "lastseen", "estdepartureairport", "estarrivalairport", "day", "airline"]
    f = pd.concat([pd.read_parquet(p, columns=cols) for p in files], ignore_index=True)
    f = f.drop_duplicates(["callsign", "firstseen"])
    f = f[~hawaiian_on_alaska(f)]
    f["d"] = pd.to_datetime(f["day"], utc=True).dt.tz_localize(None).dt.normalize()
    counts = f.groupby(["d", "airline"]).size().unstack(fill_value=0).asfreq("D").fillna(0)
    return (counts, f) if markets else counts


def hawaiian_on_alaska(f: pd.DataFrame) -> pd.Series:
    """Alaska-callsign flights touching a Hawaii airport (ICAO codes PH..). From late October 2025 Hawaiian Airlines'
    flights appear under Alaska's callsign (the two merged; in OpenSky, Alaska's Hawaii flights go from ~50 to ~900 a
    week) - left out in every year, so Alaska's growth compares like with like."""
    hi = (f.estdepartureairport.astype(str).str.startswith("PH") | f.estarrivalairport.astype(str).str.startswith("PH"))
    return f.callsign.astype(str).str.startswith("ASA") & hi


def _pair(o: pd.Series, d: pd.Series) -> pd.Series:
    ok = o.notna() & d.notna()
    o, d = o.astype(object).where(ok, "").astype(str), d.astype(object).where(ok, "").astype(str)
    return pd.Series(np.where(o < d, o + "-" + d, d + "-" + o), index=o.index).where(ok)


class LiveRivals:
    """Rivals' flying on Delta's markets, live from OpenSky (notebook 07). A market = an airport pair (either
    direction, ICAO codes, both airports seen by OpenSky). For quarter q, Delta's markets are the pairs Delta flew in
    the same quarter a year earlier (OpenSky, at least `min_flights` flights) - the same definition as the T-100
    history test. Measure: rivals' tracked flight-hours (and flights) on those markets, day by day.

    growth(q, as_of): the days of q already known (OpenSky: the next day) vs the same weekdays 52 weeks earlier;
    fewer than 7 days known: the last 28 days. Days flagged by the outage detector (full or partial, in either year)
    are left out of both sums. Rival data starts 27 Jun 2022, so the first quarter with a year-ago is Q3 2023.
    MEASURE (chosen in notebook 07 by matching T-100's rival growth on Delta's markets, Q3 2023 - Q2 2026):
      "total" (used): rivals' total tracked flights - closest to T-100 (2.0 pts average gap, correlation 0.82);
      "raw":   rivals' flying on Delta's markets (hit by the drift in how many flights OpenSky can place at both
               airports: 84% -> 90% of rivals' flights over 2023-26);
      "share x total": rivals' share of their airport-identified flying that is on Delta's markets x "total"."""
    MEASURE = "total"

    def __init__(self, delta_flights: pd.DataFrame, outages: pd.DataFrame, data_dir=None, min_flights: int = 30):
        from ..paths import DATA
        files = sorted(((data_dir or DATA) / "opensky" / "trino_rivals").glob("rivals_*.parquet"))
        if not files:
            raise FileNotFoundError("run: python scripts/fetch_opensky_rivals.py plan")
        parts, tot = [], []
        for p in files:                                             # one week at a time (17M flights in all)
            f = pd.read_parquet(p, columns=["callsign", "firstseen", "lastseen", "estdepartureairport",
                                            "estarrivalairport", "day"]).drop_duplicates(["callsign", "firstseen"])
            f = f[~hawaiian_on_alaska(f)]
            f["pair"] = _pair(f.estdepartureairport, f.estarrivalairport)
            f["d"] = pd.to_datetime(f["day"], utc=True).dt.tz_localize(None).dt.normalize()
            f["hours"] = ((pd.to_datetime(f.lastseen, utc=True) - pd.to_datetime(f.firstseen, utc=True))
                          .dt.total_seconds() / 3600).clip(0, 20)
            tot.append(f.assign(known=f.pair.notna()).groupby(["d", "known"]).agg(flights=("hours", "size"),
                                                                                   hours=("hours", "sum")))
            f = f.dropna(subset=["pair"])
            parts.append(f.groupby(["d", "pair"]).agg(flights=("hours", "size"), hours=("hours", "sum")))
        g = pd.concat(parts).groupby(level=[0, 1]).sum()
        self.daily = g                                              # (day, pair) -> rivals' flights, hours
        t = pd.concat(tot).groupby(level=[0, 1]).sum().unstack("known", fill_value=0)
        self.totals = pd.DataFrame({"hours_all": t["hours"].sum(axis=1), "hours_known": t["hours"].get(True, 0),
                                    "flights_all": t["flights"].sum(axis=1), "flights_known": t["flights"].get(True, 0)})
        days = g.index.get_level_values(0)
        counts = pd.Series(0, index=pd.date_range(days.min(), days.max()))
        dl = delta_flights.assign(d=pd.to_datetime(delta_flights.day, utc=True).dt.tz_localize(None).dt.normalize(),
                                  pair=_pair(delta_flights.origin, delta_flights.destination)).dropna(subset=["pair"])
        dq = dl.groupby([dl.d.dt.to_period("Q"), "pair"]).size()
        self.delta_markets = {q: set(s[s >= min_flights].index.get_level_values(1)) for q, s in dq.groupby(level=0)}
        bad = outages[(outages.full | outages.partial)].index
        self.bad = set(bad)
        self.first_day = counts.index[0]
        self.last_day = counts.index[-1]
        self._cache = {}

    def _series(self, q: pd.Period, col: str) -> pd.Series:
        key = (q, col)
        if key not in self._cache:
            mk = self.delta_markets.get(q - 4, set())
            x = self.daily[col]
            x = x[x.index.get_level_values(1).isin(mk)]
            self._cache[key] = x.groupby(level=0).sum().reindex(pd.date_range(self.first_day, self.last_day), fill_value=0)
        return self._cache[key]

    def growth(self, q, as_of, col: str = "flights") -> dict:
        q, as_of = pd.Period(q, "Q"), pd.Timestamp(as_of)
        s = self._series(q, col)
        last = min(as_of - pd.Timedelta(days=1), self.last_day)
        days = pd.date_range(q.start_time, q.end_time.normalize())
        obs = days[days <= last]
        if len(obs) >= 7:
            w, how = obs, "days of the quarter so far"
        else:
            w, how = pd.date_range(last - pd.Timedelta(days=27), last), "last 28 days"
        ok = lambda a: a not in self.bad and a - pd.Timedelta(days=364) not in self.bad and a - pd.Timedelta(days=364) >= self.first_day
        keep = [(a, a - pd.Timedelta(days=364)) for a in w if ok(a)]
        if len(keep) < 7:                       # too many outage days: the last 28 usable days instead
            back = pd.date_range(last - pd.Timedelta(days=90), last)[::-1]
            keep = sorted([(a, a - pd.Timedelta(days=364)) for a in back if ok(a)][:28])
            how = "last 28 usable days (outages)"
        if len(keep) < 7:
            return {"growth_pct": np.nan, "days": len(keep), "rule": how, "markets": len(self.delta_markets.get(q - 4, ()))}
        a, b = [x for x, _ in keep], [y for _, y in keep]
        T = self.totals
        raw = s.reindex(a).sum() / s.reindex(b).sum()
        share = (s.reindex(a).sum() / T[col + "_known"].reindex(a).sum()) / (s.reindex(b).sum() / T[col + "_known"].reindex(b).sum())
        total = T[col + "_all"].reindex(a).sum() / T[col + "_all"].reindex(b).sum()
        gr = {"total": total, "raw": raw, "share x total": share * total}[self.MEASURE]
        gr = (gr - 1) * 100
        return {"growth_pct": float(gr), "raw_pct": float((raw - 1) * 100), "share_x_total_pct": float((share * total - 1) * 100),
                "rivals_total_pct": float((total - 1) * 100), "days": len(keep), "rule": how,
                "markets": len(self.delta_markets.get(q - 4, ()))}
