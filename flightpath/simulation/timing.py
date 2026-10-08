"""How much of a quarter is still to come on a given date - so a scenario can change only the rest of the quarter.

A change that starts on the chosen date (a schedule cut, a fare war, a fuel spike) should leave the days already flown
as they were. To apply it to "the rest of the quarter", the simulator needs the SHARE of the quarter's flying (and of
its passengers) still to come. Days are not equal - July flies more than September - so the share is measured, not
counted in calendar days:

  flying      Delta's daily flight hours from flight tracking (OpenSky), on the same days of the quarter a year
              earlier; days flagged as OpenSky outages are filled by interpolation (notebook 07, section 7a).
  passengers  TSA checkpoint counts on the same days of the quarter a year earlier.

A year earlier is used because this year's remaining days have not happened yet, and last year's pattern was public
on any date we stand on (point in time). Notebook 05b tests the rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _quarter_days(q: pd.Period) -> pd.DatetimeIndex:
    return pd.date_range(q.start_time.normalize(), q.end_time.normalize(), freq="D")


def rest_shares(daily: pd.DataFrame, quarter, as_of) -> dict:
    """Share of the quarter still to come after `as_of` (the chosen day counts as already happened).

    `daily`: index = date; columns `delta_hours`, `tsa`, and optionally `outage_full` / `outage_partial` (bool).
    Returns {"flying", "passengers", "calendar"} as fractions in [0, 1], plus where the pattern came from."""
    q = pd.Period(str(quarter), "Q")
    as_of = pd.Timestamp(as_of).normalize()
    days = _quarter_days(q)
    elapsed = int((days <= as_of).sum())                  # days of this quarter already happened (incl. as_of)
    out = {"calendar": (len(days) - elapsed) / len(days), "pattern_from": str(q - 4)}
    if elapsed == 0 or elapsed == len(days):
        out.update(flying=1.0 if elapsed == 0 else 0.0, passengers=1.0 if elapsed == 0 else 0.0)
        return out
    ly = _quarter_days(q - 4)
    for key, col in [("flying", "delta_hours"), ("passengers", "tsa")]:
        s = daily[col].reindex(ly).astype(float)
        if col == "delta_hours":                          # OpenSky outage days: not real drops in flying
            bad = pd.Series(False, index=ly)
            for flag in ("outage_full", "outage_partial"):
                if flag in daily.columns:
                    bad |= daily[flag].reindex(ly).fillna(False).astype(bool)
            s[bad] = np.nan
        s = s.interpolate(limit_direction="both")
        pos = np.arange(len(ly))
        # day i of last year's quarter stands for day i of this quarter (same position in the quarter)
        rest = s.values[pos >= elapsed].sum() if len(ly) > elapsed else 0.0
        out[key] = float(rest / s.sum()) if s.notna().all() and s.sum() > 0 else out["calendar"]
    return out
