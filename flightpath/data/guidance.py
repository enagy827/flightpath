"""Delta's guidance, as extracted by scripts/extract_guidance.py (OpenAI), re-checked and put into one table (notebook 04).

Each extracted item is re-checked against its source filing with the SAME checks the extraction script applies
(quote found in the filing, numbers found in the quote, period not yet ended on the filing date), so the notebook
does not depend on whether the script's --recheck was run.

Guidance in words ("up mid-teens") is turned into a range with ONE stated table (WORDS), used everywhere and checked
against how results landed in notebook 04.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]


def _extractor():
    spec = importlib.util.spec_from_file_location("extract_guidance", _ROOT / "scripts" / "extract_guidance.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_extracted(data_dir=None) -> pd.DataFrame:
    """One row per extracted guidance item, with the automatic checks re-applied."""
    from ..paths import DATA
    data = Path(data_dir or DATA)
    ex = _extractor()
    srcs = {p.stem: p for d in ("delta_8k", "delta_8k_other") for p in (data / "sec" / d).glob("*")}
    rows = []
    for f in sorted((data / "derived" / "guidance").glob("*.json")):
        res = json.loads(f.read_text())
        src = srcs.get(f.stem)
        tn = ex._norm(ex.doc_text(src)) if src else None
        for it in res["items"]:
            if tn is not None:
                it = ex.check(dict(it), tn, res["filed"])
            rows.append({"file": res["file"], "filed": pd.Timestamp(res["filed"]), "kind": res["kind"],
                         "source_found": src is not None, **it})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["metric_named"] = [metric_named(m, q) for m, q in zip(df["metric"], df["quote"])]
    df["unit_ok"] = [unit_ok(m, u) for m, u in zip(df["metric"], df["unit"])]
    df["year_consistent"] = [year_consistent(p, q) for p, q in zip(df["period"], df["quote"])]
    df["passes_checks"] = (df["source_found"] & df["row_found"] & df["numbers_in_quote"] & df["future_period"]
                           & df["metric_named"] & df["unit_ok"] & df["year_consistent"])
    df["period_end"] = df["period"].map(ex._period_end)
    df["whole"] = df["quote"].map(whole_company)
    rng = df.apply(lambda r: pd.Series(as_range(r), index=["range_low", "range_high", "range_from"]), axis=1)
    return pd.concat([df, rng], axis=1)


# Two more checks, on the line the number sits on (not only that it is in the filing):
# - the quote must NAME the metric (a bare "$ ~3.07 - 3.12" can't show whether it is quarterly or full-year EPS);
# - the unit must fit the metric (revenue "growth" given in $ billions is a revenue level, not growth).
METRIC_WORDS = {"eps": r"\beps\b|earnings|per share", "operating_margin": r"margin", "pretax_margin": r"margin",
                "total_revenue_yoy": r"revenue", "passenger_revenue_yoy": r"revenue", "unit_revenue_yoy": r"revenue|rasm",
                "fuel_price_gal": r"fuel|gallon", "capacity_yoy": r"capacity|\basms?\b|seat miles",
                "nonfuel_unit_cost_yoy": r"casm|unit cost|cost per|non-?fuel", "nonfuel_cost_yoy": r"cost|expense"}
UNITS = {"eps": {"usd"}, "operating_margin": {"percent", "points"}, "pretax_margin": {"percent", "points"},
         "total_revenue_yoy": {"percent", "points"}, "passenger_revenue_yoy": {"percent", "points"},
         "unit_revenue_yoy": {"percent", "points"}, "fuel_price_gal": {"usd_per_gallon", "usd"},
         "capacity_yoy": {"percent", "points"}, "nonfuel_unit_cost_yoy": {"percent", "points"}}


def year_consistent(period, quote: str) -> bool:
    """The quote must not name a year LATER than the labelled period's year: "Delta introduces 2023 guidance for $5 to
    $6 in adjusted earnings per share" labelled as Q4 2022 is full-year 2023 guidance (found on the 14 Dec 2022 8-K).
    Earlier years are fine - they are usually the comparison base ("over 2023", "vs. 2019")."""
    py = re.search(r"(?:19|20)\d\d", str(period or ""))
    if py is None:
        return True
    return all(int(y) <= int(py.group(0)) for y in re.findall(r"\b(20\d\d)\b", quote or ""))


def metric_named(metric: str, quote: str) -> bool:
    pat = METRIC_WORDS.get(metric)
    return True if pat is None else bool(re.search(pat, quote or "", re.I))


def unit_ok(metric: str, unit: str) -> bool:
    ok = UNITS.get(metric)
    return True if ok is None else unit in ok


# word -> (low, high) in %, for growth guidance given in words. Bands: low = 1-3, mid = 3-7, high = 7-9 within
# single digits; low-teens 10-13, mid-teens 13-17, high-teens 17-19 (a band is centred on its word: mid-teens ~15).
WORDS = [
    ("flat", r"\bflat\b(?!\s*to)", (-1.0, 1.0)),
    ("low single digits", r"low[- ]single[- ]digit", (1.0, 3.0)),
    ("mid single digits", r"mid[- ]single[- ]digit", (3.0, 7.0)),
    ("high single digits", r"high[- ]single[- ]digit", (7.0, 9.0)),
    ("low double digits / low-teens", r"low[- ](?:double[- ]digit|teens)", (10.0, 13.0)),
    ("mid-teens", r"mid[- ]teens", (13.0, 17.0)),
    ("high-teens", r"high[- ]teens", (17.0, 19.0)),
]


def word_range(wording: str | None) -> tuple[float, float] | None:
    """Range (%) for guidance given in words, signed by "down"; None if the wording isn't in the table."""
    if not isinstance(wording, str) or not wording:
        return None
    w = wording.lower()
    for _, pat, (lo, hi) in WORDS:
        if re.search(pat, w):
            return (-hi, -lo) if re.search(r"\bdown\b|\bdecline|\blower\b", w) and "flat" not in pat else (lo, hi)
    return None


def as_range(row: pd.Series) -> tuple[float, float, str] | tuple[None, None, None]:
    """(low, high, how) for one guidance item: stated numbers first, then the word table."""
    lo, hi, pt = row.get("low"), row.get("high"), row.get("point")
    if pd.notna(lo) and pd.notna(hi):
        return min(lo, hi), max(lo, hi), "stated range"
    if pd.notna(pt):
        return pt, pt, "stated number"
    if pd.notna(lo) or pd.notna(hi):
        v = lo if pd.notna(lo) else hi
        return v, v, "stated bound"
    wr = word_range(row.get("wording"))
    return (wr[0], wr[1], "words -> table") if wr else (None, None, None)


_PART = re.compile(r"\b(domestic|international|mainline|regional|atlantic|pacific|latin|trans-?atlantic|"
                   r"transpacific|cargo|loyalty|premium|main cabin)\b", re.I)
_WHOLE = re.compile(r"\b(system|consolidated|total)\b", re.I)


def whole_company(quote: str) -> bool:
    """Guidance for the whole airline, not one region or business ("Domestic capacity up 2%" is a part)."""
    return bool(_WHOLE.search(quote)) or not _PART.search(quote)


def in_force(g: pd.DataFrame, metric: str, period, as_of) -> pd.Series | None:
    """The latest checked whole-company guidance item for `metric` and quarter `period` filed on or before `as_of`
    (an update replaces earlier guidance). Growth guidance measured against a named year (e.g. "vs 2019") is skipped."""
    end = pd.Period(period, "Q").end_time.normalize()
    s = g[(g.metric == metric) & (g.period_end == end) & g.period.str.contains("Q") & g.passes_checks
          & g.whole & (g.filed <= pd.Timestamp(as_of))]
    s = s[s["range_low"].notna()]
    if metric.endswith("_yoy"):          # growth vs 2019 (Delta's 2021-23 habit) is not growth vs a year earlier
        s = s[~s["vs"].fillna("").str.contains(r"\b(?:19|20)\d\d\b")]
    return None if s.empty else s.sort_values("filed").iloc[-1]
