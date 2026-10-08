"""Delta's reported quarterly numbers, parsed from earnings-release exhibits (8-K Ex. 99.1).

Downloaded by `scripts/fetch_public_data.py sec` -> ~/flightpath_data/sec/delta_8k/<filed>_<name>.htm
available_at = filing date (the release is public that morning).

The parser reads the operating-statistics table: each label is followed by the current-quarter
value and then the prior-year value. Labels have shifted slightly over the years, so each metric
lists several patterns. ALWAYS eyeball `reported_quarters()` output against a release or two.
"""
from __future__ import annotations

import html as htmllib
import re
from datetime import date, timedelta

import pandas as pd

from .base import DataSource

# metric -> regexes for its row label in the operating statistics table, tried in order (case-insensitive).
# Older releases (2009-2011) use Title Case labels, footnote markers "(2)" and "¢"/"$" cells between numbers.
LABELS = {
    "rpm_m": [r"Revenue passenger miles \(millions\)"],
    "asm_m": [r"Available seat miles \(millions\)"],
    "load_factor_pct": [r"Passenger load factor"],
    "yield_cents": [r"Passenger mile yield(?: \(cents\))?"],
    # one alternation (not a list) so the FIRST row in the document wins, as in earlier versions
    "prasm_cents": [r"Passenger revenue per available seat mile(?: \(cents\)| \(PRASM\))|PRASM \(cents\)"],
    "trasm_cents": [r"Total revenue per available seat mile(?: \(cents\)| \(TRASM\))|TRASM \(cents\)"],
    "casm_cents": [r"(?:Operating )?cost per available seat mile(?: \(cents\)| \(CASM\))|CASM \(cents\)"],
    "fuel_gallons_m": [r"Fuel gallons consumed \(millions\)"],
    "fuel_price_gal": [r"Average price per fuel gallon"],
    # Delta's price excluding mark-to-market hedge gains/losses (Note A). This is the basis of Delta's fuel guidance
    # ("fuel price, including taxes, settled hedges and refinery impact") and of its adjusted EPS.
    "fuel_price_gal_adj": [r"Average price per fuel gallon, adjusted(?: - see Note [A-C])?"],
}
# income-statement rows ($ millions). Until 2017 passenger revenue was split into mainline + regional, so the
# "Total passenger revenue" row (revenue-by-region table) is tried first.
IS_LABELS = {
    "passenger_revenue_m": [r"Total passenger revenue", r"Passenger(?=\s*\$?\s*[\d(])"],
    "total_revenue_m": [r"Total operating revenue", r"Operating revenue"],
    "fuel_expense_m": [r"Aircraft fuel and related taxes", r"Fuel expense"],
    "total_opex_m": [r"Total operating expense"],
    "operating_income_m": [r"Operating income(?:\s*/?\s*\(loss\))?|Operating \(loss\)\s*/?\s*income"],
}
NUM = r"\(?\$?\s*-?[\d,]+(?:\.\d+)?\s*%?\)?"
_TOKEN = re.compile(r"\(?-?\d[\d,]*(?:\.\d+)?\)?|[A-Za-z]{3,}")


def html_to_text(raw: str) -> str:
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    t = re.sub(r"(?i)<br\s*/?>|</(p|div|tr|li|h\d)>", "\n", t)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    t = htmllib.unescape(t).replace("\xa0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    # negative numbers split across cells: "(482 \n )" -> "(482)"
    t = re.sub(r"\(\s*(-?[\d,]+(?:\.\d+)?)\s*\)", r"(\1)", t)
    # footnote markers at the end of a label: "Available Seat Miles (millions) (2)" -> "... (millions)"
    t = re.sub(r"(?m)([A-Za-z)])[ \t]+\(\d\)[ \t]*$", r"\1", t)
    return t


def _to_float(tok: str) -> float:
    neg = tok.strip().startswith("(") and tok.strip().endswith(")")
    v = float(re.sub(r"[^\d.\-]", "", tok))
    return -v if neg else v


# $-million rows: skip same-label rows in per-seat-mile (cents) tables, e.g. "Aircraft fuel and related taxes 3.72"
MIN_ABS = {"passenger_revenue_m": 100, "total_revenue_m": 100, "fuel_expense_m": 100, "total_opex_m": 100}


def _first_two_numbers_after(label_re: str, text: str, min_abs: float = 0.0) -> tuple[float, float] | None:
    """First two numbers in the table row that starts with `label_re` (current quarter, same quarter a year
    earlier). Anchored to the start of a line so narrative sentences don't match; reads numbers until the next
    word (the next row's label), skipping "$", "%", "¢" and "pts"."""
    # older releases wrap labels across lines ("Available\nSeat Miles (millions)"): any whitespace between words
    label_re = label_re.replace(" ", r"\s+")
    for m in re.finditer(r"(?im)^\s*(?:" + label_re + r")[^\n\d$(]*", text):
        nums = []
        start = m.end()
        foot = re.match(r"\s*\(\d\)(?=\s)", text[start:start + 10])     # footnote marker on its own line
        if foot:
            start += foot.end()
        for tok in _TOKEN.finditer(text, start, start + 400):
            w = tok.group(0)
            if w[0].isalpha():
                if w.lower() in ("pts", "pt"):
                    continue
                break
            nums.append(_to_float(w))
            if len(nums) == 2:
                if min(abs(nums[0]), abs(nums[1])) >= min_abs:
                    return nums[0], nums[1]
                break
    return None


def quarter_from_text(text: str, filed: pd.Timestamp) -> pd.Period:
    """Quarter being reported: the most recent completed quarter before the filing date."""
    return (filed - pd.Timedelta(days=1)).to_period("Q") if filed.month not in (1, 4, 7, 10) \
        else (filed.to_period("Q") - 1)


def parse_release(raw_html: str, filed: str) -> dict:
    text = html_to_text(raw_html)
    filed_ts = pd.Timestamp(filed)
    row = {"filed": filed_ts, "quarter": str(quarter_from_text(text, filed_ts))}
    for k, pats in {**LABELS, **IS_LABELS}.items():
        got = None
        for pat in pats:
            got = _first_two_numbers_after(pat, text, MIN_ABS.get(k, 0.0))
            if got:
                break
        row[k] = got[0] if got else None
        row[k + "_prior_year"] = got[1] if got else None
    # Until 2017 Delta also printed a MAINLINE operating-statistics table (its own flights, excluding regional
    # partners) - the same scope as BTS T-100 for code DL, so it allows a direct like-for-like check (notebook 00)
    for k, lab in (("mainline_asm_m", LABELS["asm_m"][0]), ("mainline_rpm_m", LABELS["rpm_m"][0])):
        row[k] = row[k + "_prior_year"] = None
        for mm in re.finditer(r"(?im)^\s*(?:" + lab.replace(" ", r"\s+") + r")", text):
            ctx = re.findall(r"(?i)(Mainline|Consolidated|Regional|Six Months|Nine Months|Year Ended)",
                             text[max(0, mm.start() - 600):mm.start()])
            if ctx and ctx[-1].lower() == "mainline":
                got = _first_two_numbers_after(lab, text[mm.start():])
                if got:
                    row[k], row[k + "_prior_year"] = got
                break
    m = re.search(r"(?im)^\s*(Average\s+price\s+per\s+fuel\s+gallon[^\n\d$]*)", text)
    row["fuel_price_label"] = re.sub(r"\s+", " ", m.group(1)).strip(" ,") if m else None
    return row


def reported_quarters(data_dir=None) -> pd.DataFrame:
    """One row per earnings release (current + prior-year columns)."""
    from ..paths import DATA
    folder = (data_dir or DATA) / "sec" / "delta_8k"
    rows = []
    for f in sorted(list(folder.glob("*.htm*")) + list(folder.glob("*.txt"))):
        filed = f.name[:10]
        row = parse_release(f.read_text(encoding="utf-8", errors="ignore"), filed)
        row["file"] = f.name
        # keep only files that actually contain an operating statistics table
        if row.get("asm_m") is not None:
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = (df.sort_values(["filed", "file"], kind="stable")
            .drop_duplicates("quarter", keep="first").reset_index(drop=True))
    return fill_derived(_comparison_base(df))


def _comparison_base(df: pd.DataFrame) -> pd.DataFrame:
    """Which quarter the release's comparison column refers to. Normally the same quarter a year earlier, but
    during COVID Delta compared 2021 and 2022 quarters with 2019. Identified from the data: the comparison-column
    capacity (ASM) must match the capacity first reported for that earlier quarter (within 0.5%)."""
    df = df.copy()
    asm = pd.Series(df["asm_m"].values, index=pd.PeriodIndex(df["quarter"], freq="Q"))
    base = []
    for q, prior in zip(pd.PeriodIndex(df["quarter"], freq="Q"), df["asm_m_prior_year"]):
        hit = None
        for lag in (4, 8, 12):
            ref = asm.get(q - lag)
            if pd.notna(prior) and ref is not None and pd.notna(ref) and abs(prior / ref - 1) < 0.005:
                hit = str(q - lag)
                break
        base.append(hit if hit else str(q - 4) + "?")
    df["comparison_quarter"] = base
    return df


def fill_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Fill gaps / fix mis-parses with accounting identities (flagged in `derived_fields`).

    - fuel expense: must be $M scale; else gallons x average price per gallon
    - RPM: ASM x load factor
    - yield: passenger revenue / RPM
    """
    df = df.copy()
    df["derived_fields"] = ""

    def flag(mask, name):
        df.loc[mask, "derived_fields"] += name + " "

    bad = df["fuel_expense_m"].isna() | (df["fuel_expense_m"] < 100)
    df.loc[bad, "fuel_expense_m"] = df["fuel_gallons_m"] * df["fuel_price_gal"]
    flag(bad, "fuel_expense")
    m = df["rpm_m"].isna()
    df.loc[m, "rpm_m"] = df["asm_m"] * df["load_factor_pct"] / 100
    flag(m, "rpm")
    m = df["yield_cents"].isna()
    df.loc[m, "yield_cents"] = df["passenger_revenue_m"] / df["rpm_m"] * 100
    flag(m, "yield")
    # operating income must equal total revenue - total operating expense (a few releases put another
    # "Operating income" row first, e.g. a year-to-date or adjusted column)
    if {"total_revenue_m", "total_opex_m", "operating_income_m"} <= set(df.columns):
        implied = df["total_revenue_m"] - df["total_opex_m"]
        m = implied.notna() & ((df["operating_income_m"] - implied).abs() > 1)
        df.loc[m, "operating_income_m"] = implied[m]
        flag(m, "operating_income")
    df["derived_fields"] = df["derived_fields"].str.strip()
    return df


def reported_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Store series. delta.reported.<metric> = the quarter's value as first reported.
    delta.reported_prior_year.<metric> = the comparison column printed in this release (restated on the current
    basis): normally the same quarter a year earlier; for 2021-2022 releases it is 2019 (see `comparison_quarter`),
    so those values are NOT stored as prior-year. Year-over-year changes should use these two from the same release (like-for-like,
    e.g. across the 2018 revenue-accounting change), not two different releases (notebook 00, section 4)."""
    metrics = [m for m in list(LABELS) + list(IS_LABELS) if m in df.columns]
    # history starts 2009Q1: the Q4 2008 release straddles the Northwest merger (29 Oct 2008)
    df = df[pd.PeriodIndex(df["quarter"], freq="Q") >= pd.Period("2009Q1", "Q")]
    if "comparison_quarter" in df.columns:          # keep prior-year values only where they really are prior-year
        yoy = df["comparison_quarter"] == (pd.PeriodIndex(df["quarter"], freq="Q") - 4).astype(str)
        df = df.copy()
        for m in metrics:
            if m + "_prior_year" in df.columns:
                df.loc[~yoy, m + "_prior_year"] = None
    frames = []
    for prefix, suffix in (("delta.reported.", ""), ("delta.reported_prior_year.", "_prior_year")):
        cols = [m + suffix for m in metrics if m + suffix in df.columns]
        long = df.melt(id_vars=["quarter", "filed"], value_vars=cols, var_name="metric", value_name="value")
        long["metric"] = long["metric"].str.replace("_prior_year$", "", regex=True)
        long["period"] = pd.PeriodIndex(long["quarter"], freq="Q").start_time
        frames.append(pd.DataFrame({"series": prefix + long["metric"], "period": long["period"],
                                    "available_at": long["filed"], "value": long["value"], "source": "SEC"}))
    return pd.concat(frames, ignore_index=True).dropna(subset=["value"])


class SECSource(DataSource):
    name = "sec"
    default_lag = timedelta(days=15)

    def fetch(self, start: date, end: date) -> None:
        raise NotImplementedError("Run scripts/fetch_public_data.py sec on the laptop.")

    def parse(self) -> pd.DataFrame:
        return reported_observations(reported_quarters())


# ----------------------------------------------------------------------------- regional table
REGIONS = ["Domestic", "Atlantic", "Latin America", "Pacific"]
_PCT = r"(?:\(?-?[\d.]+\)?|[\u2014\u2013-])\s*%"      # "5%", "(7)%", or a dash meaning 0% ("— %")


def parse_regions(raw_html: str, filed: str) -> list[dict]:
    """Delta's passenger-revenue-by-region table: revenue ($M) and % change in revenue, unit revenue, yield and
    capacity. The comparison base is read from the table header ("2Q22 vs 2Q19" -> base year 2019); during
    COVID (4Q20-4Q22) Delta compared to 2019 rather than the prior year."""
    text = html_to_text(raw_html)
    filed_ts = pd.Timestamp(filed)
    q = str(quarter_from_text(text, filed_ts))
    rows = []
    for reg in REGIONS:
        m = re.search(r"(?m)^\s*" + reg + r"\s*\$?\s*([\d,]+)\s+((?:" + _PCT + r"\s*){4})", text)
        if not m:
            continue
        head = re.sub(r"\s+", " ", text[max(0, m.start() - 400):m.start()])
        b = re.findall(r"\dQ(\d\d)\s*\|?\s*(?:versus|vs\.?)\s*\|?\s*\dQ(\d\d)", head)
        base_year = 2000 + int(b[-1][1]) if b else int(q[:4]) - 1
        vals = [(-1 if p.strip().startswith("(") else 1) * float(re.sub(r"[^\d.]", "", p) or 0) for p in re.findall(_PCT, m.group(2))]
        rows.append({"quarter": q, "filed": filed_ts, "region": reg, "revenue_m": float(m.group(1).replace(",", "")),
                     "revenue_chg_pct": vals[0], "unit_revenue_chg_pct": vals[1], "yield_chg_pct": vals[2],
                     "capacity_chg_pct": vals[3], "comparison_base_year": base_year})
    return rows


def reported_regions(data_dir=None) -> pd.DataFrame:
    from ..paths import DATA
    folder = (data_dir or DATA) / "sec" / "delta_8k"
    rows = []
    for f in sorted(list(folder.glob("*.htm*")) + list(folder.glob("*.txt"))):
        if "ex9902" in f.name:
            continue
        rows += parse_regions(f.read_text(encoding="utf-8", errors="ignore"), f.name[:10])
    df = pd.DataFrame(rows)
    return df.drop_duplicates(["quarter", "region"], keep="first").reset_index(drop=True) if len(df) else df
