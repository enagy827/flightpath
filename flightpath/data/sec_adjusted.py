"""Delta's adjusted (non-GAAP) figures and below-the-line items, from the same earnings releases (notebook 04).

Delta's guidance is on an ADJUSTED basis (excluding third-party refinery sales, mark-to-market hedge and investment
results, one-time items), so revenue, costs, profit and EPS are compared on that basis. Every figure comes with the
prior-year column printed in the SAME release, so year-over-year changes are like-for-like even when Delta changes a
definition (e.g. 2026: MRO expense removed from non-fuel cost) - the same rule as notebook 00.

Read from each release:
  statistical summary (current, prior year):  TRASM adjusted, CASM-Ex (non-fuel unit cost), both in cents per ASM
  income statement (current, prior year):     profit sharing, interest expense, pre-tax income, income tax,
                                              diluted shares, diluted EPS (GAAP)
  adjusted summary table (current, prior year, recent releases): operating income, pre-tax income, net income and
                                              diluted EPS, adjusted
Derived: adjusted revenue = TRASM adjusted x ASM; non-fuel cost = CASM-Ex x ASM; other revenue = adjusted revenue -
passenger revenue.
"""
from __future__ import annotations

import re

import pandas as pd

from .sec import _TOKEN, _to_float, html_to_text, quarter_from_text

# label -> (regex, smallest absolute value accepted; skips per-seat-mile rows with the same label)
ROWS = {
    # "TRASM, adjusted - see Note A (cents)" (2018+), "TRASM, excluding refinery - see Note A (cents)" (2017)
    "trasm_adj_cents": [(r"TRASM, (?:adjusted|excluding refinery)\s*(?:-\s*see Note [A-C])?\s*\(cents\)", 1)],
    "casm_ex_cents": [(r"CASM-Ex(?:,? including profit sharing)?(?: - see Note [A-C])? \(cents\)", 1)],
    "casm_ex_label": [],
    "profit_sharing_m": [(r"Profit sharing", 1)],
    # since 2026 non-fuel cost excludes the cost of maintenance work for other airlines (MRO); shown in the
    # non-fuel cost reconciliation as "MRO expense"
    "mro_expense_m": [(r"MRO expense", 1)],
    "interest_expense_m": [(r"Interest expense, net", 1), (r"Interest expense", 1)],
    "pretax_income_m": [(r"Income \(Loss\) Before Income Taxes|\(Loss\) Income Before Income Taxes|"
                         r"Income Before Income Taxes|Loss Before Income Taxes", 1)],
    "income_tax_m": [(r"Income Tax (?:\(Provision\) Benefit|Benefit \(Provision\)|Provision|Benefit)", 0)],
    "diluted_shares_m": [(r"Diluted Weighted Average Shares Outstanding", 100)],
    "eps_diluted": [(r"Diluted (?:Earnings|\(Loss\) Earnings|Loss|Earnings \(Loss\)) Per Share", 0)],
}
ADJUSTED_FROM = "2022Q2"     # first quarter whose adjusted summary table is read (see adjusted_quarters)
_CELL = re.compile(r"\(?-?\d[\d,]*(?:\.\d+)?\)?|(?<![\w-])[—–]|[A-Za-z]{2,}")


def _cells(label_re: str, text: str, n: int, min_abs: float = 0.0, with_label: bool = False):
    """First `n` cells of the table row starting with `label_re`. A dash cell means zero."""
    label_re = label_re.replace(" ", r"\s+")
    for m in re.finditer(r"(?im)^\s*(?:" + label_re + r")[^\n\d$(\u2014\u2013]*", text):
        vals = []
        for tok in _CELL.finditer(text, m.end(), m.end() + 400):
            w = tok.group(0)
            if w[0].isalpha():
                if w.lower() in ("pts", "pt", "see", "note"):
                    continue
                break
            vals.append(0.0 if w in "\u2014\u2013" else _to_float(w))
            if len(vals) == n:
                break
        if len(vals) == n and all(v == 0 or abs(v) >= min_abs for v in vals):
            return (vals, re.sub(r"\s+", " ", m.group(0)).strip()) if with_label else vals
    return None


def _two_cells(label_re: str, text: str, min_abs: float) -> tuple[float, float] | None:
    """Current and prior-year cells of the table row starting with `label_re`."""
    got = _cells(label_re, text, 2, min_abs)
    return (got[0], got[1]) if got else None


def _adjusted_block(text: str) -> dict:
    """The summary tables at the top of recent releases (2019+). Each has a header ending in "($ in millions except
    per share and unit costs) 2Q26 2Q25". Three layouts, told apart by the words just above that header:
      "Adjusted ... Change Change"           -> adjusted table: columns 1-2 = this quarter, prior year
      "GAAP Adjusted GAAP Adjusted" + FY cols -> columns 3-4 = adjusted this quarter, prior year (5-8 = full years)
      "GAAP ... Change Change"                -> GAAP table, skipped
    Reads operating income, pre-tax income, net income and diluted EPS, adjusted."""
    out = {}
    for m in re.finditer(r"\(\$ in millions except per share", text):
        head = re.sub(r"\s+", " ", text[max(0, m.start() - 250): m.start()])
        head = head[head.rfind("NM") + 2:] if "NM" in head else head       # only the words of THIS header
        if "Adjusted" not in head:
            continue
        q4 = "GAAP" in head and bool(re.search(r"FY\s?\d{2}", text[m.start(): m.start() + 250]))
        block = text[m.start(): m.start() + 2500]
        for k, lab in (("operating_income_adj_m", r"Operating income"), ("pretax_income_adj_m", r"Pre-tax income"),
                       ("net_income_adj_m", r"Net income"), ("eps_adj", r"Diluted earnings per share")):
            got = _cells(lab, block, 4 if q4 else 2)
            if got:
                out[k], out[k + "_prior_year"] = (got[2], got[3]) if q4 else (got[0], got[1])
        if out:
            out["adjusted_table_layout"] = "Q4 (GAAP + adjusted)" if q4 else "adjusted"
            return out
    return out


def parse_adjusted(raw_html: str, filed: str) -> dict:
    text = html_to_text(raw_html)
    filed_ts = pd.Timestamp(filed)
    row = {"filed": filed_ts, "quarter": str(quarter_from_text(text, filed_ts))}
    for k, pats in ROWS.items():
        got = None
        for pat, mn in pats:
            got = _two_cells(pat, text, mn)
            if got:
                break
        if pats:
            row[k], row[k + "_prior_year"] = got if got else (None, None)
    got = _cells(ROWS["casm_ex_cents"][0][0], text, 2, 1, with_label=True)
    row["casm_ex_label"] = got[1] if got else None
    row.update(_adjusted_block(text))
    return row


def adjusted_quarters(data_dir=None) -> pd.DataFrame:
    """One row per quarter (first release that reports it), joined to the operating statistics of notebook 00."""
    from ..paths import DATA
    from .sec import reported_quarters
    folder = (data_dir or DATA) / "sec" / "delta_8k"
    rows = []
    for f in sorted(list(folder.glob("*.htm*")) + list(folder.glob("*.txt"))):
        r = parse_adjusted(f.read_text(encoding="utf-8", errors="ignore"), f.name[:10])
        r["file"] = f.name
        rows.append(r)
    adj = pd.DataFrame(rows)
    rep = reported_quarters(data_dir)
    adj = adj[adj["file"].isin(rep["file"])].drop(columns=["filed"])
    d = rep.merge(adj.drop(columns=["quarter"]), on="file", how="left")
    d["casm_ex_includes_profit_sharing"] = d["casm_ex_label"].fillna("").str.contains("including profit sharing")
    for s in ("", "_prior_year"):
        trasm = d["trasm_adj_cents" + s].fillna(d["trasm_cents" + s])     # no refinery sales: adjusted = reported
        d["revenue_adj_m" + s] = trasm * d["asm_m" + s] / 100
        d["nonfuel_cost_m" + s] = d["casm_ex_cents" + s] * d["asm_m" + s] / 100
        # 2016-2017 and Q3 2018 Delta's CASM-Ex INCLUDED profit sharing (row label says so): take it out, so non-fuel
        # cost means the same thing in every quarter (profit sharing is its own line, linked to profit)
        inc = d["casm_ex_label"].fillna("").str.contains("including profit sharing")
        d.loc[inc, "nonfuel_cost_m" + s] -= d.loc[inc, "profit_sharing_m" + s]
        d["other_revenue_adj_m" + s] = d["revenue_adj_m" + s] - d["passenger_revenue_m" + s]
    # identity check on the adjusted summary table: EPS x diluted shares = net income (within 2 cents a share).
    # Rows that fail were read from a different table layout and are dropped, not guessed.
    ok = (d["net_income_adj_m"] / d["diluted_shares_m"] - d["eps_adj"]).abs() <= 0.02
    # Before mid-2022 the releases use several table layouts and the GAAP and adjusted tables can't be told apart
    # reliably (a GAAP table can pass the identity check too), so adjusted results are used from Q2 2022 on only.
    ok &= pd.PeriodIndex(d["quarter"], freq="Q") >= pd.Period(ADJUSTED_FROM, "Q")
    d["adjusted_table_ok"] = ok
    cols = [c for c in d.columns if c.startswith(("operating_income_adj_m", "pretax_income_adj_m", "net_income_adj_m",
                                                  "eps_adj"))]
    d.loc[~ok, cols] = None
    d["quarter"] = pd.PeriodIndex(d["quarter"], freq="Q")
    d = d.set_index("quarter").sort_index()
    # a quarter whose own adjusted table couldn't be read: use the prior-year column of the release a year later
    # (public only from that later filing date - recorded so as-of rules can respect it)
    d["adjusted_from_next_year"] = False
    for q in d.index[d["eps_adj"].isna()]:
        nxt = q + 4
        if nxt in d.index and pd.notna(d.loc[nxt, "eps_adj_prior_year"]) and \
                str(d.loc[nxt, "comparison_quarter"]) == str(q):
            for c in ("operating_income_adj_m", "pretax_income_adj_m", "net_income_adj_m", "eps_adj"):
                d.loc[q, c] = d.loc[nxt, c + "_prior_year"]
            d.loc[q, "adjusted_from_next_year"] = True
            d.loc[q, "adjusted_public"] = d.loc[nxt, "filed"]
    d["adjusted_public"] = d.get("adjusted_public", pd.Series(index=d.index, dtype="datetime64[ns]")).fillna(d["filed"])
    return d
