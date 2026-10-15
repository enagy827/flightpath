"""FlightPath - a living system of Delta's quarter: start here, forecast, simulator, live data, track record, method.

Run from the flightpath folder:   streamlit run app/streamlit_app.py
Reads only app/snapshot/ (built by scripts/build_app_snapshot.py), so it starts in seconds and can be hosted.
The AI event box needs an OpenAI key: OPENAI_API_KEY in .env (local) or in the host's secrets.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flightpath.appdata import (CHANNELS_BY_METRIC, NICE, Snapshot, build_entry, chain_table, eps_bridge,  # noqa: E402
                                forecast, fuel_split, grid_dates, interpret_event, risk_light, signal, simulate)
from flightpath.paths import DATA  # noqa: E402
from flightpath.simulation.quarter import Levers, draw, evaluate, shapley  # noqa: E402

st.set_page_config(page_title="FlightPath", page_icon="✈", layout="wide")
BLUE, ORANGE, GREEN, AMBER, PINK, DKGREEN, PURPLE, RED = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                                           "#008300", "#4a3aa7", "#e34948"]
MUTED, GRID, TEXT2 = "#a3a29c", "#e4e3df", "#52514e"
LIGHT = {"red": RED, "amber": AMBER, "green": GREEN}
N = 50_000
st.markdown("""<style>
div[data-testid="stMetricValue"] {font-size: 1.9rem;}
.small {font-size: 0.85rem; color: #6b6a66;}
.dot {font-size: 1.1rem;}
.badge {display:inline-block; padding:1px 9px; border-radius:10px; font-size:0.72rem; font-weight:600;
        letter-spacing:0.04em; color:white; vertical-align:middle; margin-right:6px;}
[data-testid="stHeaderActionElements"], h1 a, h2 a, h3 a, h4 a {display:none !important;}
.b-fc {background:#2a78d6;} .b-sim {background:#4a3aa7;} .b-data {background:#6b6a66;}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------------ data
@st.cache_resource
def snap(version):
    """The snapshot, loaded once per version: `version` changes whenever a snapshot file or the shared code changes,
    so an updated copy (e.g. a new upload to the hosted app) is never served from the old one in memory."""
    S = Snapshot()
    if os.getenv("FLIGHTPATH_SHOW_IBES") != "1":        # licensed I/B/E/S consensus: off unless asked for
        S.cons_hist = None                              # (Emma, 15 Oct 2026: off for the finals, as online)
    return S


def _snap_version():
    import flightpath.appdata as _ad
    files = sorted((ROOT / "app" / "snapshot").glob("*")) + [Path(_ad.__file__)]
    return tuple((f.name, f.stat().st_size, f.stat().st_mtime_ns) for f in files if f.is_file())


S_ = snap(_snap_version())


@st.cache_resource(show_spinner="This date isn't saved in the snapshot, so FlightPath is loading the raw data to work it "
                                "out (about 2 minutes, first time only)...")
def live_ctx():
    """The raw data, for dates not in the snapshot (None if it isn't on this machine or can't be loaded)."""
    if not DATA.exists():
        return None
    try:
        from flightpath.simulation.general import Context
        ctx = Context()
        ctx.live_rivals()
        return ctx
    except Exception:
        return None


@st.cache_resource(max_entries=48)
def run(q, d, lv_items=(), modes=()):
    lv = Levers(**dict(lv_items), **dict(modes))
    e = S_.entry(q, d)
    R, S, S0 = simulate(e, N, lv)
    F = forecast(e, R, S) if e["news"] else {}
    F0 = forecast(e, R, S0) if (e["news"] and lv.active()) else F
    return {"e": e, "R": R, "S": S, "S0": S0, "F": F, "F0": F0, "lv": lv}


def fmt(m, v, sign=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    if m == "eps":
        return f"{'+' if sign and v >= 0 else '-' if v < 0 else ''}${abs(v):.2f}"
    return f"{v:+.1f}%" if (sign or m == "revenue_pct") else f"{v:.1f}%"


def density(x, color, name, opacity=0.25):
    """Smooth curve of where the runs land (kernel density on 8,000 of them; percentages stay exact in the text)."""
    from scipy.stats import gaussian_kde
    x = np.asarray(x, float)
    sub = np.random.default_rng(0).choice(x, min(8000, len(x)), replace=False)
    k = gaussian_kde(sub, bw_method=0.25)
    lo, hi = np.percentile(x, [0.5, 99.5])
    g = np.linspace(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo), 300)
    rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
    return go.Scatter(x=g, y=k(g), mode="lines", line=dict(color=color, width=2), fill="tozeroy", name=name,
                      fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{opacity})", hoverinfo="skip")


def badge(kind):
    """Which model a number comes from: blue = Forecast, purple = Simulator, grey = data."""
    lab, cls = {"fc": ("FORECAST", "b-fc"), "sim": ("SIMULATOR", "b-sim"), "data": ("DATA", "b-data")}[kind]
    return f"<span class='badge {cls}'>{lab}</span>"


def md(text, **kw):
    """st.markdown with dollar signs shown as dollar signs (not as math)."""
    st.markdown(text.replace("$", "&#36;" if kw.get("unsafe_allow_html") else "\\$"), **kw)


def layout(fig, h=320, title=None):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40 if title else 10, b=40), title=title,
                      plot_bgcolor="white", paper_bgcolor="white", font=dict(color="#0b0b0b", size=12),
                      legend=dict(orientation="h", yanchor="top", y=-0.12, x=0), hovermode="x unified")
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def api_key():
    k = os.getenv("OPENAI_API_KEY")
    if not k:
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
            k = os.getenv("OPENAI_API_KEY")
        except ImportError:
            pass
    if not k:
        try:
            k = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            k = None
    return k


# ------------------------------------------------------------------------------------------------ 1. forecast
SIG = {"below": ("▼", RED), "above": ("▲", GREEN), "within": ("●", MUTED)}
UNITS = {"eps": "", "operating_margin": " pts", "revenue_pct": " pts"}


def diff_txt(m, v):
    if m == "eps":
        return f"{'▲' if v >= 0 else '▼'} {'+' if v >= 0 else '−'}${abs(v):.2f}"
    return f"{'▲' if v >= 0 else '▼'} {'+' if v >= 0 else '−'}{abs(v):.1f} pts"


def range_bar(m, f, cons=None, actual=None):
    """Small picture: Delta's guidance range (grey band) vs the forecast's 80% range (blue bar) and the forecast (dot)."""
    lo = min(f["low"], f["p10"]) ; hi = max(f["high"], f["p90"])
    pad = (hi - lo) * 0.30
    fig = go.Figure()
    fig.add_shape(type="rect", x0=f["low"], x1=f["high"], y0=0.15, y1=0.85, fillcolor="#e4e3df", line_width=0, layer="below")
    fig.add_annotation(x=(f["low"] + f["high"]) / 2, y=1.12, text="Delta's guidance", showarrow=False, font=dict(size=10, color=TEXT2), yanchor="bottom")
    fig.add_trace(go.Scatter(x=[f["p10"], f["p90"]], y=[0.5, 0.5], mode="lines", line=dict(color=BLUE, width=7), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[f["forecast"]], y=[0.5], mode="markers", marker=dict(color="white", size=11, line=dict(color=BLUE, width=3)),
                             hovertemplate="forecast %{x}<extra></extra>"))
    # labels: ends of the 80% range beside the bar, forecast value above the circle
    fig.add_annotation(x=f["p10"], y=0.5, text=fmt(m, f["p10"]) + " ", showarrow=False, xanchor="right", font=dict(size=12, color=BLUE))
    fig.add_annotation(x=f["p90"], y=0.5, text=" " + fmt(m, f["p90"]), showarrow=False, xanchor="left", font=dict(size=12, color=BLUE))
    fig.add_annotation(x=f["forecast"], y=0.78, text=f"<b>{fmt(m, f['forecast'])}</b>", showarrow=False, yanchor="bottom",
                       font=dict(size=13, color=BLUE))
    if cons is not None:
        fig.add_trace(go.Scatter(x=[cons], y=[0.5], mode="markers", marker=dict(color=ORANGE, symbol="diamond", size=10),
                                 hovertemplate="Wall Street %{x}<extra></extra>"))
    if actual is not None:
        fig.add_trace(go.Scatter(x=[actual], y=[0.5], mode="markers", marker=dict(color="#0b0b0b", symbol="x", size=10),
                                 hovertemplate="actual %{x}<extra></extra>"))
    tick = (lambda v: f"${v:.2f}") if m == "eps" else (lambda v: f"{v:.0f}%")
    vals = [f["low"], f["high"]]
    fig.update_xaxes(range=[lo - pad, hi + pad], tickvals=vals, ticktext=[tick(v) for v in vals], showgrid=False, zeroline=False)
    fig.update_yaxes(visible=False, range=[0, 1.45])
    fig.update_layout(height=125, margin=dict(l=6, r=6, t=4, b=22), showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
    return fig


def chance_bar(f):
    b, a = f["below"], f["above"]
    i = max(100 - b - a, 0)
    seg = lambda w, c: f"<div style='width:{w}%;background:{c};height:8px'></div>" if w > 0.5 else ""
    return (f"<div style='display:flex;border-radius:4px;overflow:hidden;margin:2px 0 2px'>{seg(b, '#f2b8b7')}{seg(i, '#d6d5d0')}{seg(a, '#b7e3cf')}</div>"
            f"<div class='small'>below {b:.0f}% · inside {i:.0f}% · above {a:.0f}%</div>")


def controls():
    """Quarter and as-of date, shown at the top of the pages that use them; the choice carries across pages."""
    global q, d, P, qname, reported, known_actual
    qs = sorted(S_.dates)
    st.session_state.setdefault("q_sel", qs[-1])
    st.caption("Every number uses only data public on this date. Pick a past date to see what FlightPath said then.")
    c1, c2 = st.columns([1, 4])
    q = c1.selectbox("Quarter", qs, index=qs.index(st.session_state["q_sel"]), format_func=lambda x: f"{x[-2:]} {x[:4]}")
    if q != st.session_state["q_sel"]:
        st.session_state["q_sel"] = q
        st.session_state.pop("d_sel", None)
    saved = sorted(S_.dates[q])
    end = pd.Period(q, "Q").end_time.normalize() + pd.Timedelta(days=14)
    ds = list(pd.date_range(saved[0], max(saved[-1], min(end, pd.Timestamp.today().normalize()))))
    if st.session_state.get("d_sel") not in ds:
        st.session_state["d_sel"] = saved[-1]
    d = c2.select_slider("As of", options=ds, value=st.session_state["d_sel"], format_func=lambda x: f"{x:%d %b %Y}")
    st.session_state["d_sel"] = d
    if not S_.saved(q, d):                     # not in the snapshot: work it out from the raw data, if it's here
        ctx = live_ctx()
        if ctx is not None:
            S_.live = lambda qq, dd: build_entry(ctx, qq, dd)
            with st.spinner(f"Working out {d:%d %b %Y} from the raw data..."):
                S_.entry(q, d)
        else:
            before = [x for x in saved if x <= d]
            shown = before[-1] if before else saved[0]
            st.caption(f"{d:%d %b %Y} isn't saved in this copy of FlightPath, so the page shows {shown:%d %b %Y}, "
                       "the closest saved date before it.")
            d = shown
    P = pd.Period(q, "Q")
    qname = f"{q[-2:]} {q[:4]}"
    reported = S_.actuals.get(q)
    known_actual = reported if (reported and pd.Timestamp(reported["filed"]) <= d) else None


def page_card():
    controls()
    r = run(q, d)
    e, F, S, R = r["e"], r["F"], r["S"], r["R"]
    B = e["B"]
    st.title(f"Delta {qname}: forecast")
    md(f"{badge('fc')} As of {d:%A %d %B %Y}. **Forecast = Delta's own guidance, updated by the simulator for the news "
       f"since it was given** (fuel for EPS and margin; fuel + flight-tracking capacity for revenue).", unsafe_allow_html=True)
    with st.container(border=True):
        md("""**How the forecast works.** 
About ten days into each quarter, Delta publishes guidance: a range for EPS, operating margin and revenue growth.
Delta knows its own bookings, contracts and plans, so the middle of each range is the forecast's starting point. If
Delta updates its guidance mid-quarter, the forecast starts from the update.

From there, the forecast adds only **what has changed since Delta guided**. To measure that, the simulator runs the
quarter twice with identical assumptions: once with everything known today, once with only what was known on the
guidance date. The gap between the two runs is the effect of the news alone. For **EPS and margin**, the news is jet
fuel prices: they change fuel costs, and fares partly follow. For **revenue**, the change in how much Delta is flying,
measured from flight tracking, is added as well.

The **80% range** combines two uncertainties: how sure the simulator is about the size of the news adjustment, and how
far Delta's own guidance has typically missed in past quarters at the same point in the quarter. The actual result
has landed inside this range in about 8 of every 10 past quarters. It narrows once the quarter has ended, when all
the news is in.
""")
    if not F:
        st.warning("Delta had not given guidance for this quarter yet on this date - pick a later date.")
        return
    md("""
**What's being forecast.** Delta reports three headline numbers each quarter, and gives guidance on all three:
- **Adjusted EPS:** profit per share, excluding one-off items. The number the stock market reacts to most.
- **Operating margin:** the share of revenue left as operating profit, after fuel, wages and all other operating costs.
- **Total revenue growth:** revenue vs the same quarter last year.
""")
    cons = S_.consensus_as_of(q, d)
    cols = st.columns(3)
    for c, m in zip(cols, ["eps", "operating_margin", "revenue_pct"]):
        if m not in F:
            continue
        f = F[m]
        with c.container(border=True):
            extra = (" <span class='small' title='Revenue uses fuel + flight-tracking capacity news; adopted 7 Oct 2026 "
                     "after testing on past quarters, so it is being tested live.'>(newer method, testing live)</span>"
                     if m == "revenue_pct" else "")
            md(f"{badge('fc')}**{NICE[m]}**{extra}", unsafe_allow_html=True)
            md(f"<div style='display:flex;gap:30px;align-items:flex-end;margin-top:4px'>"
               f"<div><div class='small'>Forecast</div><div style='font-size:2rem;font-weight:600;color:{BLUE}'>{fmt(m, f['forecast'])}</div></div>"
               f"<div><div class='small'>Delta guidance midpoint</div><div style='font-size:1.5rem;color:{MUTED}'>{fmt(m, f['mid'])}</div></div></div>"
               f"<div class='small'>{diff_txt(m, f['forecast'] - f['mid'])} vs Delta guidance midpoint</div>",
               unsafe_allow_html=True)
            st.plotly_chart(range_bar(m, f, None, known_actual[m] if known_actual else None), width="stretch",
                            config={"displayModeBar": False})
            st.caption("Grey band: Delta's guidance · blue bar: forecast's 80% range · circle: forecast" +
                       (" · ✕ actual" if known_actual else ""))
            parts = {"below": f["below"], "inside": max(100 - f["below"] - f["above"], 0), "above": f["above"]}
            k = max(parts, key=parts.get)
            md(f"**{parts[k]:.0f}% chance Delta lands {k} its guidance range**")
            md(chance_bar(f), unsafe_allow_html=True)
            s_, txt = signal(f)
            arrow, col = SIG[s_]
            md(f"<span style='color:{col};font-weight:600'>{arrow} {txt}</span>", unsafe_allow_html=True)
            if known_actual:
                md(f"**Actual: {fmt(m, known_actual[m])}** (reported {pd.Timestamp(known_actual['filed']):%d %b %Y})")

    # Wall Street, as known on this date
    with st.container(border=True):
        md("##### Forecast vs Wall Street (EPS)")
        if cons:
            fe = F["eps"]
            gap = fe["forecast"] - cons["eps"]
            beat = (fe["runs"] > cons["eps"]).mean() * 100
            src = (f"I/B/E/S consensus on {d:%d %b %Y} ({cons['n']} analysts)" if cons["source"] == "ibes"
                   else "final consensus just before the report, from press coverage")
            rev = ""
            if cons.get("revenue_pct") is not None and "revenue_pct" in F:
                rev = (f" Revenue growth: Wall Street {cons['revenue_pct']:+.1f}%, our forecast {F['revenue_pct']['forecast']:+.1f}%.")
            md(f"**Wall Street expects ${cons['eps']:.2f}.** Our forecast is ${fe['forecast']:.2f}, "
               f"{abs(gap) * 100:.0f}¢ {'above' if gap >= 0 else 'below'}. Chance Delta beats the Street: **{beat:.0f}%**.{rev}  \n"
               f"<span class='small'>Source: {src}.</span>", unsafe_allow_html=True)
        else:
            why = (f"I/B/E/S data runs through {S_.meta.get('consensus_through', '-')}" if S_.cons_hist is not None
                   else "this copy shows Wall Street's figure only in the days before Delta reports")
            md(f"**Wall Street consensus:** not available for this date <span class='small'>({why}; the final figure "
               "appears in the days before Delta reports).</span>" if S_.cons_hist is not None else
               f"**Wall Street consensus:** not available for this date <span class='small'>({why}).</span>",
               unsafe_allow_html=True)

    left, right = st.columns([3, 2])
    with left:
        f = F["eps"]
        from scipy.stats import gaussian_kde
        x = np.asarray(f["runs"], float)
        kde = gaussian_kde(np.random.default_rng(0).choice(x, min(8000, len(x)), replace=False), bw_method=0.25)
        lo_, hi_ = np.percentile(x, [0.5, 99.5])
        g = np.linspace(lo_ - 0.1 * (hi_ - lo_), hi_ + 0.1 * (hi_ - lo_), 400)
        y = kde(g)
        fig = go.Figure()
        for msk, colr, name in [(g < f["low"], "rgba(227,73,72,0.28)", "below Delta's range"),
                                ((g >= f["low"]) & (g <= f["high"]), "rgba(163,162,156,0.30)", "inside"),
                                (g > f["high"], "rgba(27,175,122,0.28)", "above")]:
            if msk.any():
                fig.add_trace(go.Scatter(x=g[msk], y=y[msk], fill="tozeroy", mode="lines", line=dict(width=0),
                                         fillcolor=colr, name=name, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=g, y=y, mode="lines", line=dict(color=BLUE, width=2), showlegend=False, hoverinfo="skip"))
        top = y.max() * 1.12
        for v in (f["low"], f["high"]):
            fig.add_vline(x=v, line_color=TEXT2, line_width=1.5)
        fig.add_annotation(x=f["low"], y=top * 1.06, text=f"${f['low']:.2f} ", showarrow=False, xanchor="right", font=dict(size=11, color=TEXT2))
        fig.add_annotation(x=f["high"], y=top * 1.06, text=f" ${f['high']:.2f}", showarrow=False, xanchor="left", font=dict(size=11, color=TEXT2))
        fig.add_annotation(x=(f["low"] + f["high"]) / 2, y=top * 1.06, text="Delta guidance", showarrow=False,
                           font=dict(size=11, color=TEXT2))
        marks = [(f["forecast"], f"Forecast ${f['forecast']:.2f}", BLUE, "triangle-down")]
        if cons:
            marks.append((cons["eps"], f"Wall Street ${cons['eps']:.2f}", ORANGE, "diamond"))
        if known_actual:
            marks.append((known_actual["eps"], f"Actual ${known_actual['eps']:.2f}", "#0b0b0b", "x"))
        right_side = f["forecast"] < (f["low"] + f["high"]) / 2
        for i, (v, lab, colr, sym) in enumerate(marks):
            yy = top * (1.0 - 0.1 * i)
            fig.add_trace(go.Scatter(x=[v, v], y=[0, yy], mode="lines", line=dict(color=colr, width=1, dash="dot"),
                                     showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=[v], y=[yy], mode="markers+text", text=[f"  {lab}  "],
                                     textposition="middle left" if not right_side else "middle left",
                                     marker=dict(color=colr, size=10, symbol=sym),
                                     textfont=dict(color=colr, size=11), showlegend=False, hoverinfo="skip"))
        fig.update_yaxes(visible=False, range=[0, top * 1.12])
        fig.update_xaxes(title_text="Adjusted EPS ($)", tickprefix="$")
        fig = layout(fig, 360, "Range of likely EPS outcomes")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")
        md("<span class='small'>Shading: <span style='color:#c94a49'>red</span> = below Delta's "
           "guidance range, grey = inside, <span style='color:#159a6b'>green</span> = above.</span>", unsafe_allow_html=True)
    with right:
        gd = pd.Timestamp(e["news"]["eps"]["gd"]) - pd.Timedelta(days=1)
        jet = B["jet"]
        px = lambda t: float(jet[:t].iloc[-5:].mean())
        md("**Observed since Delta guided**")
        obs = [("Delta gave guidance", f"{gd:%d %b %Y}"),
               ("Market jet fuel", f"${px(gd):.2f} → ${px(d):.2f}/gal ({(px(d) / px(gd) - 1) * 100:+.0f}%)")]
        if "revenue_pct" in e["news"]:
            ly = B["cap"]["consolidated_q3_2025_reported"]
            then = (e["news"]["revenue_pct"]["bg_cap"] / ly - 1) * 100
            obs.append(("Delta's flying vs last year", f"{then:+.1f}% → {B['cap']['consolidated_yoy_pct']:+.1f}% now"))
        md("\n".join(f"- {a}: **{b}**" for a, b in obs))
        md(f"{badge('sim')}**Estimated by the simulator** (from the changes above)", unsafe_allow_html=True)
        fs = fuel_split(e, R, S)
        tip = lambda h: f" <span title=\"{h}\" style='cursor:help;color:{MUTED}'>ⓘ</span>"
        rows = [(f"Higher fuel cost: **{fmt('eps', fs['fuel cost'], True)} EPS**",
                 "Delta's fuel price × gallons burned, after profit sharing and tax"),
                (f"Fares that follow fuel: **{fmt('eps', fs['fares (follow fuel)'], True)} EPS**",
                 "Airlines raise fares when fuel rises; the link is measured on Delta's history"),
                (f"Total EPS effect: **{fmt('eps', F['eps']['news'], True)}** (margin {F['operating_margin']['news']:+.1f} pts)",
                 "The fuel news in total; the same news as a share of revenue moves the margin")]
        if "revenue_pct" in F:
            C = dict(B, as_of=e["news"]["revenue_pct"]["gd"])
            from flightpath.simulation.quarter import evaluate as _ev
            fuel_rev = float(np.median(S.revenue_pct.values - _ev(C, R).revenue_pct.values))
            tot = F["revenue_pct"]["news"]
            rows.append((f"Revenue effect: fuel {fuel_rev:+.1f} pts, capacity {tot - fuel_rev:+.1f} pts → **{tot:+.1f} pts**",
                         "Fuel moves revenue through fares; capacity through the seats flown"))
        md("\n".join(f"- {a}{tip(h)}" for a, h in rows), unsafe_allow_html=True)

        # what would change the call: fuel sensitivity, simulator run live for this quarter and date
        md(f"{badge('sim')}**What would change the call**", unsafe_allow_html=True)
        w = what_changes(q, d)
        lines = [(f"Fuel already priced: **{w['priced_pct']:.0f}%** of the quarter ({w['known']} of {w['known'] + w['left']} trading days)",
                  f"Delta's fuel cost follows the market price with about a {w['lag']}-day delay, so days already "
                  f"priced can no longer change.")]
        if w["left"] == 0:
            lines.append(("All of the quarter's fuel is priced, so fuel prices can no longer change the call",
                          "Only news other than fuel (or Delta's own update) could move it now."))
        else:
            lines.append((f"Jet fuel 10% higher for the rest of the quarter: **{fmt('eps', w['up10'], True)} EPS**; "
                          f"10% lower: **{fmt('eps', w['down10'], True)} EPS**",
                          "The simulator rerun with the fuel lever moved, everything else unchanged; fares partly follow fuel."))
            mid = F["eps"]["mid"]
            if abs(F["eps"]["forecast"] - mid) < 0.01:
                lines.append(("The forecast is already at Delta's guidance midpoint", ""))
            elif w["breakeven"] is None:
                lines.append((f"No fuel move between −50% and +100% would bring the forecast back to Delta's midpoint "
                              f"({fmt('eps', mid)})", "The simulator's fuel lever range is −50% to +100%."))
            else:
                lines.append((f"To get back to Delta's midpoint ({fmt('eps', mid)}): jet fuel would need to average about "
                              f"**${w['breakeven']:.2f}/gal** for the rest of the quarter (last price ${w['last']:.2f})",
                              "Found by rerunning the simulator at different fuel prices until the forecast reaches the midpoint."))
        md("\n".join(f"- {a}{tip(h) if h else ''}" for a, h in lines), unsafe_allow_html=True)

    # how the forecast moved, date by date up to the chosen date
    path = forecast_path(q, d)
    if len(path):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=path.date, y=path.high, mode="lines", line=dict(width=0, shape="hv"), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=path.date, y=path.low, mode="lines", line=dict(width=0, shape="hv"), fill="tonexty",
                                 fillcolor="rgba(163,162,156,0.30)", name="Delta's guidance", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=path.date, y=path.forecast, mode="lines+markers", line=dict(color=BLUE, width=2.5),
                                 marker=dict(size=4), name="Forecast", hovertemplate="%{x|%d %b %Y}: forecast $%{y:.2f}<extra></extra>"))
        if path.street.notna().any():
            fig.add_trace(go.Scatter(x=path.date, y=path.street, mode="lines+markers", line=dict(color=ORANGE, width=2, dash="dot"),
                                     marker=dict(size=4, symbol="diamond"), name="Wall Street",
                                     hovertemplate="%{x|%d %b %Y}: Wall Street $%{y:.2f}<extra></extra>"))
        if known_actual:
            fd = pd.Timestamp(known_actual["filed"])
            fig.add_trace(go.Scatter(x=[fd], y=[known_actual["eps"]], mode="markers+text", text=[f"Actual ${known_actual['eps']:.2f}"],
                                     textposition="middle left", marker=dict(color="#0b0b0b", symbol="x", size=11),
                                     name="Actual", hovertemplate="reported %{x|%d %b %Y}: $%{y:.2f}<extra></extra>"))
        fig.update_yaxes(tickprefix="$", tickformat=".2f")
        x_end = max(path.date.max(), pd.Timestamp(known_actual["filed"]) if known_actual else path.date.max())
        fig.update_xaxes(tickformat="%d %b", range=[path.date.min() - pd.Timedelta(days=3), x_end + pd.Timedelta(days=3)])
        fig = layout(fig, 340, "How the forecast moved (EPS)")
        fig.update_layout(legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch")
        cap = ("One point per day, each using only the data public on that date. Grey band: Delta's guidance range "
               "(it moves if Delta updates its guidance).")
        s_ = path.street
        if s_.notna().any() and s_[s_.first_valid_index():].isna().any():
            cap += (f" Gaps in the Wall Street line: the I/B/E/S data runs through {S_.meta.get('consensus_through', '-')}; "
                    "the final figure appears in the days before Delta reports.")
        st.caption(cap)

    st.subheader("Forecast summary")
    sc = summary_table(F, cons)
    st.dataframe(sc, width="stretch", hide_index=True)
    st.download_button("Download CSV", sc.to_csv(index=False).encode("utf-8"),
                       file_name=f"flightpath_{q}_{d:%Y-%m-%d}.csv", mime="text/csv")

    # past accuracy at the same point in the quarter (daily forecast path)
    when, acc = accuracy_at(q, d)
    n_all = S_.track[S_.track.metric == "eps"].quarter.nunique()
    st.subheader("How accurate has the forecast been in past quarters?")
    md(f"For each past quarter (Q3 2023 - Q2 2026), we took what the forecast said **{when}**, using only data public "
       f"that day, and compared it with what Delta reported at the end of the quarter - the same point in the quarter "
       f"as the date you picked.")
    md("- **Average miss:** how far off each one was from what Delta reported, on average - in cents per share for EPS, "
       "in percentage points (pts) for operating margin and revenue growth. **Smaller is better.** The same is shown for "
       "Delta's guidance midpoint and for Wall Street's consensus on the same date, for comparison.\n"
       "- **Actual inside the 80% range:** in how many of those quarters Delta's reported result landed inside the "
       "forecast's 80% range. About 8 in 10 is the target.")
    if not len(acc):
        st.caption(f"By {when.replace('on ', '', 1)}, Delta had already reported in every past quarter, so there is no "
                   "forecast left to score at this point. Pick an earlier date, or see the Track record page.")
    else:
        st.dataframe(acc, width="stretch", hide_index=True)
    if len(acc) and not acc["Actual inside the 80% range"].str.endswith(f"of {n_all} quarters").iloc[0]:
        st.caption("Quarters where Delta had already reported by that point are left out - by then it is no longer a forecast.")


def _fc_row(q, x, m="eps"):
    """The forecast for one date, computed as on the Forecast page (same runs, same seed)."""
    e = S_.entry(q, x)
    if not e["news"]:
        return None
    R, S, _ = simulate(e, N, Levers())
    f = forecast(e, R, S).get(m)
    return {"date": pd.Timestamp(x), **{k: f[k] for k in ("forecast", "low", "high", "p10", "p90", "simulator")}} if f else None


@st.cache_data(show_spinner=False, max_entries=64)
def forecast_path(q, d):
    """EPS forecast on every day of the quarter up to `d`, each using only what was public that day - read from
    forecast_path.csv (computed at build time, exactly as the page computes it); `d` itself is added if it isn't saved."""
    d = pd.Timestamp(d)
    if S_.paths is not None:
        p = S_.paths[(S_.paths.quarter == q) & (S_.paths.metric == "eps") & (S_.paths.date <= d)]
        p = p[["date", "forecast", "low", "high", "p10", "p90"]].copy()
    else:                                                    # older snapshot without the file: compute every day
        p = pd.DataFrame([r for x in sorted(S_.dates[q]) if x <= d and (r := _fc_row(q, x))])
    if not len(p) or d not in set(p.date):
        r = _fc_row(q, d)
        if r:
            p = pd.concat([p, pd.DataFrame([r])], ignore_index=True)
    if not len(p):
        return p
    p = p.sort_values("date").reset_index(drop=True)
    p["street"] = [(c["eps"] if (c := S_.consensus_as_of(q, x)) else np.nan) for x in p.date]
    return p


@st.cache_data(show_spinner=False, max_entries=64)
def what_changes(q, d):
    """Fuel sensitivity of the EPS forecast on this date: +/-10% on the unpriced days, and the fuel price that would
    bring the forecast back to Delta's guidance midpoint (secant search on the simulator, same draws throughout)."""
    e = S_.entry(q, d)
    B = e["B"]
    known, left = int(B["market_days_known"]), int(B["market_days_left"])
    last = float(B["jet"][:pd.Timestamp(d)].iloc[-1])
    out = {"known": known, "left": left, "lag": int(B["lag"]), "last": last,
           "priced_pct": 100 * known / max(known + left, 1), "up10": 0.0, "down10": 0.0, "breakeven": None}
    if left == 0 or not e["news"]:
        return out
    def eps(p):
        R, S, _ = simulate(e, N, Levers(fuel_pct=p))
        return forecast(e, R, S)["eps"]
    base = eps(0.0)
    f0, mid = base["forecast"], base["mid"]
    up, dn = eps(10.0)["forecast"], eps(-10.0)["forecast"]
    out["up10"], out["down10"] = up - f0, dn - f0
    slope = (up - dn) / 20
    if abs(slope) > 1e-6:
        p = (mid - f0) / slope
        for _ in range(2):                                  # refine: the link is close to, not exactly, linear
            if not -50 <= p <= 100:
                break
            p += (mid - eps(p)["forecast"]) / slope
        if -50 <= p <= 100:
            out["breakeven"] = last * (1 + p / 100)
    return out


def same_point(q, d):
    """The same point in the quarter as date `d` of quarter `q`, as a function of another quarter: the same day of the
    quarter (day 25 = start + 25 days), or, after the quarter has ended, the same number of days after its end."""
    Pq = pd.Period(q, "Q")
    end = Pq.end_time.normalize()
    if d > end:
        j = (d - end).days
        return f"{j} day{'s' if j > 1 else ''} after the quarter ended", lambda qq: pd.Period(qq, "Q").end_time.normalize() + pd.Timedelta(days=j)
    k = (d - Pq.start_time).days
    return f"on day {k} of the quarter", lambda qq: pd.Period(qq, "Q").start_time + pd.Timedelta(days=k)


def past_at(m, at) -> pd.DataFrame:
    """Every past quarter's forecast, 80% range, guidance and Wall Street at the date at(quarter), with the actual -
    quarters where Delta had already reported by that date are left out."""
    act = S_.track[S_.track.metric == m].groupby("quarter").actual.first()
    rows = []
    for qq, a_ in act.items():
        x = at(qq)
        if qq in S_.actuals and x >= pd.Timestamp(S_.actuals[qq]["filed"]):
            continue                                  # Delta had already reported by then: not a forecast any more
        r = S_.paths[(S_.paths.quarter == qq) & (S_.paths.metric == m) & (S_.paths.date == x)]
        if not len(r):
            continue
        r = r.iloc[0]
        c = np.nan
        if S_.cons_hist is not None and m in S_.cons_hist.columns:
            cr = S_.cons_hist[(S_.cons_hist.quarter == qq) & (S_.cons_hist.date == x)]
            c = float(cr.iloc[0][m]) if len(cr) and pd.notna(cr.iloc[0][m]) else np.nan
        rows.append({"quarter": qq, "actual": a_, "forecast": r.forecast, "guidance_low": r.low, "guidance_high": r.high,
                     "guidance_mid": (r.low + r.high) / 2, "p10": r.p10, "p90": r.p90, "street": c,
                     "simulator": r.get("simulator", np.nan)})
    cols = ["quarter", "actual", "forecast", "guidance_low", "guidance_high", "guidance_mid", "p10", "p90", "street", "simulator"]
    return pd.DataFrame(rows, columns=cols).set_index("quarter")


@st.cache_data(show_spinner=False)
def accuracy_at(q, d):
    """Accuracy in past quarters (Q3 2023 - Q2 2026) at the same point in the quarter as `d`."""
    when, at = same_point(q, pd.Timestamp(d))
    rows = []
    for m in ["eps", "operating_margin", "revenue_pct"]:
        t = past_at(m, at)
        if not len(t):
            continue
        miss = lambda x: (x - t.actual[x.index]).abs().mean()
        f_ = lambda v: f"{v * 100:.0f}¢" if m == "eps" else f"{v:.1f} pts"
        inside = int(((t.actual >= t.p10) & (t.actual <= t.p90)).sum())
        rows.append({"Result": NICE[m], "Forecast: average miss": f_(miss(t.forecast)),
                     "Delta guidance midpoint: average miss": f_(miss(t.guidance_mid)),
                     "Wall Street (same date): average miss": f_(miss(t.street.dropna())) if t.street.notna().any() else "not available",
                     "Actual inside the 80% range": f"{inside} of {len(t)} quarters"})
    return when, pd.DataFrame(rows)


def summary_table(F, cons):
    """Downloadable scorecard: one row per headline number for the chosen quarter and date; actuals once Delta has reported."""
    side = lambda x, g: "below" if x < g[0] else "above" if x > g[1] else "inside"
    rows = []
    for m in ["eps", "operating_margin", "revenue_pct"]:
        if m not in F:
            continue
        f = F[m]
        g = (f["low"], f["high"])
        inside = max(100 - f["below"] - f["above"], 0)
        ws = cons.get(m) if cons else None
        act = reported[m] if reported else None
        rows.append({"Quarter": qname, "As of": f"{d:%Y-%m-%d}", "Result": NICE[m],
                     "Delta's guidance": f"{fmt(m, g[0])} to {fmt(m, g[1])}",
                     "Forecast": fmt(m, f["forecast"]),
                     "80% range": f"{fmt(m, f['p10'])} to {fmt(m, f['p90'])}",
                     "Chance below / inside / above": f"{f['below']:.0f}% / {inside:.0f}% / {f['above']:.0f}%",
                     "Wall Street (as of date)": fmt(m, ws),
                     "Actual (end of quarter)": fmt(m, act) if act is not None else "not reported yet",
                     "In range?": ("yes" if f["p10"] <= act <= f["p90"] else "no") if act is not None else "-",
                     "Call vs guidance right?": ("yes" if side(act, g) == side(f["forecast"], g) else "no") if act is not None else "-"})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ 2. scenarios
LEVER_UI = [("fuel_pct", "Market jet fuel (%)", -50.0, 100.0, 1.0),
            ("capacity_pct", "Delta's capacity (%)", -30.0, 15.0, 0.5),
            ("demand_pts", "Industry demand (pts)", -30.0, 15.0, 0.5),
            ("fare_pct", "Fares (%)", -30.0, 30.0, 0.5),
            ("nonfuel_pct", "Non-fuel costs (%)", -15.0, 25.0, 0.5),
            ("outage", "Outage size (1 = July 2024)", 0.0, 5.0, 0.1)]
LEVER_NAME = {k: lab.split(" (")[0] for k, lab, *_ in LEVER_UI}

# the operating chain in plain words: (column, label, kind)
CHAIN_UI = [("capacity_pct", "How much Delta flies (seat-miles, vs last year)", "pct"),
            ("demand_pct", "Industry demand (air travel growth vs last year)", "pct"),
            ("load_factor_chg_pts", "How full the planes are (change vs last year)", "pts"),
            ("yield_pct", "Average fare per mile (vs last year)", "pct"),
            ("passenger_revenue_pct", "Ticket revenue (vs last year)", "pct"),
            ("revenue_pct", "Total revenue (vs last year)", "pct"),
            ("fuel_price", "Delta's fuel price ($ per gallon)", "usd2"),
            ("fuel_cost", "Fuel cost ($ millions)", "musd"),
            ("unit_cost_pct", "Other costs per seat-mile (vs last year)", "pct"),
            ("nonfuel_cost", "Other costs ($ millions)", "musd"),
            ("profit_sharing", "Profit sharing ($ millions)", "musd"),
            ("operating_margin", "Operating margin", "pct_level"),
            ("eps", "Adjusted EPS ($ per share)", "usd2")]
BRIDGE_NAME = {"revenue": "Revenue", "fuel cost": "Fuel cost", "non-fuel costs": "Other costs",
               "MRO cost": "Maintenance work<br>for other airlines", "profit sharing": "Profit sharing"}


def _cell(kind, v, change=False):
    if kind in ("pct", "pct_level"):
        return f"{v:+.1f}%" if (change or kind == "pct") else f"{v:.1f}%"
    if kind == "pts":
        return f"{v:+.1f} pts"
    if kind == "usd2":
        return f"{'+' if change and v >= 0 else '-' if v < 0 else ''}${abs(v):.2f}"
    return f"{'+' if change and v >= 0 else '-' if v < 0 else ''}${abs(v):,.0f}M"


MOVE = {"fuel_pct": (10.0, "+10%"), "capacity_pct": (1.0, "+1%"), "demand_pts": (1.0, "+1 pt"), "fare_pct": (1.0, "+1%"),
        "nonfuel_pct": (1.0, "+1%"), "outage": (0.5, "half of July 2024")}


@st.cache_data(show_spinner="Working out what moves EPS most...", max_entries=32)
def lever_table(q, d, whatif):
    """For each slider on its own, from the date's base case: the effect of a typical move on the forecast EPS, and the
    setting that would bring the forecast to Delta's guidance midpoint and to Wall Street's consensus (secant search on
    the simulator, same draws throughout, within the slider's range)."""
    e = S_.entry(q, d)
    if not e["news"] or "eps" not in e["news"]:
        return None
    B = e["B"]
    R = draw(B, N, 0, "joint")
    modes = dict(rest_of_quarter=not whatif, fuel_hindsight=whatif)
    fc = lambda k, x: forecast(e, R, evaluate(B, R, Levers(**{k: x}, **modes)) if x else evaluate(B, R))["eps"]
    base = fc("fuel_pct", 0.0)
    f0, mid = base["forecast"], base["mid"]
    cons = S_.consensus_as_of(q, d)
    targets = {"mid": mid} | ({"street": cons["eps"]} if cons else {})
    out = []
    for k, lab, lo, hi, _ in LEVER_UI:
        step, step_lab = MOVE[k]
        eff = fc(k, step)["forecast"] - f0
        row = {"key": k, "Slider": lab.split(" (")[0], "Typical move": step_lab, "effect": eff}
        slope = eff / step
        for t, target in targets.items():
            if abs(target - f0) < 0.005:
                row[t] = ("already there", None)
                continue
            if abs(slope) < 1e-6:
                row[t] = ("no effect", None)
                continue
            x = (target - f0) / slope
            for _ in range(3):                               # refine: the links are close to, not exactly, linear
                if not lo <= x <= hi:
                    break
                fx = fc(k, x)["forecast"]
                if abs(fx - target) < 0.002:
                    break
                x += (target - fx) / slope
            row[t] = ((x, None) if lo <= x <= hi else ("out of range", None))
        out.append(row)
    jet_last = float(B["jet"][:pd.Timestamp(d)].iloc[-1])
    return {"rows": out, "f0": f0, "mid": mid, "street": cons["eps"] if cons else None, "jet_last": jet_last}


def page_scenarios():
    controls()
    st.title(f"Delta {qname}: simulator")
    md(f"{badge('sim')} As of {d:%d %b %Y}. The simulator is a working model of Delta's business. Change one thing, like "
       f"fuel prices or flights flown, and it follows the effect through the business: how full the planes are, ticket "
       f"prices, revenue, costs, and finally earnings. Each connection is measured from Delta's own history. It runs every "
       f"quarter 50,000 times to show a range of likely outcomes, not just one number.", unsafe_allow_html=True)
    for k, *_ in LEVER_UI:
        st.session_state.setdefault(f"lv_{k}", 0.0)
    if "pending" in st.session_state:                      # apply an approved AI proposal / library event
        for k, v in st.session_state.pop("pending").items():
            st.session_state[f"lv_{k}"] = float(v)

    with st.container(border=True):
        md("##### How to use")
        md("""
1. Pick a quarter and date at the top. The simulator starts from everything that was public on that date.
2. Make a change, in one of three ways: type a news headline in the event box and let AI suggest settings; click one of Delta's past events; or move the sliders yourself.
3. Read the result: the boxes show the new EPS, margin and revenue growth; the charts below show how the change flows through the business.

**When a change starts.** Every slider changes the quarter from the date you picked to its end. What has already happened - the flying already done and the fuel already bought - stays as it was. For example, on 5 August, Delta's capacity −5% means Delta flies 5% less from 5 August to the end of September. To see what would have happened if the change had lasted the whole quarter, turn on "What if it had been different all quarter" under Modes. After a quarter has ended there are no days left to change, so the sliders only have an effect with that switch on.
""")
        c1, c2 = st.columns(2)
        with c1:
            md("""
**What the sliders mean**
- **Market jet fuel:** the jet fuel price, % above or below today's.
- **Delta's capacity:** how much more or less Delta flies (seats × miles), %.
- **Industry demand:** how much faster or slower air travel grows across the industry, in percentage points.
- **Fares:** a change in ticket prices, %. Whether it comes on top of the fare change the simulator already expects from fuel and demand, or replaces it, is set by the Extra / total mode.
- **Non-fuel costs:** wages, airports, maintenance and other costs, %.
- **Outage size:** a disruption like Delta's July 2024 IT outage (1 = the same size, 0.5 = half).
""")
        with c2:
            md("""
**Modes**
- **Extra / total** (fares slider): whether your fare change comes on top of what fuel and demand already do to fares, or replaces it.
- **Isolation:** shows only the direct effect of a change, with the links between parts of the business switched off - for example, fares no longer follow fuel and planes don't fill up differently. Comparing it with the normal result shows how much comes from those links.
- **What if it had been different all quarter:** applies every slider to the whole quarter, including the days already flown and the fuel already bought.
""")

    with st.container(border=True):
        md("**Event box** - describe a news event; AI proposes lever settings from Delta's documented past events. "
           "Nothing changes until you apply it.")
        own_key = api_key()
        if not own_key:                       # hosted without a key: each visitor uses their own, for their session only
            own_key = st.text_input("Your OpenAI API key (needed only for the event box)", type="password",
                                    help="Used only for your requests, in your session. It is not saved, logged or "
                                         "shown to anyone. Everything else in the app works without a key.")
        c1, c2 = st.columns([4, 1])
        headline = c1.text_input("Headline", placeholder="e.g. Hurricane forces Delta to cancel 1,500 flights at Atlanta over two days",
                                 label_visibility="collapsed")
        go_ai = c2.button("Interpret with AI", width="stretch", disabled=not headline)
        if go_ai:
            key = own_key
            if not key:
                st.error("Enter an OpenAI API key above to use the event box. The documented events below work without one.")
            else:
                with st.spinner("Reading the headline..."):
                    try:
                        st.session_state["proposal"] = {"headline": headline, **interpret_event(headline, S_.entry(q, d), S_.events, key)}
                    except Exception as ex:
                        msg = str(ex).replace(key, "[key]")[:200]        # never echo a key back
                        st.error(f"The AI call failed: {type(ex).__name__}: {msg}")
        p = st.session_state.get("proposal")
        if p:
            md(f"**Proposal for:** _{p['headline']}_ - confidence **{p['confidence']}**")
            moved = {k: v for k, v in p["levers"].items() if abs(v) > 1e-9}
            md("Sliders: " + (", ".join(f"{LEVER_NAME.get(k, k)} {v:+g}" for k, v in moved.items()) or "none (no measurable effect)"))
            md(f"Reasoning: {p['reasoning']}  \nBasis: {p['basis']}  \nCaveats: {p['caveats']}")
            a1, a2, _ = st.columns([1, 1, 3])
            if a1.button("Apply to the levers"):
                st.session_state["pending"] = {k: 0.0 for k, *_ in LEVER_UI} | p["levers"]
                st.session_state.pop("proposal")
                st.rerun()
            if a2.button("Discard"):
                st.session_state.pop("proposal")
                st.rerun()
        md("<span class='small'>Or apply a documented event directly:</span>", unsafe_allow_html=True)
        bc = st.columns(len(S_.events["events"]))
        for b, ev in zip(bc, S_.events["events"]):
            if b.button(ev["name"].split(",")[0].split(" (")[0], help=f"{ev['name']}. {ev['measured']}".replace("$", "\\$"), width="stretch"):
                st.session_state["pending"] = {k: 0.0 for k, *_ in LEVER_UI} | ev["levers"]
                st.rerun()

    # where to start: what moves EPS most right now (base case of this date, one slider at a time)
    whatif_now = bool(st.session_state.get("whatif", False))
    st.subheader("Where to start: what moves EPS most right now")
    ended = S_.entry(q, d)["B"]["rest_share"]["flying"] <= 0 and not whatif_now
    LT = None if ended else lever_table(q, d, whatif_now)
    if ended:
        st.info("This quarter has ended on the chosen date, so no slider can change it any more. Turn on \"What if it had "
                "been different all quarter\" under Modes to see what each slider would have done.")
    elif LT is None:
        st.info("Delta had not given guidance for this quarter yet on this date - pick a later date.")
    else:
        timing = ("over the whole quarter (the what-if switch is on)" if whatif_now else
                  "from the date you picked to the end of the quarter, like the sliders")
        md(f"Before you change anything, this table shows how sensitive the forecast is to each slider on this date. "
           f"Each row moves one slider on its own, starting from the forecast with no changes (EPS **${LT['f0']:.2f}**), "
           f"{timing}. Rows are sorted by how much they move EPS.")
        bl = [f"- **Typical move → effect on EPS:** what a change of that size does to the forecast. The sizes differ by "
              f"slider, so compare rows with that in mind; a move the same size the other way has about the opposite effect.",
              f"- **To reach Delta's midpoint (${LT['mid']:.2f}):** the setting that would bring the forecast to the middle "
              f"of Delta's guidance - how far things would have to move for Delta to hit its own number."]
        if LT["street"] is not None:
            bl.append(f"- **To reach Wall Street (${LT['street']:.2f}):** the setting that would make Wall Street's "
                      f"consensus come true - what you would have to believe for the Street to be right.")
        bl.append("- *Out of range* means not even the slider's full range gets there; *no effect* means the slider can't "
                  "move EPS on this date.")
        md("\n".join(bl))

        def setting(k, v):
            if isinstance(v[0], str):
                return v[0]
            x = v[0]
            if k == "fuel_pct":
                return f"{x:+.0f}% (about ${LT['jet_last'] * (1 + x / 100):.2f}/gal)"
            if k == "demand_pts":
                return f"{x:+.1f} pts"
            if k == "outage":
                return f"{x:.1f}× July 2024"
            return f"{x:+.1f}%"
        rows = []
        for r_ in sorted(LT["rows"], key=lambda z: -abs(z["effect"])):
            row = {"Slider": r_["Slider"], "Typical move": r_["Typical move"],
                   "Effect on EPS": f"{'+' if r_['effect'] >= 0 else '−'}{abs(r_['effect']) * 100:.0f}¢",
                   f"To reach Delta's midpoint (${LT['mid']:.2f})": setting(r_["key"], r_["mid"])}
            if LT["street"] is not None:
                row[f"To reach Wall Street (${LT['street']:.2f})"] = setting(r_["key"], r_["street"])
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    sh_ = S_.entry(q, d)["B"]["rest_share"]
    lc, rc = st.columns([1, 2])
    with lc:
        md("**Levers**")
        for k, lab, lo, hi, step in LEVER_UI:
            st.slider(lab, lo, hi, step=step, key=f"lv_{k}")
        with st.expander("Modes"):
            fare_mode = st.radio("Fares slider", ["extra", "total"], horizontal=True,
                                 help="**Extra:** your % is added on top of the fare change the simulator already expects "
                                      "from fuel and demand.  \n**Total:** fares end up exactly your % above the base case; "
                                      "the fare change the simulator would otherwise expect from fuel and demand is replaced "
                                      "by yours.")
            isolation = st.checkbox("Isolation", help="Links switched off: shows only the direct effect of your change "
                                                      "(see How to use).")
            whatif = st.checkbox("What if it had been different all quarter", key="whatif",
                                 help="Applies every slider to the whole quarter, including the days already flown and "
                                      "the fuel already bought.")
        if st.button("Reset levers"):
            st.session_state["pending"] = {k: 0.0 for k, *_ in LEVER_UI}
            st.rerun()
        if sh_["flying"] <= 0 and not whatif:
            st.info("This quarter has ended on the chosen date, so there are no days left to change. Turn on "
                    "\"What if it had been different all quarter\" under Modes to see what a change would have done.")
        elif not whatif:
            st.caption(f"Still to come on this date: {sh_['flying'] * 100:.0f}% of the quarter's flying and "
                       f"{sh_['passengers'] * 100:.0f}% of its passengers (measured on the same days a year earlier). "
                       f"The sliders change this part.")
    lv_items = tuple((k, float(st.session_state[f"lv_{k}"])) for k, *_ in LEVER_UI if st.session_state[f"lv_{k}"] != 0)
    modes = (("fare_mode", fare_mode), ("isolation", isolation), ("fuel_hindsight", whatif), ("rest_of_quarter", not whatif))
    r = run(q, d, lv_items, modes)
    S, S0, F, F0, B = r["S"], r["S0"], r["F"], r["F0"], r["e"]["B"]
    moved_any = bool(lv_items)
    with rc:
        md("Two views of the same quarter. **The simulator's quarter** is the simulator on its own, built only from "
           "public data, without Delta's guidance. In past quarters it has been about twice as far off as the forecast, "
           "so it is not used as the prediction - it is there to show how the business responds. **The forecast** starts "
           "from Delta's guidance and adds the news since then plus your change; that is the number to read. Your change "
           "moves both by about the same amount.")
        md(f"{badge('sim')}**The simulator's quarter** (public data only)", unsafe_allow_html=True)
        cs = st.columns(3)
        for c, m in zip(cs, ["eps", "operating_margin", "revenue_pct"]):
            v, v0 = float(np.median(S[m])), float(np.median(S0[m]))
            c.metric(NICE[m], fmt(m, v), delta=f"{fmt(m, v - v0, True)} from your change" if moved_any else None)
        md(f"{badge('fc')}**The forecast** (Delta's guidance + news + your change)", unsafe_allow_html=True)
        if F:
            cs = st.columns(3)
            for c, m in zip(cs, ["eps", "operating_margin", "revenue_pct"]):
                if m not in F:
                    c.caption(f"{NICE[m]}: no guidance in force")
                    continue
                f, f0 = F[m], F0[m]
                c.metric(f"{NICE[m]}", fmt(m, f["forecast"]),
                         delta=f"{fmt(m, f['forecast'] - f0['forecast'], True)} from your change" if moved_any else None,
                         delta_color="normal")
                ins = lambda x: max(100 - x["below"] - x["above"], 0)
                seg = lambda w, col: f"<div style='width:{w}%;background:{col};height:7px'></div>" if w > 0.5 else ""
                bar = (f"<div style='display:flex;border-radius:4px;overflow:hidden;margin:3px 0'>{seg(f['below'], '#f2b8b7')}"
                       f"{seg(ins(f), '#d6d5d0')}{seg(f['above'], '#b7e3cf')}</div>")
                txt = (f"<div class='small'>Guidance {fmt(m, f['low'])} to {fmt(m, f['high'])}</div>{bar}"
                       f"<div class='small'>{f['below']:.0f}% below · {ins(f):.0f}% in · {f['above']:.0f}% above</div>")
                if moved_any:
                    txt += f"<div class='small' style='color:{MUTED}'>was {f0['below']:.0f}% · {ins(f0):.0f}% · {f0['above']:.0f}%</div>"
                c.markdown(txt.replace("$", "&#36;"), unsafe_allow_html=True)
        else:
            st.info("No guidance in force on this date: showing the simulator alone.")

        # range of likely EPS outcomes
        st.write("")
        md("**Range of likely EPS outcomes**")
        md("<span class='small'>Each of the 50,000 runs ends at a slightly different EPS. Grey shows where the forecast's "
           "runs land without your change, blue with it. The shaded band is Delta's guidance range.</span>",
           unsafe_allow_html=True)
        fig = go.Figure()
        x0 = F0["eps"]["runs"] if F else S0.eps.values
        x1 = F["eps"]["runs"] if F else S.eps.values
        t0 = density(x0, "#a3a29c", "Without your change")
        top = float(np.max(t0.y))
        if moved_any:
            t1 = density(x1, BLUE, "With your change")
            top = max(top, float(np.max(t1.y)))
        if F:                                               # Delta's guidance: shaded band with labelled edges
            lo_g, hi_g = F["eps"]["low"], F["eps"]["high"]
            fig.add_vrect(x0=lo_g, x1=hi_g, fillcolor="#a3a29c", opacity=0.18, line_width=0, layer="below")
            for v, anchor in ((lo_g, "right"), (hi_g, "left")):
                fig.add_vline(x=v, line_color=TEXT2, line_width=1.5)
                fig.add_annotation(x=v, y=top * 1.16, text=f" ${v:.2f} " , showarrow=False, xanchor=anchor,
                                   font=dict(size=11, color=TEXT2))
            fig.add_annotation(x=(lo_g + hi_g) / 2, y=top * 1.16, text="Delta's guidance", showarrow=False,
                               font=dict(size=11, color=TEXT2))
        fig.add_trace(t0)
        c0 = F0["eps"]["forecast"] if F else float(np.median(x0))
        c1 = F["eps"]["forecast"] if F else float(np.median(x1))
        marks = [(c0, "Without" if moved_any else "Forecast", "#7a7974")]
        if moved_any:
            fig.add_trace(t1)
            marks.append((c1, "With", BLUE))
        for i, (v, lab, colr) in enumerate(marks):             # the middle result of each curve
            fig.add_vline(x=v, line_color=colr, line_width=1.5, line_dash="dash")
            fig.add_annotation(x=v, y=top * (1.02 - 0.1 * i), text=f" {lab}: ${v:.2f} ", showarrow=False,
                               xanchor="left", font=dict(size=11, color=colr), bgcolor="rgba(255,255,255,0.8)")
        fig.update_yaxes(showticklabels=False, range=[0, top * 1.25])
        fig.update_xaxes(tickprefix="$", tickformat=".2f", title_text="Adjusted EPS ($ per share)")
        fig = layout(fig, 340)
        fig.update_layout(showlegend=True, hovermode=False, legend=dict(orientation="h", y=1.12, x=0, yanchor="bottom"),
                          margin=dict(t=40))
        st.plotly_chart(fig, width="stretch")
        st.caption("Dashed lines: the forecast without and with your change.")

        if moved_any:
            br = eps_bridge(B, S, S0)
            parts = br.iloc[:-1]
            parts = parts[parts.abs() >= 0.005].sort_values(key=lambda v: v.abs(), ascending=True)
            e0 = F0["eps"]["forecast"] if F else float(np.median(S0.eps))
            e1 = F["eps"]["forecast"] if F else float(np.median(S.eps))
            md("**How your change reaches EPS**")
            md("<span class='small'>What your change did to each part of Delta's income statement, in dollars per share "
               "after tax.</span>", unsafe_allow_html=True)
            sgn = lambda v: f"{'+' if v >= 0 else '−'}${abs(v):.2f}"
            md(f"EPS without your change **${e0:.2f}** → with your change **${e1:.2f}** ({sgn(e1 - e0)})")
            labs = [f"{BRIDGE_NAME.get(i, i).replace('<br>', ' ')} {sgn(v)}" for i, v in parts.items()]
            fig = go.Figure(go.Bar(x=parts.values, y=labs, orientation="h",
                                   marker_color=[GREEN if v >= 0 else RED for v in parts.values], hoverinfo="skip"))
            lim = max(parts.abs().max() if len(parts) else 0.01, 0.01) * 1.15
            fig.add_vline(x=0, line_color=TEXT2, line_width=1)
            fig.update_xaxes(tickformat="+.2f", title_text="Effect on EPS ($ per share)", range=[-lim, lim])
            fig = layout(fig, 80 + 42 * max(len(parts), 1))
            fig.update_layout(hovermode=False, showlegend=False)
            st.plotly_chart(fig, width="stretch")
            st.caption("Green bars raise EPS, red bars lower it. Each bar is the middle result of the 50,000 runs, so they "
                       "can add up to slightly more or less than the total.")
            if len(lv_items) > 1:
                sv = shapley(B, r["R"], r["lv"]).drop(["base case", "scenario"])
                sv.index = [LEVER_NAME.get(i, i) for i in sv.index]
                md("**Which change mattered most**")
                md("<span class='small'>When you move more than one slider, this splits the EPS change between them, so "
                   "you can see which one mattered most. The pieces add up to the total change.</span>", unsafe_allow_html=True)
                fig = go.Figure(go.Bar(x=sv.values, y=list(sv.index), orientation="h",
                                       marker_color=[GREEN if v >= 0 else RED for v in sv.values],
                                       text=[f"{'+' if v >= 0 else '−'}${abs(v):.2f}" for v in sv.values], textposition="outside",
                                       cliponaxis=False))
                lim = max(abs(sv.values).max(), 0.01) * 1.35
                fig.update_xaxes(tickformat="+.2f", title_text="EPS change ($ per share)", range=[-lim, lim])
                fig = layout(fig, 90 + 45 * len(sv))
                fig.update_layout(hovermode=False)
                st.plotly_chart(fig, width="stretch")

        # the operating chain, in plain words
        md("**How the change flows through the business**")
        md("<span class='small'>The simulator's quarter step by step, from how much Delta flies to EPS. Base case = no "
           "change; With your change = after it; Change = the difference. Rows your change moved are highlighted. Each "
           "number is the middle result of the 50,000 runs.</span>", unsafe_allow_html=True)
        rows, hit = [], []
        for col, lab, kind in CHAIN_UI:
            b0, b1 = float(np.median(S0[col])), float(np.median(S[col]))
            ch = b1 - b0
            rows.append({"Step": lab, "Base case": _cell(kind, b0), "With your change": _cell(kind, b1),
                         "Change": _cell("pts" if kind in ("pct", "pct_level") else kind, ch, change=True)})
            tol = {"musd": 0.5, "usd2": 0.005}.get(kind, 0.05)
            hit.append(abs(ch) >= tol)
        ct = pd.DataFrame(rows)
        st.dataframe(ct.style.apply(lambda s_: ["background-color: #e8f0fb" if h else "" for h in hit], axis=0),
                     width="stretch", hide_index=True, height=35 * (len(ct) + 1) + 3)

    # save your scenario
    st.subheader("Save your scenario")
    md("Download your scenario as a spreadsheet file (CSV): the date, how each slider was set, and the results without "
       "and with your change.")
    sl = {f"Slider: {lab}": float(st.session_state[f"lv_{k}"]) for k, lab, *_ in LEVER_UI}
    rows = []
    for m in ["eps", "operating_margin", "revenue_pct"]:
        row = {"Quarter": qname, "As of": f"{d:%Y-%m-%d}",
               "Timing": "whole quarter (what-if)" if whatif else "from this date to the end of the quarter",
               "Fares mode": fare_mode, "Isolation": "on" if isolation else "off", **sl, "Result": NICE[m],
               "Simulator without your change": fmt(m, float(np.median(S0[m]))),
               "Simulator with your change": fmt(m, float(np.median(S[m])))}
        if F and m in F:
            f, f0 = F[m], F0[m]
            ins = lambda x: max(100 - x["below"] - x["above"], 0)
            row |= {"Delta's guidance": f"{fmt(m, f['low'])} to {fmt(m, f['high'])}",
                    "Forecast without your change": fmt(m, f0["forecast"]), "Forecast with your change": fmt(m, f["forecast"]),
                    "Change": fmt(m, f["forecast"] - f0["forecast"], True),
                    "80% range with your change": f"{fmt(m, f['p10'])} to {fmt(m, f['p90'])}",
                    "Chance below / inside / above (with your change)": f"{f['below']:.0f}% / {ins(f):.0f}% / {f['above']:.0f}%"}
        rows.append(row)
    st.download_button("Download scenario (CSV)", pd.DataFrame(rows).to_csv(index=False).encode("utf-8"),
                       file_name=f"flightpath_scenario_{q}_{d:%Y-%m-%d}.csv", mime="text/csv")


# ------------------------------------------------------------------------------------------------ 3. the data
# Every dataset behind FlightPath (docs/DATA_SOURCES.md and notebooks 00-07). Status: used / benchmark / tested, not used.
SOURCES = [
    ("Flight tracking - Delta", "OpenSky Network (historical database)", "Every Delta flight since 2019: 7.74 million flights, 405 weekly queries", "Next day", "Delta's capacity: the forecast's revenue line and the simulator", "Used"),
    ("Flight tracking - 5 rivals", "OpenSky Network (historical database)", "Daily flights of United, American, Southwest, Alaska and JetBlue", "Next day", "Outage detector (spots days OpenSky lost data); tested as a fare signal", "Used"),
    ("Aircraft database", "OpenSky Network", "Aircraft type for each tracked plane, which gives its seats", "Snapshot (Aug 2025)", "Seats per flight", "Used"),
    ("T-100 Segment", "U.S. DOT - Bureau of Transportation Statistics (BTS)", "Every airline's monthly flights, seats and passengers by route, 2009 - Jun 2026", "About 80 days", "The official check for flight tracking; seats per aircraft type; industry demand history", "Used"),
    ("On-Time Performance", "U.S. DOT - BTS", "Every Delta and Endeavor domestic flight, including cancellations, 2019 - Jul 2026", "About 60 days", "Confirms flight-tracking outages; measured weather disruptions for the event box", "Used"),
    ("Form 41, Schedule P-12(a)", "U.S. DOT - BTS", "Delta's monthly fuel gallons and cost, 2000 - Jun 2026", "About 85 days", "Measured the 14-day delay between market fuel prices and what Delta pays", "Used"),
    ("Earnings releases", "Delta Air Lines (SEC EDGAR, form 8-K)", "72 quarterly releases, Q4 2008 - Q2 2026: results, operating statistics, costs", "Same day", "Reported results and every link in the simulator", "Used"),
    ("Guidance", "Delta Air Lines (SEC EDGAR, 8-Ks)", "1,819 guidance items from 239 filings, extracted with AI and checked against each filing", "Same day", "The forecast's starting point", "Used"),
    ("Jet fuel spot price", "U.S. Energy Information Administration (EIA)", "Daily U.S. Gulf Coast jet fuel price", "About a week (conservative)", "Fuel news in every forecast line; the simulator's fuel cost", "Used"),
    ("Checkpoint travel numbers", "TSA (U.S. DHS)", "Daily travelers screened nationwide, 2019 - today", "Next day", "Industry demand in the simulator; tested as forecast news", "Used"),
    ("Airport database", "airportsdata (open source)", "Airport locations and countries", "n/a", "Route distances and regions", "Used"),
    ("Analyst estimates", "LSEG I/B/E/S via WRDS (Georgia Tech academic license)", "Every analyst's EPS and revenue estimate for Delta since 2009, with the date it was recorded", "As recorded (our copy runs to 20 Aug 2026)", "Wall Street's consensus on any date, for comparison", "Benchmark"),
    ("Final consensus", "Press coverage of each Delta report (hand-collected, with sources)", "Wall Street's EPS consensus just before each report", "At the report", "Benchmark near the report date", "Benchmark"),
]
DELAYS = [("Delta's filings (results, guidance)", 0), ("Flight tracking (OpenSky)", 1), ("TSA travelers", 1),
          ("Jet fuel price (EIA)", 7), ("Airline fares index (BLS)", 25), ("BTS On-Time", 60), ("BTS T-100", 80),
          ("BTS Form 41 fuel", 85)]
CHECKS = [
    ("Delta's releases read correctly", "Three accounting identities hold in every quarter 2009-2026, and Delta's adjusted figures rebuilt from 71 releases match what Delta printed within $3M.", "00, 04"),
    ("Guidance extracted correctly", "1,819 items from 239 filings, each checked against its filing; a hand check of the last nine releases agrees on all 27 figures.", "04"),
    ("TSA vs the official passenger count", "TSA travelers and BTS industry passengers move together (correlation about 1), but TSA grows about 1.4 points faster since 2023, so it is converted with the recent gap.", "03"),
    ("Wall Street's consensus rebuilt", "The consensus rebuilt for any date from I/B/E/S matches I/B/E/S's own monthly consensus to a median 0.5¢ over 164 months.", "07, section 7g"),
]
TESTED = [
    ("Rivals' flying (OpenSky)", "As news about Delta's fares", "EPS forecast worse; pointed the right way only 20 times in 37", "07, 7c"),
    ("Airline fares index (BLS)", "As revenue and EPS news", "Revenue slightly better (1.37 vs 1.71 points on day 70, not significant); EPS worse (24¢ vs 14¢ after the quarter)", "07, 7d-7e"),
    ("TSA travelers / industry demand", "As news for the forecast", "EPS forecast worse (20¢ vs 18¢ on day 20): Delta already knows its own bookings when it guides", "07, section 2"),
    ("Flight tracking for EPS", "Capacity news in the EPS forecast", "EPS worse (21¢ vs 19¢ on day 20), so it is used for revenue only, where it helps (1.56 vs 1.89 points)", "07, 7b and 7e"),
    ("The simulator on its own", "As a second opinion next to the forecast", "When the two disagreed, the actual result moved toward the simulator only 5 times in 16", "07, 7f"),
    ("Diesel and Brent crude prices (EIA)", "Downloaded as possible extra fuel signals", "Not needed: Delta's fuel cost follows jet fuel, which is used", "02"),
    ("Delta stock price (Nasdaq)", "Collected for a test of the share price on report day", "Not used yet", "-"),
    ("Earlier flight-tracking copies (OpenSky Zenodo 2019-2021, live API)", "First sources of the same flights", "Replaced by OpenSky's historical database, which they match flight for flight", "01"),
]
STATUS_COLOR = {"Used": "#e8f0fb", "Benchmark": "#fdf1e3", "Tested, not used": "#f3f2ef", "Not used": "#f3f2ef",
                "Not used yet": "#f3f2ef", "Replaced": "#f3f2ef"}


def us_holidays(years):
    out = []
    for y in years:
        nov1 = pd.Timestamp(y, 11, 1)
        out.append(("Thanksgiving", nov1 + pd.Timedelta(days=(3 - nov1.weekday()) % 7 + 21)))
        sep1 = pd.Timestamp(y, 9, 1)
        out.append(("Labor Day", sep1 + pd.Timedelta(days=(0 - sep1.weekday()) % 7)))
        may31 = pd.Timestamp(y, 5, 31)
        out.append(("Memorial Day", may31 - pd.Timedelta(days=may31.weekday())))
        out += [("4 July", pd.Timestamp(y, 7, 4)), ("Christmas", pd.Timestamp(y, 12, 25)), ("New Year", pd.Timestamp(y, 1, 1))]
    return out


@st.cache_data(show_spinner=False, max_entries=64)
def news_split(q, d):
    """What the data did to the forecast on this date: fuel news on EPS; fuel and capacity news on revenue growth."""
    r = run(q, d)
    e, F, R, S = r["e"], r["F"], r["R"], r["S"]
    if not F:
        return None
    out = {"eps_fuel": F["eps"]["news"]}
    if "revenue_pct" in F:
        C = dict(e["B"], as_of=e["news"]["revenue_pct"]["gd"])
        fuel_rev = float(np.median(S.revenue_pct.values - evaluate(C, R).revenue_pct.values))
        out |= {"rev_fuel": fuel_rev, "rev_cap": F["revenue_pct"]["news"] - fuel_rev}
    return out


def page_signals():
    controls()
    st.title("The Data")
    md(f"{badge('data')} FlightPath is built only on public data. This page shows every dataset behind it: where it "
       f"comes from, how we checked it, how soon it becomes public, and what it shows today. Every number in the app uses "
       f"a dataset only from the day it became public.", unsafe_allow_html=True)
    k = st.columns(4)
    k[0].metric("Delta flights tracked", "7.74 million", help="Every Delta mainline flight OpenSky tracked, Jan 2019 - Oct 2026")
    k[1].metric("Delta earnings releases read", "72", help="Q4 2008 - Q2 2026, from SEC EDGAR")
    k[2].metric("Guidance items extracted", "1,819", help="From 239 Delta filings, each checked against its filing")
    k[3].metric("Datasets used", f"{len(SOURCES)}", help="Plus 4 more we tried or set aside, listed at the bottom of the page")

    # 1. where the data comes from
    st.subheader("Where the data comes from")
    md("Every dataset FlightPath uses, what it gives us, how long after the fact it becomes public, and how it is used. "
       "*Used* = feeds the forecast or the simulator; *Benchmark* = only for comparison. Datasets we tried and set aside "
       "are listed at the bottom of the page.")
    src = pd.DataFrame(SOURCES, columns=["Dataset", "Publisher", "What it gives us", "Public after", "Used for", "Status"])
    st.table(src.style.apply(lambda r_: [f"background-color: {STATUS_COLOR.get(r_['Status'], '')}"] * len(r_), axis=1)
             .hide(axis="index"))

    # 2. how we know it's right
    st.subheader("How we know it's right")
    md("Each dataset was checked against an independent source before it was used. Flight tracking gets the most "
       "attention: it is the one input the official numbers can't give us in time, so it has to match them once they "
       "come out.")
    dp = json.loads((S_.path / "data_page.json").read_text()) if (S_.path / "data_page.json").exists() else {}

    md("#### From flight tracking to Delta's capacity")
    md("""
Delta reports its capacity as **seat-miles** (seats flown × miles flown) for the whole airline. Flight tracking sees
something different - individual planes in the air - so it takes six steps to turn one into the other (notebooks 01
and 06):
1. **Track every flight.** OpenSky's ground receivers pick up each Delta plane's transponder: when it took off, when it
   landed and which aircraft it was. That gives the hours each plane flew.
2. **Add seats.** Each aircraft's type gives its number of seats, from the official BTS records, so hours flown become
   seat-hours.
3. **Fill the gaps.** On days OpenSky lost data (found by the outage detector below), the missing flying is filled in
   from the same weekday.
4. **Calibrate to the official count.** OpenSky doesn't see every flight, and seat-hours aren't seat-miles. So our
   seat-hours are divided by their ratio to BTS's official seat-miles over the latest 3 months BTS has published - one
   step that corrects for both.
5. **From Delta's own planes to the whole airline.** Flight tracking covers Delta's mainline planes; Delta's reported
   total also includes flying by its regional partners. Mainline has been between 91% and 95% of the total since 2022
   (chart), so the estimate is scaled up with that share - the same quarter last year, plus its recent drift.
6. **The rest of the quarter.** While the quarter is still running, the days not flown yet are filled with the same
   days a year earlier, grown at the rate seen so far.
""")
    if "mainline_share" in dp:
        ms = pd.DataFrame(dp["mainline_share"]["quarters"])
        fig = go.Figure(go.Scatter(x=ms.quarter, y=ms.share * 100, mode="lines+markers", line=dict(color=BLUE, width=2),
                                   hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
        fig.update_yaxes(ticksuffix="%", range=[80, 100], title_text="mainline share")
        fig.update_xaxes(tickvals=[x for x in ms.quarter if x.endswith("Q1")], ticktext=[x[:4] for x in ms.quarter if x.endswith("Q1")])
        fig = layout(fig, 250, "Delta's mainline share of its total capacity (step 5)")
        fig.update_layout(hovermode="x unified", showlegend=False)
        st.plotly_chart(fig, width="stretch")
        st.caption("BTS T-100 mainline seat-miles ÷ Delta's reported seat-miles, quarter by quarter. The 2020 dip is the "
                   "pandemic.")

    if "capacity_validation" in dp:
        cv = pd.DataFrame(dp["capacity_validation"]["quarters"])
        bm = dp["capacity_validation"]["benchmarks"]
        md("#### Does it match the official count?")
        md("<span class='small'>Delta's mainline capacity (billions of seat-miles). Blue: our estimate from flight "
           "tracking, made 7 days after each quarter ended with only the data public then. Grey: the official BTS figure, "
           "published about 80 days later. The label is our miss.</span>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=cv.quarter, y=cv.bts_bn, name="Official (BTS), ~80 days later", marker_color="#d6d5d0"))
        fig.add_trace(go.Scatter(x=cv.quarter, y=cv.estimate_bn, mode="markers+text", name="Flight tracking, 7 days after the quarter",
                                 marker=dict(color=BLUE, size=10), text=[f"{v:+.1f}%" for v in cv.error_pct],
                                 textposition="top center", textfont=dict(size=10, color=BLUE)))
        fig.update_yaxes(title_text="bn seat-miles", range=[0, cv[["bts_bn", "estimate_bn"]].max().max() * 1.15])
        fig = layout(fig, 340)
        fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.1, x=0, yanchor="bottom"), margin=dict(t=40))
        st.plotly_chart(fig, width="stretch")
        st.caption("The two largest misses (Q2 2023, Q1 2024) came from flight-tracking outages in the months used to "
                   "calibrate - the problem the outage detector below fixes.")
        md("""
**Is it better than what you could do without flight tracking?** Before BTS publishes, the only other way to put a
number on a quarter is from BTS's own history. The table compares flight tracking with the two ways of doing that, all
made on the same day (7 days after the quarter ends) with only official data public by then:
- **Same as last year:** the quarter's capacity = the same quarter a year earlier.
- **Last year × latest known growth:** the same quarter a year earlier, grown at the rate of the latest quarter BTS has
  fully published.

Flight tracking misses by less than either, including on the quarters held out when the method was chosen. The gap
against "same as last year" is clearly real; against the growth method it could partly be chance with 18 quarters.
Flight tracking helps most when the airline changes fast: in 2022, as Delta rebuilt its schedule after the pandemic,
the growth method missed by 4-9% in three of four quarters, flight tracking by at most 1.6%.
""")
        keys = [("Flight tracking (FlightPath)", "OpenSky (chosen method)"),
                ("Without flight tracking: last year × latest known growth", "benchmark: last year x latest known growth"),
                ("Without flight tracking: same as last year", "benchmark: same as last year")]
        bt = pd.DataFrame({"How the quarter is estimated (7 days after it ends)": [n for n, _ in keys],
                           "Average miss, 2022 - 2026 (18 quarters)": [f"{bm[k_]['mae_all']:.2f}%" for _, k_ in keys],
                           "Average miss, held-out quarters": [f"{bm[k_]['mae_heldout']:.2f}%" for _, k_ in keys],
                           "Largest miss": [f"{bm[k_]['largest']:.2f}%" for _, k_ in keys]})
        st.table(bt.style.hide(axis="index"))

    md("#### Finding the days flight tracking lost data")
    md("OpenSky sometimes loses data for a day or more. Flight tracking would read that as Delta cancelling flights. The "
       "outage detector checks the five rival airlines: when all of them drop at once, by the same proportion, it is "
       "OpenSky losing data - a storm or an IT failure hits some airlines much more than others. 28 days were flagged "
       "in 2023-2026 (11 whole days, 17 partial), all confirmed against the government's On-Time records. Fixing them "
       "cut the forecast's capacity error on day 20 of a quarter from 3.64% to 2.24% (notebook 07, section 7a).")
    sg = S_.signals
    flg = sg[(sg.outage_full == True) | (sg.outage_partial == True)]  # noqa: E712
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sg.index, y=sg.rivals_flights, mode="lines", line=dict(color="#a3a29c", width=1),
                             name="5 rivals' tracked flights per day"))
    fig.add_trace(go.Scatter(x=flg.index, y=flg.rivals_flights, mode="markers", marker=dict(color=RED, symbol="x", size=9),
                             name="Flagged: OpenSky lost data"))
    fig.update_yaxes(title_text="flights per day")
    fig = layout(fig, 280)
    fig.update_layout(legend=dict(orientation="h", y=1.1, x=0, yanchor="bottom"), margin=dict(t=30))
    st.plotly_chart(fig, width="stretch")

    if "fuel_monthly" in dp:
        fm = pd.DataFrame(dp["fuel_monthly"]["months"])
        md("#### Market fuel price vs what Delta paid")
        md("Delta buys fuel at many airports and under contracts, so it doesn't pay the Gulf Coast market price exactly. "
           "Notebook 02 measured a 14-day delay between the market price and Delta's cost (on 114 months of BTS Form 41 "
           "data) and a way to predict Delta's price from the market price, with a typical error of about 3%. The chart "
           "shows the two moving together.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fm.month, y=fm.market_14d_earlier, name="Market jet fuel, 14 days earlier (EIA)",
                                 line=dict(color=AMBER, width=2)))
        fig.add_trace(go.Scatter(x=fm.month, y=fm.delta_paid, name="Price Delta paid (BTS Form 41)",
                                 line=dict(color=BLUE, width=2)))
        fig.update_yaxes(tickprefix="$", title_text="$ per gallon")
        fig = layout(fig, 280)
        fig.update_layout(legend=dict(orientation="h", y=1.1, x=0, yanchor="bottom"), margin=dict(t=30))
        st.plotly_chart(fig, width="stretch")
        st.caption("Monthly averages, Jan 2017 - Jun 2026; Delta's mainline price per gallon.")

    md("#### Other checks")
    st.table(pd.DataFrame(CHECKS, columns=["Check", "Result", "Notebook"]).style.hide(axis="index"))

    # 3. what we know on any given day (a general rule, shown on an example day)
    st.subheader("What we know on any given day")
    md("Standing on any day in a quarter, this is how recent the newest number from each source is - shown here for "
       "day 45, halfway through a July - September quarter (14 August). Flight tracking, TSA and fuel prices reach up "
       "to a few days ago, well inside the current quarter. Everything else stops before the quarter began: Delta's own "
       "results cover last quarter, and the government's monthly figures arrive one to three months after each month "
       "ends. So mid-quarter, the only up-to-date view of how much Delta is flying is flight tracking. FlightPath never "
       "uses a number before it was public.")
    today_ = pd.Timestamp(2026, 7, 1) + pd.Timedelta(days=44)          # example: day 45 of a July - September quarter
    def newest_month(lag):                                              # monthly data: whole months, `lag` days after each ends
        m = (today_ - pd.Timedelta(days=lag)).to_period("M")
        while m.end_time.normalize() + pd.Timedelta(days=lag) > today_:
            m -= 1
        return (today_ - m.end_time.normalize()).days, f"{m.to_timestamp():%B}"
    fresh = [("Flight tracking (OpenSky)", 1, "yesterday"), ("TSA travelers", 1, "yesterday"),
             ("Jet fuel price (EIA)", 7, "about a week ago"),
             ("Delta's reported results", 45, "last quarter (to 30 June)")]
    for name, lag in [("Airline fares index (BLS), ~25 days after each month", 25),
                      ("BTS On-Time (domestic flights), ~60 days", 60),
                      ("BTS T-100 (official capacity), ~80 days", 80), ("BTS Form 41 (fuel bought), ~85 days", 85)]:
        fresh.append((name, *newest_month(lag)))
    DAY, BACK = 45, 150
    fr = pd.DataFrame(fresh[::-1], columns=["source", "lag", "newest"]).sort_values("lag", ascending=False, kind="stable")
    inq = fr.lag < DAY
    fig = go.Figure()
    fig.add_vrect(x0=-DAY, x1=0, fillcolor=BLUE, opacity=0.07, line_width=0, layer="below",
                  annotation_text="this quarter so far", annotation_position="bottom left")
    fig.add_trace(go.Bar(y=fr.source, x=BACK - fr.lag, base=-BACK, orientation="h",
                         marker_color=[GREEN if v else "#c9c8c3" for v in inq], hoverinfo="skip",
                         text=[f"newest: {t}  " for t in fr.newest], textposition="inside", insidetextanchor="end",
                         textfont=dict(color=["white" if v else TEXT2 for v in inq], size=11)))
    fig.add_vline(x=0, line_color=TEXT2, line_width=2, annotation_text="today (day 45)", annotation_position="bottom right")
    fig.update_xaxes(range=[-BACK, 25], tickvals=[-150, -120, -90, -60, -30, 0],
                     ticktext=["150 days ago", "120", "90", "60", "30", "today"], title_text="")
    fig = layout(fig, 360)
    fig.update_layout(showlegend=False, hovermode=False, margin=dict(t=30))
    st.plotly_chart(fig, width="stretch")
    st.caption("Green: the data reaches into the current quarter. Grey: its newest number is from before the quarter "
               "began. Each bar ends at the newest day that source covers on day 45; the same rule holds on any day and "
               "in any quarter.")

    # 4. live data
    st.subheader("Live data")
    sig = S_.signals[S_.signals.index < d]
    last = {c_: sig[c_].dropna().index.max() for c_ in ["delta_hours", "rivals_flights", "jet_fuel", "tsa"]}
    md(f"The daily data as it stood on **{d:%d %b %Y}**. Data through: flight tracking {last['delta_hours']:%d %b}, "
       f"rivals {last['rivals_flights']:%d %b}, jet fuel {last['jet_fuel']:%d %b}, TSA {last['tsa']:%d %b}.")
    full = st.toggle("Show all history (from July 2023)", value=False)
    lo = pd.Timestamp("2023-07-01") if full else P.start_time - pd.Timedelta(days=45)
    w = sig[sig.index >= lo]
    qd = sig[(sig.index >= P.start_time)]
    gd = None
    e = S_.entry(q, d)
    if e["news"]:
        gd = pd.Timestamp(e["news"]["eps"]["gd"]) - pd.Timedelta(days=1)

    def qtd(col_sum):
        s = sig[col_sum]
        a = s[(s.index >= P.start_time)]
        b = s.reindex(a.index - pd.Timedelta(days=364))
        return (a.sum() / b.sum() - 1) * 100 if len(a) >= 7 and b.notna().all() else np.nan
    out_q = qd[(qd.outage_full == True) | (qd.outage_partial == True)]  # noqa: E712
    c = st.columns(4)
    raw = qtd("delta_hours")
    c[0].metric("Delta's flying this quarter", f"{e['B']['cap']['consolidated_yoy_pct']:+.1f}%",
                help="Estimated seat-miles for the whole airline vs the same quarter last year - the number the forecast "
                     "uses (flight tracking calibrated to the official count, notebook 01).")
    c[0].caption(f"raw tracked flight-hours so far: {raw:+.1f}%" if not np.isnan(raw) else "raw tracked flight-hours: -")
    c[1].metric("5 rivals' flights, quarter so far", f"{qtd('rivals_flights'):+.1f}%" if not np.isnan(qtd('rivals_flights')) else "-",
                help="All five rivals' tracked flights added together (United, American, Southwest, Alaska, JetBlue), vs "
                     "the same days a year earlier - one combined total, so the bigger airlines count for more.")
    jn = e["B"]["jet"]
    c[2].metric("Market jet fuel (last 5 trading days)", f"${jn.iloc[-5:].mean():.2f}/gal",
                delta=(f"{(jn.iloc[-5:].mean() / jn[:gd].iloc[-5:].mean() - 1) * 100:+.0f}% since Delta guided" if gd is not None else None),
                delta_color="inverse")
    c[3].metric("TSA travelers, quarter so far", f"{qtd('tsa'):+.1f}%" if not np.isnan(qtd('tsa')) else "-",
                help="TSA checkpoint counts vs the same weekdays a year earlier (all US airlines)")
    ns = news_split(q, d)
    if ns:
        parts = [f"fuel since Delta guided: **{fmt('eps', ns['eps_fuel'], True)} EPS**"]
        if "rev_cap" in ns:
            parts.append(f"Delta's flying: **{ns['rev_cap']:+.1f} pts** of revenue growth (fuel {ns['rev_fuel']:+.1f} pts)")
        md("**What this data did to the forecast:** " + "; ".join(parts) + ". Details on the Forecast page.")
    if len(out_q):
        st.caption(f"Flight-tracking outage days this quarter, found and fixed: {len(out_q)}.")

    hol = [(n, h) for n, h in us_holidays(range(w.index.min().year - 1, w.index.max().year + 1))]
    def shade(fig_):
        seen = set()
        for n, h in hol:
            for x in (h, h + pd.Timedelta(days=364)):            # the holiday, and the day compared with last year's
                if w.index.min() <= x <= w.index.max() and (n, x) not in seen:
                    seen.add((n, x))
                    fig_.add_vrect(x0=x - pd.Timedelta(days=3), x1=x + pd.Timedelta(days=3), fillcolor="#a3a29c",
                                   opacity=0.15, line_width=0, layer="below")
    md("**Flying vs a year earlier**")
    md("<span class='small'>One point per day. Each day's point is a rolling week: that day and the 6 days before it, "
       "compared with the same 7 days a year earlier (same weekdays), in %. Using a week instead of the single day keeps "
       "the line from jumping with the weekday pattern - airlines fly less on Saturdays than on Fridays. Above 0% = more "
       "flying than a year earlier. Delta: hours flown (raw flight tracking); rivals: number of flights, all five added "
       "together. When the blue line is below the orange one, Delta is growing less than its rivals. Shaded: holiday "
       "weeks - when a holiday falls on a different date than last year, the comparison jumps for a few days; that is "
       "not a real change.</span>", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w.index, y=w.delta_yoy7, name="Delta (tracked flight-hours)", line=dict(color=BLUE, width=2)))
    fig.add_trace(go.Scatter(x=w.index, y=w.rivals_yoy7, name="5 rivals (tracked flights)", line=dict(color=ORANGE, width=2)))
    ow = w[(w.outage_full == True) | (w.outage_partial == True)]  # noqa: E712
    if len(ow):
        fig.add_trace(go.Scatter(x=ow.index, y=[0] * len(ow), mode="markers", name="Flight-tracking outage day (fixed)",
                                 marker=dict(color=RED, symbol="x", size=8)))
    for x, lab, pos in [(P.start_time, "quarter start", "top left")] + ([(gd, "Delta guided", "top right")] if gd is not None else []):
        if x >= w.index.min():
            fig.add_vline(x=x, line_dash="dot", line_color=MUTED, annotation_text=lab, annotation_position=pos)
    shade(fig)
    fig.update_yaxes(ticksuffix="%")
    fig.update_xaxes(tickformat="%d %b %Y" if full else "%d %b")
    fig = layout(fig, 330)
    fig.update_layout(legend=dict(orientation="h", y=1.08, x=0, yanchor="bottom"), margin=dict(t=30))
    st.plotly_chart(fig, width="stretch")
    a, b = st.columns(2)
    with a:
        md("**Market jet fuel, U.S. Gulf Coast ($ per gallon)**")
        md("<span class='small'>The daily market price (EIA). Delta pays it with about a 14-day delay. The dotted line is "
           "when Delta gave guidance: the gap since then is the fuel news in the forecast.</span>", unsafe_allow_html=True)
        fig = go.Figure(go.Scatter(x=w.index, y=w.jet_fuel, line=dict(color=AMBER, width=2), name="jet fuel"))
        if gd is not None and gd >= w.index.min():
            g_px = float(jn[:gd].iloc[-5:].mean())
            fig.add_vline(x=gd, line_dash="dot", line_color=MUTED, annotation_text="Delta guided")
            fig.add_hline(y=g_px, line_dash="dash", line_color=MUTED, annotation_text=f"${g_px:.2f} when Delta guided",
                          annotation_position="top right", annotation_font=dict(size=11, color=TEXT2))
        fig.update_yaxes(tickprefix="$")
        fig.update_xaxes(tickformat="%d %b %Y" if full else "%d %b")
        st.plotly_chart(layout(fig, 280), width="stretch")
    with b:
        md("**TSA travelers vs a year earlier**")
        md("<span class='small'>One point per day. Each day's point is a rolling week: travelers screened that day and "
           "the 6 days before it, compared with the same 7 days a year earlier, all U.S. airlines. Used as industry demand "
           "in the simulator. Shaded: holiday weeks.</span>", unsafe_allow_html=True)
        fig = go.Figure(go.Scatter(x=w.index, y=w.tsa_yoy7, line=dict(color=GREEN, width=2), name="TSA"))
        shade(fig)
        fig.update_yaxes(ticksuffix="%")
        fig.update_xaxes(tickformat="%d %b %Y" if full else "%d %b")
        st.plotly_chart(layout(fig, 280), width="stretch")

    # 5. tested, not used
    st.subheader("What we tried and set aside")
    md("Not every dataset helped. Each one tested as news was tried on the same past quarters as the forecast, and kept "
       "only if it made the forecast more accurate; a few others were collected but turned out not to be needed.")
    st.table(pd.DataFrame(TESTED, columns=["Data", "Tried as", "Result", "Notebook"]).style.hide(axis="index"))
    st.caption("Flight data: The OpenSky Network, https://opensky-network.org. Full "
               "catalog with links and download dates: docs/DATA_SOURCES.md.")


# ------------------------------------------------------------------------------------------------ 4. track record
OFFSETS = {"20": "day 20", "45": "day 45", "70": "day 70", "end+7": "7 days after the quarter ended"}
# with the daily forecast path: every day from day 14 (just after Delta guides) to day 89 (the last day every quarter
# has), then 1-7 days after the quarter ended (Delta's earliest report came 8 days after) - the same point in every quarter
DAY_KEYS = [str(k) for k in range(14, 90)] + [f"end+{j}" for j in range(1, 8)]


def key_label(key):
    if key.startswith("end+"):
        j = int(key[4:])
        return f"{j} day{'s' if j > 1 else ''} after the quarter ended"
    return f"day {key}"


def key_at(key):
    if key.startswith("end+"):
        return lambda qq: pd.Period(qq, "Q").end_time.normalize() + pd.Timedelta(days=int(key[4:]))
    return lambda qq: pd.Period(qq, "Q").start_time + pd.Timedelta(days=int(key))


def key_x(key):
    """Position on the 'through the quarter' chart: the day, or 91 + days after the quarter ended."""
    return 91 + int(key[4:]) if key.startswith("end+") else int(key)
CONS_KEY = {"eps": "eps", "revenue_pct": "revenue_pct", "operating_margin": None}


def _when(qq, off):
    Pq = pd.Period(qq, "Q")
    return Pq.end_time.normalize() + pd.Timedelta(days=7) if off == "end+7" else Pq.start_time + pd.Timedelta(days=int(off))


@st.cache_data(show_spinner=False)
def track_rows(m, off):
    """Backtest rows for one headline number and one test date, with Wall Street's consensus on the same date."""
    if S_.paths is not None:                        # any day: from the daily forecast path
        return past_at(m, key_at(off))
    T = S_.track[(S_.track.offset.astype(str) == off) & (S_.track.metric == m)].set_index("quarter").copy()
    key = CONS_KEY[m]
    cons = pd.Series(np.nan, index=T.index)
    if key and S_.cons_hist is not None:
        for qq in T.index:
            r = S_.cons_hist[(S_.cons_hist.quarter == qq) & (S_.cons_hist.date == _when(qq, off))]
            if len(r) and pd.notna(r.iloc[0][key]):
                cons[qq] = float(r.iloc[0][key])
    T["street"] = cons
    return T


def _miss(m, v):
    """A signed miss in plain words: '12¢ too high' / '0.8 pts too low'."""
    if pd.isna(v):
        return "n/a"
    size = f"{abs(v) * 100:.0f}¢" if m == "eps" else f"{abs(v):.1f} pts"
    return "exact" if (m == "eps" and abs(v) < 0.005) or (m != "eps" and abs(v) < 0.05) else f"{size} too {'high' if v > 0 else 'low'}"


QEND_DAY = 91          # the chart puts "quarter ends" at 91 (quarters are 90-92 days long) and the days after it beyond


@st.cache_data(show_spinner=False)
def miss_by_day(m) -> pd.DataFrame:
    """Average miss of the forecast, Delta's guidance midpoint and Wall Street over the past quarters on every day of
    the quarter (DAY_KEYS: day 14 to 89, then 1-7 days after the quarter ended), from forecast_path.csv."""
    out = {}
    for key in DAY_KEYS:
        t = past_at(m, key_at(key))
        out[key_x(key)] = {"Forecast": (t.forecast - t.actual).abs().mean(),
                           "Delta's guidance midpoint": (t.guidance_mid - t.actual).abs().mean(),
                           "Wall Street": (t.street - t.actual).abs().mean() if t.street.notna().any() else np.nan,
                           "label": key_label(key)}
    return pd.DataFrame(out).T


def track_section(m, title, intro):
    st.subheader(title)
    md(intro)
    if S_.paths is not None:
        off = st.select_slider("Forecast made on", DAY_KEYS, value="45", format_func=key_label, key=f"day_{m}")
    else:
        off = st.segmented_control("Forecast made on", list(OFFSETS), default="45", format_func=lambda x: OFFSETS[x],
                                   key=f"off_{m}") or "45"
    lab = key_label(off) if S_.paths is not None else OFFSETS[off]
    lab = lab if "after" in lab else f"on {lab}"
    T = track_rows(m, off)
    n = len(T)
    side = lambda v, lo, hi: np.where(v < lo, "below", np.where(v > hi, "above", "inside"))
    e_f, e_g, e_s = T.forecast - T.actual, T.guidance_mid - T.actual, T.street - T.actual
    unit = (lambda v: f"{v * 100:.0f}¢") if m == "eps" else (lambda v: f"{v:.2f} pts")
    has_s = T.street.notna().any()
    inside = int(((T.actual >= T.p10) & (T.actual <= T.p90)).sum())
    right = int((side(T.forecast, T.guidance_low, T.guidance_high) == side(T.actual, T.guidance_low, T.guidance_high)).sum())
    cand = pd.DataFrame({"Forecast": e_f.abs(), "Delta's guidance": e_g.abs()} | ({"Wall Street": e_s.abs()} if has_s else {}))
    best = cand.idxmin(axis=1)
    wins = int((best == "Forecast").sum())

    md("**Average miss**")
    c = st.columns(4 if has_s else 3)
    c[0].metric("Forecast", unit(e_f.abs().mean()))
    c[1].metric("Delta's guidance midpoint", unit(e_g.abs().mean()))
    if has_s:
        c[2].metric("Wall Street (same day)", unit(e_s.abs().mean()))
    c[-1].metric("Actual inside the forecast's 80% range", f"{inside} of {n}")
    md(f"<span class='small'>Average miss = average distance from what Delta reported ({'cents per share' if m == 'eps' else 'percentage points'}); "
       f"smaller is better. Forecast made {lab} in each quarter, Q3 2023 - Q2 2026, using only data public that "
       f"day. The forecast was the closest of the {'three' if has_s else 'two'} in **{wins} of {n}** quarters and called the right side "
       f"of Delta's range (below / inside / above) in **{right} of {n}**. For reference, the simulator on its own, without "
       f"Delta's guidance, missed by {unit((T.simulator - T.actual).abs().mean())} on average.</span>", unsafe_allow_html=True)

    # by quarter
    fig = go.Figure()
    xs = list(T.index)
    fig.add_trace(go.Bar(x=xs, y=T.guidance_high - T.guidance_low, base=T.guidance_low, marker_color="#a3a29c", opacity=0.25,
                         name="Delta's guidance range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=T.forecast, mode="markers", marker=dict(color=BLUE, size=10), name="Forecast (with 80% range)",
                             error_y=dict(type="data", symmetric=False, array=T.p90 - T.forecast, arrayminus=T.forecast - T.p10,
                                          color=BLUE, thickness=1)))
    if has_s:
        fig.add_trace(go.Scatter(x=xs, y=T.street, mode="markers", marker=dict(color=ORANGE, symbol="line-ew-open", size=18,
                                                                                line=dict(width=3)), name="Wall Street (same day)"))
    fig.add_trace(go.Scatter(x=xs, y=T.actual, mode="markers", marker=dict(color="#0b0b0b", symbol="diamond", size=9), name="Actual"))
    fig.update_yaxes(tickprefix="$" if m == "eps" else "", ticksuffix="" if m == "eps" else "%")
    fig = layout(fig, 330)
    fig.update_layout(legend=dict(orientation="h", y=1.08, x=0, yanchor="bottom"), margin=dict(t=30))
    st.plotly_chart(fig, width="stretch")

    # who was closest each quarter
    md("**Who was closest each quarter?**")
    rows = []
    for qq in T.index:
        r_ = T.loc[qq]
        row = {"Quarter": f"{qq[-2:]} {qq[:4]}", "Actual": fmt(m, r_.actual), "Forecast": _miss(m, e_f[qq]),
               "Delta's guidance midpoint": _miss(m, e_g[qq])}
        if has_s:
            row["Wall Street"] = _miss(m, e_s[qq])
        row["Closest"] = best[qq]
        row["Inside the 80% range?"] = "yes" if r_.p10 <= r_.actual <= r_.p90 else "no"
        row["Right side of Delta's range?"] = ("yes" if side(r_.forecast, r_.guidance_low, r_.guidance_high) ==
                                               side(r_.actual, r_.guidance_low, r_.guidance_high) else "no")
        rows.append(row)
    tb = pd.DataFrame(rows)
    hl = lambda s_: [("background-color: #e8f0fb" if v == "Forecast" else "") for v in s_]
    st.table(tb.style.apply(hl, subset=["Closest"]).hide(axis="index"))
    st.caption("Each miss is how far that number was from what Delta reported, and in which direction. Closest = the "
               "smallest miss of the three. Right side = the forecast said Delta would land below, inside or above its "
               "guidance range, and it did.")
    st.download_button("Download this table (CSV)", tb.to_csv(index=False).encode("utf-8"),
                       file_name=f"flightpath_track_{m}_{off}.csv", mime="text/csv", key=f"dl_{m}")

    # how the miss shrinks through the quarter
    md("**How the miss shrinks through the quarter**")
    md("<span class='small'>Average miss on each day of the quarter. As more of the quarter becomes known, everyone "
       "gets closer - the question is who gets there first.</span>", unsafe_allow_html=True)
    sc = 100 if m == "eps" else 1
    fig = go.Figure()
    if S_.paths is not None:                       # every day of the quarter, from forecast_path.csv
        M = miss_by_day(m)
        xs, x_title = M.index, "day of the quarter, then days after it ended"
        fig.add_vline(x=QEND_DAY, line=dict(color="#a3a29c", width=1, dash="dot"))
        fig.add_annotation(x=QEND_DAY, y=1, yref="paper", text="quarter ends", showarrow=False, xanchor="left",
                           font=dict(size=10, color=TEXT2))
    else:                                          # older snapshot: the four test dates only
        M = pd.DataFrame({name: [(track_rows(m, o)[col] - track_rows(m, o).actual).abs().mean() for o in OFFSETS]
                          for name, col in [("Forecast", "forecast"), ("Delta's guidance midpoint", "guidance_mid"),
                                            ("Wall Street", "street")]})
        xs, x_title = ["day 20", "day 45", "day 70", "after the quarter"], None
    for name, colr, dash in [("Forecast", BLUE, "solid"), ("Delta's guidance midpoint", "#7a7974", "dot"), ("Wall Street", ORANGE, "dash")]:
        v = np.array(M[name], float) * sc
        if np.isnan(v).all():
            continue
        fig.add_trace(go.Scatter(x=xs, y=v, mode="lines", name=name, line=dict(color=colr, width=2.5, dash=dash), connectgaps=False,
                                 customdata=M["label"] if "label" in M else None,
                                 hovertemplate=(f"%{{customdata}} - " if "label" in M else "") +
                                 f"{name}: %{{y:.{0 if m == 'eps' else 2}f}}{'¢' if m == 'eps' else ' pts'}<extra></extra>"))
    if x_title:
        fig.update_xaxes(title_text=x_title, tickvals=[14, 20, 30, 45, 60, 70, 80, 91, 98],
                         ticktext=["14", "20", "30", "45", "60", "70", "80", "end", "+7 days"])
    fig.update_yaxes(title_text="average miss (¢ per share)" if m == "eps" else "average miss (pts)", rangemode="tozero")
    fig = layout(fig, 280)
    fig.update_layout(legend=dict(orientation="h", y=1.1, x=0, yanchor="bottom"), margin=dict(t=30), hovermode="x unified")
    st.plotly_chart(fig, width="stretch")


def live_test(q):
    """A call locked before Delta reported (notebook 08), scored against what Delta reported once it has."""
    L, act = S_.locks[q], S_.actuals.get(q)
    made = pd.Timestamp(L["made_at_utc"].replace(" UTC", ""))
    qn = f"{q[-2:]} {q[:4]}"
    with st.container(border=True):
        md(f"##### Live test: {qn} - called before Delta reported")
        if act:
            filed = pd.Timestamp(act["filed"])
            md(f"Locked on {made:%d %b %Y}, {(filed.normalize() - made.normalize()).days} days before Delta reported, "
               f"and fingerprinted so it can't be changed afterwards. Scored against Delta's adjusted results of "
               f"{filed:%d %b %Y}.")
        else:
            md(f"Locked on {made:%d %b %Y}, before Delta reports, and fingerprinted so it can't be changed afterwards. "
               "It will be scored here once Delta reports.")
        rows = []
        for m in ["eps", "operating_margin", "revenue_pct"]:
            f = L["forecast"][m]
            lo, hi = f["guidance"]
            rng = lambda a, b: f"{fmt(m, a)} to {fmt(m, b).lstrip('$')}"     # one $ per cell (two would render as math)
            row = {"Result": NICE[m], "Delta's guidance": rng(lo, hi),
                   "Locked forecast": fmt(m, f["value"]), "80% range": rng(f["p10"], f["p90"])}
            if act:
                a = act[m]
                side = lambda v: "below" if v < lo else "above" if v > hi else "inside"
                row.update({"Actual": fmt(m, a), "Forecast off by": _miss(m, f["value"] - a),
                            "Guidance midpoint off by": _miss(m, (lo + hi) / 2 - a),
                            "Inside the 80% range?": "yes" if f["p10"] <= a <= f["p90"] else "no",
                            "Right side of Delta's range?": "yes" if side(f["value"]) == side(a) else "no"})
            rows.append(row)
        st.table(pd.DataFrame(rows).style.hide(axis="index"))
        ws = S_.consensus_eps(q)
        notes = []
        if act and ws is not None:
            notes.append(f"Wall Street's final EPS consensus before the report (press): {fmt('eps', ws)}, "
                         f"{_miss('eps', ws - act['eps'])}.")
        notes.append(f"Fingerprint (SHA-256 of the lock record): {L['sha256']}")
        st.caption("  \n".join(notes))


def page_track():
    st.title("Track record")
    ws_txt = ("Delta's own guidance and Wall Street's consensus on the same day (LSEG I/B/E/S)" if S_.cons_hist is not None
              else "Delta's own guidance (the comparison with Wall Street's consensus is summarised in the write-up)")
    md(f"{badge('fc')} How the forecast would have done in every past quarter, Q3 2023 - Q2 2026, using only the data "
       f"public on each day - misses included - next to {ws_txt}. Pick when the forecast was made in each section: "
       f"early in the quarter little is known yet, after the quarter almost everything is.", unsafe_allow_html=True)
    for lq in sorted(S_.locks, reverse=True):
        live_test(lq)
    track_section("eps", "Adjusted EPS", "Profit per share, excluding one-off items.")
    track_section("operating_margin", "Operating margin",
                  "The share of revenue left as operating profit. No Wall Street consensus is available for margin in our "
                  "data, so the comparison is with Delta's guidance only.")
    track_section("revenue_pct", "Total revenue growth",
                  "Revenue vs the same quarter last year. This line uses flight-tracking capacity as well as fuel; it was "
                  "chosen after testing on these same quarters (notebook 07, 7e), so treat it as provisional until it keeps "
                  "working on new quarters - Q3 2026 is the first. Delta gave no revenue guidance for Q3 2023.")

    # did the forecast see Delta's guidance changes coming?
    st.subheader("Did the forecast see Delta's guidance changes coming?")
    md("Sometimes Delta changes its guidance in the middle of a quarter, in a filing between earnings reports - for "
       "example, cutting its EPS range when fuel prices jump. A useful forecast should already be pointing that way "
       "before Delta says so. For each of these changes, we looked at the forecast 45, 30, 14 and 1 day before it.")
    md("- **Pointing that way:** the forecast was on that side of the middle of Delta's range (leaning toward the change).\n"
       "- **Clearly outside the range:** the forecast was beyond Delta's range in that direction - a strong signal, "
       "shown on the Forecast page as ▼ / ▲.\n"
       "- **45+ days before** means it was already pointing that way at the earliest point we checked.")
    U = S_.updates_test
    ev = U[U.tag == "before update"].dropna(subset=["direction"])
    ev = ev[ev.direction != 0]
    def early(g):
        if not len(g):
            return "no"
        k = int(g.days_before.max())
        return "45+ days before" if k >= 45 else f"{k} day{'s' if k > 1 else ''} before"
    rows = []
    for (qq, u, m), g in ev.groupby(["quarter", "update_date", "metric"]):
        dirn = g.direction.iloc[0]
        clear = g[np.sign(np.where(g.forecaster < g.low, -1, np.where(g.forecaster > g.high, 1, 0))) == dirn]
        lean = g[np.sign(g.forecaster - g.mid) == dirn]
        rows.append({"Delta changed its guidance on": f"{pd.Timestamp(u):%d %b %Y}", "Number": NICE[m],
                     "Delta moved it": "down" if dirn < 0 else "up", "Forecast pointing that way": early(lean),
                     "Forecast clearly outside the range": early(clear)})
    st.table(pd.DataFrame(rows).style.hide(axis="index"))
    ctl = U[U.tag == "no update"]
    fa = ((ctl.forecaster < ctl.low) | (ctl.forecaster > ctl.high)).mean() * 100
    st.caption(f"False alarms: in quarters where Delta did not change its guidance, the forecast sat outside Delta's range "
               f"{fa:.0f}% of the time. (Notebook 07 reports 5% for the engine as it was locked on 7 Oct; the app uses the "
               f"updated engine with the flight-tracking outage detector.)")


# ------------------------------------------------------------------------------------------------ 5. how it works
DECISIONS = [
    ("Fuel news, in all three numbers", "Used",
     "EPS 14-18¢ off vs 14-19¢ for Delta's guidance alone, and never worse. Fuel is the one input we see as precisely as Delta does.",
     "Not significant (smallest p ≈ 0.21): fuel moved little after Delta guided in most of these quarters", "07, section 2"),
    ("Flight-tracking capacity news, for revenue", "Used (revenue only)",
     "With fuel news: 1.56 vs 1.89 points off on average, closer than guidance in 7 of 11 quarters.",
     "p ≈ 0.05 (one average per quarter). Chosen after earlier tests on the same quarters, so somewhat flattering", "07, 7e"),
    ("Flight-tracking capacity news, for EPS and margin", "Not used",
     "EPS 21¢ vs 19¢ off on day 20, even with OpenSky's outages fixed: Delta's guidance already plans its costs around its schedule.",
     "Significantly worse (p ≈ 0.02)", "07, 7b"),
    ("Demand news (TSA travelers)", "Not used",
     "EPS 20¢ vs 19¢ off on day 20: Delta already knows its own bookings when it guides.",
     "Not significant (p ≈ 0.51)", "07, section 2"),
    ("The simulator's own number as the forecast", "Used as the engine, not the call",
     "On its own it misses EPS by 31-34¢ vs 14-19¢ for guidance. When it disagreed most with the forecast, the actual moved toward it only 5 times in 16.",
     "No relation (p ≈ 0.26)", "06; 07, 7f"),
    ("OpenSky outage detector", "Used",
     "Watches five rivals' flights: when all drop together, it is a gap in the tracking, not Delta flying less. Capacity estimate error on day 20: 3.6% → 2.2%.",
     "Check of the input (no p-value)", "07, 7a"),
    ("Revenue range widened 1.75x", "Used",
     "The actual lands inside the 80% range in 36 of 44 quarter-dates (82%), up from 26 (59%); 77% when each quarter is left out in turn. EPS (90%) and margin (81%) needed no change.",
     "Coverage check (no p-value)", "07, 7h"),
    ("Simulator changes start on the chosen date", "Used (Simulator)",
     "The share of the quarter still to come is measured from last year's daily flying and passengers, not counted in calendar days (on average 1-1.5 points apart, up to 4.5 in a first quarter). Five checks pass.",
     "Design check (no p-value)", "05b"),
    ("Wall Street's consensus (I/B/E/S)", "Benchmark",
     "Early in the quarter the forecast keeps pace: EPS 17.6¢ vs 17.6¢ off on day 20, revenue 1.79 vs 1.97 points. Seven days after the quarter analysts are well ahead: EPS 14.0¢ vs 7.2¢.",
     "Not significant either way (p ≥ 0.12)", "07, 7g"),
]


def page_how():
    st.title("Method & evidence")
    md("""
The details behind the other pages: how the pieces fit together, where AI is used, why the forecast is built the way
it is, and its limits. Every number is reproducible from the project notebooks (00-08). The data itself, how it was
checked and what was set aside are on **The Data** page.
""")
    st.subheader("How the pieces fit together")
    md("""
Each box is a link measured on Delta's own history (notebooks 01-04). The simulator runs all the links together
50,000 times, each time drawing every uncertain piece (measurement errors, the size of each link, surprises), so it
gives a range of outcomes rather than one number.

The forecast starts from the middle of Delta's guidance and adds what the news since then is worth. To measure the
news, the simulator runs twice on the same draws: once with today's inputs and once with the inputs as they stood when
Delta guided. The difference between the two runs is the news; the simulator's own errors cancel out. The range
combines the spread of that difference with how far Delta's guidance has missed in the past at the same point in the
quarter.
""")
    st.graphviz_chart("""
digraph { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef4fc", color="#2a78d6", fontname="Helvetica", fontsize=11];
 edge [color="#a3a29c"];
 cap [label="Capacity\\n(OpenSky flight tracking,\\nBTS, Delta's plan)"]; dem [label="Demand\\n(TSA, BTS)"]; fuel [label="Jet fuel\\n(EIA)"];
 lf [label="Load factor"]; y [label="Fares / yield"]; rev [label="Revenue"]; fc [label="Fuel cost"]; nf [label="Non-fuel costs"];
 ps [label="Profit sharing"]; eps [label="EPS", fillcolor="#2a78d6", fontcolor="white"];
 cap -> lf; dem -> lf; lf -> y; fuel -> y; cap -> rev; lf -> rev; y -> rev; fuel -> fc; cap -> fc; cap -> nf; rev -> ps; fc -> ps; nf -> ps; ps -> eps; rev -> eps; fc -> eps; nf -> eps;
 g [label="Delta's guidance\\n(AI-read from SEC filings)", fillcolor="#fdf0ea", color="#eb6834"]; g -> eps [style=dashed, label="forecast = guidance + news"];
}""")
    st.subheader("Where AI is used")
    md("""
- **Reading Delta's filings:** OpenAI read 239 of Delta's SEC filings (82 earnings releases and 157 other 8-K filings)
  and pulled out **1,819 guidance statements**; each is checked against its filing (quote present, numbers in the
  quote, period not over, metric named, units fit) - 69.5% of dated items pass, and only those are used.
- **The event box:** turns a news headline into simulator levers, sized only from numbers in the headline, Delta's
  documented past events (the July 2024 outage, the 2023 fuel spike, the 2025 demand slowdown) and a table of every
  large weather disruption at Delta since 2019 measured in BTS On-Time data (only those public by the chosen date). It
  must show its arithmetic and sources, and a person approves before anything runs.
""")
    st.subheader("Why the forecast is built this way")
    md("""
Each choice was tested on the same past quarters (Q3 2023 - Q2 2026) against Delta's guidance midpoint on its own.
The **p-value** is the chance of a difference at least this large if the choice really made no difference; below
0.05 is the usual bar for "significant". With 11-12 quarters most differences are not significant, so a tie goes to
the simpler choice.
""")
    st.table(pd.DataFrame(DECISIONS, columns=["Choice", "Decision", "Evidence", "Significance", "Notebook"]).style.hide(axis="index"))
    st.caption("Datasets tried and set aside (rivals' flying, the airfare price index, diesel and crude prices): The Data page.")
    st.subheader("Limits")
    md("""
- 12 quarters of backtest (Q3 2023 - Q2 2026) - enough to reject ideas, thin for proving small gains.
- No edge over Wall Street's **final** EPS consensus yet; the edge is timing (mid-quarter) and the revenue line.
- The revenue method (fuel + capacity news, wider range) is newer and was chosen on the same quarters it is scored on;
  Q3 2026 is its first new quarter.
- The simulator's own number is not used for the call: it explains the forecast and tests changes, while the call
  comes from Delta's guidance plus the news.
- Our copy of Wall Street's consensus (I/B/E/S) runs to 20 Aug 2026; after that the app uses the final consensus just
  before the report, from press coverage.
- Flight tracking misses some flights and has its own outages (detected and corrected); it is calibrated to BTS with a
  ~3-month delay.
- Simulator changes use last year's daily pattern for the rest of the quarter, and a change is the same size every day
  it lasts.

""")
    st.subheader("How to use FlightPath")
    md("""
- **Use the forecast as a mid-quarter check on Delta's guidance.** In most quarters it stays close to guidance. When it
  moves outside Delta's range, something big has changed since Delta guided (so far, usually fuel). That's the time to
  look closer.
- **Read the range, not just the middle.** The 80% range shows how far off the forecast has been in the past. A
  forecast inside Delta's range with a wide spread means there's no clear signal yet.
- **Know when Wall Street is the better guide.** Early in the quarter the forecast keeps pace with analysts. Late in the
  quarter they pull ahead, so near the report, weight the consensus more.
- **Use the simulator to size a shock, not to forecast.** Use it to see what a fuel spike, a storm or a schedule cut
  would do, and where it hits the income statement. Its own number is not the call.
- **Treat the revenue line as provisional.** It's the newest part and has been tested on 11 quarters.
- FlightPath is a research tool built on public data, not investment advice.

*Flight data: The OpenSky Network, https://opensky-network.org.*
""")


# ------------------------------------------------------------------------------------------------ 0. start here
def page_start():
    st.title("FlightPath")
    md("#### A living system of Delta's quarter")
    md("""
FlightPath follows **Delta Air Lines' current quarter while it is still happening**, using only public data, and
answers three questions an investor cares about before Delta reports: **where will the results land, what could
move them, and how far can this be trusted?**
""")
    with st.container(border=True):
        md("""
**What FlightPath predicts.** For each Delta quarter, FlightPath predicts the three numbers Delta guides on and
reports: **adjusted earnings per share (EPS)**, **operating margin** and **total revenue growth** vs the same quarter
last year. The simulator also shows the steps in between: capacity (seats flown), how full the planes are, fares, fuel
and other costs, and profit sharing.
""")
    st.subheader("Two models, one system")
    a, b = st.columns(2)
    with a.container(border=True):
        md(f"{badge('sim')}", unsafe_allow_html=True)
        md("""
**The simulator** is a working model of Delta's business. Change one thing, like fuel prices or flights flown, and it
follows the effect through the business: how full the planes are, ticket prices, revenue, costs, and finally earnings.
Each connection is measured from Delta's own history. It runs every quarter 50,000 times to show a range of likely
outcomes, not just one number.
""")
    with b.container(border=True):
        md(f"{badge('fc')}", unsafe_allow_html=True)
        md("""
**The forecast** is Delta's plan, updated. Delta knows its own bookings and contracts, so its guidance is the best starting point. The forecast takes that plan
and moves it by **what the simulator says the news since then is worth**: jet fuel for EPS and margin, plus flight
tracking of Delta's capacity for revenue.
""")
    md("""
**In one line:** Delta's guidance is the plan; the simulator measures how much the world has changed since the plan
was made, and what it would take to break it; the forecast is the plan, updated.

**Where it helps most.** In most quarters little changes after Delta guides, so the forecast stays close to guidance.
It earns its keep when something big moves. It flagged Delta's September 2023 guidance cut by late July, about 7
weeks before Delta announced it. In Q3 2026 it was at $1.95 by late July while Wall Street sat at $2.25; analysts reached about $1.88 by
October. It can size what a storm or a fuel spike is likely to do before Delta reports it. Its revenue line, using
flight tracking, beat Delta's own guidance in 7 of 11 past quarters. Early in the quarter it keeps pace with Wall
Street's consensus on EPS and is slightly better on revenue, though the differences are too small to be statistically
significant. By the end of the quarter, analysts are well ahead.
""")
    cards = {"Forecast": ("fc", "Starts from Delta's most recent guidance and adds what has happened since, using public "
                          "data on fuel prices and flights flown, to track whether and how the expected result is changing."),
             "Simulator": ("sim", "Enter your own changes to fuel, capacity, demand, fares or costs, or describe a news event, "
                           "and see how each part of the business responds, step by step, down to EPS."),
             "The Data": ("data", "See every data source behind FlightPath, how we checked it, when each becomes public, and "
                          "the latest daily data: Delta's flights from flight tracking, jet fuel prices and air travelers."),
             "Track record": ("fc", "See how the forecast would have done in every past quarter (Q3 2023 – Q2 2026), next to "
                              "Delta's guidance, Wall Street and what Delta actually reported.")}
    for heading, names in [("Use it", ["Forecast", "Simulator"]), ("Check it", ["The Data", "Track record"])]:
        st.subheader(heading)
        for c, name in zip(st.columns(2), names):
            kind, txt = cards[name]
            with c.container(border=True):
                tags = badge("fc") + badge("sim") if kind == "both" else badge(kind)
                md(f"{tags}**{name}**", unsafe_allow_html=True)
                md(txt)
                st.page_link(PAGE_OBJ[name], label=f"Open {name.lower()} →")
    st.caption("How to use FlightPath, how the models fit together, where AI is used, why the forecast is built this way, "
               "and its limits: Method & evidence.")


# ------------------------------------------------------------------------------------------------ navigation + sidebar
PAGE_OBJ = {"Start here": st.Page(page_start, title="Start here", url_path="start", default=True),
            "Forecast": st.Page(page_card, title="Forecast", url_path="forecast"),
            "Simulator": st.Page(page_scenarios, title="Simulator", url_path="simulator"),
            "The Data": st.Page(page_signals, title="The Data", url_path="data"),
            "Track record": st.Page(page_track, title="Track record", url_path="track-record"),
            "Method & evidence": st.Page(page_how, title="Method & evidence", url_path="method")}
nav = st.navigation(list(PAGE_OBJ.values()), position="hidden")

with st.sidebar:
    md("## ✈ FlightPath")
    st.caption("A living system of Delta's quarter")
    st.page_link(PAGE_OBJ["Start here"], label="Start here")
    md("<span class='small'>USE IT</span>", unsafe_allow_html=True)
    for name in ["Forecast", "Simulator"]:
        st.page_link(PAGE_OBJ[name], label=name)
    md("<span class='small'>DETAILS</span>", unsafe_allow_html=True)
    for name in ["The Data", "Track record", "Method & evidence"]:
        st.page_link(PAGE_OBJ[name], label=name)
    st.divider()
    st.caption(f"Snapshot built {S_.meta['built']}.  \nData through: " +
               "; ".join(f"{k} {v}" for k, v in S_.meta["data_through"].items()))

q = d = P = qname = reported = known_actual = None          # set by controls() on the pages that use them
nav.run()
