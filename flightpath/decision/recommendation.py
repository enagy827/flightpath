"""Decision card: what FlightPath thinks vs. Delta's guidance, how confident, and what drives it.

Direction rule (pre-registered, not tuned): operating-impact FSS > +0.5 -> "above guidance" (long bias),
< -0.5 -> "below guidance" (short bias), otherwise "in line" (no trade). Conviction: |FSS| < 0.5 low,
0.5-1.5 medium, > 1.5 high. Not position sizing - that's Longaeva's call.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..simulation.shocks import PRESET_SCENARIOS
from .expectations import baseline
from .signal import operating_impact, surprise


@dataclass
class DecisionCard:
    quarter: str
    as_of: str
    baseline_source: str
    view: str
    conviction: str
    operating_fss: float
    p_revenue_beats_baseline: float
    revenue_vs_baseline_pct: float
    fuel_price_model: float
    fuel_price_baseline: float | None
    key_numbers: dict = field(default_factory=dict)
    drivers: list = field(default_factory=list)
    scenarios: list = field(default_factory=list)
    caveats: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _conviction(z: float) -> str:
    a = abs(z)
    return "low" if a < 0.5 else ("medium" if a < 1.5 else "high")


def build_card(engine, quarter: str, as_of, run_scenarios: bool = True) -> tuple[DecisionCard, object]:
    st = engine.run(quarter, as_of)
    base = baseline(engine.store, quarter, as_of)
    rev = surprise(st.passenger_revenue, base["passenger_revenue"])
    op = operating_impact(st, base)
    z = op["fss"]
    view = "above guidance (long bias)" if z > 0.5 else ("below guidance (short bias)" if z < -0.5 else "in line (no trade)")
    nc = st.meta["nowcast"]
    q = lambda x, p: float(np.quantile(x, p))
    key = {
        "ASM (bn)": (float(np.median(st.asm)) / 1e9, q(st.asm, .1) / 1e9, q(st.asm, .9) / 1e9),
        "Load factor (%)": (float(np.median(st.load_factor)) * 100, q(st.load_factor, .1) * 100, q(st.load_factor, .9) * 100),
        "Yield (cents)": (float(np.median(st.yield_)), q(st.yield_, .1), q(st.yield_, .9)),
        "Passenger revenue ($bn)": (float(np.median(st.passenger_revenue)) / 1e9, q(st.passenger_revenue, .1) / 1e9, q(st.passenger_revenue, .9) / 1e9),
        "Fuel price ($/gal)": (float(np.median(st.fuel_price)), q(st.fuel_price, .1), q(st.fuel_price, .9)),
        "Fuel expense ($bn)": (float(np.median(st.fuel_expense)) / 1e9, q(st.fuel_expense, .1) / 1e9, q(st.fuel_expense, .9) / 1e9),
    }
    drivers = [
        f"TSA throughput {nc['demand']['tsa_yoy'] * 100:+.1f}% YoY ({nc['demand']['tsa_share_observed']:.0%} of quarter observed)",
        f"Capacity estimate {np.median(st.asm) / nc['demand']['asm_prior_year'] * 100 - 100:+.1f}% YoY "
        f"(method: {', '.join(nc['capacity']['components'])})",
        f"Fuel: model ${np.median(st.fuel_price):.2f}/gal vs guidance "
        + (f"${base['fuel_price']:.2f}" if base.get("fuel_price") else "n/a"),
        f"Revenue gap vs baseline ${op['revenue_gap_mean'] / 1e6:+,.0f}M; fuel cost gap ${op['fuel_gap_mean'] / 1e6:+,.0f}M",
    ]
    scen = []
    if run_scenarios:
        for sc in PRESET_SCENARIOS[1:]:
            s2 = engine.run(quarter, as_of, scenario=sc)
            o2 = operating_impact(s2, base)
            scen.append({"scenario": sc.name, "description": sc.description,
                         "operating_impact_$M": o2["model_mean"] / 1e6,
                         "change_vs_base_$M": (o2["model_mean"] - op["model_mean"]) / 1e6})
    card = DecisionCard(
        quarter=quarter, as_of=str(as_of), baseline_source=base["source"], view=view,
        conviction=_conviction(z), operating_fss=z, p_revenue_beats_baseline=rev["p_above"],
        revenue_vs_baseline_pct=rev["pct_vs_baseline"], fuel_price_model=float(np.median(st.fuel_price)),
        fuel_price_baseline=base.get("fuel_price"), key_numbers=key, drivers=drivers, scenarios=scen,
        caveats=["Passenger revenue baseline = guided adjusted revenue growth + latest passenger-vs-total gap.",
                 "Non-fuel costs not modeled (assumed in line with guidance).",
                 "Backtest: ~3% passenger revenue error, 7 quarters - small sample."])
    return card, st
