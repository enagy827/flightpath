"""Simulation engine: fit modules point-in-time, nowcast, then propagate n_paths through the chain.

    capacity -> fuel -> demand (load factor, RPM) -> pricing (yield) -> revenue

Scenarios (simulation/shocks.py) set hooks in state.meta (fuel_price_multiplier, demand_shock,
capacity_multiplier) so a shock flows only downstream through the business mechanisms.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..modules.capacity import CapacityModule
from ..modules.demand import DemandModule
from ..modules.fuel import FuelModule
from ..modules.pricing import PricingModule
from ..modules.revenue import RevenueModule
from .state import BusinessState


class SimulationEngine:
    def __init__(self, store, n_paths: int = 10_000, seed: int = 42, capacity: CapacityModule | None = None):
        self.store = store
        self.n_paths = n_paths
        self.seed = seed
        self.modules = {"capacity": capacity or CapacityModule(), "fuel": FuelModule(),
                        "demand": DemandModule(), "pricing": PricingModule(), "revenue": RevenueModule()}
        self.as_of = None

    def fit(self, as_of) -> None:
        for m in self.modules.values():
            m.fit(self.store, as_of)
        self.as_of = as_of

    def run(self, quarter: str, as_of=None, scenario=None, seed: int | None = None) -> BusinessState:
        as_of = as_of or self.as_of
        if self.as_of is None or pd.Timestamp(as_of) != pd.Timestamp(self.as_of):
            self.fit(as_of)
        # use Delta's capacity guidance when it's public at as_of (validated as more reliable than trend)
        from ..decision.expectations import guidance_for
        g = guidance_for(quarter, as_of)
        cap = self.modules["capacity"]
        if g and g.get("capacity_yoy"):
            lo, hi = g["capacity_yoy"]
            cap.guidance_yoy, cap.guidance_sd = (lo + hi) / 2, max((hi - lo) / 2 / 1.28, 0.005)
        else:
            cap.guidance_yoy = None
        st = BusinessState("DAL", quarter, pd.Timestamp(as_of).date(), self.n_paths)
        st.meta["nowcast"] = {k: m.nowcast(self.store, as_of, quarter) for k, m in self.modules.items()}
        if scenario is not None:
            scenario.apply(st)
        rng = np.random.default_rng(self.seed if seed is None else seed)
        shocks = {}   # independent shocks by default; correlated draws can be injected here
        st = self.modules["capacity"].simulate(st, shocks, rng)
        st.asm = st.asm * st.meta.get("capacity_multiplier", 1.0)
        for k in ("fuel", "demand", "pricing", "revenue"):
            st = self.modules[k].simulate(st, shocks, rng)
        apply_disruption(st)
        return st


# Operational disruption: unlike a planned capacity cut, cancelled flights lose their passengers
# (no load-factor/yield offset) and add recovery costs (crew, hotels, refunds, compensation).
# Cost calibrated to the July 2024 outage: ~2% of the quarter's mainline flying lost (BTS T-100) and
# Delta cited roughly $500M total impact (VERIFY figure in Delta's Q3-2024 release/call); ~$280M of that
# is lost revenue at ~2% of passenger revenue, leaving ~$110M of extra cost per 1% of flying lost.
DISRUPTION_COST_PER_PCT = 110e6


def apply_disruption(st) -> None:
    share = st.meta.get("disruption_share", 0.0)
    if not share:
        return
    st.meta["disruption_lost_revenue"] = st.passenger_revenue * share
    st.passenger_revenue = st.passenger_revenue * (1 - share)
    st.rpm = st.rpm * (1 - share)
    st.fuel_gallons = st.fuel_gallons * (1 - share)
    st.fuel_expense = st.fuel_expense * (1 - share)
    st.effective_asm = st.asm * (1 - share)
    st.disruption_cost = np.full(st.n_paths, share * 100 * DISRUPTION_COST_PER_PCT)
