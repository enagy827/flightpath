"""Counterfactual scenarios. Each sets a hook the modules read, so the shock propagates downstream only."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Scenario:
    name: str
    fuel_price_multiplier: float = 1.0     # market jet fuel x this for the rest of the quarter
    demand_shock: float = 0.0              # added to TSA YoY growth (log points), e.g. -0.05
    capacity_multiplier: float = 1.0       # PLANNED capacity change (Delta flies more/less by choice)
    disruption_share: float = 0.0          # share of the quarter's flying lost to an operational disruption
    description: str = ""
    extra: dict = field(default_factory=dict)

    def apply(self, state) -> None:
        state.meta["fuel_price_multiplier"] = self.fuel_price_multiplier
        state.meta["demand_shock"] = self.demand_shock
        state.meta["capacity_multiplier"] = self.capacity_multiplier
        state.meta["disruption_share"] = self.disruption_share
        state.meta["scenario"] = self.name


PRESET_SCENARIOS = [
    Scenario("Base case"),
    Scenario("Jet fuel +20%", fuel_price_multiplier=1.2, description="Market jet fuel 20% higher for the rest of the quarter"),
    Scenario("Demand -5%", demand_shock=-0.05, description="Industry passenger growth 5 points weaker"),
    Scenario("Major IT outage / hub meltdown", disruption_share=0.02,
             description="~2% of the quarter's flights cancelled, like July 2024 (~6,000 mainline departures)"),
    Scenario("Capacity +2% vs plan", capacity_multiplier=1.02, description="Delta flies 2% more than expected"),
]
