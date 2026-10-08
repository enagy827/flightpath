"""Revenue module (identity): Passenger Revenue = RPM x Passenger Yield   (proposal eq. 3)."""
from .base import Module


class RevenueModule(Module):
    name = "revenue"
    inputs = ("rpm", "yield_")
    outputs = ("passenger_revenue",)

    def fit(self, store, as_of):
        pass

    def nowcast(self, store, as_of, quarter):
        return {}

    def simulate(self, state, shocks, rng):
        state.passenger_revenue = state.rpm * state.yield_ / 100.0     # miles x cents/mile -> $
        return state
