"""Disruption module. Promoted to P1: now fed by real-time data, not just weather.

Inputs:
  - OpenSky: flights FLOWN per hub per day; a drop vs. same-weekday baseline = disruption
    (OpenSky does not record cancellations directly)
  - BTS on-time performance: true cancellations (lagged; backtest + calibration)
  - FAA: ground delay programs, ground stops; OPSNET / ASPM daily ops + delays at hubs
  - events.py: IT outages, labor, fleet groundings (sized from historical analogs)
  - NOAA weather (P2): only as an early warning feeding the same channel
Effects: cancellations reduce effective ASM (feedback to capacity) and add costs
(crew, customer compensation, rebooking). Validate on July 2024 IT outage case study.
"""
from .base import Module


class DisruptionModule(Module):
    name = "disruption"
    inputs = ("asm",)
    outputs = ("cancel_rate", "effective_asm", "disruption_cost")

    def fit(self, store, as_of):
        raise NotImplementedError  # TODO: cost per cancelled departure from past events

    def nowcast(self, store, as_of):
        raise NotImplementedError

    def simulate(self, state, shocks, rng):
        raise NotImplementedError
