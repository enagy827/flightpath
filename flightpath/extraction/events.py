"""News -> events -> simulation shocks.

The edge is not reading the headline (everyone does) - it is translating an event into
a QUANTIFIED effect on Delta's quarter within minutes, consistently, with uncertainty.

Pipeline:
  1. detect: news/FAA/8-K documents -> LLM classifies: is this a Delta-relevant event?
  2. tag (LLM, qualitative only): event_type, affected driver(s), hubs, expected duration,
     severity bucket, first-seen timestamp. The LLM does NOT output dollar impacts.
  3. size (quant, not LLM): magnitude from historical analogs - past events of the same
     type, measured in OpenSky / BTS data (e.g. cancellations per day of a hub ground stop).
  4. shock: convert into a Scenario / shock on the relevant module(s) for the current quarter.

Demo case study: July 2024 IT outage (CrowdStrike) - Delta cancelled thousands of flights
over several days and later cited a large financial impact. Show how FlightPath would have
sized it from flight-tracking data before Delta disclosed the cost.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class EventType(str, Enum):
    IT_OUTAGE = "it_outage"
    WEATHER = "weather"
    ATC_CONSTRAINT = "atc_constraint"          # ground stops, staffing, shutdown
    LABOR = "labor"                            # strike threat, contract, union drive
    FLEET = "fleet"                            # groundings, engine issues, delivery delays
    GUIDANCE_UPDATE = "guidance_update"        # mid-quarter 8-K / conference comments
    COMPETITOR_CAPACITY = "competitor_capacity"  # competitor cuts, exits, bankruptcy
    DEMAND_SHOCK = "demand_shock"              # macro, geopolitical, health
    OTHER = "other"


class Event(BaseModel):
    event_type: EventType
    headline: str
    quote: str                      # verbatim supporting text
    drivers: list[str]              # e.g. ["capacity", "disruption", "pricing"]
    hubs: list[str] = []
    expected_duration_days: float | None = None
    severity: str                   # "low" | "medium" | "high" - qualitative only
    first_seen: datetime            # earliest publication across sources
    sources: list[str]
    confidence: float = Field(ge=0, le=1)


def detect_and_tag(documents: list, llm) -> list[Event]:
    """Steps 1-2. Dedupe the same event across outlets (cluster by time + type + hub). TODO."""
    raise NotImplementedError


def size_event(event: Event, analogs) -> dict:
    """Step 3: distribution of impact (e.g. lost ASM, extra cost) from historical analogs. TODO."""
    raise NotImplementedError


def event_to_scenario(event: Event, impact: dict):
    """Step 4: returns simulation.shocks.Scenario applied to the current quarter. TODO."""
    raise NotImplementedError
