"""Evidence schema returned by the extractor (validated with pydantic)."""
from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class Driver(str, Enum):
    DEMAND_LEISURE = "demand_leisure"
    DEMAND_CORPORATE = "demand_corporate"
    DEMAND_PREMIUM = "demand_premium"
    DEMAND_INTERNATIONAL = "demand_international"
    PRICING = "pricing"
    CAPACITY = "capacity"
    FUEL = "fuel"
    LABOR = "labor"
    OPERATIONS = "operations"
    OTHER_RISK = "other_risk"


class Direction(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class Evidence(BaseModel):
    driver: Driver
    direction: Direction
    claim: str                        # one-sentence paraphrase
    quote: str                        # verbatim supporting text (required)
    numeric_value: float | None = None  # ONLY if stated verbatim in quote
    numeric_unit: str | None = None
    period_referenced: str | None = None  # e.g. "Q3 2025", "second half"
    confidence: float = Field(ge=0, le=1)
    source_doc: str                   # file / URL
    available_at: date                # publication date of the document
