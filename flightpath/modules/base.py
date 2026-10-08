"""Interface every business module implements."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import numpy as np

from ..simulation.state import BusinessState
from ..timeline import PointInTimeStore


class Module(ABC):
    name: str
    inputs: tuple[str, ...]   # state fields this module reads
    outputs: tuple[str, ...]  # state fields this module writes

    @abstractmethod
    def fit(self, store: PointInTimeStore, as_of: date) -> None:
        """Estimate parameters using ONLY data available at as_of."""

    @abstractmethod
    def nowcast(self, store: PointInTimeStore, as_of: date, quarter: str) -> dict:
        """Estimate for `quarter` (e.g. "2026Q3") using only data visible at as_of."""

    @abstractmethod
    def simulate(self, state: BusinessState, shocks: dict[str, np.ndarray],
                 rng: np.random.Generator) -> BusinessState:
        """Propagate n_paths through this mechanism, writing `outputs` into state."""
