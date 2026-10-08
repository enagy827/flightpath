"""Forecast accuracy and calibration metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def mape(forecast: pd.Series, realized: pd.Series) -> float:
    raise NotImplementedError


def interval_coverage(results: pd.DataFrame, level: float) -> float:
    """Share of realized values inside the central `level` interval (target: ~= level). TODO."""
    raise NotImplementedError


def crps(samples: np.ndarray, realized: float) -> float:
    """Continuous ranked probability score from simulated paths (properscoring). TODO."""
    raise NotImplementedError


def pit_histogram(results: pd.DataFrame) -> np.ndarray:
    """Probability integral transform values; uniform = well calibrated. TODO."""
    raise NotImplementedError


def skill_vs_benchmark(model_err: pd.Series, bench_err: pd.Series) -> dict:
    """Relative error + Diebold-Mariano test. TODO."""
    raise NotImplementedError
