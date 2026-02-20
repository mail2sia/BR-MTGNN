# src/drift.py
# Simple drift detection utilities (CUSUM) and interval widening helper.
from __future__ import annotations
import numpy as np
from typing import Tuple

def cusum_detect(residuals: np.ndarray, k: float = 0.5, h: float = 8.0) -> Tuple[bool, float]:
    """CUSUM test on residuals (1D array). Returns (drift_flag, score)."""
    x = np.asarray(residuals, dtype=float)
    x = x - np.mean(x) if x.size > 0 else x
    gp = 0.0; gn = 0.0; score = 0.0
    for xi in x:
        gp = max(0.0, gp + xi - k)
        gn = max(0.0, gn - xi - k)
        score = max(score, gp, gn)
    return (score > h), float(score)

def widen_intervals(lo: np.ndarray, hi: np.ndarray, factor: float = 1.2):
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) * factor
    return mid - half, mid + half
