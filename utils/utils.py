"""Shared utilities for electronics lab analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def frequency_of_square_wave(
    time_us: np.ndarray | pd.Series,
    voltage: np.ndarray | pd.Series,
) -> tuple[float, float]:
    """Estimate frequency from time and voltage using median period between rising edges.

    Time is assumed to be in microseconds. Returns (freq_Hz, period_us).
    Returns (nan, nan) if fewer than two rising edges are found.
    """
    t = np.asarray(time_us)
    v = np.asarray(voltage)
    mid = (v.max() + v.min()) / 2
    above = v > mid
    rising = np.where(np.diff(above.astype(int)) == 1)[0]

    if len(rising) >= 2:
        period_us = float(np.median(np.diff(t[rising])))
        period_s = period_us * 1e-6
        freq_Hz = 1.0 / period_s
        print(
            f"Input period (median): {period_us:.2f} μs  →  frequency: {round(freq_Hz):.0f} Hz"
        )
        return freq_Hz, period_us

    print("Could not find enough rising edges to compute frequency.")
    return float("nan"), float("nan")
