"""Shared utilities for electronics lab analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint
from pathlib import Path


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


def load_oscilloscope_data(data_file: str, data_dir: Path) -> pd.DataFrame:
    """Load oscilloscope data from CSV file and convert time to microseconds."""
    ureg = pint.UnitRegistry()
    voltage_df = pd.read_csv(data_dir / f"{data_file}.csv")
    s_to_us = ureg.Quantity(1.0, ureg.second).to(ureg.microsecond).magnitude
    voltage_df["t_in"] = (voltage_df["t_in"] - voltage_df["t_in"].min()) * s_to_us
    voltage_df["t_out"] = (voltage_df["t_out"] - voltage_df["t_out"].min()) * s_to_us
    return voltage_df


def plot_oscilloscope_data(
    time_in: np.ndarray | pd.Series,
    voltage_in: np.ndarray | pd.Series,
    time_out: np.ndarray | pd.Series,
    voltage_out: np.ndarray | pd.Series,
    file_out: str,
    output_dir: Path,
) -> None:
    """Plot oscilloscope data."""
    plt.figure(figsize=(10, 6))
    plt.plot(time_in, voltage_in, label="Voltage In", color="goldenrod")
    plt.plot(time_out, voltage_out, label="Voltage Out", color="blue")
    plt.xlabel(r"Time $t$ (μs)")
    plt.ylabel(r"Voltage Out $V_{\mathrm{Out}}$ (V)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f"{file_out}.png", dpi=150)
    plt.show()


def find_closest_index(series: pd.Series[float], target_value: float) -> int:
    """Find the index of the value closest to target_value in the series."""
    return int(series.sub(target_value).abs().idxmin())


def calculate_charge_tau(
    time_series: pd.Series,
    voltage_series: pd.Series,
    start_idx: int,
    end_idx: int,
) -> tuple[float, float, float]:
    """Calculate tau for charging phase.

    Returns: (tau_time, tau_voltage, tau_voltage_calc)
    """
    end_volt = float(voltage_series.iloc[end_idx])
    tau_volt_calc = np.round((1 - (1 / np.e)) * end_volt, 3)
    tau_idx = find_closest_index(voltage_series[start_idx:end_idx], tau_volt_calc)
    tau_time = float(time_series.iloc[tau_idx])
    tau_volt = float(voltage_series.iloc[tau_idx])
    return (tau_time, tau_volt, tau_volt_calc)


def calculate_discharge_tau(
    time_series: pd.Series,
    voltage_series: pd.Series,
    start_idx: int,
    end_idx: int,
) -> tuple[float, float, float]:
    """Calculate tau for discharging phase.

    Returns: (tau_time, tau_voltage, tau_voltage_calc)
    """
    start_volt = float(voltage_series.iloc[start_idx])
    tau_volt_calc = np.round((1 / np.e) * start_volt, 3)
    tau_idx = find_closest_index(voltage_series[start_idx:end_idx], tau_volt_calc)
    tau_time = float(time_series.iloc[tau_idx])
    tau_volt = float(voltage_series.iloc[tau_idx])
    return (tau_time, tau_volt, tau_volt_calc)
