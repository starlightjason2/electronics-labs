"""Shared utilities for electronics lab analysis."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint
from matplotlib.markers import MarkerStyle


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


def plot_tau_charging(
    ax,
    time_series: pd.Series,
    voltage_series: pd.Series,
    start_idx: int,
    end_idx: int,
    color: str = "red",
    time_unit: str = "microsecond",
) -> float:
    """Plot tau measurement for charging phase.

    Returns: tau value in specified time units
    """
    ureg = pint.UnitRegistry()
    start_time = float(time_series.iloc[start_idx])
    tau_time, tau_volt, tau_volt_calc = calculate_charge_tau(
        time_series, voltage_series, start_idx, end_idx
    )
    tau = ureg.Quantity(tau_time - start_time, getattr(ureg, time_unit))

    ax.plot(
        tau_time,
        tau_volt_calc,
        "o",
        color=color,
        label=f"$\\tau = {tau.magnitude:.1f}$ μs",
    )
    ax.plot(
        [start_time, tau_time],
        [tau_volt, tau_volt],
        color=color,
        linestyle="--",
        alpha=0.7,
    )
    return tau.magnitude


def plot_tau_discharging(
    ax,
    time_series: pd.Series,
    voltage_series: pd.Series,
    start_idx: int,
    end_idx: int,
    color: str = "green",
    time_unit: str = "microsecond",
) -> float:
    """Plot tau measurement for discharging phase.

    Returns: tau value in specified time units
    """
    ureg = pint.UnitRegistry()
    start_time = float(time_series.iloc[start_idx])
    tau_time, tau_volt, tau_volt_calc = calculate_discharge_tau(
        time_series, voltage_series, start_idx, end_idx
    )
    tau = ureg.Quantity(tau_time - start_time, getattr(ureg, time_unit))

    ax.plot(
        tau_time,
        tau_volt_calc,
        "o",
        color=color,
        label=f"$\\tau = {tau.magnitude:.1f}$ μs",
    )
    ax.plot(
        [start_time, tau_time],
        [tau_volt, tau_volt],
        color=color,
        linestyle="--",
        alpha=0.7,
    )
    return tau.magnitude


# RC filter frequency response (Bode / phase) — shared by lab3_part4 and lab4_part2


def low_pass_phase_theoretical(omega_values: np.ndarray, omega_c: float) -> np.ndarray:
    """Phase for first-order low-pass: φ = -arctan(ω/ω_c)."""
    return -np.arctan(omega_values / omega_c)


def high_pass_phase_theoretical(omega_values: np.ndarray, omega_c: float) -> np.ndarray:
    """Phase for first-order high-pass: φ = arctan(ω_c/ω)."""
    return np.arctan(omega_c / omega_values)


def low_pass_gain_theoretical(omega_values: np.ndarray, omega_c: float) -> np.ndarray:
    """Gain in dB for first-order low-pass: 20 log10(1/√(1+(ω/ω_c)²))."""
    return 20 * np.log10(1 / np.sqrt(1 + (omega_values / omega_c) ** 2))


def high_pass_gain_theoretical(omega_values: np.ndarray, omega_c: float) -> np.ndarray:
    """Gain in dB for first-order high-pass."""
    return 20 * np.log10(
        (omega_values / omega_c) / np.sqrt(1 + (omega_values / omega_c) ** 2)
    )


def _log_extrapolate_frequency(
    f_hz: pd.Series, values: pd.Series, target_value: float, closest_idx: int
) -> float:
    """Estimate characteristic frequency by linear fit in (log10(f), y) and solve for target y."""
    if len(f_hz) < 2:
        return float(f_hz.iloc[closest_idx])

    other = closest_idx + 1 if closest_idx < len(f_hz) - 1 else closest_idx - 1
    x1 = math.log10(float(f_hz.iloc[closest_idx]))
    x2 = math.log10(float(f_hz.iloc[other]))
    y1 = float(values.iloc[closest_idx])
    y2 = float(values.iloc[other])

    if y2 == y1:
        return float(f_hz.iloc[closest_idx])
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    x_target = (target_value - b) / m
    return 10**x_target


def create_phase_plot(
    df: pd.DataFrame,
    omega_c: float,
    filter_type: str,
    output_dir: Path,
    *,
    dpi: int = 150,
) -> None:
    """Create phase plot (theory + experimental) and save. Uses lab4 standard: logspace(0,6),
    phi = ±τ·f·dT (minus for low-pass, plus for high-pass), -45° line with find_closest_index.
    """
    output_dir.mkdir(exist_ok=True)
    sign = -1 if filter_type == "low" else 1
    df["phi"] = sign * math.tau * df["f"] * df["dT"]

    plt.figure(figsize=(10, 6))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("$\\phi$ (degrees)")
    plt.grid(True, which="both", alpha=0.3)

    f_values = np.logspace(0, 6, num=100)
    omega_values = math.tau * f_values
    critical_omega = math.pi / 4 if filter_type == "high" else -math.pi / 4
    critical_deg = float(np.degrees(critical_omega))

    eq_text = ""
    if filter_type == "high":
        phase_theoretical = high_pass_phase_theoretical(omega_values, omega_c)
        eq_text = r"$\phi(f) = \arctan\left(\dfrac{f_c}{f}\right)$"
    elif filter_type == "low":
        phase_theoretical = low_pass_phase_theoretical(omega_values, omega_c)
        eq_text = r"$\phi(f) = -\arctan\left(\dfrac{f}{f_c}\right)$"

    plt.semilogx(
        f_values,
        np.degrees(phase_theoretical),
        label=f"{filter_type.capitalize()} Pass Phase (Theoretical)",
    )
    plt.scatter(
        df["f"],
        np.degrees(df["phi"]),
        marker=MarkerStyle("o"),
        label="Phase (Experimental)",
    )

    eq_text += f"\n$\\omega_c = {omega_c:.2f}$ Hz"
    eq_text += f"\n$f_{{c,theoretical}} = {(omega_c / math.tau):.2f}$ Hz"
    plt.text(
        0.05,
        0.05,
        eq_text,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat"),
    )

    phi_deg = pd.Series(np.degrees(df["phi"]))
    closest_idx = find_closest_index(phi_deg, critical_deg)
    f_c_exp = _log_extrapolate_frequency(df["f"], phi_deg, critical_deg, closest_idx)
    plt.axhline(
        y=critical_deg,
        color="red",
        linestyle="--",
        label="45° characteristic line",
    )
    plt.axvline(
        x=f_c_exp,
        color="steelblue",
        linestyle="--",
        label=f"$f_{{c,experimental}}$ = {f_c_exp:.0f} Hz",
    )

    plt.legend()
    plt.savefig(output_dir / f"{filter_type.lower()}_pass_phase_plot.png", dpi=dpi)
    plt.close()


def create_bode_plot(
    df: pd.DataFrame,
    omega_c: float,
    filter_type: str,
    output_dir: Path,
    *,
    dpi: int = 150,
) -> None:
    """Create Bode (gain) plot (theory + experimental) and save. Uses lab4 standard: logspace(0,6),
    -3 dB line with find_closest_index and axvline, equation at bottom with ω_c in Hz.
    """
    output_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain (dB)")
    plt.grid(True, which="both", alpha=0.3)

    critical_db = -3
    closest_idx = find_closest_index(df["db"], critical_db)
    f_c_exp = _log_extrapolate_frequency(df["f"], df["db"], critical_db, closest_idx)
    plt.axhline(
        y=critical_db,
        color="red",
        linestyle="--",
        label="-3 dB characteristic line",
    )
    plt.axvline(
        x=f_c_exp,
        color="steelblue",
        linestyle="--",
        label=f"$f_{{c,experimental}}$ = {f_c_exp:.0f} Hz",
    )

    f_values = np.logspace(0, 6, num=100)
    omega_values = math.tau * f_values

    if filter_type == "high":
        gain_theoretical = high_pass_gain_theoretical(omega_values, omega_c)
        eq_text = r"$A(f) = 20\log_{10}\left(\dfrac{2\pi\,f/\omega_c}{\sqrt{1+(2\pi\,f/\omega_c)^2}}\right)$"
    else:
        gain_theoretical = low_pass_gain_theoretical(omega_values, omega_c)
        eq_text = (
            r"$A(f) = 20\log_{10}\left(\dfrac{1}{\sqrt{1+(2\pi\,f/\omega_c)^2}}\right)$"
        )

    plt.semilogx(f_values, gain_theoretical, label="Gain (Theoretical)")
    plt.scatter(df["f"], df["db"], marker=MarkerStyle("o"), label="Gain (Experimental)")

    eq_text += f"\n$\\omega_{{c,theoretical}} = {omega_c:.2f}$ Hz"
    eq_text += f"\n$f_{{c,theoretical}} = {(omega_c / math.tau):.2f}$ Hz"
    plt.text(
        0.05,
        0.05,
        eq_text,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat"),
    )

    plt.legend()
    plt.savefig(output_dir / f"{filter_type}_pass_bode_plot.png", dpi=dpi)
    plt.close()
