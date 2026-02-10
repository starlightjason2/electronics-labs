from __future__ import annotations

import math
import matplotlib

matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.paths import get_paths

paths = get_paths(__file__)


low_pass_df = pd.read_csv(paths.data_dir / "freq_response_low_pass.csv")
low_pass_df["db"] = 20 * np.log10(low_pass_df["v_out"] / low_pass_df["v_in"])

high_pass_df = pd.read_csv(paths.data_dir / "freq_response_high_pass.csv")
high_pass_df["db"] = 20 * np.log10(high_pass_df["v_out"] / high_pass_df["v_in"])

omega_c = 1 / (62e-6)


def create_phase_plot(df: pd.DataFrame, type="low" or "high"):
    df["phi"] = -math.tau * df["f"] * df["dT"]

    plt.figure(figsize=(10, 6))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("$\\phi$ (degrees)")
    plt.grid(True, which="both", alpha=0.3)

    f_values = np.logspace(0, 6, num=100)
    omega_values = math.tau * f_values
    phase = phase_theoretical(omega_values)

    # theory
    plt.semilogx(
        f_values,
        np.degrees(phase),
        label=f"{type.capitalize()} Pass Phase (Theoretical)",
    )
    # experimental
    plt.scatter(
        df["f"], np.degrees(df["phi"]), marker="o", label="Phase (Experimental)"
    )

    plt.axhline(y=-45, color="red", linestyle="--")
    plt.legend()

    plt.savefig(paths.output_dir / f"{type.lower()}_pass_phase_plot.png", dpi=150)


def phase_theoretical(omega_values: np.ndarray):
    return -np.arctan(omega_values / omega_c)


def low_pass_gain_theoretical(omega_values: np.ndarray):
    return 20 * np.log10(1 / np.sqrt(1 + (omega_values / omega_c) ** 2))


def high_pass_gain_theoretical(omega_values: np.ndarray):
    return 20 * np.log10(
        (omega_values / omega_c) / np.sqrt(1 + (omega_values / omega_c) ** 2)
    )


def create_bode_plot(df: pd.DataFrame, type="low" or "high"):
    paths.output_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 6))

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain (dB)")
    plt.grid(True, which="both", alpha=0.3)
    plt.axhline(y=-3, color="red", linestyle="--")

    f_values = np.logspace(0, 6, num=100)
    omega_values = math.tau * f_values
    gain_theoretical = np.zeros(len(omega_values))
    if type == "high":
        gain_theoretical = high_pass_gain_theoretical(omega_values)
    elif type == "low":
        gain_theoretical = low_pass_gain_theoretical(omega_values)

    # theory
    plt.semilogx(f_values, gain_theoretical, label="Gain (Theoretical)")
    # experimental
    plt.scatter(df["f"], df["db"], marker="o", label="Gain (Experimental)")

    plt.legend()
    plt.savefig(paths.output_dir / f"{type}_pass_bode_plot.png", dpi=150)


create_bode_plot(low_pass_df, "low")
create_phase_plot(low_pass_df, "low")
create_bode_plot(high_pass_df, "high")
