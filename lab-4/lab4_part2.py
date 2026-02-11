from __future__ import annotations

import math
import matplotlib

matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.paths import get_paths
from utils.utils import find_closest_index

paths = get_paths(__file__)


high_pass_df = pd.read_csv(paths.data_dir / "freq_response_high_pass.csv")
high_pass_df["db"] = 20 * np.log10(high_pass_df["v_out"] / high_pass_df["v_in"])

omega_c = 1 / (20e-6)


def create_phase_plot(df: pd.DataFrame, omega_c: float, type="low"):
    df["phi"] = math.tau * df["f"] * df["dT"]

    plt.figure(figsize=(10, 6))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("$\\phi$ (degrees)")
    plt.grid(True, which="both", alpha=0.3)

    f_values = np.logspace(2, 6, num=100)
    phase_theoretical = np.zeros(len(f_values))
    omega_values = math.tau * f_values
    critical_omega = math.pi / 4

    if type == "high":
        phase_theoretical = high_pass_phase_theoretical(omega_values, omega_c)
        eq_text = r"$\phi(f) = \arctan\left(\dfrac{\omega_c}{2\pi\,f}\right)$"
    else:  # low
        critical_omega = -critical_omega
        phase_theoretical = low_pass_phase_theoretical(omega_values, omega_c)
        eq_text = r"$\phi(f) = -\arctan\left(\dfrac{2\pi\,f}{\omega_c}\right)$"

    # theory
    plt.semilogx(
        f_values,
        np.degrees(phase_theoretical),
        label=f"{type.capitalize()} Pass Phase (Theoretical)",
    )
    # experimental
    plt.scatter(
        df["f"], np.degrees(df["phi"]), marker="o", label="Phase (Experimental)"
    )

    # Add equation text
    eq_text += f"\n$\\omega_c = {omega_c:.2f}$ Hz"
    plt.text(
        0.05,
        0.05,
        eq_text,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat"),
    )

    closest_idx = find_closest_index(
        pd.Series(np.degrees(df["phi"])), np.degrees(critical_omega)
    )
    plt.axhline(
        y=np.degrees(critical_omega),
        color="red",
        linestyle="--",
        label=f"45° at {df['f'].iloc[closest_idx]:.0f} Hz",
    )
    plt.axvline(
        x=df["f"].iloc[closest_idx],
        color="red",
        linestyle="--",
    )

    plt.legend()
    plt.savefig(paths.output_dir / f"{type.lower()}_pass_phase_plot.png", dpi=150)


def low_pass_phase_theoretical(omega_values: np.ndarray, omega_c: float):
    return -np.arctan(omega_values / omega_c)


def high_pass_phase_theoretical(omega_values: np.ndarray, omega_c: float):
    return np.arctan(omega_c / omega_values)


def low_pass_gain_theoretical(omega_values: np.ndarray, omega_c: float):
    return 20 * np.log10(1 / np.sqrt(1 + (omega_values / omega_c) ** 2))


def high_pass_gain_theoretical(omega_values: np.ndarray, omega_c: float):
    return 20 * np.log10(
        (omega_values / omega_c) / np.sqrt(1 + (omega_values / omega_c) ** 2)
    )


def create_bode_plot(df: pd.DataFrame, omega_c: float, type="low"):
    paths.output_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 6))

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain (dB)")
    plt.grid(True, which="both", alpha=0.3)

    critical_db = -3
    closest_idx = find_closest_index(df["db"], critical_db)
    plt.axhline(
        y=critical_db,
        color="red",
        linestyle="--",
        label=f"-3 dB at {df['f'].iloc[closest_idx]:.0f} Hz",
    )
    plt.axvline(
        x=df["f"].iloc[closest_idx],
        color="red",
        linestyle="--",
    )

    f_values = np.logspace(2, 6, num=100)
    omega_values = math.tau * f_values
    gain_theoretical = np.zeros(len(omega_values))

    if type == "high":
        gain_theoretical = high_pass_gain_theoretical(omega_values, omega_c)
        eq_text = r"$A(f) = 20\log_{10}\left(\dfrac{2\pi\,f/\omega_c}{\sqrt{1+(2\pi\,f/\omega_c)^2}}\right)$"
    else:  # low
        gain_theoretical = low_pass_gain_theoretical(omega_values, omega_c)
        eq_text = (
            r"$A(f) = 20\log_{10}\left(\dfrac{1}{\sqrt{1+(2\pi\,f/\omega_c)^2}}\right)$"
        )

    # theory
    plt.semilogx(f_values, gain_theoretical, label="Gain (Theoretical)")
    # experimental
    plt.scatter(df["f"], df["db"], marker="o", label="Gain (Experimental)")

    # Add equation text
    eq_text += f"\n$\\omega_c = {omega_c:.2f}$ Hz"
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
    plt.savefig(paths.output_dir / f"{type}_pass_bode_plot.png", dpi=150)


create_bode_plot(high_pass_df, omega_c, "high")
create_phase_plot(high_pass_df, omega_c, "high")
