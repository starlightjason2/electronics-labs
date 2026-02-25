from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import math


from utils.paths import get_paths
from utils.utils import create_bode_plot, create_phase_plot

paths = get_paths(__file__)

paths.output_dir.mkdir(exist_ok=True)


df = pd.read_csv(paths.data_dir / "paradigms_lab1.csv").sort_values(by="f")


# unit scaling
df["omega"] = 2 * math.pi * df["f"]
df["v_out"] = df["v_out"] / 1000
df["dT"] = df["dT"] * 1e-6


def theoretical():
    x_values = np.linspace(0, omega_0 * 2, 1000)
    phi_values = [
        (math.pi / 2) + np.arctan2(-2 * beta * omega_d, omega_0**2 - omega_d**2)
        for omega_d in x_values
    ]
    admittance_values = [
        (omega_d / L)
        / (np.sqrt(((omega_0**2 - omega_d**2) ** 2) + ((2 * beta * omega_d) ** 2)))
        for omega_d in x_values
    ]

    return (x_values, phi_values, admittance_values)


R = 51
C = 0.018e-6
L = 22e-3

omega_0 = 1 / np.sqrt(L * C)
beta = R / (2 * L)
x_values, phi_values, admittance_values = theoretical()


def phase_plot(df):
    fig, ax = plt.subplots()

    df["phi"] = df["omega"] * df["dT"]

    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlabel("$\\omega_d$ (Hz)")
    ax.set_yticks([-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2])
    ax.set_yticklabels([r"$-\pi/2$", r"$-\pi/4$", r"$0$", r"$\pi/4$", r"$\pi/2$"])
    ax.set_ylabel("$\\phi$ (rad)")

    ax.scatter(
        df["omega"],
        df["phi"],
        marker=MarkerStyle("o"),
        label="Phase (Experimental)",
    )
    ax.plot(x_values, phi_values, label="Phase (Theoretical)")

    ax.axvline(x=omega_0, label="$\\omega_0$ [Hz]", linestyle="--")

    ax.legend()
    fig.savefig(paths.output_dir / "phase_plot.png")


def admittance_plot(df):
    df["A"] = df["v_out"] / df["v_in"]

    fig, ax = plt.subplots()

    ax.set_xlabel("$\\omega_d$ (Hz)")
    ax.set_ylabel("Admittance")
    ax.grid(True, which="both", alpha=0.3)

    ax.scatter(
        df["omega"], df["A"], marker=MarkerStyle("o"), label="Admittance (Experimental)"
    )
    ax.plot(x_values, admittance_values, label="Admittance (Theoretical)")
    ax.axvline(x=omega_0, label="$\\omega_0$ [Hz]", linestyle="--")

    ax.legend()
    fig.savefig(paths.output_dir / "amplitude_plot.png")


phase_plot(df)
admittance_plot(df)
