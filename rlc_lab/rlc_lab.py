from __future__ import annotations

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.axes import Axes
from matplotlib.markers import MarkerStyle

from utils.paths import get_paths

paths = get_paths(__file__)
paths.output_dir.mkdir(exist_ok=True)

df = pd.read_csv(paths.data_dir / "paradigms_lab1.csv").sort_values(by="f")
R, L, C = 51, 22e-3, 0.018e-6
omega_0 = 1 / np.sqrt(L * C)
beta = R / (2 * L)

df["omega"] = 2 * math.pi * df["f"]
# df["v_in"] = df["v_in"] / 2
df["v_out"] = df["v_out"] / 1000
df["dT"] = df["dT"] * 1e-6
df["phi"] = df["omega"] * df["dT"]
df["A"] = (df["v_out"] / df["v_in"]) / R


def theoretical_values() -> tuple[np.ndarray, list[float], list[float]]:
    x = np.linspace(0, omega_0 * 2, 1000)
    return (x, [phase_at_omega(w) for w in x], [admittance_at_omega(w) for w in x])  # type: ignore


def admittance_at_omega(omega_d: float | np.ndarray) -> float | np.ndarray:
    """Return |Y| in S at angular frequency omega_d."""
    return (omega_d / L) / np.sqrt(
        (omega_0**2 - omega_d**2) ** 2 + (2 * beta * omega_d) ** 2
    )


def phase_at_omega(omega_d: float | np.ndarray) -> float | np.ndarray:
    """Return phase in rad at angular frequency omega_d."""
    return (math.pi / 2) + np.arctan2(-2 * beta * omega_d, omega_0**2 - omega_d**2)


x_values, phi_values, admittance_values = theoretical_values()

OMEGA_0_LABEL = "$\\omega_0$ (Resonant Frequency)"


def setup_omega_axis(ax: Axes, n_beta: int = 30) -> None:
    x_lo, x_hi = omega_0 - n_beta * beta, omega_0 + n_beta * beta
    ax.set_xlim(x_lo, x_hi)
    ax.xaxis.set_major_locator(
        ticker.FixedLocator(np.arange(x_lo, x_hi + 1e-9, 10 * beta))
    )
    ax.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(x_lo, x_hi + 1e-9, beta)))
    ax.tick_params(axis="x", which="major", length=7, width=1, color="black")
    ax.tick_params(
        axis="x", which="minor", length=4, width=1, color="gray", direction="inout"
    )
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x / 1000:.1f}")
    ax.set_xlabel("$\\omega$ ($10^3$ rad/s)")
    ax.axvspan(
        omega_0 - beta,
        omega_0 + beta,
        linestyle="--",
        color="darkorange",
        linewidth=1,
        alpha=0.3,
        label="$\\omega_0 \\pm \\beta$ (Band Pass)",
    )


def phase_plot(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    ax.grid(True, which="both", alpha=0.3)
    setup_omega_axis(ax)
    ax.set_ylabel("$\\phi$ (rad)")
    ax.set_yticks([-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2])
    ax.set_yticklabels([r"$-\pi/2$", r"$-\pi/4$", r"$0$", r"$\pi/4$", r"$\pi/2$"])
    ax.plot(x_values, phi_values, label="Phase (Theoretical)")
    ax.axvline(x=omega_0, label=OMEGA_0_LABEL, linestyle="--", color="red")
    ax.scatter(
        df["omega"],
        df["phi"],
        marker=MarkerStyle("o"),
        label="Phase (experimental)",
        zorder=10,
    )
    ax.legend()
    fig.savefig(paths.output_dir / "phase_plot.png", dpi=600)


def admittance_plot(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    ax.set_ylabel("Admittance $|Y|$ (S)")
    ax.grid(True, which="both", alpha=0.3)
    ax.plot(x_values, admittance_values, label="Admittance (Theoretical)")
    ax.set_ylim(0, np.max(admittance_values) / np.sqrt(2) * 1.8)
    setup_omega_axis(ax)
    ax.axvline(x=omega_0, label=OMEGA_0_LABEL, linestyle="--", color="red")
    ax.scatter(
        df["omega"],
        df["A"],
        marker=MarkerStyle("o"),
        label="Admittance (Experimental)",
        zorder=10,
    )
    ax.legend()
    fig.savefig(paths.output_dir / "amplitude_plot.png", dpi=600)


def oscilloscope_traces_plot(
    omegas: np.ndarray,
    *,
    n_periods: float = 1.5,
    v_in_ref: float = 1.0,
) -> None:
    """Subplots of V_E and V_R vs ωt at given omegas, with theoretical_values |Y| and φ."""
    omegas = np.atleast_1d(omegas)
    n, ncols = len(omegas), min(3, len(omegas))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows))
    axes = np.atleast_1d(np.array([axes]) if n == 1 else axes).flatten()

    theta_max = n_periods * 2 * math.pi
    theta = np.linspace(0, theta_max, 1000)
    ticks = np.arange(0, theta_max + 1e-9, math.pi / 2)
    tick_labels = [f"{t/np.pi:.1f}$\\pi$" if t != 0 else "0" for t in ticks]
    phase_ref = 1.5 * math.pi  # V_E peak in first period

    for idx, omega in enumerate(omegas):
        beta_diff_factor = np.abs((omega_0 - omega) / beta)
        ax = axes[idx]
        admittance, phi = admittance_at_omega(omega), phase_at_omega(omega)
        v_in = v_in_ref * np.cos(theta)
        v_out = v_in_ref * admittance * R * np.cos(theta + phi)

        ax.plot(theta, v_in, label="$V_{E}$", color="C0", linewidth=2)
        ax.plot(theta, v_out, label="$V_{R}$", color="C1", linewidth=1)

        ax.plot(
            [phase_ref, phase_ref - phi],
            [0, 0],
            color="red",
            linewidth=1.5,
            solid_capstyle="butt",
        )
        ax.text(
            0.98,
            0.98,
            f"$\\phi = {(phi / math.pi):.2f}\\pi$ rad",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="right",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9
            ),
        )

        ax.set_xlabel("$\\omega t$ (rad)")
        ax.set_xlim(0, theta_max)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
        ax.set_ylabel("Voltage (V)")

        if beta_diff_factor == 1:
            ax.axhline(y=np.sqrt(2), color="C2", label="$V=\\pm1/\\sqrt{2}$")
            ax.axhline(y=-np.sqrt(2), color="C2")

        sign = (
            ""
            if omega == omega_0
            else f" {('+' if omega > omega_0 else '-')}{beta_diff_factor:.0f} \\beta"
        )
        ax.set_title(
            f"$\\omega = \\omega_0{sign} = {(omega/1000):.1f}\\cdot 10^3$ rad/s"
        )
        ax.grid(True, alpha=0.3)

    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="upper center", ncols=2, frameon=True
    )
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])  # type: ignore

    fig.savefig(paths.output_dir / "oscilloscope_traces.png", dpi=600)
    plt.close(fig)


phase_plot(df)
admittance_plot(df)
oscilloscope_traces_plot(
    np.array(
        [
            omega_0 - beta,
            omega_0,
            omega_0 + beta,
            omega_0 - 10 * beta,
            omega_0 + 10 * beta,
            omega_0 + 100 * beta,
        ]
    )
)

# Part 1 data subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))

ax1.plot(df["f"] / 1000, df["v_in"], marker="o")
ax1.set_ylabel("$V_{in}$ (V)")
ax1.set_xlabel("Frequency (kHz)")
ax1.grid(True, alpha=0.3)
ax1.set_title("Input Voltage vs Frequency")

ax2.plot(df["f"] / 1000, df["v_out"], marker="o", color="C1")
ax2.set_ylabel("$V_{out}$ (V)")
ax2.set_xlabel("Frequency (kHz)")
ax2.grid(True, alpha=0.3)
ax2.set_title("Output Voltage vs Frequency")

ax3.plot(df["f"] / 1000, df["dT"] * 1e6, marker="o", color="C2")
ax3.set_ylabel("$\\Delta t$ ($\\mu$s)")
ax3.set_xlabel("Frequency (kHz)")
ax3.grid(True, alpha=0.3)
ax3.set_title("Time Shift vs Frequency")

fig.tight_layout()
fig.savefig(paths.output_dir / "part1_data.png", dpi=600)
plt.close(fig)
