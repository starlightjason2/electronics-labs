from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from numpy import interp

from utils.paths import get_paths
from scipy.signal import find_peaks
from utils.utils import (
    find_closest_index,
    load_oscilloscope_data,
    format_sig_figs,
    add_equation_text,
)
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

paths = get_paths(__file__)

low_pass_df = pd.read_csv(
    paths.data_dir / "final_lab4_part4_10microsec_200hz_input_4pp.csv"
)
fft_df = pd.read_csv(paths.data_dir / "final_fft_lab4_sinc.csv")
fft_df["f_in"] = fft_df["f_in"] / 1000  # Hz -> kHz


def sinc(freq, A, tau):
    return np.abs(A * np.sinc(tau * freq))


popt, _ = curve_fit(
    sinc,
    fft_df["f_in"],
    fft_df["amplitude"],
    p0=[fft_df["amplitude"].max(), 1e-4],
    bounds=([0, 0], [np.inf, np.inf]),
    method="trf",
    maxfev=20000,
)
A_fit, tau_fit = popt

# --- PLOTTING ---
ax1: Axes
ax2: Axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Ax1 (Time Domain)
ax1.plot(
    low_pass_df["t_in"],
    low_pass_df["v_in"],
    label="Voltage In",
    color="goldenrod",
)
ax1.plot(
    low_pass_df["t_out"],
    low_pass_df["v_out"],
    label="$V_R$ (RL Circuit)",
    color="blue",
    alpha=0.5,
)
ax1.set_xlabel(r"Time $t$ (s)")
ax1.set_ylabel(r"Voltage $V_{\mathrm{In}}$ (V)")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)

# Ax2 (FFT + fit)
ax2.plot(
    fft_df["f_in"], fft_df["amplitude"], color="blue", linewidth=1, label="Experimental"
)

n_fit = np.linspace(fft_df["f_in"].min(), fft_df["f_in"].max(), 1000)
ax2.plot(
    n_fit,
    sinc(n_fit, *popt),
    color="tomato",
    linestyle="--",
    linewidth=1,
    label=add_equation_text(
        "$A_n = |A\\,\\mathrm{sinc}(\\tau f)|$", {"A": A_fit, "\\tau": (tau_fit, "ms")}
    ),
)

ax2.set_xlabel("Frequency (kHz)")
ax2.set_ylabel("Fourier Coefficient")
ax2.grid(True, alpha=0.3)
ax2.legend()

fig.tight_layout()
fig.savefig(paths.output_dir / "fft_low_pass_200Hz.png", dpi=150)
plt.close()

# --- Bode plot from pulse response ---
import math
from utils.utils import high_pass_gain_theoretical

bode_df = pd.read_csv(paths.data_dir / "final_lab4_part4_bodeplot.csv")
bode_df = bode_df[bode_df["f_in"] > 0]  # drop DC
bode_df["db"] = 20 * np.log10(bode_df["amplitude"])

# find experimental f_c at -3 dB via interpolation
f_c_exp = float(np.interp(-3, bode_df["db"].to_numpy(), bode_df["f_in"].to_numpy()))

fig2, ax3 = plt.subplots(figsize=(7, 5))
ax3.plot(bode_df["f_in"], bode_df["db"], label="LabView Bode Plot", zorder=5)

omega_c = 1 / (20e-6)
f_th = np.linspace(bode_df["f_in"].min(), bode_df["f_in"].max(), 500)
ax3.plot(
    f_th,
    high_pass_gain_theoretical(math.tau * f_th, omega_c),
    color="tomato",
    alpha=0.7,
    linewidth=2,
    label="High-Pass Theoretical",
)
ax3.axhline(y=-3, color="red", linestyle="--", alpha=0.7, label="$-3$ dB")
ax3.axvline(
    x=f_c_exp,
    color="steelblue",
    linestyle="--",
    alpha=0.7,
    label=f"$f_{{c,\\mathrm{{exp}}}}$ = {f_c_exp:.0f} Hz",
)
ax3.set_xscale("log")
ax3.set_xlabel("Frequency (Hz)")
ax3.set_ylabel("Gain (dB)")
ax3.grid(True, which="both", alpha=0.3)
ax3.legend(fontsize=8)
fig2.tight_layout()
fig2.savefig(paths.output_dir / "bode_pulse_response.png", dpi=150)
plt.close()
