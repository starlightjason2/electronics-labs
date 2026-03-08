from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

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

low_pass_df = pd.read_csv(paths.data_dir / "low_pass_pulse_200Hz.csv")
fft_df = pd.read_csv(paths.data_dir / "fft_low_pass_pulse_200Hz.csv")
fft_df = fft_df.dropna(subset=["f_in", "amplitude"])
fft_df["f_in"] = fft_df["f_in"] / 1000  # Hz -> kHz

f_peak = fft_df.loc[fft_df["amplitude"].idxmax(), "f_in"]  # ~4 kHz
fft_df = fft_df[fft_df["f_in"] >= f_peak]  # drop everything below peak

peak_indices, _ = find_peaks(fft_df["amplitude"], height=0.005)
harmonic_spacing = (
    fft_df["f_in"].iloc[peak_indices[1]] - fft_df["f_in"].iloc[peak_indices[0]]
)
tau_init = 1.0 / harmonic_spacing


def sinc_envelope(freq, A, tau):
    return np.abs(A * np.sinc(tau * freq))


popt, _ = curve_fit(
    sinc_envelope,
    fft_df["f_in"].iloc[peak_indices],
    fft_df["amplitude"].iloc[peak_indices],
    p0=[fft_df["amplitude"].max(), tau_init],
    bounds=([0, 0], [np.inf, np.inf]),
    method="trf",
    maxfev=20000,
)
A_fit, tau_fit = popt

# --- PLOTTING ---
ax1: Axes
ax2: Axes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Ax1 (Time Domain)
ax1.plot(
    low_pass_df["t_in"], low_pass_df["v_in"], label="Voltage In", color="goldenrod"
)
ax1.set_xlabel(r"Time $t$ (s)")
ax1.set_ylabel(r"Voltage $V_{\mathrm{In}}$ (V)")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)

# Ax2 (FFT + fit)
ax2.scatter(fft_df["f_in"], fft_df["amplitude"], label="Oscilloscope FFT Coefficients")

n_fit = np.linspace(fft_df["f_in"].min(), fft_df["f_in"].max(), 1000)
ax2.plot(
    n_fit,
    sinc_envelope(n_fit, *popt),
    color="tomato",
    linewidth=2,
    label=add_equation_text(
        "$A_n = |A\\,\\mathrm{sinc}(\\tau f)|$", {"A": A_fit, "\\tau": tau_fit}
    ),
)

ax2.set_xlabel("Frequency (kHz)")
ax2.set_ylabel("Fourier Coefficient")
ax2.grid(True, alpha=0.3)
ax2.legend()

fig.tight_layout()
fig.savefig(paths.output_dir / "fft_low_pass_200Hz.png", dpi=150)
