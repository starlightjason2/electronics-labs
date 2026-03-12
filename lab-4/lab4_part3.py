from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from utils.paths import get_paths
from utils.utils import load_oscilloscope_data, format_sig_figs, add_equation_text
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

paths = get_paths(__file__)

f0 = 400
low_pass_df = load_oscilloscope_data("low_pass_400Hz", paths.data_dir)
fft_df = pd.read_csv(paths.data_dir / "fft_low_pass_400Hz.csv")
fft_df["n"] = fft_df["f_in"] / f0


def square_wave_func(n, magnitude):
    """Square wave Fourier series coefficient function."""
    return magnitude * (1 / (n * np.pi)) * (2 - 2 * (-1) ** n)


ax1: Axes
ax2: Axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(
    low_pass_df["t_in"], low_pass_df["v_in"], label="Voltage In", color="goldenrod"
)
ax1.set_xlabel(r"Time $t$ (μs)")
ax1.set_ylabel(r"Voltage $V_{\mathrm{Out}}$ (V)")
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)

ax2.scatter(fft_df["n"], fft_df["amplitude"], zorder=100, label="FFT Experimental")
ax2.set_xlabel("$n$th Harmonic")
ax2.set_ylabel("Fourier Coefficient")
ax2.set_xticks(np.arange(0, int(fft_df["n"].max()) + 1, 1))
ax2.grid(True, alpha=0.3)

# fit square wave function (only odd harmonics where signal exists)
fft_odd = fft_df[fft_df["n"] % 2 == 1].copy()
initial_scale = fft_odd["amplitude"].iloc[0] / square_wave_func(
    fft_odd["n"].iloc[0], 1.0
)
popt, _ = curve_fit(
    square_wave_func, fft_df["n"], fft_df["amplitude"], p0=[initial_scale]
)
magnitude_fit = popt[0]

# plot fit with stem
n_plot = np.arange(1, 10, 1)
ax2.scatter(
    n_plot,
    square_wave_func(n_plot, magnitude_fit),
    s=120,
    facecolors="none",
    edgecolors="tomato",
    linestyles="dashed",
    linewidths=1.5,
    zorder=50,
    label=add_equation_text(
        "$A_n = |A| \\cdot \\frac{1}{n\\pi} \\cdot (2 - 2(-1)^n)$",
        {"|A|": magnitude_fit},
    ),
)
ax2.legend(loc="upper right")

fig.tight_layout()
fig.savefig(paths.output_dir / f"fft_low_pass_400Hz.png", dpi=150)
