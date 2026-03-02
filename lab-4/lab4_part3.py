from __future__ import annotations

import numpy as np
import pandas as pd

from utils.paths import get_paths
from utils.utils import (
    find_closest_index,
    load_oscilloscope_data,
)
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

paths = get_paths(__file__)

f0 = 400
omega_c = 1 / (20e-6)

low_pass_df = load_oscilloscope_data("low_pass_400Hz", paths.data_dir)
fft_df = pd.read_csv(paths.data_dir / "fft_low_pass_400Hz.txt.csv")
fft_df["n"] = fft_df["f_in"] / f0

ax1: Axes
ax2: Axes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

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
ax2.grid(True, alpha=0.3)

amp = np.max(low_pass_df["v_in"])
n = np.arange(1, 10, 1)
fft_coeffs = (1 / (n * np.pi)) * (2 - 2 * (-1) ** n)
scale = fft_df["amplitude"][find_closest_index(fft_df["f_in"], f0)] / fft_coeffs[0]
ax2.stem(
    n,
    fft_coeffs * scale,
    linefmt="tomato",
    markerfmt="ro",
    basefmt=" ",
    label="FFT Theoretical",
)
ax2.legend()

plt.tight_layout()
plt.savefig(paths.output_dir / f"fft_low_pass_400Hz.png", dpi=150)
plt.close()
