from __future__ import annotations

import numpy as np
import pandas as pd

from utils.paths import get_paths
from utils.utils import (
    create_bode_plot,
    create_phase_plot,
    plot_oscilloscope_data,
    load_oscilloscope_data,
)
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

paths = get_paths(__file__)

low_pass_df = pd.read_csv(
    paths.data_dir / "low_pass_pulse_200Hz.csv",
)
fft_df = pd.read_csv(paths.data_dir / "fft_low_pass_pulse_200Hz.csv")
low_pass_df["t_in"] = low_pass_df["t_in"] - low_pass_df["t_in"].min()

omega_c = 1 / (20e-6)

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

ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Fourier Coefficient")
ax2.grid(True, alpha=0.3)

# experimental
ax2.scatter(fft_df["f_in"], fft_df["amplitude"], label="Experimental FT")

fig.legend()
fig.tight_layout()
fig.savefig(paths.output_dir / f"fft_low_pass_200Hz.png", dpi=150)
