from __future__ import annotations

import numpy as np
import pandas as pd

from utils.paths import get_paths
from utils.utils import find_closest_index, load_oscilloscope_data
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

paths = get_paths(__file__)

low_pass_df = pd.read_csv(
    paths.data_dir / "low_pass_pulse_200Hz.csv",
)
fft_df = pd.read_csv(paths.data_dir / "fft_low_pass_pulse_200Hz.csv")
low_pass_df["t_in"] = low_pass_df["t_in"] - low_pass_df["t_in"].min()


f0 = 200
threshold = np.max(low_pass_df["v_in"]) / 2
pulse_samples = (low_pass_df["v_in"] > threshold).sum()
dt = low_pass_df["t_in"].diff().median()
pulse_width = pulse_samples * dt
T = low_pass_df["t_in"].max()

fft_df["n"] = fft_df["f_in"] / f0

ax1: Axes
ax2: Axes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))


ax1.plot(
    low_pass_df["t_in"], low_pass_df["v_in"], label="Voltage In", color="goldenrod"
)
ax1.set_xlabel(r"Time $t$ (s)")
ax1.set_ylabel(r"Voltage $V_{\mathrm{Out}}$ (V)")

ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)

ax2.set_xlabel("$n$th Harmonic")
ax2.set_ylabel("Fourier Coefficient")
ax2.grid(True, alpha=0.3)

# experimental
ax2.scatter(fft_df["n"], fft_df["amplitude"], zorder=100, label="FFT Experimental")
# theory
threshold = np.max(low_pass_df["v_in"]) / 2
duty_cycle = pulse_width / T
fft_coeffs = np.abs(duty_cycle * np.sinc(fft_df["n"] * duty_cycle))
scale = fft_df["amplitude"].max() / fft_coeffs.max()
print(duty_cycle)
ax2.plot(
    fft_df["n"],
    fft_coeffs * scale,
    color="tomato",
)


fig.legend()
fig.tight_layout()
fig.savefig(paths.output_dir / f"fft_low_pass_200Hz.png", dpi=150)
