from __future__ import annotations

import math
import os
import sys

if os.environ.get("XDG_SESSION_TYPE") == "wayland":
    os.environ.setdefault("QT_QPA_PLATFORM", "wayland")

import matplotlib

matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint
from pathlib import Path
from scipy import constants as sc

from utils.paths import get_paths

paths = get_paths(__file__)
DATA_FILE = "freq_response_low_pass"

voltage_df = pd.read_csv(paths.data_dir / f"{DATA_FILE}.csv")
voltage_df["db"] = 20 * np.log10(voltage_df["v_out"] / voltage_df["v_in"])
voltage_df["phi"] = (math.tau) - (
    math.tau * voltage_df["f"] * voltage_df["dT"] * (180 / math.pi)
)

print(voltage_df)

plt.figure(figsize=(10, 6))
plt.semilogx(voltage_df["f"], voltage_df["db"], marker="o")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Gain (dB)")
plt.grid(True, which="both", alpha=0.3)
paths.output_dir.mkdir(exist_ok=True)
plt.savefig(paths.output_dir / f"low_pass_bode_plot.png", dpi=150)
# plt.show()


plt.figure(figsize=(10, 6))
plt.axvline(x=15000)
plt.semilogx(voltage_df["f"], voltage_df["phi"], marker="o")
plt.xlabel("Frequency (Hz)")
plt.ylabel("$phi$")
plt.grid(True, which="both", alpha=0.3)
paths.output_dir.mkdir(exist_ok=True)
plt.savefig(paths.output_dir / f"low_pass_phase_plot.png", dpi=150)
plt.show()
