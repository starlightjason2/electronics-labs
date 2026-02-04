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
from utils.utils import frequency_of_square_wave

paths = get_paths(__file__)
DATA_FILE = "low_pass_square_20kHz"
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

voltage_df = pd.read_csv(paths.data_dir / f"{DATA_FILE}.csv")

# Offset time so minimum is 0, then convert s → μs (factor from pint)
s_to_us = (1.0 * ureg.second).to(ureg.microsecond).magnitude
voltage_df["t_in"] = (voltage_df["t_in"] - voltage_df["t_in"].min()) * s_to_us
voltage_df["t_out"] = (voltage_df["t_out"] - voltage_df["t_out"].min()) * s_to_us

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the data
plt.plot(voltage_df["t_in"], voltage_df["v_in"], label="Voltage In", color="goldenrod")
plt.plot(voltage_df["t_out"], voltage_df["v_out"], label="Voltage Out", color="blue")


plt.title(f"Voltage vs Time (20 kHz input)")
plt.xlabel(r"Time $t$ (μs)")
plt.ylabel(r"Voltage Out $V_{\mathrm{Out}}$ (V)")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
paths.output_dir.mkdir(exist_ok=True)
plt.savefig(paths.output_dir / f"{Path(DATA_FILE)}.png", dpi=150)
plt.show()

# TODO: apply triaangle wave to input
