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
from utils.utils import load_oscilloscope_data, plot_tau_charging, plot_tau_discharging

paths = get_paths(__file__)
DATA_FILE = "low_pass_square_1kHz"

voltage_df = load_oscilloscope_data(DATA_FILE, paths.data_dir)
voltage_df["v_in"] = voltage_df["v_in"] - voltage_df["v_in"].min()
voltage_df["v_out"] = voltage_df["v_out"] - voltage_df["v_out"].min()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(voltage_df["t_in"], voltage_df["v_in"], label="Voltage In", color="goldenrod")
ax.plot(voltage_df["t_out"], voltage_df["v_out"], label="Voltage Out", color="blue")

plot_tau_charging(ax, voltage_df["t_out"], voltage_df["v_out"], 1, 252)
plot_tau_discharging(ax, voltage_df["t_out"], voltage_df["v_out"], 253, 502)

ax.set_xlabel(r"Time $t$ (μs)")
ax.set_ylabel(r"Voltage Out $V_{\mathrm{Out}}$ (V)")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
paths.output_dir.mkdir(exist_ok=True)
plt.savefig(paths.output_dir / f"{DATA_FILE}.png", dpi=150)
plt.show()
