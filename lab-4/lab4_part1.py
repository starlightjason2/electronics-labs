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

low_pass_df = load_oscilloscope_data("low_pass_square_5kHz", paths.data_dir)
low_pass_df["v_in"] = low_pass_df["v_in"] - low_pass_df["v_in"].min()
low_pass_df["v_out"] = low_pass_df["v_out"] - low_pass_df["v_out"].min()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(low_pass_df["t_in"], low_pass_df["v_in"], label="Voltage In", color="goldenrod")
ax.plot(low_pass_df["t_out"], low_pass_df["v_out"], label="Voltage Out", color="blue")

plot_tau_charging(ax, low_pass_df["t_out"], low_pass_df["v_out"], 1, 502)
plot_tau_discharging(ax, low_pass_df["t_out"], low_pass_df["v_out"], 502, 1001)

ax.set_xlabel(r"Time $t$ (μs)")
ax.set_ylabel(r"Voltage Out $V_{\mathrm{Out}}$ (V)")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
paths.output_dir.mkdir(exist_ok=True)
plt.savefig(paths.output_dir / f"low_pass_square_5kHz.png", dpi=150)


high_pass_df = load_oscilloscope_data("high_pass_square_200Hz", paths.data_dir)
high_pass_df["v_in"] = high_pass_df["v_in"] - high_pass_df["v_in"].min()
high_pass_df["v_out"] = high_pass_df["v_out"] - high_pass_df["v_out"].min()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
    high_pass_df["t_in"], high_pass_df["v_in"], label="Voltage In", color="goldenrod"
)
ax.plot(high_pass_df["t_out"], high_pass_df["v_out"], label="Voltage Out", color="blue")

# plot_tau_charging(ax, high_pass_df["t_out"], high_pass_df["v_out"], 1, 1252)
plot_tau_discharging(ax, high_pass_df["t_out"], high_pass_df["v_out"], 1, 1253)

ax.set_xlabel(r"Time $t$ (μs)")
ax.set_ylabel(r"Voltage Out $V_{\mathrm{Out}}$ (V)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
paths.output_dir.mkdir(exist_ok=True)
plt.savefig(paths.output_dir / f"high_pass_square_200Hz.png", dpi=150)
