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
from utils.utils import load_oscilloscope_data, calculate_charge_tau, calculate_discharge_tau

paths = get_paths(__file__)
DATA_FILE = "low_pass_square_1kHz"
ureg = pint.UnitRegistry()

voltage_df = load_oscilloscope_data(DATA_FILE, paths.data_dir)
voltage_df["v_in"] = voltage_df["v_in"] - voltage_df["v_in"].min()
voltage_df["v_out"] = voltage_df["v_out"] - voltage_df["v_out"].min()

plt.figure(figsize=(10, 6))
plt.plot(voltage_df["t_in"], voltage_df["v_in"], label="Voltage In", color="goldenrod")
plt.plot(voltage_df["t_out"], voltage_df["v_out"], label="Voltage Out", color="blue")

charge_start_idx = 1
charge_end_idx = 252
charge_start_time = float(voltage_df["t_out"].iloc[charge_start_idx])

tau_charge_time, tau_charge_volt, tau_charge_volt_calc = calculate_charge_tau(
    voltage_df["t_out"], voltage_df["v_out"], charge_start_idx, charge_end_idx
)

plt.plot(tau_charge_time, tau_charge_volt_calc, "ro")
plt.plot(
    [charge_start_time, tau_charge_time],
    [tau_charge_volt, tau_charge_volt],
    color="red",
    linestyle="--",
    alpha=0.7,
)
tau1 = ureg.Quantity(tau_charge_time - charge_start_time, ureg.microsecond)
plt.text(
    tau_charge_time + 100,
    tau_charge_volt,
    f"$\\tau_1 = {tau1.magnitude:.2f}$ μs",
    ha="center",
    color="red",
)


discharge_start_idx = 253
discharge_end_idx = 502
discharge_start_time = float(voltage_df["t_out"].iloc[discharge_start_idx])

tau_discharge_time, tau_discharge_volt, tau_discharge_volt_calc = calculate_discharge_tau(
    voltage_df["t_out"], voltage_df["v_out"], discharge_start_idx, discharge_end_idx
)


plt.plot(tau_discharge_time, tau_discharge_volt, "go")
plt.plot(
    [discharge_start_time, tau_discharge_time],
    [tau_discharge_volt, tau_discharge_volt],
    color="green",
    linestyle="--",
    alpha=0.7,
)
tau2 = ureg.Quantity(tau_discharge_time - discharge_start_time, ureg.microsecond)
plt.text(
    tau_discharge_time + 100,
    tau_discharge_volt,
    f"$\\tau_2 = {tau2.magnitude:.2f}$ μs",
    ha="center",
    color="green",
)


plt.xlabel(r"Time $t$ (μs)")
plt.ylabel(r"Voltage Out $V_{\mathrm{Out}}$ (V)")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
paths.output_dir.mkdir(exist_ok=True)
plt.savefig(paths.output_dir / f"{DATA_FILE}.png", dpi=150)
plt.show()
