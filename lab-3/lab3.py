import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import constants as sc
import math

SAVE_DIR = "./"


def find_closest(series, target_value):
    """Find the index of the value closest to target_value in the series"""
    return series.sub(target_value).abs().idxmin()


# part 1 data
voltage_df = pd.read_csv("./lab3_part1.csv")

# Offset time data so minimum starts at 0 and convert to microseconds
voltage_df["t1"] = (voltage_df["t1"] - voltage_df["t1"].min()) * 1e6
voltage_df["t2"] = (voltage_df["t2"] - voltage_df["t2"].min()) * 1e6

# Shift voltage data up by minimum value
voltage_df["v1"] = voltage_df["v1"] - voltage_df["v1"].min()
voltage_df["v2"] = voltage_df["v2"] - voltage_df["v2"].min()

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the data
plt.plot(voltage_df["t1"], voltage_df["v1"], label="Voltage In", color="goldenrod")
plt.plot(voltage_df["t2"], voltage_df["v2"], label="Voltage Out", color="blue")

charge_start_idx = 1
charge_end_idx = 252
charge_start_time = voltage_df["t2"].iloc[charge_start_idx]
charge_start_volt = voltage_df["v2"].iloc[charge_start_idx]
charge_end_time = voltage_df["t2"].iloc[charge_end_idx]
charge_end_volt = voltage_df["v2"].iloc[charge_end_idx]

# Find time at tau voltage level for drop
tau_charge_volt_calc = (1 - (1 / math.e)) * charge_end_volt
tau_charge_volt_idx = find_closest(
    voltage_df["v2"][charge_start_idx:charge_end_idx], tau_charge_volt_calc
)
tau_charge_time = voltage_df["t2"].iloc[tau_charge_volt_idx]
tau_charge_volt = voltage_df["v2"].iloc[tau_charge_volt_idx]

plt.plot(tau_charge_time, tau_charge_volt_calc, "ro")
plt.plot(
    [charge_start_time, tau_charge_time],
    [tau_charge_volt, tau_charge_volt],
    color="red",
    linestyle="--",
    alpha=0.7,
)
plt.text(
    tau_charge_time + 100,
    tau_charge_volt,
    f"$\\tau_1 = {(tau_charge_time-charge_start_time):.2f}$ μs",
    ha="center",
    color="red",
)


# Second half starts from drop_idx
discharge_start_idx = 253
discharge_end_idx = 502
discharge_start_time = voltage_df["t2"].iloc[discharge_start_idx]
discharge_start_volt = voltage_df["v2"].iloc[discharge_start_idx]
discharge_end_time = voltage_df["t2"].iloc[discharge_start_idx]
discharge_end_volt = voltage_df["v2"].iloc[discharge_start_idx]


tau_discharge_volt_calc = (1 / math.e) * (discharge_start_volt)
tau_discharge_idx = find_closest(
    voltage_df["v2"].iloc[discharge_start_idx:discharge_end_idx],
    tau_discharge_volt_calc,
)
tau_discharge_volt = voltage_df["v2"].iloc[tau_discharge_idx]
tau_discharge_time = voltage_df["t2"].iloc[tau_discharge_idx]


plt.plot(tau_discharge_time, tau_discharge_volt, "go")
plt.plot(
    [tau_discharge_time, discharge_end_time],
    [tau_discharge_volt, tau_discharge_volt],
    color="green",
    linestyle="--",
    alpha=0.7,
)
plt.text(
    tau_discharge_time + 100,
    tau_discharge_volt,
    f"$\\tau_2 = {(tau_discharge_time-discharge_start_time):.2f}$ μs",
    ha="center",
    color="green",
)

plt.title("Voltage vs Time")
plt.xlabel(r"Time $t$ (μs)")
plt.ylabel(r"Voltage Out $V_{\mathrm{Out}}$ (V)")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig("voltage_vs_time.png")
plt.show()
