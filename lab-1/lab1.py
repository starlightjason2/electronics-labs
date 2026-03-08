from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from utils.paths import get_paths
from utils.utils import log_func, line_func

paths = get_paths(__file__)


def voltage_divider_fit(x, r_1, r_2, v_S):
    return x * v_S * r_2 / (x * r_2 + r_1 * x + r_1 * r_2)


df = pd.read_csv(paths.data_dir / "lab_data.csv")

# Resistor
fig, ax = plt.subplots()
resistor_curr = [67.0, 66.1, 65.4, 64.0, 63.0, 62.1, 61.0]
resistor_volt = [0.67, 0.66, 0.65, 0.64, 0.63, 0.62, 0.61]
ax.scatter(resistor_curr, resistor_volt)
popt, _ = curve_fit(line_func, resistor_curr, resistor_volt)
x_fit = np.linspace(min(resistor_curr), max(resistor_curr), 100)
ax.plot(x_fit, line_func(x_fit, *popt), "--")
ax.set_xlabel("Current (mA)")
ax.set_ylabel("Voltage (V)")
ax.grid(True, alpha=0.3)
paths.output_dir.mkdir(exist_ok=True)
plt.savefig(paths.output_dir / "resistor_current_vs_voltage.png", dpi=150)
plt.close()

# LEDs
fig, ax = plt.subplots()
diode_df = df[["Diode I", "Diode V"]].dropna()
red_df = df[["Red LED I", "Red LED V"]].dropna()
blue_df = df[["Blue LED I", "Blue LED V"]].dropna()

ax.scatter(diode_df["Diode I"], diode_df["Diode V"], label="Diode", color="green")
popt, _ = curve_fit(log_func, diode_df["Diode I"], diode_df["Diode V"])
x_fit = np.linspace(diode_df["Diode I"].min(), diode_df["Diode I"].max(), 100)
ax.plot(x_fit, log_func(x_fit, *popt), "--", color="green")

ax.scatter(red_df["Red LED I"], red_df["Red LED V"], label="Red LED", color="red")
popt, _ = curve_fit(log_func, red_df["Red LED I"], red_df["Red LED V"])
x_fit = np.linspace(red_df["Red LED I"].min(), red_df["Red LED I"].max(), 100)
ax.plot(x_fit, log_func(x_fit, *popt), "--", color="red")

ax.scatter(blue_df["Blue LED I"], blue_df["Blue LED V"], label="Blue LED", color="blue")
popt, _ = curve_fit(log_func, blue_df["Blue LED I"], blue_df["Blue LED V"])
x_fit = np.linspace(blue_df["Blue LED I"].min(), blue_df["Blue LED I"].max(), 100)
ax.plot(x_fit, log_func(x_fit, *popt), "--", color="blue")

ax.set_xlabel("Current (A)")
ax.set_ylabel("Voltage (V)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig(paths.output_dir / "led_current_vs_voltage.png", dpi=150)
plt.close()

# Voltage divider
fig, ax = plt.subplots()
vd_df = df[["Voltage Divider R", "Voltage Divider V"]].dropna()
ax.scatter(vd_df["Voltage Divider R"], vd_df["Voltage Divider V"])
x_fit = np.linspace(0, 10000, 100)
ax.plot(x_fit, voltage_divider_fit(x_fit, 100, 100, 5), "-", label="Theoretical Voltage $V_L$")
ax.set_xlabel("Resistance (Ω)")
ax.set_ylabel("Voltage (V)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig(paths.output_dir / "voltage_divider.png", dpi=150)
plt.close()

# Total Resistance
fig, ax = plt.subplots()
tr_df = df[["Total Resistance V", "Total Resistance I"]].dropna()
ax.scatter(tr_df["Total Resistance V"], tr_df["Total Resistance I"])
popt, _ = curve_fit(line_func, tr_df["Total Resistance V"], tr_df["Total Resistance I"])
x_fit = np.linspace(tr_df["Total Resistance V"].min(), tr_df["Total Resistance V"].max(), 100)
ax.plot(x_fit, line_func(x_fit, *popt), "--")
ax.set_xlabel("Voltage (V)")
ax.set_ylabel("Current (mA)")
ax.grid(True, alpha=0.3)
plt.savefig(paths.output_dir / "total_resistance.png", dpi=150)
plt.close()
