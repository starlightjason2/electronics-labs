from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit

from utils.paths import get_paths
from utils.utils import log_func, line_func, format_sig_figs, add_equation_text

paths = get_paths(__file__)

lab5_df = pd.read_csv(paths.data_dir / "part2.csv")
lab1_df = pd.read_csv(Path(paths.script_dir).parent / "lab-1" / "data" / "lab_data.csv")
lab1_df = lab1_df[["Diode I", "Diode V"]].dropna()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Part 2 data
ax1.scatter(lab5_df["I_diode"], lab5_df["V_potentiometer"], label="Experimental Data")
popt, _ = curve_fit(line_func, lab5_df["I_diode"], lab5_df["V_potentiometer"])
m, b = popt
x_fit = np.linspace(lab5_df["I_diode"].min(), lab5_df["I_diode"].max(), 100)
ax1.plot(
    x_fit,
    line_func(x_fit, *popt),
    "--",
    label=add_equation_text(
        "$V = m\\,I+b$",
        {"m": m, "b": b},
    ),
)

ax1.set_xlabel("Current (mA)")
ax1.set_ylabel("Voltage (V)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Lab 1 diode data
ax2.scatter(lab1_df["Diode I"], lab1_df["Diode V"], label="Experimental Data")
popt, _ = curve_fit(log_func, lab1_df["Diode I"], lab1_df["Diode V"])
a, b = popt
x_fit = np.linspace(lab1_df["Diode I"].min(), lab1_df["Diode I"].max(), 100)
ax2.plot(
    x_fit,
    log_func(x_fit, *popt),
    "--",
    label=add_equation_text("$V = a\\,\\ln(b\\,I)$", {"a": a, "b": b}),
)
ax2.set_xlabel("Current (A)")
ax2.set_ylabel("Voltage (V)")
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.tight_layout()
paths.output_dir.mkdir(exist_ok=True)
fig.savefig(paths.output_dir / "part2.png", dpi=150)
