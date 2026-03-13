from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from utils.paths import get_paths
from utils.utils import line_func, format_sig_figs, add_equation_text

paths = get_paths(__file__)
paths.output_dir.mkdir(exist_ok=True)

# Part 1b
df1b = pd.read_csv(paths.data_dir / "part1b.csv")
fig, ax = plt.subplots()
ax.scatter(df1b["v_+"], df1b["v_-"])
popt, _ = curve_fit(line_func, df1b["v_+"], df1b["v_-"])
m, b = popt
x_fit = np.linspace(df1b["v_+"].min(), df1b["v_+"].max(), 100)
ax.plot(
    x_fit,
    line_func(x_fit, *popt),
    label=add_equation_text(
        "$V = m\\,I+b$",
        {"m": m, "b": (b, "V")},
    ),
)
ax.set_xlabel("v_+ (V)")
ax.set_ylabel("v_- (V)")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(paths.output_dir / "part1b.png", dpi=300)
plt.close()

# Noninverting op-amp oscilloscope traces
opamp_files = [
    ("noninverting_opamp_1kHz_50mVpp.csv", "1 kHz, 50 mVpp"),
    ("noninverting_opamp_1kHz_300mVpp.csv", "1 kHz, 300 mVpp"),
    ("noninverting_opamp_30kHz_30mVpp.csv", "30 kHz, 30 mVpp"),
    ("noninverting_opamp_30kHz_100mVpp.csv", "30 kHz, 100 mVpp"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, (fname, title) in zip(axes.flat, opamp_files):
    df = pd.read_csv(paths.data_dir / fname)
    cols = df.columns
    t_in, v_in, t_out, v_out = cols[0], cols[1], cols[2], cols[3]

    # convert time to μs
    scale = 1e6
    ax.plot(df[t_in] * scale, df[v_in], label="$V_{\\mathrm{in}}$", color="goldenrod")
    ax.plot(df[t_out] * scale, df[v_out], label="$V_{\\mathrm{out}}$", color="blue")
    ax.set_title(title)
    ax.set_xlabel("Time (μs)")
    ax.set_ylabel("Voltage (V)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(paths.output_dir / "noninverting_opamp.png", dpi=150)
plt.close()
