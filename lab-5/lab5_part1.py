from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from utils.paths import get_paths
from utils.utils import line_func, format_sig_figs, add_equation_text

paths = get_paths(__file__)

# Part 1a
df1a = pd.read_csv(paths.data_dir / "part1a.csv")
fig, ax = plt.subplots()
ax.scatter(df1a["v_+"], df1a["v_-"])
ax.set_xlabel("v_+ (V)")
ax.set_ylabel("v_- (V)")
ax.grid(True, alpha=0.3)
paths.output_dir.mkdir(exist_ok=True)
plt.savefig(paths.output_dir / "part1a.png", dpi=300)
plt.close()

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
        {"m": m, "b": b},
    ),
)
ax.set_xlabel("v_+ (V)")
ax.set_ylabel("v_- (V)")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(paths.output_dir / "part1b.png", dpi=300)
plt.close()
