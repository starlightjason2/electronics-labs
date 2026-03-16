from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
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
ax.set_xlabel("$V_+$ (V)")
ax.set_ylabel("$V_-$ (V)")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(paths.output_dir / "part1b.png", dpi=600)
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
    time_scale = 1e6
    volt_scale = 1e3

    df[v_out] = df[v_out] * volt_scale
    df[v_in] = df[v_in] * volt_scale
    df[t_in] = df[t_in] * time_scale
    df[t_out] = df[t_out] * time_scale

    ax.plot(
        df[t_in],
        df[v_in],
        label="$V_{\\mathrm{in}}$",
        color="goldenrod",
    )
    ax.plot(
        df[t_out],
        df[v_out],
        label="$V_{\\mathrm{out}}$",
        color="blue",
    )

    # dashed horizontal lines at amplitude (positive peak) of each signal
    amp_in = (df[v_in].max() - df[v_in].min()) / 2
    amp_out = (df[v_out].max() - df[v_out].min()) / 2
    center_in = (df[v_in].max() + df[v_in].min()) / 2
    center_out = (df[v_out].max() + df[v_out].min()) / 2
    gain = amp_out / amp_in
    ax.axhline(
        center_in + amp_in,
        color="goldenrod",
        linestyle="--",
        label="$A_{\\mathrm{in}}$ =" + f" $\\pm${(amp_in):.0f} mV",
    )
    ax.axhline(
        center_in - amp_in,
        color="goldenrod",
        linestyle="--",
    )

    ax.axhline(
        center_out + amp_out,
        color="blue",
        linestyle="--",
        label="$A_{\\mathrm{out}}$ =" + f" $\\pm${(amp_out):.0f} mV",
    )
    ax.axhline(
        center_out - amp_out,
        color="blue",
        linestyle="--",
    )

    ax.set_title(title)
    ax.set_xlabel("Time (μs)")
    ax.set_ylabel("Voltage (V)")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], visible=False))
    labels.append(f"$G = {gain:.2f}$")
    ax.legend(
        handles,
        labels,
        fontsize=8,
        loc="lower right",
        framealpha=1,
    )
    ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(paths.output_dir / "noninverting_opamp.png", dpi=600)
plt.close()
