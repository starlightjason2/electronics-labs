from __future__ import annotations

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.axes import Axes
from matplotlib.markers import MarkerStyle
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from utils.paths import get_paths
from utils.utils import (
    format_sig_figs,
    add_equation_text,
)

paths = get_paths(__file__)
paths.output_dir.mkdir(exist_ok=True)

df = pd.read_csv(paths.data_dir / "data.csv").dropna()
print(df)
# reflection coefficient
df["R"] = df["V_ref"] / df["V_in"]
# transmission coefficient
df["T"] = df["V_term"] / df["V_in"]

length = 156  # meters
period = 1.58  # Time between pulses (micro s)
speed = 1.97e8  # m/s

fig, ax = plt.subplots()

r0 = 39 / 1000
z0 = 53.5
c0 = 100e-12
v = 1.97e8
# Single-pass attenuation: e^{-Γ L / 2v} where Γ/2v = R0 C0 v / 2
alpha = 0.5 * r0 * c0 * v * length  # exponent for one traversal
damp_1x = np.exp(-alpha)  # transmission (single pass)
damp_2x = np.exp(-2 * alpha)  # reflection (round trip)
print(f"single-pass damping = {damp_1x:.4f}, round-trip damping = {damp_2x:.4f}")


def theory_R(z1, z2):
    return (z2 - z1) / (z1 + z2)


def theory_T(z1, z2):
    return 1 + theory_R(z1, z2)


r_values = np.linspace(0, 160, 1000)
ax.scatter(df["R_term"], df["R"], label="$R=V_R / V_I$", color="C0")
ax.plot(
    r_values,
    [-theory_R(r, z0) for r in r_values],
    label=r"$R_{\mathrm{theory}}$ w/o damping",
    color="C0",
)

ax.plot(
    r_values,
    [theory_R(z0, r) * damp_2x for r in r_values],
    label=r"$R_{\mathrm{theory}}$ with damping",
    color="C0",
    linestyle="--",
)

ax.scatter(df["R_term"], df["T"], label="$T=V_T / V_I$", color="C1")
ax.plot(
    r_values,
    [theory_T(z0, r) for r in r_values],
    label=r"$T_{\mathrm{theory}}$ w/o damping",
    color="C1",
)
ax.plot(
    r_values,
    [theory_T(z0, r) * damp_1x for r in r_values],
    label=r"$T_{\mathrm{theory}}$ with damping",
    color="C1",
    linestyle="--",
)

ax.set_xlabel("Terminating Resistance $R$ ($\\Omega$)")
ax.set_ylabel("Reflection/Transmission Coefficient $R, \ T$ (V / V)")

ax.legend()
ax.grid(True, alpha=0.3)
# fig.tight_layout()
fig.savefig(paths.output_dir / "wave_propagation_damping.png", dpi=600)
plt.close()
