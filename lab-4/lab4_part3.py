from __future__ import annotations

import numpy as np
import pandas as pd

from utils.paths import get_paths
from utils.utils import (
    create_bode_plot,
    create_phase_plot,
    plot_oscilloscope_data,
    load_oscilloscope_data,
)
import matplotlib.pyplot as plt

paths = get_paths(__file__)


low_pass_df = load_oscilloscope_data("low_pass_400Hz", paths.data_dir)

omega_c = 1 / (20e-6)

plt.figure(figsize=(10, 6))
plt.plot(
    low_pass_df["t_in"], low_pass_df["v_out"], label="Voltage In", color="goldenrod"
)
plt.xlabel(r"Time $t$ (μs)")
plt.ylabel(r"Voltage $V_{\mathrm{Out}}$ (V)")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig(paths.output_dir / f"low_pass_400Hz.png", dpi=150)
