from __future__ import annotations

import numpy as np
import pandas as pd

from utils.paths import get_paths
from utils.utils import create_bode_plot, create_phase_plot

paths = get_paths(__file__)


high_pass_df = pd.read_csv(paths.data_dir / "freq_response_high_pass.csv")
high_pass_df["db"] = 20 * np.log10(high_pass_df["v_out"] / high_pass_df["v_in"])

omega_c = 1 / (20e-6)

create_bode_plot(high_pass_df, omega_c, "high", paths.output_dir)
create_phase_plot(high_pass_df, omega_c, "high", paths.output_dir)
