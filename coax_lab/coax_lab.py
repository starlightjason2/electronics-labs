from __future__ import annotations

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.axes import Axes
from matplotlib.markers import MarkerStyle

from utils.paths import get_paths

paths = get_paths(__file__)
paths.output_dir.mkdir(exist_ok=True)

df = pd.read_csv(paths.data_dir / "data.csv").dropna()

# reflection coefficient
df["R"] = df["V_ref"] / df["V_in"]
# transmission coefficient
df["T"] = df["V_term"] / df["V_in"]

length = 312  # meters
period = 1.58  # Time between pulses (micro s)
speed = 1.97e8  # m/s

fig, ax = plt.subplots()
