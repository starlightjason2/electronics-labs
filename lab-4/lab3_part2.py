from __future__ import annotations

import math
import os
import sys

if os.environ.get("XDG_SESSION_TYPE") == "wayland":
    os.environ.setdefault("QT_QPA_PLATFORM", "wayland")

import matplotlib

matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint
from pathlib import Path
from scipy import constants as sc

from utils.paths import get_paths
from utils.utils import load_oscilloscope_data, plot_oscilloscope_data

paths = get_paths(__file__)

DATA_FILE = "high_pass_square_510Hz"
voltage_df = load_oscilloscope_data(DATA_FILE, paths.data_dir)
plot_oscilloscope_data(
    voltage_df["t_in"],
    voltage_df["v_in"],
    voltage_df["t_out"],
    voltage_df["v_out"],
    DATA_FILE,
    paths.output_dir,
)

DATA_FILE = "high_pass_triangle_200Hz"
voltage_df = load_oscilloscope_data(DATA_FILE, paths.data_dir)
plot_oscilloscope_data(
    voltage_df["t_in"],
    voltage_df["v_in"],
    voltage_df["t_out"],
    voltage_df["v_out"],
    DATA_FILE,
    paths.output_dir,
)
