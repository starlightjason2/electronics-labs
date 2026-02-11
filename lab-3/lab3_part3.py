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
ureg = pint.UnitRegistry()


square__wave_df = load_oscilloscope_data("low_pass_square_20kHz", paths.data_dir)
plot_oscilloscope_data(
    time_in=square__wave_df["t_in"],
    voltage_in=square__wave_df["v_in"],
    voltage_out=square__wave_df["v_out"],
    time_out=square__wave_df["t_out"],
    file_out="low_pass_square_20kHz",
    output_dir=paths.output_dir,
)


triange_wave_df = load_oscilloscope_data("low_pass_triangle_200kHz", paths.data_dir)
plot_oscilloscope_data(
    time_in=triange_wave_df["t_in"],
    voltage_in=triange_wave_df["v_in"],
    voltage_out=triange_wave_df["v_out"] * 10,
    time_out=triange_wave_df["t_out"],
    file_out="low_pass_triangle_200kHz",
    output_dir=paths.output_dir,
)
