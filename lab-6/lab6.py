"""
Lab 6: RC oscillator circuits.
Load oscilloscope traces (output and probe TP), plot voltage vs time, compare to τ = RC.
CSV columns are normalized from oscilloscope 'Time - Plot 0/1, Amplitude - Plot 0/1' to
t_out, v_out, t_probe, v_probe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from utils.paths import get_paths

paths = get_paths(__file__)
paths.output_dir.mkdir(exist_ok=True)

# (filename, C_F, R_ohm) for theoretical τ = R*C and period comparison
DATASETS = [
    ("lab6_75nF_3.2kOhm.csv", 75e-9, 3200),
    ("lab6_73nF_3.2kOhm.csv", 73e-9, 3200),
    ("lab6_9.3microF_3.2kOhm.csv", 9.3e-6, 3200),
]

OSCILLOSCOPE_COLUMNS = {
    "Time - Plot 0": "t_out",
    "Amplitude - Plot 0": "v_out",
    "Time - Plot 1": "t_probe",
    "Amplitude - Plot 1": "v_probe",
}


def load_oscillator_csv(path):
    """Load CSV and normalize headers to t_out, v_out, t_probe, v_probe."""
    df = pd.read_csv(path)
    df = df.rename(columns=OSCILLOSCOPE_COLUMNS)

    return df.dropna(how="all")


def estimate_period_s(df, channel="v_out"):
    """Estimate oscillation period (s) from peak spacing."""
    v = df[channel].to_numpy()
    t = df["t_out"].to_numpy()
    peaks, _ = find_peaks(v, height=(v.max() + v.min()) / 2, distance=5)
    if len(peaks) < 2:
        return None
    periods = np.diff(t[peaks])
    return float(np.median(periods))


def plot_traces(df, title_suffix, tau_s, out_path):
    """Plot output and probe voltage vs time in side-by-side subplots.

    tau_s = R*C for reference.
    """
    fig, (ax_out, ax_probe) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

    t_out_us = df["t_out"].to_numpy() * 1e6  # s -> μs
    t_probe_us = df["t_probe"].to_numpy() * 1e6

    # Left: output node
    ax_out.plot(t_out_us, df["v_out"], color="C0")
    ax_out.set_xlabel("Time (μs)")
    ax_out.set_ylabel("Voltage (V)")
    ax_out.grid(True, alpha=0.3)

    # Right: probe / TP node
    ax_probe.plot(t_probe_us, df["v_probe"], color="C1")
    ax_probe.set_xlabel("Time (μs)")
    ax_probe.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=150)
    plt.close()


def main():
    for fname, C_F, R_ohm in DATASETS:
        filepath = paths.data_dir / fname
        if not filepath.exists():
            continue
        df = load_oscillator_csv(filepath)
        tau_s = R_ohm * C_F
        label = fname.replace(".csv", "").replace("lab6_", "")
        out_name = f"oscillator_{label}.png"
        plot_traces(df, label, tau_s, paths.output_dir / out_name)


if __name__ == "__main__":
    main()
