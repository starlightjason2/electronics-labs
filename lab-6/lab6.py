"""Lab 6: RC oscillator circuits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from utils.paths import get_paths
from utils.utils import add_equation_text

paths = get_paths(__file__)
paths.output_dir.mkdir(exist_ok=True)

# (filename, capacitance in F, resistance in Ω)
DATASETS = [
    ("lab6_75nF_3.2kOhm.csv", 75e-9, 3200),
    ("lab6_73nF_3.2kOhm.csv", 73e-9, 3200),
    ("lab6_9.3microF_3.2kOhm.csv", 9.3e-6, 3200),
]

# Oscilloscope CSV column mapping
COLS = {
    "Time - Plot 0": "t_out",
    "Amplitude - Plot 0": "v_out",
    "Time - Plot 1": "t_probe",
    "Amplitude - Plot 1": "v_probe",
}


def exp_decay(t, A, tau, V0):
    """V(t) = V0 + A * exp(-t / tau)."""
    return V0 + A * np.exp(-t / tau)


def exp_charge(t, A, tau, V0):
    """V(t) = V0 + A * (1 - exp(-t / tau))."""
    return V0 + A * (1 - np.exp(-t / tau))


def fit_period(t, v_probe, trans):
    """Fit exponential to both half-cycles of the first period.

    Auto-selects charge vs decay model based on segment shape.
    Returns list of (slice, popt, model).
    """
    results = []
    for j in range(2):
        if j + 1 >= len(trans):
            break

        sl = slice(trans[j], trans[j + 1])
        t_seg = t[sl] - t[trans[j]]
        v_seg = v_probe[sl]
        tau_guess = t_seg[-1] / 3

        # Peak in the middle = charge curve, otherwise decay
        v_mid = v_seg[len(v_seg) // 2]
        charges_up = v_mid > max(v_seg[0], v_seg[-1])

        if charges_up:
            model = exp_charge
            p0 = [v_mid - v_seg[0], tau_guess, v_seg[0]]
        else:
            model = exp_decay
            p0 = [v_seg[0] - v_seg[-1], tau_guess, v_seg[-1]]

        amp_max = 1.2 * (np.max(v_probe) - np.min(v_probe)) / 2
        bounds = ([-amp_max, 1e-10, -np.inf], [amp_max, t_seg[-1] * 5, np.inf])
        p0[0] = np.clip(p0[0], -amp_max * 0.99, amp_max * 0.99)

        popt, _ = curve_fit(
            model,
            t_seg,
            v_seg,
            p0=p0,
            bounds=bounds,
            maxfev=10000,
        )
        results.append((sl, popt, model))

    return results


def plot_traces(t, v_out, v_probe, trans, out_path):
    """Plot output and probe traces with exponential fit on the first period."""

    t_ms = t * 1e3
    fig, (ax_out, ax_probe) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    ax_out: plt.Axes  # type: ignore[attr-defined]
    ax_probe: plt.Axes  # type: ignore[attr-defined]

    # Output square wave
    ax_out.plot(t_ms, v_out, color="C0", label="Op-amp Output")
    v_out_margin = 1.2 * max(abs(v_out.min()), abs(v_out.max()))
    ax_out.set(
        xlabel="Time (ms)", ylabel="Voltage (V)", ylim=(-v_out_margin, v_out_margin)
    )
    ax_out.grid(True, alpha=0.3)

    # Probe (TP)
    ax_probe.plot(t_ms, v_probe, color="C1", label="Test Probe")
    v_margin = 1.2 * max(abs(v_probe.min()), abs(v_probe.max()))
    ax_probe.set(xlabel="Time (ms)", ylim=(-v_margin, v_margin))
    ax_probe.grid(True, alpha=0.3)

    # Overlay exponential fit on first period
    if len(trans) >= 3:
        fits = fit_period(t, v_probe, trans)
        colors = ["darkgreen", "purple"]

        for i, (sl, popt, model) in enumerate(fits):
            A, tau, V0 = popt
            model_name = "Charge" if model is exp_charge else "Discharge"

            # Extend the fit beyond the segment, fading out
            t_start = t[sl.start]
            t_end = t[sl.stop - 1] if sl.stop <= len(t) else t[-1]
            t_extend = np.linspace(0, t_ms[-1] / 1e3 - t_start, 500)
            t_plot = (t_extend + t_start) * 1e3
            v_plot = model(t_extend, *popt)

            # Solid on fitted region, faded beyond
            in_seg = t_plot <= t_end * 1e3
            ax_probe.plot(
                t_plot[in_seg],
                v_plot[in_seg],
                color=colors[i],
                ls="--",
                lw=1.8,
                label=add_equation_text(
                    rf"{model_name}: $V_0+Ae^{{-t/\tau}}$",
                    {"A": (A, "V"), r"\tau": (tau, "s"), "V_0": (V0, "V")},
                ),
            )
            ax_probe.plot(
                t_plot[~in_seg],
                v_plot[~in_seg],
                color=colors[i],
                ls="--",
                lw=1.2,
                alpha=0.7,
            )

    fig.legend(
        loc="upper center",
        ncol=3,
        columnspacing=3,
        edgecolor="none",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.84])  # type: ignore
    fig.savefig(out_path, dpi=600)
    plt.close()


def main():
    for fname, C, R in DATASETS:
        fpath = paths.data_dir / fname
        if not fpath.exists():
            continue

        df = pd.read_csv(fpath).rename(columns=COLS).dropna(how="all")
        t = df["t_out"].to_numpy()
        v_out = df["v_out"].to_numpy()
        v_probe = df["v_probe"].to_numpy()
        trans = np.where(np.diff(np.sign(v_out)))[0]

        label = fname.replace(".csv", "").replace("lab6_", "")
        plot_traces(
            t, v_out, v_probe, trans, paths.output_dir / f"oscillator_{label}.png"
        )


if __name__ == "__main__":
    main()
