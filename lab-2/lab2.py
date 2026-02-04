import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path

SAVE_DIR = "./"


# Exponential fit
def exp_func(x, a, b):
    return a * np.exp(b * x)


# Line fit
def line_func(x, m, b):
    return m * x + b


class DataPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class DataSet:
    def __init__(
        self,
        data_points: list[DataPoint],
        title="",
        # fit to line by default
        fit_func=line_func,
    ):
        self.title = title
        self.data_points = data_points
        self.fit_func = fit_func

    def get_x(self):
        return [point.x for point in self.data_points]

    def get_y(self):
        return [point.y for point in self.data_points]

    def graph(self):
        plt.scatter(self.get_x(), self.get_y(), label=self.title)

    def linear_fit(self, equation_latex: str):
        x_values, y_values = self.get_x(), self.get_y()
        curve, _ = curve_fit(self.fit_func, x_values, y_values)
        m, b = curve
        x_fit = np.linspace(min(x_values), max(x_values), 100)
        y_fit = self.fit_func(x_fit, *curve)
        plt.plot(x_fit, y_fit, "--", label=f"{self.title} fit")

        # Add equation text to plot
        equation = f"${equation_latex}$".replace("m", f"{m:.2f}").replace(
            "b", f"{b:.2f}"
        )

        plt.text(
            0.05,
            0.95,
            equation,
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            verticalalignment="top",
            fontsize=14,
        )


class Graph:
    def __init__(self, data: list[DataSet], title: str, x_label: str, y_label: str):
        self.data = data
        self.x_label = x_label
        self.y_label = y_label
        self.title = title

    def save(self, fit_equation="", save=True):
        plt.close("all")
        for data_set in self.data:
            data_set.graph()

            if fit_equation:
                data_set.linear_fit(fit_equation)

        plt.title(self.title)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)

        if len(self.data) > 1:
            plt.legend()

        if save:
            file_name = f"{self.title}.png".lower().replace(" ", "_")
            plt.savefig(Path(SAVE_DIR) / file_name)


# Transistor Data
transsitor_current = Graph(
    title="Transistor Current Characteristics",
    x_label="Base Current (microA)",
    y_label="Collector Current (mA)",
    data=[
        DataSet(
            data_points=[
                DataPoint(0.2, 314),
                DataPoint(1.04, 196),
                DataPoint(1.51, 137),
                DataPoint(1.81, 100),
                DataPoint(2, 78),
                DataPoint(2.14, 65),
                DataPoint(2.25, 46),
                DataPoint(2.33, 36),
                DataPoint(2.42, 24),
                DataPoint(2.47, 17),
            ],
            title="Collector vs Base Current",
            fit_func=line_func,
        ),
    ],
)
transsitor_current.save(fit_equation=r"I_C = m \, I_B + b")

# Base-Emitter Voltage vs Base Current
base_emitter = Graph(
    title="Collector Current vs. Base-Emitter Voltage",
    y_label=r"Base-Emitter Voltage $V_{BE}$ (V)",
    x_label=r"Collector Current $I_C$ ($\mu$A)",
    data=[
        DataSet(
            data_points=[
                DataPoint(196, 0.614),
                DataPoint(137, 0.624),
                DataPoint(100, 0.630),
                DataPoint(78, 0.632),
                DataPoint(65, 0.634),
                DataPoint(46, 0.635),
                DataPoint(36, 0.638),
                DataPoint(24, 0.642),
                DataPoint(17, 0.645),
            ],
        ),
    ],
)
base_emitter.save()
