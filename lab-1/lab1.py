import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path

SAVE_DIR = "./output"


def voltage_divider_fit(x, r_1, r_2, v_S):
    return x * v_S * r_2 / (x * r_2 + r_1 * x + r_1 * r_2)


# diode fit
def log_func(x, a, b):
    return a * (np.log(b * x))


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
        color=None,
        # fit to line by default
        fit_func=line_func,
    ):
        self.title = title
        self.data_points = data_points
        self.fit_func = fit_func
        self.color = color

    def get_x(self):
        return [point.x for point in self.data_points]

    def get_y(self):
        return [point.y for point in self.data_points]

    def graph(self):
        plt.scatter(self.get_x(), self.get_y(), label=self.title, color=self.color)

    def fit(self, label=""):
        x_values, y_values = self.get_x(), self.get_y()
        curve, curve_params = curve_fit(self.fit_func, x_values, y_values)
        x_fit = np.linspace(min(x_values), max(x_values), 100)
        y_fit = self.fit_func(x_fit, *curve)
        plt.plot(x_fit, y_fit, "--", label=f"{self.title} fit", color=self.color)


class Graph:
    def __init__(
        self, data: list[DataSet], x_label: str, y_label: str, title: str = ""
    ):
        self.data = data
        self.x_label = x_label
        self.y_label = y_label
        self.title = title

    def save(self, fit=False, save=True):
        for data_set in self.data:
            data_set.graph()

            if fit:
                data_set.fit()

        plt.title(self.title)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)

        if len(self.data) > 1:
            plt.legend()

        if save:
            file_name = self.title.lower().replace(" ", "_")
            plt.savefig(Path(SAVE_DIR) / file_name)
        plt.close("all")


# Resistor
resistor = Graph(
    title="Resistor Current vs Voltage",
    x_label="Current (mAmps)",
    y_label="Voltage (Volts)",
    data=[
        DataSet(
            data_points=[
                DataPoint(67.0, 0.67),
                DataPoint(66.1, 0.66),
                DataPoint(65.4, 0.65),
                DataPoint(64.0, 0.64),
                DataPoint(63.0, 0.63),
                DataPoint(62.1, 0.62),
                DataPoint(61.0, 0.61),
            ],
            fit_func=line_func,
        ),
    ],
)
resistor.save(fit=True)


# LEDs
leds = Graph(
    title="LED Current vs Voltage",
    x_label="Current (Amps)",
    y_label="Voltage (Volts)",
    data=[
        DataSet(
            data_points=[
                DataPoint(0.54, 0.57),
                DataPoint(0.5, 0.567),
                DataPoint(00.48, 0.565),
                DataPoint(00.45, 0.562),
                DataPoint(00.41, 0.558),
                DataPoint(00.38, 0.554),
                DataPoint(00.35, 0.552),
                DataPoint(00.31, 0.547),
                DataPoint(00.27, 0.541),
                DataPoint(0.6, 0.577),
                DataPoint(00.72, 0.585),
                DataPoint(00.83, 0.593),
                DataPoint(01.03, 0.602),
                DataPoint(01.31, 0.614),
                DataPoint(01.66, 0.625),
                DataPoint(01.92, 0.632),
                DataPoint(02.39, 0.642),
            ],
            title="Diode",
            color="green",
            fit_func=log_func,
        ),
        DataSet(
            data_points=[
                DataPoint(0.16, 1.753),
                DataPoint(0.18, 1.755),
                DataPoint(0.2, 1.76),
                DataPoint(0.25, 1.771),
                DataPoint(0.3, 1.778),
                DataPoint(0.36, 1.785),
                DataPoint(0.45, 1.794),
                DataPoint(0.6, 1.807),
                DataPoint(0.75, 1.817),
                DataPoint(0.88, 1.824),
                DataPoint(1.15, 1.836),
                DataPoint(1.55, 1.851),
            ],
            title="Red LED",
            color="red",
            fit_func=log_func,
        ),
        DataSet(
            data_points=[
                DataPoint(0.25, 2.536),
                DataPoint(0.29, 2.541),
                DataPoint(0.32, 2.546),
                DataPoint(0.41, 2.557),
                DataPoint(0.5, 2.566),
                DataPoint(0.62, 2.577),
                DataPoint(0.81, 2.591),
                DataPoint(1.02, 2.605),
                DataPoint(1.52, 2.632),
                DataPoint(3.32, 2.707),
                DataPoint(2.26, 2.665),
                DataPoint(1.67, 2.640),
            ],
            title="Blue LED",
            color="blue",
            fit_func=log_func,
        ),
    ],
)
leds.save(fit=True)

voltage_divider = Graph(
    title="Voltage divider",
    x_label="Resistance (Ohms)",
    y_label="Voltage (Volts)",
    data=[
        DataSet(
            data_points=[
                DataPoint(180.4, 2.022),
                DataPoint(551, 2.330),
                DataPoint(1062, 2.44),
                DataPoint(1793, 2.484),
                DataPoint(2610, 2.500),
                DataPoint(3370, 2.520),
                DataPoint(4650, 2.527),
                DataPoint(7003, 2.537),
            ],
        ),
    ],
)
graph = DataSet(
    data_points=[
        DataPoint(r_L, voltage_divider_fit(r_L, 100, 100, 5))
        for r_L in np.linspace(0, 10000, 100)
    ]
)
plt.plot(graph.get_x(), graph.get_y(), "-", label="Theoretical Voltage $V_L$")
plt.legend()
voltage_divider.save()

total_resistance = Graph(
    title="Total Resistance",
    x_label="Voltage (V)",
    y_label="Current (mA)",
    data=[DataSet(data_points=[DataPoint(0.6661, 0), DataPoint(0.580, 10.23)])],
)
total_resistance.save(fit=True)
