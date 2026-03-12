import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from utils import get_paths

paths = get_paths(__file__)

fig, ax = plt.subplots()

amp = 1
length = 1
x_values = np.linspace(0, length, 1000)
speed = 0.4
period = length / speed
fps = 24
animation_len = int(2 * period)

n = np.arange(1, 25)
t = np.linspace(0, animation_len, fps * animation_len)


def psi(x, t):
    return (
        amp
        * (9 / (np.pi**2 * n**2))
        * np.sin(n * np.pi / 3)
        * np.sin(n * np.pi * x[:, None] / length)
        * np.cos(n * np.pi * speed * t / length)
    ).sum(axis=1)


(wave,) = ax.plot(x_values, psi(x_values, t[0]))
ax.set(
    xlim=[0, length],
    ylim=[-amp * 2, amp * 2],
    xlabel="x [m]",
    ylabel="$\\psi(x,t)$ [m]",
)


def update(frame):
    wave.set_ydata(psi(x_values, t[frame]))
    return [wave]


ani = animation.FuncAnimation(fig=fig, func=update, frames=len(t), interval=30)
ani.save(paths.output_dir / "wavefunction2.gif")
