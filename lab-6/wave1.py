import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from utils import get_paths

paths = get_paths(__file__)

fig, ax = plt.subplots()

amp = 1
k = 2 * np.pi
omega = np.pi

freq = omega / (2 * np.pi)
wavelength = k / (2 * np.pi)
period = 1 / freq
speed = wavelength * freq

x_values = np.linspace(0, wavelength, 1000)
fps = 24
animation_len = int(period)

n = np.arange(1, 100)
t = np.linspace(0, animation_len, fps * animation_len)

omega = k * speed
phi = np.pi / 2


def psi(x, t):
    return amp * np.cos(-k * x + omega * t + phi)


(wave,) = ax.plot(x_values, psi(x_values, t[0]))
ax.set(
    xlim=[0, wavelength],
    ylim=[-amp * 2, amp * 2],
    xlabel="x [m]",
    ylabel="$\\psi(x,t)$ [m]",
)


def update(frame):
    wave.set_ydata(psi(x_values, t[frame]))
    return [wave]


ani = animation.FuncAnimation(fig=fig, func=update, frames=len(t), interval=30)
ani.save(paths.output_dir / "wavefunction1.gif")
