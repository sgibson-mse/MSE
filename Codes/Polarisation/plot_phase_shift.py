import numpy as np
import matplotlib.pyplot as pyplot

from matplotlib.animation import FuncAnimation, writers

pi = np.pi
fig, ax = pyplot.subplots(2, 1)
x_data_pol, y_data_pol = [], []

T_data_field = np.linspace(0, 4 * pi, 1500)

ln1, = ax[1].plot([], [], 'ro', animated=True)
ln2, ln3, = ax[0].plot([], [], 'r', [], [], 'b', animated=True)

ln = [ln1, ln2, ln3]

ax[1].set_xlim(-1, 1)
ax[1].set_ylim(-1, 1)
ax[1].set_xlabel("$E_x$ Field (Arb. Units)")
ax[1].set_ylabel("$E_y$ Field (Arb. Units)")
ax[0].set_xlim(0, 2)
ax[0].set_ylim(-1, 1)
ax[0].set_xlabel("Time (cycles)")
ax[0].set_ylabel("E field (Arb. Units)")

fig.tight_layout()


def xfunc(T):
    if T < 2 * pi:
        return np.sin(T)
    elif T < 4 * pi:
        return np.sin(T - (0.25 * (T - 2 * pi)))
    elif T < 6 * pi:
        return np.sin(T - (pi / 2))
    else:
        return np.sin(T - (0.25 * (T - 6 * pi) + pi / 2))


def Ex_func(T, T0):
    if T0 < 2 * pi:
        return np.sin(T)
    elif T0 < 4 * pi:
        return np.sin(T - (0.25 * (T0 - 2 * pi)))
    elif T0 < 6 * pi:
        return np.sin(T - (pi / 2))
    else:
        return np.sin(T - (0.25 * (T0 - 6 * pi) + pi / 2))

def yfunc(T):
    return np.sin(T)

def frame_update(frame):
    x_data_pol.append(xfunc(frame))
    y_data_pol.append(yfunc(frame))

    if len(x_data_pol) > 10:
        x_data_pol.pop(0)
        y_data_pol.pop(0)

    Ey_data_field = np.sin(T_data_field)
    Ex_data_field = Ex_func(T_data_field, frame)
    ln[0].set_data(x_data_pol, y_data_pol)
    ln[1].set_data(T_data_field / (2 * pi), Ey_data_field)
    ln[2].set_data(T_data_field / (2 * pi), Ex_data_field)
    return ln

animate = FuncAnimation(fig, frame_update, frames=np.linspace(0, 12 * pi, 1500), blit=True, repeat=False)
#animate.save("Polarisation.mp4", writers['ffmpeg'](fps=30))
pyplot.show()