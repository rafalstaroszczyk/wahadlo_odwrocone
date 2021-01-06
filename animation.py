import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani


def init():
    ax.set_xlim(-1.5 * l, 1.5 * l)
    ax.set_ylim(-1.5 * l, 1.5 * l)
    return line,


def animate_frames(i):
    line.set_data([x1[i], x2[i]], [y1[i], y2[i]])
    return line,


def animate():
    global x1, x2, y1, y2, line, ax, anim
    frame_data = np.loadtxt("frame_data")
    x1 = frame_data[:, 3]
    y1 = frame_data[:, 8]
    x2 = x1 + l * np.sin(frame_data[:, 2])
    y2 = y1 - l * np.cos(frame_data[:, 2])

    fig, ax = plt.subplots()
    line, = plt.plot([], [], animated=True)

    anim = ani.FuncAnimation(fig, animate_frames, frames=range(len(frame_data[:, 0])), init_func=init, blit=True, interval=int(1000 * (frame_data[1, 1] - frame_data[0, 1])))


if __name__ == '__main__':
    l = 1
    animate()
    plt.show()