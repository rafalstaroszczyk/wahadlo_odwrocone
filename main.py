import numpy as np

def epsilon(t, phi):
    global a, f, g, l, dt
    return 3 / 2 * (4 * np.pi ** 2 * a * f ** 2 * np.sin(2 * np.pi * f * t) - g) / l * np.sin(phi)


dt = 0.0001
t = 10
n = int(t / dt)
phi0 = 0.99 * np.pi
omega0 = 0
a = 0.01
l = 1
g = 10
f = 100
data = np.zeros((n, 5))  # data: i, t, phi, omega, epsilon
data[0, 2] = phi0
data[0, 3] = omega0
data[0, 4] = -3 / 2 * g / l * np.sin(phi0)
for i in range(n-1):
    data[i + 1, 0:4] = data[i, 0:4] + np.array((1, dt, data[i, 3] * dt, data[i, 4] * dt))
    data[i + 1,  4] = epsilon(data[i, 1], data[i, 2])
offset = np.zeros((1, n))
offset[:] = np.pi
data[:, 2] = offset - data[:, 2]
np.savetxt("data", data, fmt="%f")
