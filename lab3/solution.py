import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import odeint


def odesys(y, t, m, g, alpha, c, J, k):
    dy = np.zeros(4)
    dy[0] = y[2]
    dy[1] = y[3]
    a1 = J + m * y[0]**2 * np.sin(alpha)**2
    a2 = m
    b1 = -2 * m * y[0] * y[2] * y[3] * np.sin(alpha)**2 - c * y[1]
    b2 = m * y[0] * y[3]**2 * \
        np.sin(alpha)**2 - m * g * np.cos(alpha) - k * y[2]

    dy[2] = b2 / a2
    dy[3] = b1 / a1

    return dy


m = 1
J = 3
alpha = math.pi / 6
k = 10
c = 10
g = 9.81

s0 = 0
phi0 = math.pi / 6
ds0 = 20
dphi0 = 0

y0 = [s0, phi0, ds0, dphi0]

Steps = 1000
t_fin = 20
t = np.linspace(0, t_fin, Steps)

Y = odeint(odesys, y0, t, (m, g, alpha, c, J, k))

s = Y[:, 0]
phi = Y[:, 1]

fig = plt.figure(figsize=[15, 7])
ax = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

ax.plot(t, s)
ax2.plot(t, phi)

plt.show()
