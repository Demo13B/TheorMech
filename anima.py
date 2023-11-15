import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def Rot2D(X, Y, Alpha):
    RX = X * np.cos(Alpha) - Y * np.sin(Alpha)
    RY = X * np.sin(Alpha) + Y * np.cos(Alpha)
    return RX, RY


t = sp.Symbol('t')

r = sp.cos(6 * t)
phi = t + 0.2 * sp.cos(3 * t)

x = r * sp.cos(phi)
y = r * sp.sin(phi)
Vx = sp.diff(x, t)
Vy = sp.diff(y, t)
Wx = sp.diff(Vx, t)
Wy = sp.diff(Vy, t)

T = np.linspace(0, 10, 2000)

X = np.zeros_like(T)
Y = np.zeros_like(T)
VX = np.zeros_like(T)
VY = np.zeros_like(T)
WX = np.zeros_like(T)
WY = np.zeros_like(T)

for i in np.arange(len(T)):
    X[i] = sp.Subs(x, t, T[i])
    Y[i] = sp.Subs(y, t, T[i])
    VX[i] = sp.Subs(Vx, t, T[i])
    VY[i] = sp.Subs(Vy, t, T[i])
    WX[i] = sp.Subs(Wx, t, T[i])
    WY[i] = sp.Subs(Wy, t, T[i])

fig = plt.figure()

ax1 = fig.add_subplot(1, 1, 1)
ax1.axis('equal')
ax1.set(xlim=[-10, 10], ylim=[-10, 10])

ax1.plot(X, Y)

P, = ax1.plot(X[0], Y[0], marker='o')
VLine, = ax1.plot([X[0], X[0] + VX[0]], [Y[0], Y[0] + VY[0]], 'r')
WLine, = ax1.plot([X[0], X[0] + WX[0]], [Y[0], Y[0] + WY[0]], 'b')

ArrowX = np.array([-0.1, 0, -0.1])
ArrowY = np.array([0.1, 0, -0.1])

RVArrowX, RVArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[0], VX[0]))
RWArrowX, RWArrowY = Rot2D(ArrowX, ArrowY, math.atan2(WY[0], WX[0]))
VArrow, = ax1.plot(RVArrowX + X[0] + VX[0], RVArrowY + Y[0] + VY[0], 'r')
WArrow, = ax1.plot(RWArrowX + X[0] + WX[0], RWArrowY + Y[0] + WY[0], 'b')


def anima(i):
    P.set_data(X[i], Y[i])

    VLine.set_data([X[i], X[i] + VX[i]], [Y[i], Y[i] + VY[i]])
    RVArrowX, RVArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[i], VX[i]))
    VArrow.set_data(RVArrowX + X[i] + VX[i], RVArrowY + Y[i] + VY[i])

    WLine.set_data([X[i], X[i] + WX[i]], [Y[i], Y[i] + WY[i]])
    RWArrowX, RWArrowY = Rot2D(ArrowX, ArrowY, math.atan2(WY[i], WX[i]))
    WArrow.set_data(RWArrowX + X[i] + WX[i], RWArrowY + Y[i] + WY[i])
    return P, VLine, VArrow, WLine, WArrow


anim = FuncAnimation(fig, anima, frames=2000, interval=10, repeat=False)

plt.show()
