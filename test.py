import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def Rot2D(X, Y, Alpha):
    RX = X * np.cos(Alpha) - Y * np.sin(Alpha)
    RY = X * np.sin(Alpha) + Y * np.cos(Alpha)
    return RX, RY


R = 4
Omega = 1
t = sp.Symbol('t')

x = R * (Omega * t - sp.sin(Omega * t))
y = R * (1 - sp.cos(Omega * t))
xC = Omega * R * t
Vx = R * (Omega - Omega * sp.cos(Omega * t))
Vy = R * Omega * sp.sin(Omega * t)
Wx = R * Omega * Omega * sp.sin(Omega * t)
Wy = R * Omega * Omega * sp.cos(Omega * t)

T = np.linspace(0, 10, 1000)

X = np.zeros_like(T)
Y = np.zeros_like(T)
XC = np.zeros_like(T)
YC = R
VX = np.zeros_like(T)
VY = np.zeros_like(T)
WX = np.zeros_like(T)
WY = np.zeros_like(T)

for i in np.arange(len(T)):
    X[i] = sp.Subs(x, t, T[i])
    Y[i] = sp.Subs(y, t, T[i])
    XC[i] = sp.Subs(xC, t, T[i])
    VX[i] = sp.Subs(Vx, t, T[i])
    VY[i] = sp.Subs(Vy, t, T[i])
    WX[i] = sp.Subs(Wx, t, T[i])
    WY[i] = sp.Subs(Wy, t, T[i])

fig = plt.figure()

ax1 = fig.add_subplot(1, 1, 1)
ax1.axis('equal')
ax1.set(xlim=[-R, 12 * R], ylim=[-R, 3 * R])

ax1.plot(X, Y)
ax1.plot([X.min(), X.max()], [0, 0], 'black')

Phi = np.linspace(0, 2 * math.pi, 100)
# Circ, = ax1.plot(XC[0] + R * np.cos(Phi), YC + R * np.sin(Phi), 'g')

P, = ax1.plot(X[0], Y[0], marker='o')
VLine, = ax1.plot([X[0], X[0] + VX[0]], [Y[0], Y[0] + VY[0]], 'r')
WLine, = ax1.plot([X[0], X[0] + WX[0]], [Y[0], Y[0] + WY[0]], 'b')

ArrowX = np.array([-0.2 * R, 0, -0.2 * R])
ArrowY = np.array([0.1 * R, 0, -0.1 * R])

RVArrowX, RVArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[0], VX[0]))
VArrow, = ax1.plot(RVArrowX + X[0] + VX[0], RVArrowY + Y[0] + VY[0], 'r')


def anima(i):
    P.set_data(X[i], Y[i])
    # Circ.set_data(XC[i] + R * np.cos(Phi), YC + R * np.sin(Phi))
    VLine.set_data([X[i], X[i] + VX[i]], [Y[i], Y[i] + VY[i]])
    WLine.set_data([X[i], X[i] + WX[i]], [Y[i], Y[i] + WY[i]])
    RVArrowX, RVArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[i], VX[i]))
    VArrow.set_data(RVArrowX + X[i] + VX[i], RVArrowY + Y[i] + VY[i])
    return P, VLine, WLine, VArrow


anim = FuncAnimation(fig, anima, frames=1000, interval=2, repeat=False)

plt.show()
