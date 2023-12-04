import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math


def Rot2D(X, Y, Alpha):
    RX = X * np.cos(Alpha) - Y * np.sin(Alpha)
    RY = X * np.sin(Alpha) + Y * np.cos(Alpha)
    return RX, RY


# Условия
m = 1
J = 3
alpha = math.pi / 6
k = 10
c = 10
l = 3
s0 = 0
phi0 = math.pi / 6

Steps = 1001
t_fin = 2
t = np.linspace(0, t_fin, Steps)

Phi = 2 * math.pi * t

X_Box = np.array([0, 0, 0, 0, 0])
Y_Box = np.array([-1, 1, 1, -1, -1])
Z_Box = np.array([1, 1, 2, 2, 1])

X = np.arange(5005)
X = X.reshape((1001, 5))

X = np.zeros_like(X, float)
Y = np.zeros_like(X, float)
Z = np.zeros_like(X, float)

X[0] = X_Box
Y[0] = Y_Box
Z[0] = Z_Box

for i in range(1, 1001):
    A = Rot2D(X[0], Y[0], Phi[i])
    X[i] = np.array(A[0])
    Y[i] = np.array(A[1])
    print(X[i], Y[i])
    Z[i] = Z[0]

X_Axis = [0, 0, 10, 0, 0]
Y_Axis = [7, 0, 0, 0, 0]
Z_Axis = [0, 0, 0, 0, 2]


fig = plt.figure(figsize=[15, 7])
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.axis('equal')
ax.set(xlim=[-5, 15], ylim=[-4, 10], zlim=[0, 2])

ax.plot(X_Axis, Y_Axis, Z_Axis, color='black')

Drawed_Box = ax.plot(X[0], Y[0], Z[0])[0]


def anima(i):
    Drawed_Box.set_data(X[i], Y[i])
    Drawed_Box.set_3d_properties(Z[i])
    return [Drawed_Box]


anim = FuncAnimation(fig, anima, frames=len(t), interval=2, repeat=False)

plt.show()
