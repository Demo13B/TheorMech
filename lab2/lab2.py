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

Steps = 1000
t_fin = 2
t = np.linspace(0, t_fin, Steps)

Phi = 2 * math.pi * t

Plate_X_Start = np.array([0, 0, 0, 0, 0])
Plate_Y_Start = np.array([-1, 1, 1, -1, -1])
Plate_Z_Start = np.array([1, 1, 2, 2, 1])

Plate_X = np.arange(Steps * 5)
Plate_X = Plate_X.reshape((Steps, 5))

Plate_X = np.zeros_like(Plate_X, float)
Plate_Y = np.zeros_like(Plate_X, float)
Plate_Z = np.zeros_like(Plate_X, float)

Plate_X[0] = Plate_X_Start
Plate_Y[0] = Plate_Y_Start
Plate_Z[0] = Plate_Z_Start

for i in range(1, Steps):
    A = Rot2D(Plate_X[0], Plate_Y[0], Phi[i])
    Plate_X[i] = np.array(A[0])
    Plate_Y[i] = np.array(A[1])
    Plate_Z[i] = Plate_Z[0]

X_Axis = [0, 0, 10, 0, 0]
Y_Axis = [7, 0, 0, 0, 0]
Z_Axis = [0, 0, 0, 0, 2]


fig = plt.figure(figsize=[15, 7])
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.axis('equal')
ax.set(xlim=[-5, 15], ylim=[-4, 10], zlim=[0, 2])

ax.plot(X_Axis, Y_Axis, Z_Axis, color='black')

Drawed_Box = ax.plot(Plate_X[0], Plate_Y[0], Plate_Z[0])[0]


def anima(i):
    Drawed_Box.set_data(Plate_X[i], Plate_Y[i])
    Drawed_Box.set_3d_properties(Plate_Z[i])
    return [Drawed_Box]


anim = FuncAnimation(fig, anima, frames=len(t), interval=2, repeat=False)

plt.show()
