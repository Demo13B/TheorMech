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
v0 = 0
w0 = math.pi / 6

# Условия для отрисовки
Steps = 1000
t_fin = 2
t = np.linspace(0, t_fin, Steps)

# Получаем phi(t)
Phi = 2 * math.pi * t

# Задаем параметры для отрисовки пластины
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

# Рассчет положений пластины
for i in range(1, Steps):
    A = Rot2D(Plate_X[0], Plate_Y[0], Phi[i])
    Plate_X[i] = np.array(A[0])
    Plate_Y[i] = np.array(A[1])
    Plate_Z[i] = Plate_Z[0]

# Задаем штифт
X_Axis = [0, 0]
Y_Axis = [0, 0]
Z_Axis = [0, 3]

# Создаем график
fig = plt.figure(figsize=[15, 7])
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.axis('equal')
ax.set(xlim=[-4, 4], ylim=[-4, 4], zlim=[0, 4])

# Отрисовка штифта
ax.plot(X_Axis, Y_Axis, Z_Axis, color='black', linestyle='--')

# Отрисовка Пластины
Drawed_Plate = ax.plot(Plate_X[0], Plate_Y[0], Plate_Z[0])[0]


def anima(i):
    Drawed_Plate.set_data(Plate_X[i], Plate_Y[i])
    Drawed_Plate.set_3d_properties(Plate_Z[i])
    return [Drawed_Plate]


anim = FuncAnimation(fig, anima, frames=len(t), interval=2, repeat=False)

plt.show()
