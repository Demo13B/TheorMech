import numpy as np
import sympy as sp
from mpmath import mp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math


def Rot(X, Y, Alpha):
    RX = X * np.cos(Alpha) - Y * np.sin(Alpha)
    RY = X * np.sin(Alpha) + Y * np.cos(Alpha)
    return RX, RY


def S(Y0, Z0, s, angle):
    MY = Y0 + s * mp.sin(angle)
    MZ = Z0 + s * mp.cos(angle)
    return MY, MZ


# Условия
m = 1
J = 3
alpha = math.pi / 6
k = 10
c = 10
l = 3
v0 = 0
w0 = math.pi / 6

# Рассчет размеров пластины
height = 2
length = height * math.tan(alpha)
remaining_height = l - height

# Рассчет времени
Steps = 1000
t_fin = 20
t = np.linspace(0, t_fin, Steps)

# Координаты phi(t), s(t)
Phi = 2 * math.pi * t
s = np.sin(math.pi * t)

# Задание параметов для отрисовки пластины
Plate_X_Start = np.array([0, 0, 0, 0, 0])
Plate_Y_Start = np.array(
    [-length / 2, length / 2, length / 2, -length/2, -length/2])
Plate_Z_Start = np.array(
    [remaining_height / 2, remaining_height / 2, height + remaining_height / 2, height + remaining_height / 2, remaining_height / 2])

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
    A = Rot(Plate_X[0], Plate_Y[0], Phi[i])
    Plate_X[i] = np.array(A[0])
    Plate_Y[i] = np.array(A[1])
    Plate_Z[i] = Plate_Z[0]

# Задание положения штифта
Pin_X = [0, 0]
Pin_Y = [0, 0]
Pin_Zl = [0, remaining_height / 2]
Pin_Zc = [remaining_height / 2, height + remaining_height / 2]
Pin_Zu = [height + remaining_height / 2, height + remaining_height]

# Задаеник положения канал
Channel_X_Start = np.array([0, 0])
Channel_Y_Start = np.array([-length/2, length/2])
Channel_Z_Start = np.array(
    [remaining_height / 2, height + remaining_height / 2])

Channel_X = np.arange(Steps * 2)
Channel_X = Channel_X.reshape((Steps, 2))


Channel_X = np.zeros_like(Channel_X, float)
Channel_Y = np.zeros_like(Channel_X, float)
Channel_Z = np.zeros_like(Channel_X, float)

Channel_X[0] = Channel_X_Start
Channel_Y[0] = Channel_Y_Start
Channel_Z[0] = Channel_Z_Start

# Рассчет положений канала
for i in range(1, Steps):
    A = Rot(Channel_X[0], Channel_Y[0], Phi[i])
    Channel_X[i] = np.array(A[0])
    Channel_Y[i] = np.array(A[1])
    Channel_Z[i] = Channel_Z[0]

# Рассчет положений точки
Point_X = np.arange(Steps, dtype=float)
Point_Y = np.arange(Steps, dtype=float)
Point_Z = np.arange(Steps, dtype=float)

Point_X = np.zeros_like(Point_X, float)
Point_Y = np.zeros_like(Point_Y, float)
Point_Z = np.zeros_like(Point_Z, float)

Point_X[0] = 0
Point_Y[0] = 0
Point_Z[0] = height / 2 + remaining_height / 2

# Рассчет положения точки
for i in range(1, Steps):
    A = S(Point_Y[0], Point_Z[0], s[i], alpha)
    Point_Y[i] += A[0]
    Point_Z[i] += A[1]

    B = Rot(Point_X[i], Point_Y[i], Phi[i])
    Point_X[i] = np.array(B[0])
    Point_Y[i] = np.array(B[1])
    Point_Z[i] = Point_Z[i]


# Создание модели
fig = plt.figure(figsize=[15, 7])
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.axis('equal')
ax.set(xlim=[- l, l], ylim=[- l, l], zlim=[0, l])

# Отрисовка штифта
ax.plot(Pin_X, Pin_Y, Pin_Zl, color='black')
ax.plot(Pin_X, Pin_Y, Pin_Zc, color='black', linestyle='--')
ax.plot(Pin_X, Pin_Y, Pin_Zu, color='black')

# Отрисовка канала
Channel = ax.plot(Channel_X[0], Channel_Y[0], Channel_Z[0],
                  linestyle='--', color='black')[0]

# Отрисовка Пластины
Drawed_Plate = ax.plot(Plate_X[0], Plate_Y[0], Plate_Z[0])[0]

# Отрисовка точки
Point = ax.plot(Point_X[0], Point_Y[0], Point_Z[0], marker='.')[0]


def anima(i):
    Drawed_Plate.set_data(Plate_X[i], Plate_Y[i])
    Drawed_Plate.set_3d_properties(Plate_Z[i])

    Channel.set_data(Channel_X[i], Channel_Y[i])
    Channel.set_3d_properties(Channel_Z[i])

    Point.set_data(Point_X[i], Point_Y[i])
    Point.set_3d_properties(Point_Z[i])

    return [Point]


anim = FuncAnimation(fig, anima, frames=len(t), interval=20, repeat=False)

plt.show()
