import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Функция поворота стрелки


def Rot2D(X, Y, Alpha):
    RX = X * np.cos(Alpha) - Y * np.sin(Alpha)
    RY = X * np.sin(Alpha) + Y * np.cos(Alpha)
    return RX, RY


t = sp.Symbol('t')

# Задание условия
r = sp.cos(6 * t)
phi = t + 0.2 * sp.cos(3 * t)

# Рассчет формул
x = r * sp.cos(phi)
y = r * sp.sin(phi)
Vx = sp.diff(x, t) * 0.2
Vy = sp.diff(y, t) * 0.2
Wx = sp.diff(Vx, t) * 0.2
Wy = sp.diff(Vy, t) * 0.2

# Формирование векторов значений
T = np.linspace(0, 10, 2000)

X = np.zeros_like(T)
Y = np.zeros_like(T)
VX = np.zeros_like(T)
VY = np.zeros_like(T)
WX = np.zeros_like(T)
WY = np.zeros_like(T)

# Заполнение векторов значений
for i in np.arange(len(T)):
    X[i] = sp.Subs(x, t, T[i])
    Y[i] = sp.Subs(y, t, T[i])
    VX[i] = sp.Subs(Vx, t, T[i])
    VY[i] = sp.Subs(Vy, t, T[i])
    WX[i] = sp.Subs(Wx, t, T[i])
    WY[i] = sp.Subs(Wy, t, T[i])

# Создаем фигуру
fig = plt.figure()

ax1 = fig.add_subplot(1, 1, 1)
ax1.axis('equal')
ax1.set(xlim=[-2, 2], ylim=[-2, 2])

ax1.plot(X, Y)

# Добавляем векторы
P, = ax1.plot(X[0], Y[0], marker='o')
RLine, = ax1.plot([0, X[0]], [0, Y[0]], 'g')
VLine, = ax1.plot([X[0], X[0] + VX[0]], [Y[0], Y[0] + VY[0]], 'r')
WLine, = ax1.plot([X[0], X[0] + WX[0]], [Y[0], Y[0] + WY[0]], 'b')

# Добавляем стрелку
ArrowX = np.array([-0.1, 0, -0.1])
ArrowY = np.array([0.1, 0, -0.1])

# Добавляем стрелки для векторов
RRArrowX, RRArrowY = Rot2D(ArrowX, ArrowY, math.atan2(Y[0], X[0]))
RVArrowX, RVArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[0], VX[0]))
RWArrowX, RWArrowY = Rot2D(ArrowX, ArrowY, math.atan2(WY[0], WX[0]))
VArrow, = ax1.plot(RVArrowX + X[0] + VX[0], RVArrowY + Y[0] + VY[0], 'r')
WArrow, = ax1.plot(RWArrowX + X[0] + WX[0], RWArrowY + Y[0] + WY[0], 'b')
RArrow, = ax1.plot(RRArrowX + X[0], RRArrowX + Y[0], 'g')

# Функция анимации


def anima(i):
    P.set_data(X[i], Y[i])

    RLine.set_data([0, X[i]], [0, Y[i]])
    RRArrowX, RRArrowY = Rot2D(ArrowX, ArrowY, math.atan2(Y[i], X[i]))
    RArrow.set_data(RRArrowX + X[i], RRArrowY + Y[i])

    VLine.set_data([X[i], X[i] + VX[i]], [Y[i], Y[i] + VY[i]])
    RVArrowX, RVArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[i], VX[i]))
    VArrow.set_data(RVArrowX + X[i] + VX[i], RVArrowY + Y[i] + VY[i])

    WLine.set_data([X[i], X[i] + WX[i]], [Y[i], Y[i] + WY[i]])
    RWArrowX, RWArrowY = Rot2D(ArrowX, ArrowY, math.atan2(WY[i], WX[i]))
    WArrow.set_data(RWArrowX + X[i] + WX[i], RWArrowY + Y[i] + WY[i])
    return P, RLine, RArrow, VLine, VArrow, WLine, WArrow


# Запуск анимации
anim = FuncAnimation(fig, anima, frames=2000, interval=10, repeat=False)

# Показать фигуру
plt.show()
