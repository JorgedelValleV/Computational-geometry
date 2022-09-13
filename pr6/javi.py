# -*- coding: utf-8 -*-
"""
Created on Wed Apr 8 23:53:36 2020
@author: robert monjo
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.spatial import ConvexHull, convex_hull_plot_2d
# https://matplotlib.org/3.1.0/tutorials/colors/colormaps.htm
# =================== APARTADO 1 =====================
# q = variable de posición, dq0 = \dot{q}(0) = valor inicial de la derivada
# d = granularidad del parámetro temporal
def deriv(q, dq0, d):
    #dq = np.empty([len(q)])
    dq = (q[1:len(q)]-q[0:(len(q)-1)])/d
    dq = np.insert(dq, 0, dq0) # dq = np.concatenate(([dq0],dq))
    return dq
# Ecuación de un sistema dinámico continuo
# Ejemplo de oscilador simple
def F(q):
    k = 1
    ddq = -2*q*(q**2-1)
    return ddq
# Resolución de la ecuación dinámica \ddot{q} = F(q), obteniendo la órbita q(t)
# Los valores iniciales son la posición q0 := q(0) y la derivada dq0 := \dot{q}(0)
def orb(n, q0, dq0, F, args=None, d=0.001):
    #q = [0.0]*(n+1)
    q = np.empty([n+1])
    q[0] = q0
    q[1] = q0 + dq0*d
    for i in np.arange(2, n+1):
        args = q[i-2]
        q[i] = - q[i-2] + d**2*F(args) + 2*q[i-1]
    return q # np.array(q),
#################################################################
# ESPACIO FÁSICO D_{0,inf}
#################################################################
# Granularidad t = nd tal que d \in [10^-4,10e-3], n \in N
d = 10**(-4)
n = int(32/d)
# Condiciones iniciales:
seq_q0 = np.linspace(0., 1., num=10)
seq_dq0 = np.linspace(0., 2., num=10)
# Pintamos el espacio de fases
def simplectica(q0, dq0, F, d, n, col=0, marker='-'):
    q = orb(n, q0=q0, dq0=dq0, F=F, d=d)
    dq = deriv(q, dq0=dq0, d=d)
    p = dq/2
    plt.plot(q, p, marker, c=plt.get_cmap("winter")(col))
fig = plt.figure(figsize=(8, 5))
fig.subplots_adjust(hspace=0.4, wspace=0.2)
ax = fig.add_subplot(1, 1, 1)
for i in range(len(seq_q0)):
    for j in range(len(seq_dq0)):
        q0 = seq_q0[i]
        dq0 = seq_dq0[j]
        col = (1+i+j*(len(seq_q0)))/(len(seq_q0)*len(seq_dq0))
        #ax = fig.add_subplot(len(seq_q0), len(seq_dq0), 1+i+j*(len(seq_q0)))
        simplectica(q0=q0, dq0=dq0, F=F, col=col, marker=',', d=d, n=n)
ax.set_xlabel("q(t)", fontsize=12)
ax.set_ylabel("p(t)", fontsize=12)
fig.savefig('Simplectic.png', dpi=250)
plt.show()
# =================== APARTADO 1 =====================

#################################################################
# Diagrama de fases (q, p) para un tiempo determinado
#################################################################
def diagrama_fases_t(t, d, show=True):
    # Granularidad t = nd tal que d \in [10e-4,10e-3], n \in N
    n = int(t/d)
    # Condiciones iniciales:
    seq_q0 = np.linspace(0., 1., num=20)
    seq_dq0 = np.linspace(0., 2, num=20)
    q2 = np.array([])
    p2 = np.array([])
    for i in range(len(seq_q0)):
        for j in range(len(seq_dq0)):
            q0 = seq_q0[i]
            dq0 = seq_dq0[j]
            q = orb(n, q0=q0, dq0=dq0, F=F, d=d)
            dq = deriv(q, dq0=dq0, d=d)
            p = dq/2
            q2 = np.append(q2, q[-1])
            p2 = np.append(p2, p[-1])
            if show:
                plt.plot(q[-1], p[-1], marker=".", markersize=10, markeredgecolor="steelblue", markerfacecolor="steelblue")
            # if (i == 19 or j == 0):
            # plt.plot(q[-1], p[-1], marker=".", markersize= 10, markeredgecolor="steelblue",markerfacecolor="steelblue")
            # else:
            # plt.plot(q[-1], p[-1], marker=".", markersize= 10, markeredgecolor="orange",markerfacecolor="orange")
    # plt.scatter(q2, p2, marker=".") # otra forma
    if show:
        plt.rcParams["legend.markerscale"] = 6
        plt.xlabel("q(t)", fontsize=12)
        plt.ylabel("p(t)", fontsize=12)
        plt.savefig('Fases_0.25_naranja.png', dpi=250)
        plt.show()
    return q2, p2
def area(q2, p2, show=True):
    # Total envolutra convexa
    X = np.array([q2, p2]).T
    hull = ConvexHull(X)
    
    if show:
        fig = convex_hull_plot_2d(hull)
        fig.savefig('Convexa.png', dpi=250)
    X_area = hull.volume
    print("Área total de la envoltura convexa:", hull.volume)
    # Envoltura convexa parte inferior
    X_bottom = np.array([q2[::20], p2[::20]]).T
    hull_bottom = ConvexHull(X_bottom)
    if show:
        fig2 = convex_hull_plot_2d(hull_bottom)
        fig2.savefig('Convexa_bot.png', dpi=250)
    X_bottom = hull_bottom.volume
    print("Área de la parte inferior:", hull_bottom.volume)
    # Envoltura convexa parte derecha
    X_right = np.array([q2[-20:], p2[-20:]]).T
    hull_right = ConvexHull(X_right)
    if show:
        fig3 = convex_hull_plot_2d(hull_right)
        fig3.savefig('Convexa_right.png', dpi=250)
    X_right = hull_right.volume
    print("Área de la parte derecha:", hull_right.volume)
    # El área buscada es la resta de la total menos las otras
    return X_area - X_bottom - X_right
# Primero probamos para delta = 10^{-3}
t = 1/4
d = 10**(-3)
q2, p2 = diagrama_fases_t(t, d, show=True)
area_total = area(q2, p2, show=True)
print("Área de D_t para t = 1/4:", area_total)
# Si probamos para delta = 10^{-4}
t = 1/4
d = 10**(-4)
q2, p2 = diagrama_fases_t(t, d, False)
area_total = area(q2, p2, False)
print("Área de D_t para t = 1/4:", area_total)
# Observación: la estimación del error, utilizando los valores de
# t = 1/4 y delta = 10^{-3}, es de 0.001.

# =================== APARTADO 3 =====================
def animate(t):
    ax = plt.axes()
    plt.xlim(-2.2, 2.2)
    plt.ylim(-1.2, 1.2)
    q2, p2 = diagrama_fases_t(t, d=10**(-3.5), show=False)
    # ,c=plt.get_cmap("winter")(0)
    ax.scatter(q2, p2, c=q2, cmap="winter", marker=".")
    return ax,
def init():
    return animate(0.1),
fig = plt.figure(figsize=(6, 6))
ani = animation.FuncAnimation(
fig, animate, np.arange(0.1, 5, 0.1), init_func=init)
ani.save("ejercicio_iii_winter.mp4", fps=25)
