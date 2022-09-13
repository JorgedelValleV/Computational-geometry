# coding: utf-8
from matplotlib import animation
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
"""
2-sphere
"""
# 0.1 para quitar el polo norte y que se vea
u = np.linspace(0.1, np.pi, 30)
v = np.linspace(0, 2 * np.pi, 60)
x = np.outer(np.sin(u), np.sin(v))
y = np.outer(np.sin(u), np.cos(v))
z = np.outer(np.cos(u), np.ones_like(v))
# Quitamos un 0.004 para no coger el polo norte
t2 = np.linspace(0, (np.pi/2)-0.004, 5000)
x2 = np.cos(t2)
y2 = np.sin(40*t2)*np.sin(t2)
z2 = np.cos(40*t2)*np.sin(t2)
c2 = x2 + y2
fig = plt.figure()
ax = plt.axes(projection='3d')
#ax.plot(x2, y2, z2, '-b',c="gray")
ax.plot_surface(x, y, z, rstride=1, cstride=1,
cmap='viridis', edgecolor='none')
ax.plot(x2, y2, z2, '-', c="gray", zorder=3)
#ax.plot_wireframe(x2, y2, z2)
ax.set_title('surface')

"""
2-esfera proyectada
"""
def proj(x, z, z0=1, alpha=0.5):
    z0 = z*0+z0
    eps = 1e-16
    x_trans = x/(abs(z0-z)**alpha+eps)
    return(x_trans)
# Nótese que añadimos un épsilon para evitar dividi entre 0!!
z0 = 1
fig = plt.figure(figsize=(12, 12))
fig.subplots_adjust(hspace=0.4, wspace=0.2)

c2 = np.sqrt(x2**2+y2**2)
col = plt.get_cmap("hot")(c2/np.max(c2))

ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.plot_surface(x, y, z, rstride=1, cstride=1,cmap='viridis', edgecolor='none', alpha=0.9)
#ax.plot(x2, y2, z2, '-',c="gray",zorder=3)
ax.scatter(x2, y2, z2, '-b', c=col, zorder=3, s=0.1)
ax.set_title('2-sphere')

ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.set_xlim3d(-1, 1)
ax.set_ylim3d(-1, 1)
ax.set_zlim3d(0, 2)
ax.plot_surface(proj(x, z, z0=z0), proj(y, z, z0=z0), z*0+1, rstride=3,cstride=10, cmap='viridis', alpha=0.5, edgecolor='purple',)
#ax.plot(proj(x2,z2,z0=z0), proj(y2,z2,z0=z0), 1, '-b',c="grey",zorder=1)
ax.scatter(proj(x2, z2, z0=z0), proj(y2, z2, z0=z0),1+0.035, '-', c=col, zorder=3, s=0.1)
ax.set_title('Stereographic projection')
plt.show()
fig.savefig('stereo2.png', dpi=250) # save the figure to file
plt.close(fig)
def proj2(x, z, t, z0=-1):
    z0 = z*0+z0
    eps = 1e-16
    x_trans = (2*x)/(2*(1-t)+(1-z)*t + eps)
    return(x_trans)
# Nótese que añadimos un épsilon para evitar dividi entre 0!!
def animate(t):
    xt = proj2(x, z, t)
    yt = proj2(y, z, t)
    zt = -t + z*(1-t)
    x2t = proj2(x2, z2, t)
    y2t = proj2(y2, z2, t)
    z2t = -t + z2*(1-t)
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    ax.plot_surface(xt, yt, zt, rstride=1, cstride=1, alpha=0.5,
    cmap='viridis', edgecolor='none')
    # ax.plot(x2t, y2t, z2t, '-', c="gray")
    ax.scatter(x2t, y2t, z2t, '-', c=col, zorder=3, s=0.1)
    return ax,
def init():
    return animate(0),
fig = plt.figure(figsize=(6, 6))
ani = animation.FuncAnimation(fig, animate, np.arange(
0, 1.001, 0.025), init_func=init, interval=30)
# Llegamos hasta 1.001 para ver el instante en 1
ani.save("ejercicio_ii.mp4", fps=10)