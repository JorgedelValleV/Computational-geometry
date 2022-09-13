#from mpl_toolkits import mplot3d
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

u = np.linspace(0.1, np.pi, 30)
v = np.linspace(0, 2 * np.pi, 60)
x = np.outer(np.sin(u), np.sin(v))
y = np.outer(np.sin(u), np.cos(v))
z = np.outer(np.cos(u), np.ones_like(v))

t2 = np.linspace(0.001, 1,5000)
x2 = np.sin(80 * t2/2)**3
y2 = np.cos(80 * t2/2)**3
z2 = np.sqrt(1-x2**2-y2**2)
c2 = x2 + y2

"""
2-esfera proyectada
"""

def proj(x,z,z0=1,alpha=0.5):
    z0 = z*0+z0
    eps = 1e-16
    x_trans = x/(abs(z0-z)**alpha+eps)
    return(x_trans)

z0 = 1

fig = plt.figure(figsize=(12,12))
fig.subplots_adjust(hspace=0.4, wspace=0.2)

c2 = np.sqrt(x2**2+y2**2)
col = plt.get_cmap("hot")(c2/np.max(c2))

ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='plasma', edgecolor='none',alpha=0.9)
ax.plot(x2, y2, z2, '-b',c="gray",zorder=3)
ax.set_title('2-sphere');

ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.set_xlim3d(-1,1)
ax.set_ylim3d(-1,1)
ax.set_zlim3d(0, 2)
ax.plot_surface(proj(x,z,z0=z0), proj(y,z,z0=z0), z*0+1, rstride=1, cstride=1,cmap='viridis', alpha=0.5, edgecolor='purple')
ax.scatter(proj(x2, z2, z0=z0), proj(y2, z2, z0=z0),1+0.1, '-', c=col, zorder=3, s=0.1)
ax.set_title('Stereographic projection');

ax = fig.add_subplot(2, 2, 3)
ax.plot(x2,y2, 0, '-b',c="blue",zorder=1)
ax.plot(proj(x2,z2,z0=z0), proj(y2,z2,z0=z0), 0, '-b',c="red",zorder=1)
ax.set_title('deformation');

plt.show()
fig.savefig('stereo1.png', dpi=250)
plt.close(fig) 

z0=-1
from matplotlib import animation

def param(v,z,t,z0=-1):
    z0 = z*0+z0
    eps = 1e-16
    v_trans = (2*v) / (2*(1-t)+(1-z)*t + eps)
    return(v_trans)

def animate(t):
    xt = param(x, z, t)
    yt = param(y, z, t)
    zt = -t + z*(1-t)
    x2t = param(x2, z2, t)
    y2t = param(y2, z2, t)
    z2t = -t + z2*(1-t)
    
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(-6, 6)
    ax.set_ylim3d(-6, 6)
    ax.set_zlim3d(-3, 3)
    ax.plot_surface(xt, yt, zt, rstride=1, cstride=1, alpha=0.5, cmap='viridis', edgecolor='none')
    ax.scatter(x2t, y2t, z2t, '-', c=col, zorder=3, s=0.1)
    return ax,

def init():
    return animate(0),

fig = plt.figure(figsize=(6, 6))
ani = animation.FuncAnimation(fig, animate, np.arange(0, 1.001, 0.025), init_func=init, interval=20)
ani.save("anim1.mp4", fps=5)
