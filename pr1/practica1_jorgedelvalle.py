# -*- coding: utf-8 -*-
"""
Práctica 1 : Jorge del Valle Vázquez
"""
import os
import matplotlib.pyplot as plt
import numpy as np
#import math as mt
workpath = "C:/"
os.getcwd()
files = os.listdir(workpath)

def logistica(x,r):
    return r*x*(1-x);
def fn(x0,f,n,r):
    x = x0
    for j in range(n):
        x = f(x)
    return(x)
def orbita(x0,f,N,r):
    orb = np.empty([N])
    orb[0]= x0
    for i in range(1,N):
        orb[i] = f(orb[i-1],r)
    return(orb)
def periodo(suborb, epsilon=0.001):
    N=len(suborb)
    for i in np.arange(2,N-1,1):
        if abs(suborb[N-1] - suborb[N-i]) < epsilon :
            break
    return(i-1)
def atrac(f,x0,r, N0=200, N=50, epsilon=0.001):
    orb = orbita(x0,f,N0,r)
    ult = orb[-1*np.arange(N,0,-1)]
    per = periodo(ult, epsilon)
    V0 = np.sort([ult[N-1-i] for i in range(per)])
    return V0

"""
Sugerencias
"""
#Tomamos delta 0.0005 y analizamos el intervalo centrado en xo
def estabilidad(x0, r, N0 = 200, N = 50, h = 0.2):
    x0_ = np.arange(x0 - h,x0 + h, 0.0005)
    V0s = []
    for x in x0_:
        V0s.append(atrac(logistica, x, r))
    plt.figure(figsize=(10,10))
    for i in range(len(x0_)):
        for j in V0s[i]:
            plt.plot(x0_[i], j,'ro', markersize=1)
    #dibujamos para cada valor de x0 el conjunto de atractores
    plt.show()
#Tomamos DELTA 0.0005 y analizamos el intervalo centrado en r
def bifurcaciones(x0,r,  N0= 200, N = 50, h = 0.2):
    rss = np.arange(r - h,r + h, 0.0005)
    V0s = np.empty([N,len(rss)])*float("nan")
    for i in range(len(rss)):
        V0 = atrac(logistica,x0, rss[i], N0, N)
        V0s[range(len(V0)),i] = V0
    plt.figure(figsize=(10,10))
    for j in range(N):
        plt.plot(rss, V0s[j,], 'ro', markersize=1)
    plt.xlabel = "r"
    plt.ylabel = "V0"
    plt.axvline(x=r, ls="--")
    #dibujamos para cada valor de r el conjunto de atractores
    plt.show()
    
def error(f, r, V0,  N0 = 200, N = 50, epsilon = 0.001):
    errores = []
    V0_ult=V0[-1]
    '''
    a partir del ultimo de los elementos del conjunto atractor calculamos los siguientes valores , cuantos? el periodo
    calculamos componente a componente el error entre el V0 de entrada y el conjunto atractor obtenido de aplicar f "periodo" veces a tal cjto
    nos quedamos con el maximo de entre estos errores
    si tal maximo es 0 eso indica que nuestro calculo inicial carecia de error
    si tal maximo permanece invariante por dos iteraciones obtenemos así el error del calculo inicial
    '''
    for j in range(N):
        orb = orbita(V0_ult, f, len(V0), r)
        V0_ult = orb[-1]
        orb_ult_ord = np.sort(orb[-len(V0):])
        errores.append(max([abs(V0[i]- orb_ult_ord[i]) for i in range(len(V0))]))
        if errores[-1] == 0 or (len(errores) >= 2 and errores[-1] == errores[-2]):
            break
    return ( orb_ult_ord, errores[-1] )
    
"""
    i
"""
print("Primer conjunto: (x=0.5) (r=3.20) (N=30)")
plt.plot(orbita(0.5, logistica, 30, 3.20 ))
V0 = atrac(logistica, 0.5, 3.20)
estabilidad(0.5, 3.20)
bifurcaciones(0.5, 3.20) 
print(V0, error(logistica,3.20,  V0))

print("Segundo conjunto: (x=0.5) (r=3.50) (N=30)")
plt.plot(orbita(0.5, logistica, 30, 3.50))
V0 = atrac(logistica, 0.5, 3.50)
estabilidad(0.5, 3.50)
bifurcaciones( 0.5,3.50)
print(V0, error(logistica,3.50,  V0))

"""
    ii
"""
print("Valores de r ∈ (3.544, 4) para los cuales el conjunto atractor tiene 8 elementos")
rss = np.arange(3.544,4, 0.0005)
r_ = []
for i in range(len(rss)):
    V0 = atrac(logistica, 0.5, rss[i])
    # comprobamos que el cjto atractor tenga 8 elementos
    if len(V0)==8 : 
        r_.append(rss[i])

#seleccionamos uno de los r para los cuales hay 8 elem en cjto atractor
r8= np.random.choice(r_)
plt.plot(orbita(0.5, logistica, 50,r8))
V0 = atrac(logistica, 0.5, r8)
estabilidad(0.5, r8)
bifurcaciones( 0.5,r8)
print(r_,r8)
print(V0, error(logistica,r8,  V0))
