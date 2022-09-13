##Mario Jimenez Gutierrez
import os
import matplotlib.pyplot as plt
import numpy as np#import math as mt

workpath = "C:/"
os.getcwd()
files = os.listdir(workpath)



def logistica(x, r):
    return r*x*(1-x);

def fn(x0,f,n, r):
    x = x0
    for j in range(n):
        x = f(x, r)
    return(x)

def orbita(x0,f,N, r):
    orb = np.empty([N])
    for i in range(N):
        orb[i] = fn(x0, f, i, r)
    return orb

def periodo(suborb, epsilon=0.001):
    N=len(suborb)
    for i in np.arange(2,N-1,1):
        if abs(suborb[N-1] - suborb[N-i]) < epsilon :
            break
    return(i-1)
    
def atrac(f, x0,r, N0 = 200, N = 50, epsilon=0.001):
    orb = orbita(x0,f,N0,r)
    ult = orb[-1*np.arange(N,0,-1)]
    per = periodo(ult, epsilon)
    V0 = np.sort([ult[N-1-i] for i in range(per)])
    return V0

"""Como se indica en las sugerencias, calula para valores cerca de x0 su conjunto
    atractor"""
def analisis_estabilidad(r, x0, N0 = 200, N = 50, delta = 0.2):
    x0s = np.arange(x0 - delta,x0 + delta, 0.0005)
    atractores = []
    
    for valor in x0s:
        atractores.append(atrac(logistica, valor, r))
    
    plt.figure(figsize=(10,10))
    
    for i in range(len(x0s)):
        for j in atractores[i]:
            plt.plot(x0s[i], j,'ro', markersize=1)
    
    plt.show()
    
    
def maximo(l):
    m = l[0]
    for i in range(len(l)):
        if l[i]>m:
            m = l[i]
    return m
def errores_aux(f, r, V0,  N0 = 200, N = 50, epsilon = 0.001):
    period = len(V0)
    ultimo = V0[-1]
    errores = []
    for j in range(N):
        orbit = orbita(ultimo, f, period, r)
        ultimo = orbit[-1]
        ult_aux = np.sort(orbit[-period:])
        errores.append(maximo([abs(V0[i]- ult_aux[i]) for i in range(period)]))
        if errores[-1] == 0 or (len(errores) >= 2 and errores[-1] == errores[-2]):
            break
    return ( ult_aux, errores[-1] )
    
    """ Como se indica en las sugerencias, calula para valores cerca de r su conjunto
    atractor"""
def analisis_bifurcaciones(x0,r,  N0= 200, N = 50, delta = 0.2):
    rss = np.arange(r - delta,r + delta, 0.0005)
    V0s = np.empty([N,len(rss)])
    V0s *=float("nan")
    for i in range(len(rss)):
        V0 = atrac(logistica,x0, rss[i], N0, N)
        V0s[range(len(V0)),i] = V0
    
    plt.figure(figsize=(10,10))
    
    for j in range(N):
        plt.plot(rss, V0s[j,], 'ro', markersize=1)
    plt.xlabel = "r"
    plt.ylabel = "V0"

    plt.axvline(x=r, ls="--")
    plt.show()
    


"""Apartado 1.a"""
print("Apartado 1.a|n")
plt.plot(orbita(0.5, logistica, 30, 3.18 ))
atractores = atrac(logistica, 0.5, 3.18)
print(atractores) 

print(errores_aux(logistica,3.18,  atractores))

analisis_estabilidad(3.18, 0.5)
analisis_bifurcaciones(0.5, 3.18) 



"""Apartado 1.b"""
print("Apartado 1.b|n")
plt.plot(orbita(0.5, logistica, 30, 3.50))
atractores = atrac(logistica, 0.5, 3.50)
print(atractores)
print(errores_aux(logistica,3.50,  atractores))
analisis_estabilidad(3.50, 0.5)
analisis_bifurcaciones( 0.5,3.50)
"""Apartado 2"""
print("Apartado 2|n")
rss = np.arange(3.544,4, 0.0005)




"""Apartado 2"""
print("Apartado 2|n")

rss = np.arange(3.544,4, 0.0005)
valor_aux = 0 ##un valor que tiene tamano 8
print("Valores de r con conjunto atractor de 8 elementos")
lista = []
for i in range(len(rss)):
    r = rss[i]
    V0 = atrac(logistica, 0.5, r)
    if len(V0)==8 : 
        lista.append(r)
        valor_aux = r

print(lista)
plt.plot(orbita(0.5, logistica, 30,valor_aux))
atractores = atrac(logistica, 0.5, valor_aux)
print(atractores)
print(errores_aux(logistica,valor_aux,  atractores))
analisis_estabilidad(valor_aux, 0.5)
analisis_bifurcaciones( 0.5,valor_aux)
