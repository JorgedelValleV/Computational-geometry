# -*- coding: utf-8 -*-
"""
Referencias:
    
    Fuente primaria del reanálisis
    https://psl.noaa.gov/data/gridded/data.ncep.reanalysis2.pressure.html
    
    Altura geopotencial en niveles de presión
    https://psl.noaa.gov/cgi-bin/db_search/DBListFiles.pl?did=59&tid=97457&vid=1498
    
    Temperatura en niveles de presión:
    https://psl.noaa.gov/cgi-bin/db_search/DBListFiles.pl?did=59&tid=97457&vid=4237

    Temperatura en niveles de superficie:
    https://psl.noaa.gov/cgi-bin/db_search/DBListFiles.pl?did=59&tid=97457&vid=1497
    
"""
import datetime as dt  # Python standard library datetime  module
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
#from scipy.io import netcdf as nc
from sklearn.decomposition import PCA
workpath = ""
import os
#workpath = "C:/NCEP"
os.getcwd()
#os.chdir(workpath)
#files = os.listdir(workpath)
import math
f = Dataset(workpath + "air.2021.nc", "r", format="NETCDF4")
#f = Dataset(workpath + "/air.2m.gauss.2021.nc", "r", format="NETCDF4")
#f = nc.netcdf_file(workpath + "/air.2221.nc", 'r')
#print(f.history)
#print(f.dimensions)
#print(f.variables)
time = f.variables['time'][:].copy()
time_bnds = f.variables['time_bnds'][:].copy()
time_units = f.variables['time'].units
level = f.variables['level'][:].copy()
lats = f.variables['lat'][:].copy()
lons = f.variables['lon'][:].copy()
air21 = f.variables['air'][:].copy()
air_units = f.variables['air'].units
#air_scale = f.variables['air'].scale_factor
#air_offset = f.variables['air'].add_offset
#print(air21.shape)
f.close()

f = Dataset(workpath + "air.2022.nc", "r", format="NETCDF4")
time = f.variables['time'][:].copy()
time_bnds = f.variables['time_bnds'][:].copy()
time_units = f.variables['time'].units
air22 = f.variables['air'][:].copy()
f.close()

f = Dataset(workpath + "hgt.2021.nc", "r", format="NETCDF4")
time21 = f.variables['time'][:].copy()
time_bnds = f.variables['time_bnds'][:].copy()
time_units = f.variables['time'].units
hgt21 = f.variables['hgt'][:].copy()
hgt_units = f.variables['hgt'].units
#hgt_scale = f.variables['hgt'].scale_factor
#hgt_offset = f.variables['hgt'].add_offset
#print(hgt21.shape)
f.close()

#f = nc.netcdf_file(workpath + "/" + files[0], 'r')
f = Dataset(workpath + "hgt.2022.nc", "r", format="NETCDF4")
time22 = f.variables['time'][:].copy()
time_bnds = f.variables['time_bnds'][:].copy()
time_units = f.variables['time'].units
hgt22 = f.variables['hgt'][:].copy()
f.close()

"""
Ejemplo de evolución temporal de un elemento de aire

plt.plot(time21, air21[:, 0, 0, 0], c='r')
plt.show()
"""

"""
Distribución espacial de la temperatura en el nivel de 1000hPa, para el primer día

plt.contour(lons, lats, air21[0,0,:,:])
plt.show()
"""



##############################################################################
# dimension 365 * (73*144)
hgt21b = hgt21[:,level==500.,:,:].reshape(len(time21),len(lats)*len(lons))
air21b = air21[:,level==500.,:,:].reshape(len(time21),len(lats)*len(lons))
n_components=4
X = hgt21b
Y = hgt21b.transpose()
pca = PCA(n_components=n_components)
Element_pca0 = pca.fit_transform(Y)
Element_pca0 = Element_pca0.transpose(1,0).reshape(n_components,len(lats),len(lons))
print(pca.components_[0],pca.explained_variance_ratio_)
'''
#Interpretar el siguiente resultado
pca.fit(Y)
print(pca.explained_variance_ratio_)
out = pca.singular_values_

#Interpretar el siguiente resultado
pca.fit(X)
print(pca.explained_variance_ratio_)
out = pca.singular_values_

#X = air21b
X = hgt21b
State_pca = pca.fit_transform(X)
Element_pca1 = pca.fit(X).components_
Element_pca1 = Element_pca1.reshape(n_components,len(lats),len(lons))
print(pca.explained_variance_ratio_)
'''

#Ejercicio de la práctica - Opción 1
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 5):
    ax = fig.add_subplot(2, 2, i)
    ax.text(0.5, 90, 'PCA-'+str(i),
           fontsize=18, ha='center')
    plt.contour(lons, lats, Element_pca0[i-1,:,:])
plt.show()
'''
#Ejercicio de la práctica - Opción 2
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 5):
    ax = fig.add_subplot(2, 2, i)
    ax.text(0.5, 90, 'PCA-'+str(i),
           fontsize=18, ha='center')
    plt.contour(lons, lats, Element_pca1[i-1,:,:])
plt.show()
'''
##############################################################################
'''
#time_idx = 237  
# Python and the reanalaysis are slightly off in time so this fixes that problem
# offset = dt.timedelta(hours=0)
# List of all times in the file as datetime objects
dt_time21 = [dt.date(1800, 1, 1) + dt.timedelta(hours=t) for t in time21]
np.min(dt_time21)
np.max(dt_time21)
'''
#hgt21b = hgt21[:,:,np.logical_and(30 < lats, lats < 50),np.logical_or(340 < lons, lons < 20)]
hgt21b = hgt21[:,:,np.logical_and(30 < lats, lats < 50),:]
hgt21b = hgt21b[:,:,:,np.logical_or(340 < lons, lons < 20)]
dt_time22 = [dt.date(1800, 1, 1) + dt.timedelta(hours=t) for t in time22]
#np.min(dt_time22)
#np.max(dt_time22)
dia0_fecha = dt.date(2022, 1, 11)
dia0_ind = dt_time22.index(dia0_fecha)
#dia0 = hgt22[dia0_ind,:,np.logical_and(30 < lats, lats < 50),np.logical_or(340 < lons, lons < 20)]
dia0 = hgt22[dia0_ind,:,:,:]
dia0 = dia0[:,np.logical_and(30 < lats, lats < 50),:]
dia0 = dia0[:,:,np.logical_or(340 < lons, lons < 20)]
#dia0 = (latitud,longitud)
def dist_euclidea(dia):
    s = 0
    for i in range(len(dia0[0])):
        for j in range(len(dia0[0][0])):
            s += 0.5*((dia0[level == 500.][0][i][j] - dia[level == 500.][0][i][j])**2)
            s += 0.5*((dia0[level == 1000.][0][i][j] - dia[level == 1000.][0][i][j])**2)
    return math.sqrt(s)
dias = 4
distancias = [(i, dist_euclidea(hgt21b[i])) for i in range(len(hgt21b))]
distancias_ord = sorted(distancias, key=lambda e : e[1])
'''
def clave_ord(tupla):
  return tupla[1]
distances = sorted(distances, key=clave_ord)
'''
distancias_best = distancias_ord[0:dias]
print('(dia,distancias) mas analogas',distancias_best)
temp_sum = air21[distancias_best[0][0]][level == 1000]
#dim (1,73,144)
for i in range(1,dias):
    temp_sum = np.add(temp_sum, air21[distancias_best[i][0]][level == 1000])
temp_media = temp_sum*(1/dias)
dia0 = (air22[dia0_ind][level == 1000])*(-1)# -1 para restar a temp_media
error_abs_med = (np.sum(abs(np.add(temp_media,dia0))))/(73*144)
print('error absoluto medio',error_abs_med)
