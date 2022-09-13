# -*- coding: utf-8 -*-
"""
Plantilla 1 de la práctica 3

Referencia: 
    https://scikit-learn.org/stable/modules/clustering.html
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
"""

import numpy as np

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.datasets import make_blobs
from scipy.spatial import ConvexHull, convex_hull_plot_2d
#from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
'''
# #############################################################################
# Aquí tenemos definido el sistema X de 1000 elementos de dos estados
# construido a partir de una muestra aleatoria entorno a unos centros:
centers = [[-0.5, 0.5], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=1000, centers=centers, cluster_std=0.4,
                            random_state=0)
#Si quisieramos estandarizar los valores del sistema, haríamos:
#from sklearn.preprocessing import StandardScaler
#X = StandardScaler().fit_transform(X)  

#Envolvente convexa, envoltura convexa o cápsula convexa 
hull = ConvexHull(X)
convex_hull_plot_2d(hull)

plt.plot(X[:,0],X[:,1],'ro', markersize=1)
plt.show()

# #############################################################################
# Los clasificamos mediante el algoritmo KMeans
n_clusters=2

#Usamos la inicialización aleatoria "random_state=0"
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
labels = kmeans.labels_
silhouette = metrics.silhouette_score(X, labels)

# Etiqueta de cada elemento (punto)
print(kmeans.labels_)
# Índice de los centros de vencindades o regiones de Voronoi para cada elemento (punto) 
print(kmeans.cluster_centers_)
#Coeficiente de Silhouette
print("Silhouette Coefficient: %0.3f" % silhouette)


# #############################################################################
# Predicción de elementos para pertenecer a una clase:
problem = np.array([[-1.5, -1], [1.5, -1]])
clases_pred = kmeans.predict(problem)
print(clases_pred)

# #############################################################################
# Representamos el resultado con un plot

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

print(colors,unique_labels)

plt.figure(figsize=(8,4))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=5)

plt.plot(problem[:,0],problem[:,1],'o', markersize=12, markerfacecolor="red")

plt.title('Fixed number of KMeans clusters: %d' % n_clusters)
plt.show()
'''

# #############################################################################
# Aquí tenemos definido el sistema X de 1000 elementos de dos estados
# construido a partir de una muestra aleatoria entorno a unos centros:
centers = [[-0.5, 0.5], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=1000, centers=centers, cluster_std=0.4,
                            random_state=0)

k_=[]
for k in range(2,16):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    silhouette = metrics.silhouette_score(X, kmeans.labels_)
    #Coeficiente de Silhouette
    k_.append(silhouette)
    print("Silhouette Coefficient: %0.3f" % silhouette)
    #plt.plot(k, silhouette, 'o', markersize=5)
silhouette_best=max(k_)
k_best=k_.index(silhouette_best)+2
print (k_best)
plt.plot(range(2,16), k_)
plt.title('Silhouette graph searching best k')
plt.show()

#Usamos la inicialización aleatoria "random_state=0"
kmeans = KMeans(n_clusters=k_best, random_state=0).fit(X)
labels = kmeans.labels_

# Índice de los centros de vencindades o regiones de Voronoi para cada elemento (punto) 
print(kmeans.cluster_centers_)

# #############################################################################
# Representamos el resultado con un plot
'''
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

plt.figure(figsize=(8,4))

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=5)

centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0],centroids[:, 1],marker="x",s=169,linewidths=3,color="g",zorder=10,)

plt.title('Fixed number of KMeans clusters: %d' % k_best)
plt.show()'''
# Step size of the mesh. Decrease to increase the quality of the VQ.
h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
AUX=np.c_[xx.ravel(), yy.ravel()]
Z = kmeans.predict(AUX)

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8,4))
plt.clf()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Paired,
    aspect="auto",
    origin="lower",
)
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=5)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=169,
    linewidths=3,
    color="w",
    zorder=10,
)
plt.title('Fixed number of KMeans clusters: %d' % k_best)
plt.show()


# #############################################################################
# Predicción de elementos para pertenecer a una clase:
problem = np.array([[0,0], [0, -1]])
clases_pred = kmeans.predict(problem)
# mostramos la prediccion de los centroides para clarificar la  pertenencia
print(clases_pred,kmeans.predict(centroids))