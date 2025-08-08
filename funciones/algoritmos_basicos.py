from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn_extra.cluster import KMedoids
from sklearn.datasets import load_iris
import pandas as pd

def makeAndPlotBlobs(n_samples=2000,random_state=7,figsize=(10, 5)):
    # extra code – the exact arguments of make_blobs() are not important
    blob_centers = np.array([[ 0.2,  2.3], [-1.5 ,  2.3], [-2.8,  1.8],
                            [-2.8,  2.8], [-2.8,  1.3]])
    blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
    X, y = make_blobs(n_samples=n_samples, centers=blob_centers, cluster_std=blob_std,
                    random_state=random_state)

    plt.figure(figsize=figsize)
    plt.scatter(X[:, 0], X[:, 1], c="#1a8d4a", s=2)
    plt.xlabel("Gasto mensual en compras")
    plt.ylabel("Frecuencia de compras")
    plt.gca().set_axisbelow(True)
    plt.grid()
    plt.title("Datos originales (escalados)")
    plt.show()

def plot_data(X,c=None,cmap="Set1"):
    plt.scatter(X[:, 0], X[:, 1],
                s=20,
                facecolors='white',
                edgecolors='black',
                alpha=0.8,
                linewidths=0.5)
    if c is not None:
        plt.scatter(X[:, 0], X[:, 1],
                    c=c, s=20, alpha=0.8,
                    edgecolors='black', linewidths=0.5,cmap=cmap)
    

def plot_centroids(centroids, weights=None, circle_color="#ffffff", cross_color='#aa1123'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=35, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=2, linewidths=12,
                color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(clusterer, X, resolution=1000,
                             show_centroids=True, show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=1,cmap="viridis")
    plt.contour(xx, yy, Z, linewidths=1, colors='k')
    plot_data(X)

    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("Gasto mensual en compras")
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("Frecuencia de compras", rotation=90)
    else:
        plt.tick_params(labelleft=False)

def clusteringAndPlot(K=5, n_samples=2000, random_state=7, figsize=(10, 5)):
    blob_centers = np.array([[ 0.2,  2.3], [-1.5 ,  2.3], [-2.8,  1.8],
                             [-2.8,  2.8], [-2.8,  1.3]])
    blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
    X, y = make_blobs(n_samples=n_samples, centers=blob_centers,
                      cluster_std=blob_std, random_state=random_state)

    kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
    y_pred = kmeans.fit_predict(X)

    plt.figure(figsize=figsize)
    plot_decision_boundaries(kmeans, X)
    plt.title("Clustering usando K-means")
    plt.show()


def showConvergencia(K=5,n_samples=2000, random_state=7, figsize=(12, 5), numpyseed=21):
    blob_centers = np.array([[0.2, 2.3], [-1.5, 2.3], [-2.8, 1.8],
                             [-2.8, 2.8], [-2.8, 1.3]])
    blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
    X, y = make_blobs(n_samples=n_samples, centers=blob_centers,
                      cluster_std=blob_std, random_state=random_state)

    np.random.seed(numpyseed)
    centros = np.random.permutation(X)[:5]

    kmeans_iter1 = KMeans(n_clusters=K, init=centros, n_init=1, max_iter=1, random_state=5)
    kmeans_iter4 = KMeans(n_clusters=K, init=centros, n_init=1, max_iter=500, random_state=5)

    kmeans_iter1.fit(X)
    kmeans_iter4.fit(X)

    # Figura con solo dos subplots en una fila
    plt.figure(figsize=figsize)

    # Subplot 421 → convertido en el primer subplot
    plt.subplot(1, 2, 1)
    plot_data(X)
    plot_centroids(kmeans_iter1.cluster_centers_, circle_color='r', cross_color='w')
    plt.ylabel("Frecuencia de compras", rotation=90)
    plt.tick_params(labelbottom=False)
    plt.title("Inicialización de centroides \n(de manera aleatoria)")

    # Subplot 428 → convertido en el segundo subplot
    plt.subplot(1, 2, 2)
    plot_decision_boundaries(kmeans_iter4, X, show_ylabels=False, show_xlabels=True)
    plt.title("Etiquetado final de los puntos\n(tras finalizar el algoritmo)")

    plt.tight_layout()
    plt.show()

def plotOriginalConverDiver(K=5,n_samples=2000, random_state=7, figsize=(12, 5), numpyseed=21,cmap="Set1"):

    blob_centers = np.array([[0.2, 2.3], [-1.5, 2.3], [-2.8, 1.8],
                             [-2.8, 2.8], [-2.8, 1.3]])
    blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
    X, y = make_blobs(n_samples=n_samples, centers=blob_centers,
                      cluster_std=blob_std, random_state=random_state)
    
    np.random.seed(21)
    centros = np.random.permutation(X)[:5]

    kmeans_iter1_conver = KMeans(n_clusters=K, init=centros, n_init=1, max_iter=1, random_state=5)
    kmeans_iter4_conver = KMeans(n_clusters=K, init=centros, n_init=1, max_iter=500, random_state=5)

    kmeans_iter1_conver.fit(X)
    kmeans_iter4_conver.fit(X)

    np.random.seed(58)
    centros = np.random.permutation(X)[:5]

    kmeans_iter1_diver = KMeans(n_clusters=K, init=centros, n_init=1, max_iter=1, random_state=5)
    kmeans_iter4_diver = KMeans(n_clusters=K, init=centros, n_init=1, max_iter=500, random_state=5)

    kmeans_iter1_diver.fit(X)
    kmeans_iter4_diver.fit(X)

    plt.figure(figsize=figsize)

    plt.subplot(1, 3, 1)
    plot_data(X,c=y,cmap=cmap)
    plot_centroids(blob_centers, circle_color='r', cross_color='w')
    ##agrego titulo y etiquetas
    plt.title("Grupos originales")
    plt.xlabel("Gasto mensual en compras")
    plt.ylabel("Frecuencia de compras")

    plt.subplot(1, 3, 2)
    plot_data(X,c=kmeans_iter4_conver.labels_,cmap=cmap)
    plot_centroids(kmeans_iter4_conver.cluster_centers_, circle_color='r', cross_color='w')
    plot_decision_boundaries(kmeans_iter4_conver, X, show_ylabels=False, show_xlabels=True)
    plt.title("Convergencia adecuada")
    plt.xlabel("Gasto mensual en compras")
    plt.ylabel("Frecuencia de compras", rotation=90)

    plt.subplot(1, 3, 3)
    plot_data(X,c=kmeans_iter4_diver.labels_,cmap=cmap)
    plot_centroids(kmeans_iter4_diver.cluster_centers_, circle_color='r', cross_color='w')
    plot_decision_boundaries(kmeans_iter4_diver, X, show_ylabels=False, show_xlabels=True)
    plt.title("Convergencia inadecuada")
    plt.xlabel("Gasto mensual en compras")
    plt.ylabel("Frecuencia de compras", rotation=90)

    plt.show()

def plotDataOutliers(figsize=(8, 4)):

    # Generar datos sintéticos con un outlier
    # np.random.seed(0)
    cluster1 = np.random.normal(loc=[2, 2], scale=0.5, size=(50, 2))
    cluster2 = np.random.normal(loc=[7, 2], scale=0.5, size=(50, 2))
    outliers = np.array([[25, 8],[32,8.5]])  # punto extremo
    X = np.vstack([cluster1, cluster2, outliers])

    # Aplicar K-means
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    kmeans_centroids = kmeans.cluster_centers_

    # Visualización comparativa
    fig, axes = plt.subplots(1, 1, figsize=figsize)

    # K-means plot
    axes.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='Set2_r', s=50, alpha=0.6, label='Datos normales')
    # axes.scatter(kmeans_centroids[:, 0], kmeans_centroids[:, 1], c='black', marker='X', s=100, label='Centroide')
    axes.scatter(outliers[:, 0], outliers[:, 1], c='red', s=50, label='Outliers', edgecolor='black')
    axes.set_title("Datos con Outliers")
    axes.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    #activo grid
    plt.close()

def kmeansOutliers(K=2,figsize=(8,4)):
    cluster1 = np.random.normal(loc=[2, 2], scale=0.5, size=(50, 2))
    cluster2 = np.random.normal(loc=[7, 2], scale=0.5, size=(50, 2))
    outliers = np.array([[25, 8],[32,8.5]])  # punto extremo
    X = np.vstack([cluster1, cluster2, outliers])

    # Aplicar K-means
    kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    kmeans_centroids = kmeans.cluster_centers_


    # Visualización comparativa
    fig, axes = plt.subplots(1, 1, figsize=figsize)

    # K-means plot
    axes.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='Set2_r', s=30)
    axes.scatter(kmeans_centroids[:, 0], kmeans_centroids[:, 1], c='black', marker='X', s=100, label='Centroide')
    axes.scatter(outliers[:, 0], outliers[:, 1], c='red', s=50, label='Outliers', edgecolor='black')
    axes.set_title("Kmeans agrupando datos con outliers")
    axes.legend()

    plt.tight_layout()
    plt.grid()
    plt.show()
    plt.close()

def kmedoidsOutliers(K=2,figsize=(8,4)):
    cluster1 = np.random.normal(loc=[2, 2], scale=0.5, size=(50, 2))
    cluster2 = np.random.normal(loc=[7, 2], scale=0.5, size=(50, 2))
    outliers = np.array([[25, 8],[32,8.5]])  # punto extremo
    X = np.vstack([cluster1, cluster2, outliers])

    # Aplicar K-medoids
    kmedoids = KMedoids(n_clusters=K, random_state=0)
    kmedoids_labels = kmedoids.fit_predict(X)
    kmedoids_centroids = kmedoids.cluster_centers_

    # Visualización comparativa
    fig, axes = plt.subplots(1, 1, figsize=figsize)

    # K-medoids plot
    axes.scatter(X[:, 0], X[:, 1], c=kmedoids_labels, cmap='Set2_r', s=30)
    axes.scatter(kmedoids_centroids[:, 0], kmedoids_centroids[:, 1], c='black', marker='X', s=100, label='Centroide')
    axes.scatter(outliers[:, 0], outliers[:, 1], c='red', s=50, label='Outliers', edgecolor='black')
    axes.set_title("Kmedoids agrupando datos con outliers")
    axes.legend()

    plt.tight_layout()
    plt.grid()
    plt.show()
    plt.close()

def transform_and_get_iris():
    """
    Columna original	Nombre simulado	            Descripción de negocio
    sepal length (cm)	media_visitas_diarias	    Cantidad promedio de visitas diarias al producto (popularidad)
    sepal width (cm)	precio_unitario	            Precio en USD del producto
    petal length (cm)	unidades_vendidas_mensual	Unidades vendidas por mes
    petal width (cm)	valoracion_media	        Valoración media de usuarios (1 a 5)

    labels = Tipo de cliente que más compra ese producto

    Código orig	       Descripción del cliente	        ¿Qué busca?
    0	               Cliente explorador	            Navega mucho, compra poco. Interesado en novedades
                                                        y precios bajos.	Buen precio, variedad, promociones
    1	               Cliente comprometido	            Compra de forma recurrente productos específicos.
                                                        Fiel a ciertos productos.	Alta valoración, confianza, calidad
    2	               Cliente impulsivo	            Compra mucho y rápido. Le importa menos el precio si
                                                        el producto es atractivo.	Ventas altas, buena valoración
        
    """
    iris = load_iris()
    ##cambio las columnas
    df = pd.DataFrame(iris.data, columns=
                      [
                          "media_visitas_diarias",
                          "precio_unitario",
                          "unidades_vendidas_mensuales",
                          "valoracion_media"
                      ])
    df["segmento"] = iris.target

    ##pasamos la columna valoración media en un rango de 1 a 5
    df["valoracion_media"] = (df["valoracion_media"] - df["valoracion_media"].min()) / \
                              (df["valoracion_media"].max() - df["valoracion_media"].min())
    df["valoracion_media"] = df["valoracion_media"] * 4 + 1
    return df

def plot3Ddata(elev=10, azim=160, figsize=(12, 6)):

    # Cargar el dataset Iris
    iris = transform_and_get_iris()
    X = iris[[
                          "media_visitas_diarias",
                          "precio_unitario",
                          "unidades_vendidas_mensuales",
                          "valoracion_media"
                      ]].values
    y = iris["segmento"].values
    target_names = ["media_visitas_diarias", "precio_unitario", "unidades_vendidas_mensuales", "valoracion_media"]

    # Selección de características
    x_axis = 0  # media_visitas_diarias
    y_axis = 1  # precio_unitario
    z_axis = 2  # unidades_vendidas_mensuales
    colors = ['r', 'g', 'b']

    # Crear figura y ejes 3D
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    for i, color, label in zip(np.unique(y), colors, target_names):
        ax.scatter(
            X[y == i, x_axis],
            X[y == i, y_axis],
            X[y == i, z_axis],
            s=50,
            alpha=0.5,
            color=color,
            label=label
        )

    # Etiquetas
    ax.set_xlabel(target_names[x_axis])
    ax.set_ylabel(target_names[y_axis])
    ax.set_zlabel(target_names[z_axis])
    ax.set_title('Datos en 3D')
    ax.legend()

    ax.view_init(elev=elev, azim=azim)  # elev: altura de la cámara, azim: rotación horizontal

    plt.tight_layout()
    plt.show()