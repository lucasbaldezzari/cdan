from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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