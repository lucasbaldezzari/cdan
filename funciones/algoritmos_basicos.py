from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_moons, make_circles
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn_extra.cluster import KMedoids
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import pandas as pd

def makeAndPlotBlobs(n_samples=2000,random_state=7,figsize=(10, 5)):
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

def clusteringAndPlot(K=5, n_samples=2000, random_state=7, figsize=(10, 5), getInercia = False):
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

    if getInercia:
        print(f"Inercia para K={K}: {kmeans.inertia_:.2f}")
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

    fig, axes = plt.subplots(1, 1, figsize=figsize)
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

    fig, axes = plt.subplots(1, 1, figsize=figsize)
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


def getElbow(n_samples=2000,random_state=7,figsize=(10, 5), max_k = 30, printear=True):
    blob_centers = np.array([[ 0.2,  2.3], [-1.5 ,  2.3], [-2.8,  1.8],
                            [-2.8,  2.8], [-2.8,  1.3]])
    blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
    X, y = make_blobs(n_samples=n_samples, centers=blob_centers, cluster_std=blob_std, random_state=random_state)

    K = np.arange(1, max_k + 1)
    kmeans_inertia = []
    for k in K:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(X)
        kmeans_inertia.append(kmeans.inertia_)
        if printear:
            print(f"Agrupando para K={k}, Inercia={kmeans.inertia_:.2f}")
    plt.figure(figsize=figsize)
    plt.plot(K, kmeans_inertia, marker='o', color="#2387fa")
    plt.xlabel("Número de clústers (K)")
    plt.ylabel("Inercia (suma de distancias dentro de clústers)")
    plt.title("Inercia de K-means para diferentes valores de K")
    plt.xticks(K)
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig("/mnt/data/kmeans_inertia_plot.png")
    plt.show()


def makeClientesSinEscalar(n_samples=600, random_state=7):
    """
    Genera un set de clientes con dos variables en escalas MUY diferentes:

    - gasto_mensual: en pesos, valores del orden de 20.000 a 180.000
    - frecuencia_compras: cantidad de compras en el mes, valores de 1 a 30

    Sirve para mostrar por qué hay que escalar antes de usar K-means.
    """
    centros = np.array([[30_000, 4], [30_000, 22], [150_000, 6], [150_000, 24]])
    ##make_blobs sólo acepta un desvío por grupo, no uno por variable, así que
    ##armamos los grupos a mano para poder darle a cada variable su propia escala
    desvios = np.array([12_000, 2.5])

    rng = np.random.RandomState(random_state)
    por_grupo = n_samples // len(centros)

    X = np.vstack([centro + rng.normal(0, desvios, size=(por_grupo, 2))
                   for centro in centros])
    y = np.repeat(np.arange(len(centros)), por_grupo)

    ##ni el gasto ni la cantidad de compras pueden ser negativos
    X = np.clip(X, [1_000, 1], None)

    return X, y


def compararEscalado(K=4, n_samples=600, random_state=7, figsize=(13, 5)):
    """
    Compara el resultado de K-means sobre datos SIN escalar y sobre los mismos
    datos estandarizados con StandardScaler.

    En el gráfico de la izquierda el gasto (miles de pesos) domina por completo
    la distancia euclidiana y la frecuencia de compras queda ignorada.
    """
    X, _ = makeClientesSinEscalar(n_samples=n_samples, random_state=random_state)

    ##agrupamos sobre los datos crudos
    kmeans_crudo = KMeans(n_clusters=K, n_init=10, random_state=42)
    labels_crudo = kmeans_crudo.fit_predict(X)

    ##agrupamos sobre los datos estandarizados
    escalador = StandardScaler()
    X_escalado = escalador.fit_transform(X)
    kmeans_escalado = KMeans(n_clusters=K, n_init=10, random_state=42)
    labels_escalado = kmeans_escalado.fit_predict(X_escalado)

    ##llevamos los centroides del espacio escalado de vuelta a las unidades originales
    centros_escalado = escalador.inverse_transform(kmeans_escalado.cluster_centers_)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].scatter(X[:, 0], X[:, 1], c=labels_crudo, cmap="Set2", s=18, alpha=0.8)
    axes[0].scatter(kmeans_crudo.cluster_centers_[:, 0], kmeans_crudo.cluster_centers_[:, 1],
                    c="black", marker="X", s=140, label="Centroide")
    axes[0].set_title(f"SIN escalar (K={K})\nla distancia la decide el gasto")

    axes[1].scatter(X[:, 0], X[:, 1], c=labels_escalado, cmap="Set2", s=18, alpha=0.8)
    axes[1].scatter(centros_escalado[:, 0], centros_escalado[:, 1],
                    c="black", marker="X", s=140, label="Centroide")
    axes[1].set_title(f"CON StandardScaler (K={K})\nambas variables pesan igual")

    for ax in axes:
        ax.set_xlabel("Gasto mensual (pesos)")
        ax.set_ylabel("Frecuencia de compras (compras por mes)")
        ax.legend()
        ax.set_axisbelow(True)
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    print("Rango del gasto mensual     : "
          f"{X[:, 0].min():>10,.0f} a {X[:, 0].max():>10,.0f}  "
          f"(amplitud {X[:, 0].max() - X[:, 0].min():,.0f})")
    print("Rango de la frec. de compras: "
          f"{X[:, 1].min():>10,.1f} a {X[:, 1].max():>10,.1f}  "
          f"(amplitud {X[:, 1].max() - X[:, 1].min():,.1f})")


def plotFormasNoEsfericas(K=2, n_samples=1000, random_state=7, figsize=(12, 9)):
    """
    Muestra dos conjuntos de datos con grupos que NO son esféricos (lunas y
    círculos concéntricos) y qué hace K-means sobre ellos.

    La fila de arriba son los grupos reales, la de abajo lo que encuentra K-means.
    """
    X_lunas, y_lunas = make_moons(n_samples=n_samples, noise=0.06,
                                  random_state=random_state)
    X_circ, y_circ = make_circles(n_samples=n_samples, noise=0.04, factor=0.45,
                                  random_state=random_state)

    datasets = [("Distribución de lunas", X_lunas, y_lunas),
                ("Distribución circular", X_circ, y_circ)]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    for col, (nombre, X, y) in enumerate(datasets):
        kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
        y_pred = kmeans.fit_predict(X)

        axes[0, col].scatter(X[:, 0], X[:, 1], c=y, cmap="Set2", s=12, alpha=0.8)
        axes[0, col].set_title(f"{nombre}\nGrupos reales")

        axes[1, col].scatter(X[:, 0], X[:, 1], c=y_pred, cmap="Set2", s=12, alpha=0.8)
        axes[1, col].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                             c="black", marker="X", s=140, label="Centroide")
        axes[1, col].set_title(f"{nombre}\nK-means con K={K}")
        axes[1, col].legend()

    for ax in axes.ravel():
        ax.set_xlabel("Variable 1")
        ax.set_ylabel("Variable 2")
        ax.set_axisbelow(True)
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def compararMetricas(K=2, n_samples=400, random_state=7, figsize=(13, 5)):
    """
    Compara K-medoids usando distancia euclidiana y distancia Manhattan sobre
    un conjunto de datos con forma de cruz.

    Se usa una cruz porque es un caso donde la métrica cambia el resultado: la
    "circunferencia" de la distancia Manhattan es un rombo y no un círculo.
    """
    rng = np.random.RandomState(random_state)
    n = n_samples // 2
    ##brazo horizontal y brazo vertical de la cruz
    brazo_h = np.c_[rng.uniform(-6, 6, n), rng.normal(0, 0.6, n)]
    brazo_v = np.c_[rng.normal(0, 0.6, n), rng.uniform(-6, 6, n)]
    X = np.vstack([brazo_h, brazo_v])

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, metrica, nombre in zip(axes, ["euclidean", "manhattan"],
                                   ["Euclidiana", "Manhattan"]):
        kmedoids = KMedoids(n_clusters=K, metric=metrica, random_state=42)
        labels = kmedoids.fit_predict(X)
        centros = kmedoids.cluster_centers_

        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="Set2", s=18, alpha=0.8)
        ax.scatter(centros[:, 0], centros[:, 1], c="black", marker="X", s=140,
                   label="Medoide")
        ax.set_title(f"K-medoids con distancia {nombre}")
        ax.set_xlabel("Variable 1")
        ax.set_ylabel("Variable 2")
        ax.legend()
        ax.set_axisbelow(True)
        ax.grid(True)

        print(f"Distancia {nombre:<11} -> medoides en {np.round(centros, 2).tolist()}")

    plt.tight_layout()
    plt.show()

    print("\nOJO: el costo de agrupamiento de cada métrica NO se puede comparar "
          "con el de\nla otra, porque están medidos en unidades distintas. "
          "Elegir la métrica es una\ndecisión sobre QUÉ significa 'parecido' "
          "en nuestros datos, no algo que se optimice.")


def makeClientesCategoricos():
    """
    Devuelve un mini conjunto de 10 clientes con 5 respuestas del tipo sí/no.
    Son datos categóricos: no tiene sentido "promediarlos".
    """
    preguntas = ["compra_online", "usa_cupones", "es_suscriptor",
                 "compra_en_ofertas", "devolvio_algun_producto"]

    datos = np.array([
        [1, 1, 0, 1, 0],   ##Cliente 1
        [1, 1, 0, 1, 1],   ##Cliente 2
        [1, 1, 0, 1, 0],   ##Cliente 3
        [1, 0, 1, 0, 0],   ##Cliente 4
        [0, 0, 1, 0, 0],   ##Cliente 5
        [0, 0, 1, 0, 1],   ##Cliente 6
        [1, 1, 1, 1, 0],   ##Cliente 7
        [0, 1, 0, 1, 0],   ##Cliente 8
        [0, 0, 1, 0, 0],   ##Cliente 9
        [1, 0, 1, 0, 1],   ##Cliente 10
    ])

    return pd.DataFrame(datos, columns=preguntas,
                        index=[f"Cliente {i}" for i in range(1, len(datos) + 1)])


def distanciaHamming(K=2, figsize=(8, 6), mostrarDatos=True):
    """
    Agrupa clientes descriptos por respuestas sí/no usando distancia de Hamming.

    K-means no serviría acá: el promedio de "sí" y "no" no significa nada. En
    cambio K-medoids acepta una matriz de distancias ya calculada.
    """
    df = makeClientesCategoricos()

    if mostrarDatos:
        print("Respuestas de cada cliente (1 = sí, 0 = no):\n")
        print(df.to_string())
        print()

    ##pdist con 'hamming' devuelve la PROPORCIÓN de respuestas distintas,
    ##por eso multiplicamos por la cantidad de preguntas para tener la CANTIDAD
    n_preguntas = df.shape[1]
    distancias = squareform(pdist(df.values, metric="hamming")) * n_preguntas

    plt.figure(figsize=figsize)
    sns.heatmap(distancias, annot=True, fmt=".0f", cmap="viridis_r",
                xticklabels=df.index, yticklabels=df.index,
                cbar_kws={"label": "Respuestas en las que difieren"})
    plt.title("Distancia de Hamming entre clientes")
    plt.tight_layout()
    plt.show()

    ##K-medoids puede trabajar directamente con la matriz de distancias
    kmedoids = KMedoids(n_clusters=K, metric="precomputed", random_state=42)
    labels = kmedoids.fit_predict(distancias)

    df_resultado = df.copy()
    df_resultado["grupo"] = labels

    print(f"\nAgrupamiento con K-medoids (K={K}) sobre la distancia de Hamming:\n")
    print(df_resultado.sort_values("grupo").to_string())

    print("\nLos medoides (clientes que representan a cada grupo) son:\n")
    print(df.iloc[kmedoids.medoid_indices_].to_string())

    return df_resultado


def medoidesComoCasosReales(K=3, figsize=(11, 5)):
    """
    Aplica K-medoids sobre el set de productos/clientes con nombres de negocio y
    muestra las FILAS REALES elegidas como medoides.

    Es la diferencia práctica con K-means: el centroide de K-means es un promedio
    que puede no corresponder a ningún caso existente; el medoide siempre existe.
    """
    df = transform_and_get_iris()
    variables = ["media_visitas_diarias", "precio_unitario",
                 "unidades_vendidas_mensuales", "valoracion_media"]

    ##escalamos porque las 4 variables están en unidades distintas
    X = StandardScaler().fit_transform(df[variables].values)

    kmedoids = KMedoids(n_clusters=K, random_state=42)
    labels = kmedoids.fit_predict(X)

    kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
    kmeans.fit(X)

    df_resultado = df.copy()
    df_resultado["grupo"] = labels

    print(f"Cantidad de casos en cada grupo (K={K}):\n")
    print(df_resultado["grupo"].value_counts().sort_index().to_string())

    print("\n" + "=" * 70)
    print("MEDOIDES: son filas REALES del conjunto de datos")
    print("=" * 70 + "\n")
    medoides = df.iloc[kmedoids.medoid_indices_][variables].copy()
    medoides.index = [f"Grupo {i} (fila {idx})"
                      for i, idx in enumerate(kmedoids.medoid_indices_)]
    print(medoides.round(2).to_string())

    print("\n" + "=" * 70)
    print("CENTROIDES de K-means: son promedios, no tienen por qué existir")
    print("=" * 70 + "\n")
    ##volvemos los centroides a las unidades originales para poder compararlos
    centros = StandardScaler().fit(df[variables].values).inverse_transform(
        kmeans.cluster_centers_)
    centroides = pd.DataFrame(centros, columns=variables,
                              index=[f"Grupo {i}" for i in range(K)])
    print(centroides.round(2).to_string())

    ##graficamos dos de las variables para ver dónde caen medoides y centroides
    plt.figure(figsize=figsize)
    plt.scatter(df["unidades_vendidas_mensuales"], df["valoracion_media"],
                c=labels, cmap="Set2", s=40, alpha=0.7, edgecolors="black",
                linewidths=0.4)
    plt.scatter(medoides["unidades_vendidas_mensuales"], medoides["valoracion_media"],
                c="black", marker="X", s=200, label="Medoide (caso real)")
    plt.scatter(centroides["unidades_vendidas_mensuales"], centroides["valoracion_media"],
                c="red", marker="P", s=200, label="Centroide K-means (promedio)")
    plt.xlabel("Unidades vendidas mensuales")
    plt.ylabel("Valoración media")
    plt.title(f"K-medoids con K={K}: medoides reales vs centroides promediados")
    plt.legend()
    plt.gca().set_axisbelow(True)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return df_resultado