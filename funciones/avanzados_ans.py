import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

def showDataKmeans(K=2, n_samples=2000, random_seed=0, showCentroides = False, figsize=(12, 6), showGrupos = False):
    # Generar datos sintéticos de círculos con un círculo en el centro y otro alrededor
    X_circles, y_circles = make_circles(n_samples=n_samples, noise=0.05, factor=0.5, random_state=random_seed)
    X_moons, y_moons = make_moons(n_samples=n_samples, noise=0.05, random_state=random_seed)

    # Aplicar KMeans
    kmeans = KMeans(n_clusters=K, random_state=random_seed)
    y_kmeans_circles = kmeans.fit_predict(X_circles)
    centers_circles = kmeans.cluster_centers_

    y_kmeans_moons = kmeans.fit_predict(X_moons)
    centers_moons = kmeans.cluster_centers_


    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Visualizar los resultados
    if showGrupos:
        axes[0].scatter(X_circles[:, 0], X_circles[:, 1], c=y_kmeans_circles, s=50, cmap='viridis', alpha=0.7)
    else:
        axes[0].scatter(X_circles[:, 0], X_circles[:, 1], c='#a12345', s=50, alpha=0.7)
    if showCentroides:
        axes[0].scatter(centers_circles[:, 0], centers_circles[:, 1], c='red', s=200, marker='X', alpha=0.7)
    axes[0].set_title(f"Círculos agrupadas con Kmeans (K={K})")
    axes[0].set_xlabel("Frecuencia de visitas")
    axes[0].set_ylabel("Tiempo de permanencia")

    if showGrupos:
        axes[1].scatter(X_moons[:, 0], X_moons[:, 1], c=y_kmeans_moons, s=50, cmap='viridis', alpha=0.7)
    else:
        axes[1].scatter(X_moons[:, 0], X_moons[:, 1], c='#a12345', s=50, alpha=0.7)
    if showCentroides:
        axes[1].scatter(centers_moons[:, 0], centers_moons[:, 1], c='red', s=200, marker='X', alpha=0.7)
    axes[1].set_title(f"Lunas agrupadas con Kmeans (K={K})")
    axes[1].set_xlabel("Frecuencia de visitas")
    axes[1].set_ylabel("Tiempo de permanencia")

    plt.show()

def plotComparativa():
    X_moons, _ = make_moons(n_samples=500, noise=0.05, random_state=42)
    X_circles, _ = make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=42)
    X_blobs, _ = make_blobs(n_samples=500, centers=2, cluster_std=0.6, random_state=42)

    datasets = [
        ("Separables linealmente (blobs)", X_blobs),
        ("No separables linealmente (moons)", X_moons),
        ("No separables linealmente (circles)", X_circles),
    ]

    # Crear figura
    fig, axes = plt.subplots(len(datasets), 3, figsize=(12, 10))
    for row, (title, X) in enumerate(datasets):
        # Datos originales
        axes[row, 0].scatter(X[:, 0], X[:, 1], c='gray', edgecolor='k', alpha=0.7)
        axes[row, 0].set_title(f"{title}\nDatos originales")

        # K-means
        km = KMeans(n_clusters=2, random_state=42)
        y_km = km.fit_predict(X)
        axes[row, 1].scatter(X[:, 0], X[:, 1], c=y_km, cmap='viridis', edgecolor='k', alpha=0.7)
        axes[row, 1].set_title("K-means")

        # DBSCAN con outliers en rojo
        db = DBSCAN(eps=0.2, min_samples=5)
        y_db = db.fit_predict(X)
        # Colorear outliers en rojo
        mask_outliers = y_db == -1
        mask_clusters = ~mask_outliers
        axes[row, 2].scatter(X[mask_clusters, 0], X[mask_clusters, 1], c=y_db[mask_clusters], cmap='viridis', edgecolor='k', alpha=0.7)
        axes[row, 2].scatter(X[mask_outliers, 0], X[mask_outliers, 1], c='red', edgecolor='k', label='Outliers', alpha=0.7)
        axes[row, 2].set_title("DBSCAN (rojo = outliers)")
        axes[row, 2].legend(loc='upper right', fontsize='small')

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

def circlesWithDBSCAN(epsilon=0.2, min_points=5, showClustering = False, n_samples=2000, random_seed=0, figsize=(10, 8),
                      showOutliers=False, title = "Datos circulares sin agrupar"):
    X, _ = make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=random_seed)
    db = DBSCAN(eps=epsilon, min_samples=min_points)
    y_db = db.fit_predict(X)

    plt.figure(figsize=figsize)
    if showClustering:
        plt.scatter(X[:, 0], X[:, 1], c=y_db, cmap='viridis', edgecolor='k', alpha=0.7)
    else:
        plt.scatter(X[:, 0], X[:, 1], color="#a2112c", edgecolor='k', alpha=0.7)
    if showOutliers:
        # Mostrar outliers
        mask_outliers = y_db == -1
        plt.scatter(X[mask_outliers, 0], X[mask_outliers, 1], c='red', edgecolor='k', label='Outliers', alpha=0.7)
        plt.legend(loc='upper right', fontsize='small')
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

def circlesNoisyWithDBSCAN(epsilon=0.1, min_points=10, showClustering = False, n_samples=2000, random_seed=0, figsize=(10, 8),
                           showOutliers=False, title = "Datos circulares ruidosos sin agrupar"):

    X, _ = make_circles(n_samples=n_samples, factor=0.5, noise=0.1, random_state=random_seed)
    db = DBSCAN(eps=epsilon, min_samples=min_points)
    y_db = db.fit_predict(X)

    plt.figure(figsize=figsize)
    if showClustering:
        plt.scatter(X[:, 0], X[:, 1], c=y_db, cmap='Set3', edgecolor='k', alpha=0.7)
    else:
        plt.scatter(X[:, 0], X[:, 1], color="#606ac3", edgecolor='k', alpha=0.7)
    if showOutliers:
        # Mostrar outliers
        mask_outliers = y_db == -1
        plt.scatter(X[mask_outliers, 0], X[mask_outliers, 1], c='red', edgecolor='k', label='Outliers', alpha=0.7)
        plt.legend(loc='upper right', fontsize='small')
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

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

def realdataWithDBSCAN(epsilon=0.1, min_points=10, columnasInteres = ["media_visitas_diarias","unidades_vendidas_mensuales"],
                       showClustering = False, n_samples=2000, random_seed=0, figsize=(10, 8),
                       showOutliers=False, title = "Datos circulares ruidosos sin agrupar"):

    data = transform_and_get_iris()
    X = data[columnasInteres].values
    db = DBSCAN(eps=epsilon, min_samples=min_points)
    y_db = db.fit_predict(X)

    # Graficar los resultados
    plt.figure(figsize=figsize)
    if showClustering:
        plt.scatter(X[:, 0], X[:, 1], c=y_db, cmap='viridis', edgecolor='k', alpha=0.7)
    else:
        plt.scatter(X[:, 0], X[:, 1], color="#03bcb0", edgecolor='k', alpha=0.7)
    if showOutliers:
        # Mostrar outliers
        mask_outliers = y_db == -1
        plt.scatter(X[mask_outliers, 0], X[mask_outliers, 1], c='red', edgecolor='k', label='Outliers', alpha=0.7)
        plt.legend(loc='upper right', fontsize='small')
    plt.title(title)
    plt.xlabel(columnasInteres[0])
    plt.ylabel(columnasInteres[1])
    plt.show()

def dendo1stExample(n_samples=50, centers=4, cluster_std=1.4, random_state=10,
                show_blobs=True, show_dendo=False, figsize=(16, 6), show_true_groups=False, show_dendo_colors=False,
                umbral_corte=None, above_threshold_color="black"):

    X2, y2_true = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=random_state)

    Z = linkage(X2, method='ward')

    y_cut = None
    k_groups = None
    if umbral_corte is not None:
        # Clusters por distancia (horizontal cut)
        y_cut = fcluster(Z, t=umbral_corte, criterion='distance')
        k_groups = len(np.unique(y_cut))

    if show_blobs and not show_dendo:
        plt.figure(figsize=figsize)
        if show_true_groups:
            # Colorea por etiquetas verdaderas si se pidió y no hay corte
            plt.scatter(X2[:, 0], X2[:, 1], c=y2_true, cmap='viridis',
                        s=100, edgecolor='k')
            plt.legend([f"Grupos verdaderos (k={len(np.unique(y2_true))})"], loc='lower left')
        elif y_cut is not None:
            # Colorea por clusters obtenidos del corte
            plt.scatter(X2[:, 0], X2[:, 1], c=y_cut, cmap='tab10',
                        s=100, edgecolor='k')
            plt.title(f"Datos de testeo - Se pintan grupos para umbral {umbral_corte} - N° grupos {k_groups}")
        else:
            # Un solo color
            plt.scatter(X2[:, 0], X2[:, 1], color="#0303bc", edgecolor='k',
                        alpha=0.7, s=100, label='Datos')
            plt.legend(loc='lower left')
            plt.title("Datos de testeo sin agrupar")

        plt.xlabel("Característica 1")
        plt.ylabel("Característica 2")
        plt.grid()

    if show_dendo and not show_blobs:
        plt.figure(figsize=figsize)
        if umbral_corte is None:
            # Sin colores (todo en negro)
            dendrogram(Z, color_threshold=0, above_threshold_color="black")
            plt.title("Dendrograma sin línea de corte")
        else:
            # Colores por debajo del umbral; por encima, color fijo
            dendrogram(Z, color_threshold=umbral_corte,
                       above_threshold_color=above_threshold_color)
            plt.axhline(y=umbral_corte, color='red', linestyle='--')
            plt.title(f"Dendrograma (umbral t={umbral_corte})")

        plt.xlabel("Muestras")
        plt.ylabel("Distancia")
        # plt.grid()

    if show_blobs and show_dendo:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        if show_true_groups and y_cut is None:
            # Colorea por etiquetas verdaderas si se pidió y no hay corte
            axes[0].scatter(X2[:, 0], X2[:, 1], c=y2_true, cmap='viridis',
                        s=100, edgecolor='k')
            axes[0].set_title("Datos de testeo - Se pintan grupos originales")
        elif y_cut is not None:
            # Colorea por clusters obtenidos del corte
            axes[0].scatter(X2[:, 0], X2[:, 1], c=y_cut, cmap='tab10',
                        s=100, edgecolor='k')
            axes[0].set_title(f"Datos de testeo - Se pintan grupos para umbral {umbral_corte} - N° grupos {k_groups}")
        else:
            # Un solo color
            axes[0].scatter(X2[:, 0], X2[:, 1], color="#0303bc", edgecolor='k',
                        alpha=0.7, s=100, label='Datos')
            axes[0].legend(loc='lower left')
            axes[0].set_title(f"Datos de testeo sin agrupar")

        
        axes[0].set_xlabel("Característica 1")
        axes[0].set_ylabel("Característica 2")
        axes[0].grid()

        if umbral_corte is None:
            # Sin colores (todo en negro)
            dendrogram(Z, color_threshold=0, above_threshold_color="black")
            axes[1].set_title("Dendrograma para datos de testeo (sin línea de corte)")
        else:
            # Colores por debajo del umbral; por encima, color fijo
            dendrogram(Z, color_threshold=umbral_corte,
                       above_threshold_color=above_threshold_color)
            axes[1].axhline(y=umbral_corte, color='red', linestyle='--')
            axes[1].set_title(f"Dendrograma para datos de testeo (umbral corte = {umbral_corte})")

        
        axes[1].set_xlabel("Muestras")
        axes[1].set_ylabel("Distancia")

    plt.show()

def realdataWithDendo(columnasInteres = ["media_visitas_diarias","unidades_vendidas_mensuales"],
                      show_real_data=True, show_dendo=True, figsize=(16, 6), show_true_groups=False,
                      show_dendo_colors=False, umbral_corte=None, above_threshold_color="black", show_scores = False,
                      estandarizar=False):

    data = transform_and_get_iris()
    y_true = data["segmento"].values
    X_raw = data[columnasInteres].values

        # Estandarización (recomendado con KMeans y Ward)
    if estandarizar:
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
    else:
        X = X_raw

    Z = linkage(X, method='ward')
    y_cut = None
    k_groups = None
    silhouette_avg = None
    db_score = None
    if umbral_corte is not None:
        # Clusters por distancia (horizontal cut)
        y_cut = fcluster(Z, t=umbral_corte, criterion='distance')
        k_groups = len(np.unique(y_cut))
        silhouette_avg = silhouette_score(X, y_cut)
        db_score = davies_bouldin_score(X, y_cut)


    if show_real_data and not show_dendo:
        plt.figure(figsize=figsize)
        if show_true_groups:
            # Colorea por etiquetas verdaderas si se pidió y no hay corte
            plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis',
                        s=100, edgecolor='k')
            plt.legend([f"Grupos verdaderos (k={len(np.unique(y_true))})"], loc='lower left')
        elif y_cut is not None:
            # Colorea por clusters obtenidos del corte
            plt.scatter(X[:, 0], X[:, 1], c=y_cut, cmap='tab10',
                        s=100, edgecolor='k')
            if not show_scores:
                plt.title(f"Datos de testeo - Se pintan grupos para umbral {umbral_corte} - N° grupos {k_groups}")
            else:
                plt.title(f"Datos de testeo - Se pintan grupos para umbral {umbral_corte} - N° grupos {k_groups}\nSilhouette: {silhouette_avg:.2f} - DB: {db_score:.2f}")
        else:
            # Un solo color
            plt.scatter(X[:, 0], X[:, 1], color="#0303bc", edgecolor='k',
                        alpha=0.7, s=100, label='Datos')
            plt.legend(loc='lower left')
            plt.title("Datos de testeo sin agrupar")

        plt.xlabel("Característica 1")
        plt.ylabel("Característica 2")
        plt.grid()

    if show_dendo and not show_real_data:
        plt.figure(figsize=figsize)
        if umbral_corte is None:
            # Sin colores (todo en negro)
            dendrogram(Z, color_threshold=0, above_threshold_color="black")
            plt.title("Dendrograma sin línea de corte")
        else:
            # Colores por debajo del umbral; por encima, color fijo
            dendrogram(Z, color_threshold=umbral_corte,
                       above_threshold_color=above_threshold_color)
            plt.axhline(y=umbral_corte, color='red', linestyle='--')
            if not show_scores:
                plt.title(f"Dendrograma (umbral t={umbral_corte})")
            else:
                plt.title(f"Dendrograma (umbral t={umbral_corte})\nSilhouette: {silhouette_avg:.2f} - DB: {db_score:.2f}")

        plt.xlabel("Muestras")
        plt.ylabel("Distancia")
        # plt.grid()

    if show_real_data and show_dendo:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        if show_true_groups and y_cut is None:
            # Colorea por etiquetas verdaderas si se pidió y no hay corte
            axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis',
                        s=100, edgecolor='k')
            axes[0].set_title("Datos reales - Se pintan grupos originales")
        elif y_cut is not None:
            # Colorea por clusters obtenidos del corte
            axes[0].scatter(X[:, 0], X[:, 1], c=y_cut, cmap='tab10',
                        s=100, edgecolor='k')
            if not show_scores:
                axes[0].set_title(f"Datos reales - Se pintan grupos para umbral {umbral_corte} - N° grupos {k_groups}")
            else:
                axes[0].set_title(f"Datos reales - Se pintan grupos para umbral {umbral_corte} - N° grupos {k_groups}\nSilhouette: {silhouette_avg:.2f} - DB: {db_score:.2f}")
        else:
            # Un solo color
            axes[0].scatter(X[:, 0], X[:, 1], color="#0303bc", edgecolor='k',
                        alpha=0.7, s=100, label='Datos')
            axes[0].legend(loc='lower left')
            axes[0].set_title(f"Datos reales sin agrupar")

        
        axes[0].set_xlabel("Característica 1")
        axes[0].set_ylabel("Característica 2")
        axes[0].grid()

        if umbral_corte is None:
            # Sin colores (todo en negro)
            dendrogram(Z, color_threshold=0, above_threshold_color="black")
            axes[1].set_title("Dendrograma para datos reales (sin línea de corte)")
        else:
            # Colores por debajo del umbral; por encima, color fijo
            dendrogram(Z, color_threshold=umbral_corte,
                       above_threshold_color=above_threshold_color)
            axes[1].axhline(y=umbral_corte, color='red', linestyle='--')
            if not show_scores:
                axes[1].set_title(f"Dendrograma para datos reales (umbral corte = {umbral_corte})")
            else:
                axes[1].set_title(f"Dendrograma para datos reales (umbral corte = {umbral_corte})\nSilhouette: {silhouette_avg:.2f} - DB: {db_score:.2f}")

        axes[1].set_xlabel("Muestras")
        axes[1].set_ylabel("Distancia")

    plt.show()

def clustering_accuracy(y_true, y_pred):
    """
    Calcula la precisión de clustering (porcentaje de puntos correctamente asignados)
    emparejando clusters predichos con clases reales vía algoritmo Húngaro.
    """
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)

    # Matriz de confusión entre etiquetas verdaderas y clusters
    cm = confusion_matrix(y_true, y_pred, labels=labels_true)
    
    row_ind, col_ind = linear_sum_assignment(-cm)  # maximizamos coincidencias
    correct = cm[row_ind, col_ind].sum()
    
    return correct / len(y_true)

def comparing_algorithms(K, epsilon, min_samples, umbral_corte=None,
                         columnasInteres = ["media_visitas_diarias","unidades_vendidas_mensuales"],
                         estandarizar=False, random_state=42):
    # Aquí puedes implementar la comparación de diferentes algoritmos
    data = transform_and_get_iris()
    y_true = data["segmento"].values
    X_raw = data[columnasInteres].values

    # Estandarización (recomendado con KMeans y Ward)
    if estandarizar:
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
    else:
        X = X_raw

    Z = linkage(X, method='ward')

    # K-means
    kmeans = KMeans(n_clusters=K)
    kmeans.fit(X)
    y_kmeans = kmeans.labels_

    # DBSCAN
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    y_dbscan = dbscan.fit_predict(X)
    k_dbscan = len(np.unique(y_dbscan[y_dbscan != -1]))  # Excluye outliers (-1)

    # Clustering Jerárquico
    if umbral_corte is not None:
        # Clusters por distancia (horizontal cut)
        y_hierarchical = fcluster(Z, t=umbral_corte, criterion='distance')
        k_groups = len(np.unique(y_hierarchical))
        silhouette_avg = silhouette_score(X, y_hierarchical)
        db_score = davies_bouldin_score(X, y_hierarchical)


    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes[0,0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=100, edgecolor='k')
    axes[0,0].set_title("Datos reales - Grupos originales")

    # K-means
    axes[0,1].scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='Blues', s=100, edgecolor='k')
    axes[0,1].set_title(f"K-means - Grupos detectados - K={K}")

    # DBSCAN
    axes[1,0].scatter(X[:, 0], X[:, 1], c=y_dbscan, cmap='Reds', s=100, edgecolor='k')
    axes[1,0].set_title(f"DBSCAN - Grupos detectados {k_dbscan}")

    # Clustering Jerárquico
    axes[1,1].scatter(X[:, 0], X[:, 1], c=y_hierarchical, cmap='Greens', s=100, edgecolor='k')
    axes[1,1].set_title(f"Jerárquico - Grupos detectados - Umbral {umbral_corte} - Grupos {k_groups}")

    print("*********** MÉTRICAS ***********")
    for y_pred, algo_name in zip([y_kmeans, y_dbscan, y_hierarchical],
                                  ["K-means", "DBSCAN", "Jerárquico"]):
        silhouette_avg = silhouette_score(X, y_pred)
        db_score = davies_bouldin_score(X, y_pred)
        print(f"{algo_name} - Silhouette: {silhouette_avg:.2f} - DB: {db_score:.2f}")
    print("*********************************************",end="\n\n")

    print("*********** PRECISIÓN DE CLUSTERING ***********")
    modelos = {
        "K-means": y_kmeans,
        "DBSCAN": y_dbscan,
        "Jerárquico (K)": y_hierarchical
    }

    for nombre, y_pred in modelos.items():
        acc = clustering_accuracy(y_true, y_pred)
        print(f"{nombre} - Precisión de clustering: {acc*100:.2f}%")
    print("*********************************************")

    plt.show()

    return