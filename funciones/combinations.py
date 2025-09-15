import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder # to encode categorical v
import pandas as pd
import plotly.express as px # for data visualization
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
##importo kmeans, clustering jerarquico
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_wine, load_iris
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def getWineDF(seed=42, test_size=0.2):
    wine = load_wine()
    X = wine.data
    y = wine.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    feature_names = wine.feature_names
    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_train['target'] = y_train
    
    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test['target'] = y_test

    return df_train, df_test

def correlationForWine(figsie=(10, 8)):
    # Cargar el conjunto de datos de vino
    data = load_wine()
    X = data.data
    feature_names = data.feature_names

    # Crear un DataFrame de pandas para facilitar la manipulación y visualización
    df = pd.DataFrame(X, columns=feature_names)

    # Calcular la matriz de correlación
    corr = df.corr()

    # Crear un mapa de calor usando seaborn
    plt.figure(figsize=figsie)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True, linewidths=.5)
    plt.title('Matriz de correlaciones del conjunto de datos de Wine')
    plt.show()

def pairPlotofWine(features=None, figsize=(10, 10)):
    # Cargar el conjunto de datos de vino
    data = load_wine()
    X = data.data
    y = data.target

    df = pd.DataFrame(X, columns=data.feature_names)
    df['target'] = y

    if features is not None:
        X = X[:, features]
        feature_names = [data.feature_names[i] for i in features]
    else:
        cols_propuestas = ["alcohol", "malic_acid", "magnesium", "color_intensity", "proline"]
        print("No se han especificado features, se usarán las siguientes:",cols_propuestas)
        #dropping las features que no están en el dataset
        feature_names = [col for col in cols_propuestas if col in data.feature_names]
        df = df[feature_names + ['target']]

    ## Crear un pairplot usando seaborn
    sns.set_theme(style="ticks")
    pair_plot = sns.pairplot(df, hue='target', diag_kind='kde', markers=["o", "s", "D"], palette='Set1', height=figsize[0]/len(feature_names))

    # Ajustar el tamaño de la figura
    pair_plot.figure.set_size_inches(figsize)

    plt.suptitle('Gráfico de dispersión y KDE del conjunto de datos de Wine', y=1.02)
    plt.show()

def classifyWineFeatures(feature1, feature2, seed = 0, cmap = "plasma", test_size=0.4, figsize=(8,5)):
    if not feature1 or not feature2:
        raise ValueError("Se deben especificar dos features para clasificar.")
    if feature1 not in load_wine().feature_names or feature2 not in load_wine().feature_names:
        raise ValueError(f"Las features deben ser del conjunto de datos de vino: {load_wine().feature_names}")
    
    # Cargar el conjunto de datos de vino
    df_train, df_test = getWineDF(seed=seed, test_size=test_size)

    # Seleccionar las dos características especificadas
    X_train = df_train[[feature1, feature2]].values
    y_train = df_train['target'].values

    X_test = df_test[[feature1, feature2]].values
    y_test = df_test['target'].values

    # Estandarizar las características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entrenar un clasificador SVM
    svm_classifier = SVC(kernel='linear', random_state=seed)
    svm_classifier.fit(X_train_scaled, y_train)

    # Hacer predicciones en el conjunto de prueba
    y_pred = svm_classifier.predict(X_test_scaled)

    # Evaluar el rendimiento del clasificador
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Cantidad de datos para set de testeo y entrenamiento: {len(y_test)}, {len(y_train)}")
    print(f'Accuracy: {accuracy:.4f}')
    print('Classification Report:')
    print(report)
    
    # Visualizar la frontera de decisión
    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=figsize)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    scatter = plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, edgecolors='k', cmap=cmap)

    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title('Frontera de decisión del clasificador SVM - Datos de TESTEO')
    plt.legend(*scatter.legend_elements(), title="Clases")

    ##agrego gráfico de la matriz de confusión
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Matriz de Confusión')
    plt.show()

def classifyWinePCA(n_comps = 2, seed=0, cmap="plasma", test_size=0.4, figsize=(8,5)):
    # Cargar el conjunto de datos de vino
    df_train, df_test = getWineDF(seed=seed, test_size=test_size)

    # Separar características y etiquetas
    X_train = df_train.drop(columns=['target']).values
    y_train = df_train['target'].values
    X_test = df_test.drop(columns=['target']).values
    y_test = df_test['target'].values
    feature_names = df_train.drop(columns=['target']).columns

    # Estandarizar las características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Aplicar PCA
    pca = PCA(n_components=n_comps, random_state=seed)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(f'Varianza explicada por las {n_comps} componentes principales: {np.sum(pca.explained_variance_ratio_):.4f}')

    # Entrenar un clasificador SVM
    svm_classifier = SVC(kernel='linear', random_state=seed)
    svm_classifier.fit(X_train_pca, y_train)
    # predicciones en el conjunto de prueba
    y_pred = svm_classifier.predict(X_test_pca)

    # rendimiento del clasificador
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Cantidad de datos para set de testeo y entrenamiento: {len(y_test)} y {len(y_train)}, respectivamente")
    print(f'Accuracy: {accuracy:.4f}')
    print('Classification Report:')
    print(report)

    ##gráfico de la varianza explicada
    plt.figure(figsize=figsize)
    plt.bar(range(1, n_comps + 1), pca.explained_variance_ratio_, alpha=0.7, align='center', label='Varianza explicada individual')
    plt.step(range(1, n_comps + 1), np.cumsum(pca.explained_variance_ratio_), where='mid', label='Varianza explicada acumulada')
    plt.xlabel('Número de componentes principales')
    plt.ylabel('Varianza explicada')
    plt.title('Varianza explicada por las componentes principales')
    plt.legend(loc='best')

    ##gráfico de dispersión de las dos primeras componentes principales
    plt.figure(figsize=figsize)
    scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, edgecolors='k', cmap=cmap)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('Proyección PCA del conjunto de datos de $Wine$ - Datos de ENTRENAMIENTO')
    plt.legend(*scatter.legend_elements(), title="Clases")

    # Visualizar la frontera de decisión
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    if n_comps > 2:
        print("Advertencia: No se puede graficar la frontera de decisión en más de 2 dimensiones.")
    else:
        Z = svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.figure(figsize=figsize)
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
        scatter = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, edgecolors='k', cmap=cmap)
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.title('Frontera de decisión del clasificador SVM - Datos de TESTEO (PCA)')
        plt.legend(*scatter.legend_elements(), title="Clases")


    ##agrego gráfico de la matriz de confusión
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Matriz de Confusión')

    plt.show()

def classifyWineLDA(n_comps = 2, seed=0, cmap="plasma", test_size=0.4, show_variance=True, figsize=(8,5)):
    if n_comps >= 3:
        raise ValueError("LDA sólo puede reducir a un máximo de (n_classes - 1) dimensiones. En este caso, n_classes=3, por lo que n_comps debe ser 1 o 2.")
    
    # Cargar el conjunto de datos de vino
    df_train, df_test = getWineDF(seed=seed, test_size=test_size)

    # Separar características y etiquetas
    X_train = df_train.drop(columns=['target']).values
    y_train = df_train['target'].values
    X_test = df_test.drop(columns=['target']).values
    y_test = df_test['target'].values
    feature_names = df_train.drop(columns=['target']).columns

    # Estandarizar las características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Aplicar LDA
    lda = LDA(n_components=n_comps)
    X_train_lda = lda.fit_transform(X_train_scaled, y_train)
    X_test_lda = lda.transform(X_test_scaled)

    # Entrenar un clasificador SVM
    svm_classifier = SVC(kernel='linear', random_state=seed)
    svm_classifier.fit(X_train_lda, y_train)

    # Hacer predicciones en el conjunto de prueba
    y_pred = svm_classifier.predict(X_test_lda)

    # Evaluar el rendimiento del clasificador
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Cantidad de datos para set de testeo y entrenamiento: {len(y_test)} y {len(y_train)}, respectivamente")
    print(f'Accuracy: {accuracy:.4f}')
    print('Classification Report:')
    print(report)

    ##gráfico de la varianza explicada
    if show_variance:
        plt.figure(figsize=(8, 5))
        plt.bar(range(1, n_comps + 1), lda.explained_variance_ratio_, alpha=0.7, align='center', label='Varianza explicada individual')
        plt.step(range(1, n_comps + 1), np.cumsum(lda.explained_variance_ratio_), where='mid', label='Varianza explicada acumulada')
        plt.xlabel('Número de componentes LDA')
        plt.ylabel('Varianza explicada')
        plt.title('Varianza explicada por las componentes LDA')
        plt.legend(loc='best')
        
    ##gráfico de dispersión de las dos primeras componentes LDA
    plt.figure(figsize=figsize)
    scatter = plt.scatter(X_train_lda[:, 0], X_train_lda[:, 1], c=y_train, edgecolors='k', cmap=cmap)
    plt.xlabel('Componente LDA 1')
    plt.ylabel('Componente LDA 2')
    plt.title('Proyección LDA del conjunto de datos de $Wine$ - Datos de ENTRENAMIENTO')
    plt.legend(*scatter.legend_elements(), title="Clases")

    # Visualizar la frontera de decisión
    x_min, x_max = X_train_lda[:, 0].min() - 1, X_train_lda[:, 0].max() + 1
    y_min, y_max = X_train_lda[:, 1].min() - 1, X_train_lda[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    if n_comps > 2:
        print("Advertencia: No se puede graficar la frontera de decisión en más de 2 dimensiones.")
    else:
        Z = svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.figure(figsize=figsize)
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
        scatter = plt.scatter(X_test_lda[:, 0], X_test_lda[:, 1], c=y_test, edgecolors='k', cmap=cmap)
        plt.xlabel('Componente LDA 1')
        plt.ylabel('Componente LDA 2')
        plt.title('Frontera de decisión del clasificador SVM - Datos de TESTEO (LDA)')
        plt.legend(*scatter.legend_elements(), title="Clases")

    ##agrego gráfico de la matriz de confusión
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Matriz de Confusión')
    
    plt.show()


## ***************************************************************************************
### **************************** IRIS TRANSFORMED **********************************************

def transformIrisDataset():
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

def splitIrisTransformedDF(test_size=0.3, seed=42):
    df = transformIrisDataset()
    X = df.drop(columns=['segmento']).values
    y = df['segmento'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    feature_names = df.drop(columns=['segmento']).columns
    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_train['segmento'] = y_train
    
    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test['segmento'] = y_test

    return df_train, df_test

def pairPlotofIrdataset(features=None, figsize=(10, 10), standard_scaler=False):
    df = transformIrisDataset()

    if features is not None:
        X = df[features]
        feature_names = [features[i] for i in range(len(features)) if features[i] in df.columns]
    else:
        cols_propuestas = ["media_visitas_diarias", "precio_unitario", "unidades_vendidas_mensuales", "valoracion_media"]
        print("No se han especificado features, se usarán las siguientes:",cols_propuestas)
        #dropping las features que no están en el dataset
        feature_names = [col for col in cols_propuestas if col in df.columns]
        # X = df[feature_names + ['segmento']]
        X = df[feature_names]

    if standard_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = pd.DataFrame(X, columns=feature_names)

    ## Crear un pairplot usando seaborn
    sns.set_theme(style="ticks")
    # pair_plot = sns.pairplot(X, hue='segmento', diag_kind='kde', markers=["o", "s", "D"], palette='Set1', height=figsize[0]/len(feature_names))
    pair_plot = sns.pairplot(X, diag_kind='kde', height=figsize[0]/len(feature_names))

    # Ajustar el tamaño de la figura
    pair_plot.figure.set_size_inches(figsize)

    plt.suptitle('Gráfico de dispersión y KDE del conjunto de datos de Iris (simulado)', y=1.02)
    plt.show()

def kmeansIrdatasetAnalysis(feature1, feature2, figsize=(10, 6), cmap="plasma", max_clusters=10, seed=42, test_size=0.3):
    """Función para analizar la cantidad óptima de clusters usando el método del codo y el coeficiente de silueta.
    
    Se usan las dos features especificadas para aplicar KMeans y evaluar la calidad del clustering.
    
    Además se grafica el gráfico del codo, un gráfico de dispersión.

    NO se aplica Kmeans, sólo se analiza para determinar la cantidad óptima de clusters.
    """
    if not feature1 or not feature2:
        raise ValueError("Se deben especificar dos features para clasificar.\n" \
        "Podes usar las siguientes: 'media_visitas_diarias', 'precio_unitario', 'unidades_vendidas_mensuales', 'valoracion_media'")
    if feature1 not in transformIrisDataset().columns or feature2 not in transformIrisDataset().columns:
        raise ValueError(f"Las features deben ser algunas de 'media_visitas_diarias', 'precio_unitario', 'unidades_vendidas_mensuales', 'valoracion_media'")
    
    # Cargar el conjunto de datos de iris transformado
    df_train, df_test = splitIrisTransformedDF(seed=seed, test_size=test_size)

    # Seleccionar las dos características especificadas
    X_train = df_train[[feature1, feature2]].values
    y_train = df_train['segmento'].values

    X_test = df_test[[feature1, feature2]].values
    y_test = df_test['segmento'].values

    # Estandarizar las características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    inertia = []
    silhouette_scores = []
    K = range(2, max_clusters + 1)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=seed)
        kmeans.fit(X_train_scaled)
        inertia.append(kmeans.inertia_)
        
        # Calcular el coeficiente de silueta sólo si hay más de 1 cluster
        if k > 1:
            from sklearn.metrics import silhouette_score
            score = silhouette_score(X_train_scaled, kmeans.labels_)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(0)

    ## Gráfico de dispersión de los datos originales para visualización
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train[:, 0], X_train[:, 1])
    plt.title('Gráfico de Dispersión - Datos de Entrenamiento')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.show()

    # Graficar el método del codo
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.plot(K, inertia, 'bx-')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Inercia')
    plt.title('Método del Codo para determinar k óptimo')

    # Graficar el coeficiente de silueta
    plt.subplot(1, 2, 2)
    plt.plot(K, silhouette_scores, 'bx-')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Coeficiente de Silueta')
    plt.title('Coeficiente de Silueta para diferentes k')

    plt.tight_layout()
    plt.show()

##aplica aglomerative clustering para obtener el dendrograma sobre el set de datos de iris transformado de training
def deondoIrdataset(feature1, feature2, used_all_features = False, figsize=(10, 6), cmap="plasma", max_clusters=10, seed=42, test_size=0.3):
    if not used_all_features:
        if not feature1 or not feature2:
            raise ValueError("Se deben especificar dos features para clasificar.\n"
            "Podes usar las siguientes: 'media_visitas_diarias', 'precio_unitario', 'unidades_vendidas_mensuales', 'valoracion_media'")
        if feature1 not in transformIrisDataset().columns or feature2 not in transformIrisDataset().columns:
            raise ValueError(f"Las features deben ser algunas de 'media_visitas_diarias', 'precio_unitario', 'unidades_vendidas_mensuales', 'valoracion_media'")
    
    df_train, df_test = splitIrisTransformedDF(seed=seed, test_size=test_size)
    if used_all_features:
        X_train = df_train.drop(columns=['segmento']).values
    else:
        X_train = df_train[[feature1, feature2]].values

    # Aplicar aglomerative clustering
    
    Z = linkage(X_train, method='ward')
    #grafico el dendo con las lineas todas en negro, es decir, sin agrupar.
    plt.figure(figsize=figsize)
    dendrogram(Z, color_threshold=0)
    plt.title('Dendrogram - Aglomerative Clustering')
    plt.xlabel('Muestras')
    plt.ylabel('Distancia')
    plt.show()

def kmeansIrdatasetClustering(feature1, feature2, n_clusters=3, seed=42, cmap="plasma", test_size=0.3):
    if not feature1 or not feature2:
        raise ValueError("Se deben especificar dos features para clasificar.")
    if feature1 not in transformIrisDataset().columns or feature2 not in transformIrisDataset().columns:
        raise ValueError(f"Las features deben ser algunas de 'media_visitas_diarias', 'precio_unitario', 'unidades_vendidas_mensuales', 'valoracion_media'")
    
    # Cargar el conjunto de datos de iris transformado
    df_train, df_test = splitIrisTransformedDF(seed=seed, test_size=test_size)

    # Seleccionar las dos características especificadas
    X_train = df_train[[feature1, feature2]].values

    # Estandarizar las características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Aplicar KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    kmeans.fit(X_train_scaled)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    ##obtengo métricas de evaluación del clustering
    silhouette_avg = silhouette_score(X_train_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_train_scaled, labels)
    inertia = kmeans.inertia_
    print(f"Métricas de Evaluación para KMeans (k={n_clusters}):")
    print(f" - Silhouette Score: {round(silhouette_avg, 4)}")
    print(f" - Davies-Bouldin Score: {round(davies_bouldin, 4)}")
    print(f" - Inertia: {round(inertia, 4)}")

    ##grafico de dos columnas por una fila. Izquierda datos de entenamiento originales, izquierda datos de entrenamiento con clusters pintados
    ##por las labels obtenidas por kmeans
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c='gray', edgecolors='k',s=100, alpha=0.7)
    plt.title('Datos de Entrenamiento Originales')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], s=100, c=labels, cmap=cmap, edgecolors='k', alpha=0.7)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=150, label='Centroides')
    plt.title(f'KMeans Clustering (k={n_clusters}) - Datos de Entrenamiento')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.legend()
    plt.tight_layout()

    plt.show()

##función para generar clustering a partir de dendograma. Se usa el umbral que se desee
def jerarquicoIrdatasetClustering(feature1, feature2, umbral, used_all_features = False, seed=42, cmap="plasma", test_size=0.3):
    if not used_all_features:
        if not feature1 or not feature2:
            raise ValueError("Se deben especificar dos features para clasificar.\n"
            "Podes usar las siguientes: 'media_visitas_diarias', 'precio_unitario', 'unidades_vendidas_mensuales', 'valoracion_media'")
        if feature1 not in transformIrisDataset().columns or feature2 not in transformIrisDataset().columns:
            raise ValueError(f"Las features deben ser algunas de 'media_visitas_diarias', 'precio_unitario', 'unidades_vendidas_mensuales', 'valoracion_media'")
        
    if not umbral or umbral <= 0:
        raise ValueError("Se debe especificar un umbral positivo para cortar el dendrograma.")
    
    if umbral > 24:
        print("El umbral debe ser menor o igual a 24 para este conjunto de datos y mayor a 0.")
        print("Se setea el umbral a 24.")
        umbral = 24

    df_train, df_test = splitIrisTransformedDF(seed=seed, test_size=test_size)
    if used_all_features:
        X_train = df_train.drop(columns=['segmento']).values
    else:
        X_train = df_train[[feature1, feature2]].values

    # Estandarizar las características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Aplicar Agglomerative Clustering
    agglom = AgglomerativeClustering(n_clusters=None, distance_threshold=umbral)
    labels = agglom.fit_predict(X_train_scaled)
    n_clusters = len(np.unique(labels))
    print(f"Cantidad de clusters formados con umbral {umbral}: {n_clusters}")
    # Obtener los centroides de los clusters
    centroids = np.array([X_train_scaled[labels == i].mean(axis=0) for i in range(n_clusters)])
    # Calcular métricas de evaluación del clustering
    silhouette_avg = silhouette_score(X_train_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_train_scaled, labels)
    print(f"Métricas de Evaluación para Agglomerative Clustering (umbral={umbral}):")
    print(f" - Silhouette Score: {round(silhouette_avg, 4)}")
    print(f" - Davies-Bouldin Score: {round(davies_bouldin, 4)}")

    ##grafico de tres columnas. Izquierda datos de entenamiento originales, en el medio
    ##el dendograma con las líneas pintadas en base al umbral, también generar una lína horizontal roja con estilo -- en el umbral
    ##el gráfico de la derecha con los datos de entrenamiento con clusters pintados
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c='gray', edgecolors='k',s=100, alpha=0.7)
    plt.title('Datos de Entrenamiento Originales')
    plt.xlabel(feature1)
    plt.ylabel(feature2)    
    plt.subplot(1, 3, 2)
    Z = linkage(X_train_scaled, method='ward')
    dendrogram(Z, color_threshold=umbral)
    plt.axhline(y=umbral, color='r', linestyle='--', label='Umbral')
    plt.title('Dendrogram - Aglomerative Clustering')
    plt.xlabel('Muestras')
    plt.ylabel('Distancia')
    plt.legend()
    plt.subplot(1, 3, 3)
    scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], s=100, c=labels, cmap=cmap, edgecolors='k', alpha=0.7)
    if n_clusters > 1:
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=150, label='Centroides')
    plt.title(f'Agglomerative Clustering (umbral={umbral}) - Datos de Entrenamiento')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.legend()
    plt.tight_layout()
    plt.show()

##función para segmentar usando kmeans o jerarquico y luego entrenar un LDA para clasificar
def classifyClusteringLDA(method="kmeans", feature1=None, feature2=None, n_clusters=3, umbral=10,
                                used_all_features = False, seed=42, cmap="Pastel1", test_size=0.3, figsize=(8,5)):
    if method not in ["kmeans", "jerarquico"]:
        raise ValueError("El método debe ser 'kmeans' o 'jerarquico'.")
    
    if method == "kmeans":
        if not feature1 or not feature2:
            raise ValueError("Se deben especificar dos features para clasificar.")
        if feature1 not in transformIrisDataset().columns or feature2 not in transformIrisDataset().columns:
            raise ValueError(f"Las features deben ser algunas de 'media_visitas_diarias', 'precio_unitario', 'unidades_vendidas_mensuales', 'valoracion_media'")
    
    if method == "jerarquico" and not used_all_features:
        if not feature1 or not feature2:
            raise ValueError("Se deben especificar dos features para clasificar.\n"
            "Podes usar las siguientes: 'media_visitas_diarias', 'precio_unitario', 'unidades_vendidas_mensuales', 'valoracion_media'")
        if feature1 not in transformIrisDataset().columns or feature2 not in transformIrisDataset().columns:
            raise ValueError(f"Las features deben ser algunas de 'media_visitas_diarias', 'precio_unitario', 'unidades_vendidas_mensuales', 'valoracion_media'")
        
    if method == "jerarquico":
        if not umbral or umbral <= 0:
            raise ValueError("Se debe especificar un umbral positivo para cortar el dendrograma.")
        
        if umbral > 24:
            print("El umbral debe ser menor o igual a 24 para este conjunto de datos y mayor a 0.")
            print("Se setea el umbral a 24.")
            umbral = 24

    df_train, df_test = splitIrisTransformedDF(seed=seed, test_size=test_size)
    if used_all_features:
        X_train = df_train.drop(columns=['segmento']).values
        X_test = df_test.drop(columns=['segmento']).values
    else:
        X_train = df_train[[feature1, feature2]].values
        X_test = df_test[[feature1, feature2]].values

    cats_names = {0: "Cliente explorador",
                  1: "Cliente comprometido",
                  2: "Cliente impulsivo"}

    y_test = df_test['segmento'].values

    #estandarizo
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #aplico el método de clustering
    if method == "kmeans":
        cluster_model = KMeans(n_clusters=n_clusters, random_state=seed)
        cluster_labels = cluster_model.fit_predict(X_train_scaled)
        centroids = cluster_model.cluster_centers_
        print(f"Cantidad de clusters formados con KMeans: {n_clusters}")
    else: #jerarquico
        cluster_model = AgglomerativeClustering(n_clusters=None, distance_threshold=umbral)
        cluster_labels = cluster_model.fit_predict(X_train_scaled)
        n_clusters = len(np.unique(cluster_labels))
        print(f"Cantidad de clusters formados con Agglomerative Clustering y umbral {umbral}: {n_clusters}")
        # Obtener los centroides de los clusters
        centroids = np.array([X_train_scaled[cluster_labels == i].mean(axis=0) for i in range(n_clusters)])

    #entreno un LDA con las labels obtenidas por el clustering
    lda = LDA(n_components=2)
    lda.fit(X_train_scaled, cluster_labels)
    
    ##clasifico los datos de testeo
    y_pred = lda.predict(X_test_scaled)
    ##obtengo las probabilidades de pertenencia a cada clase
    y_prob = lda.predict_proba(X_test_scaled)

    # Evaluar el rendimiento del clasificador
    cm = confusion_matrix(y_test, y_pred)
    ##alineando etiquetas de los clusters con las etiquetas reales
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    y_pred_aligned = np.array([mapping[label] for label in y_pred])
    cm_aligned = confusion_matrix(y_test, y_pred_aligned)

    accuracy = accuracy_score(y_test, y_pred_aligned)
    report = classification_report(y_test, y_pred_aligned)
    print(f"Cantidad de datos para set de testeo y entrenamiento: {len(y_test)} y {len(X_train_scaled)}, respectivamente")
    print("Accuracy:", round(accuracy,4)*100, "%")
    print("Reporte de clasificación:")
    print(report)

    ##gráfico de dispersión con los datos originales y los clasificados de manera superpuesta.
    ##se usan distintos marcadores para diferenciar los puntos
    plt.figure(figsize=figsize)
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], s=100, c=cluster_labels, cmap=cmap,
                edgecolors='k', alpha=0.5, label='Datos Originales', marker='o')
    plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], s=100, c=y_pred, cmap=cmap, edgecolors='k', alpha=1, label='Datos Clasificados', marker='s')
    plt.title(f'{method.capitalize()} Clustering + LDA - Datos Originales vs Clasificados')
    plt.xlabel(feature1 if not used_all_features else "Feature 1")
    plt.ylabel(feature2 if not used_all_features else "Feature 2")
    plt.legend()
    plt.grid()

    ##agrego gráfico de la matriz de confusión
    plt.figure(figsize=(6,5))
    #heatmap con seaborn de la matriz de confusión. Uso cats_names para poner los nombres de las clases
    sns.heatmap(cm_aligned, annot=True, fmt='d', cmap='Blues',
                xticklabels=[cats_names[i] for i in range(len(cats_names))],
                yticklabels=[cats_names[i] for i in range(len(cats_names))])
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.show()

def classifyClusteringSVM(method="kmeans", feature1=None, feature2=None, n_clusters=3, umbral=10,
                                used_all_features = False, seed=42, cmap="Pastel1", test_size=0.3, figsize=(8,5)):
    if method not in ["kmeans", "jerarquico"]:
        raise ValueError("El método debe ser 'kmeans' o 'jerarquico'.")
    
    if method == "kmeans":
        if not feature1 or not feature2:
            raise ValueError("Se deben especificar dos features para clasificar.")
        if feature1 not in transformIrisDataset().columns or feature2 not in transformIrisDataset().columns:
            raise ValueError(f"Las features deben ser algunas de 'media_visitas_diarias', 'precio_unitario', 'unidades_vendidas_mensuales', 'valoracion_media'")
    
    if method == "jerarquico" and not used_all_features:
        if not feature1 or not feature2:
            raise ValueError("Se deben especificar dos features para clasificar.\n"
            "Podes usar las siguientes: 'media_visitas_diarias', 'precio_unitario', 'unidades_vendidas_mensuales', 'valoracion_media'")
        if feature1 not in transformIrisDataset().columns or feature2 not in transformIrisDataset().columns:
            raise ValueError(f"Las features deben ser algunas de 'media_visitas_diarias', 'precio_unitario', 'unidades_vendidas_mensuales', 'valoracion_media'")
        
    if method == "jerarquico":
        if not umbral or umbral <= 0:
            raise ValueError("Se debe especificar un umbral positivo para cortar el dendrograma.")
        
        if umbral > 24:
            print("El umbral debe ser menor o igual a 24 para este conjunto de datos y mayor a 0.")
            print("Se setea el umbral a 24.")
            umbral = 24

    df_train, df_test = splitIrisTransformedDF(seed=seed, test_size=test_size)
    if used_all_features:
        X_train = df_train.drop(columns=['segmento']).values
        X_test = df_test.drop(columns=['segmento']).values
    else:
        X_train = df_train[[feature1, feature2]].values
        X_test = df_test[[feature1, feature2]].values

    cats_names = {0: "Cliente explorador",
                  1: "Cliente comprometido",
                  2: "Cliente leal"}
    
    y_test = df_test['segmento'].values
    #estandarizo
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    #aplico el método de clustering
    if method == "kmeans":
        cluster_model = KMeans(n_clusters=n_clusters, random_state=seed)
        cluster_labels = cluster_model.fit_predict(X_train_scaled)
        centroids = cluster_model.cluster_centers_
        print(f"Cantidad de clusters formados con KMeans: {n_clusters}")
    else: #jerarquico
        cluster_model = AgglomerativeClustering(n_clusters=None, distance_threshold=umbral)
        cluster_labels = cluster_model.fit_predict(X_train_scaled)
        n_clusters = len(np.unique(cluster_labels))
        print(f"Cantidad de clusters formados con Agglomerative Clustering y umbral {umbral}: {n_clusters}")
        # Obtener los centroides de los clusters
        centroids = np.array([X_train_scaled[cluster_labels == i].mean(axis=0) for i in range(n_clusters)])

    #entreno un SVM con las labels obtenidas por el clustering
    svm_classifier = SVC(kernel='linear', random_state=seed)
    svm_classifier.fit(X_train_scaled, cluster_labels)
    ##clasifico los datos de testeo
    y_pred = svm_classifier.predict(X_test_scaled)
    ##obtengo las probabilidades de pertenencia a cada clase
    y_prob = svm_classifier.decision_function(X_test_scaled)
    # Evaluar el rendimiento del clasificador
    cm = confusion_matrix(y_test, y_pred)
    ##alineando etiquetas de los clusters con las etiquetas reales
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    y_pred_aligned = np.array([mapping[label] for label in y_pred])
    cm_aligned = confusion_matrix(y_test, y_pred_aligned)
    accuracy = accuracy_score(y_test, y_pred_aligned)
    report = classification_report(y_test, y_pred_aligned)
    print(f"Cantidad de datos para set de testeo y entrenamiento: {len(y_test)} y {len(X_train_scaled)}, respectivamente")
    print("Accuracy:", round(accuracy,4)*100, "%")
    print("Reporte de clasificación:")
    print(report)

    ##gráfico de dispersión con los datos originales y los clasificados de manera superpuesta.
    ##se usan distintos marcadores para diferenciar los puntos
    # plt.figure(figsize=figsize)
    # plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], s=100, c=cluster_labels, cmap=cmap,
    #             edgecolors='k', alpha=0.5, label='Datos Originales', marker='o')
    # plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], s=100, c=y_pred, cmap=cmap, edgecolors='k', alpha=1, label='Datos Clasificados', marker='s')
    # plt.title(f'{method.capitalize()} Clustering + LDA - Datos Originales vs Clasificados')
    # plt.xlabel(feature1 if not used_all_features else "Feature 1")
    # plt.ylabel(feature2 if not used_all_features else "Feature 2")
    # plt.legend()
    # plt.grid()

    ##grafico la frontera de decisión del SVM y pongo los datos originales y los clasificados
    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01)) 
    Z = svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=figsize)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], s=100, c=cluster_labels, cmap=cmap,
                edgecolors='k', alpha=0.5, label='Datos Originales', marker='o')
    plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], s=100, c=y_pred, cmap=cmap, edgecolors='k', alpha=1, label='Datos Clasificados', marker='s')
    plt.xlabel(feature1 if not used_all_features else "Feature 1")
    plt.ylabel(feature2 if not used_all_features else "Feature 2")
    plt.title(f'Frontera de decisión del clasificador SVM - Datos Clasificados ({method.capitalize()} + SVM)')
    plt.legend()
    plt.grid()

    ##agrego gráfico de la matriz de confusión
    plt.figure(figsize=(6,5))
    #heatmap con seaborn de la matriz de confusión. Uso cats_names para poner los nombres de las clases
    sns.heatmap(cm_aligned, annot=True, fmt='d', cmap='Blues',
                xticklabels=[cats_names[i] for i in range(len(cats_names))],
                yticklabels=[cats_names[i] for i in range(len(cats_names))])
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.show()

## ************************************************************************************************************************
## PARTE 3

def loadSegmentedData(test_size=0.3, seed=42):
    try:
        df = pd.read_csv('cdan//datos//Producto_Dataset_Segmentado.csv')
    except Exception as e:
        df = pd.read_csv('datos\\Producto_Dataset_Segmentado.csv')
    
    # Dividir en características y etiquetas
    X = df.drop(columns=['product_category']).values
    y = df['product_category'].values
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y
                                                    , test_size=test_size, random_state=seed, stratify=y)
    
    feature_names = df.drop(columns=['product_category']).columns

    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_train['product_category'] = y_train
    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test['product_category'] = y_test

    ##drop columna product_id
    if 'product_id' in df_train.columns:
        df_train = df_train.drop(columns=['product_id'])
        df_test = df_test.drop(columns=['product_id'])

    return df_train, df_test

def basicAnalysisSegmentedDataset(pairplot_height=2):
    df_train, df_test = loadSegmentedData()
    ##quito filas Outlier de la columna product_category
    df_train = df_train[df_train['product_category'] != "Outlier"]
    df_test = df_test[df_test['product_category'] != "Outlier"]

    feature_names = df_train.drop(columns=['product_category']).columns
    #convierto a numéricas las columnas que se puedan
    for col in feature_names:
        df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
        df_test[col] = pd.to_numeric(df_test[col], errors='coerce')

    num_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df_train.columns if c not in num_cols]

    print("************************************************************************")
    print("Categorías de productos en el dataset segmentado:")
    ##cantidad por clase
    class_counts = df_train['product_category'].value_counts(dropna=False).rename_axis('product_category').reset_index(name='count')
    print("************************************************************************",end="\n\n")

    print("************************************************************************")
    print("Estadísticas descriptivas del dataset segmentado:")
    print(df_train[num_cols].describe().T)
    print("************************************************************************",end="\n\n")

    # Skewness/Kurtosis
    sk_kurt = pd.DataFrame({
        'skew': df_train[num_cols].skew(numeric_only=True),
        'kurtosis': df_train[num_cols].kurtosis(numeric_only=True)
    }).sort_values('skew', key=lambda s: s.abs(), ascending=False)

    print("************************************************************************")
    print("Skewness y Kurtosis del dataset segmentado:")
    print(sk_kurt)
    print("************************************************************************",end="\n\n")

    ##heatmap de la correlación entre las variables numéricas
    plt.figure(figsize=(8, 6))
    corr = df_train[num_cols].corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", square=True, cbar_kws={"shrink": .8})
    plt.title("Matriz de correlación del dataset segmentado")
    #ponto etiquetas del eje x en 45 grados
    plt.xticks(rotation=45, ha='right')
    # plt.show()

    p = sns.pairplot(df_train[num_cols + ['product_category']], hue='product_category',
                     diag_kind='kde', markers=["o", "s", "D"], palette='Set1',height=pairplot_height, aspect=1.2)
    # Rotar solo las labels (nombres de variables) en los ejes
    for ax in p.axes[-1, :]:  # última fila → etiquetas del eje x
        ax.set_xlabel(ax.get_xlabel(),fontsize=9)

    for ax in p.axes[:, 0]:  # primera columna → etiquetas del eje y
        ax.set_ylabel(ax.get_ylabel(), fontsize=9)

    plt.suptitle('Gráfico de dispersión y KDE del conjunto de datos segmentado', y=1.02)
    # plt.tight_layout()
    plt.show()

def processSegmenDataset(return_data = False, n_components_pca = 2, apply_pca = False, seed=42,
                          show_plots = True, remove_columns = []):
    """
    Función para aplicar procesamiento al dataset segmentado.

    1. Separar target (product_category) de features.
    2. Imputación: no hay faltantes → se omite.
    3. Escalado: aplicar StandardScaler sobre todas las numéricas.
    4. Opcional: aplicar PCA para reducir redundancia de las 4 variables altamente correlacionadas.
    5. LDA: usarlo tanto como clasificador directo como transformador supervisado (máx. 2 dimensiones para A/B/C).
    """

    df_train, df_test = loadSegmentedData(seed=seed)
    ##quito filas Outlier de la columna product_category
    df_train = df_train[df_train['product_category'] != "Outlier"]
    df_test = df_test[df_test['product_category'] != "Outlier"]

    #remuevo las columnas que se pidan
    if remove_columns:
        df_train = df_train.drop(columns=remove_columns, errors='ignore')
        df_test = df_test.drop(columns=remove_columns, errors='ignore')

    feature_names = df_train.drop(columns=['product_category']).columns

    #convierto a numéricas las columnas que se puedan
    for col in feature_names:
        df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
        df_test[col] = pd.to_numeric(df_test[col], errors='coerce')

    num_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df_train.columns if c not in num_cols]

    print("Aplicando StandardScaler a las variables numéricas...")
    scaler = StandardScaler()
    df_train[num_cols] = scaler.fit_transform(df_train[num_cols])
    df_test[num_cols] = scaler.transform(df_test[num_cols])

    ##1. Separar target (product_category) de features.
    X_train = df_train[feature_names].values
    y_train = df_train['product_category'].values
    X_test = df_test[feature_names].values
    y_test = df_test['product_category'].values

    ##y_train e y_test contienen "A", "B" y "C". Las convierto a 0, 1 y 2
    label_mapping = {'A': 0, 'B': 1, 'C': 2}
    y_train = np.array([label_mapping[label] for label in y_train])
    y_test = np.array([label_mapping[label] for label in y_test])

    if apply_pca:
        ##4. Opcional: aplicar PCA para reducir redundancia de las 4 variables altamente correlacionadas.
        print(f"Aplicando PCA para reducir a {n_components_pca} componentes principales...")
        pca = PCA(n_components=n_components_pca, random_state=seed)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        print("PCA aplicado.")
        print("Varianza explicada por cada componente principal:", pca.explained_variance_ratio_)
        print("Varianza total explicada por los componentes principales:", np.sum(pca.explained_variance_ratio_))

    ##5. LDA para aumentar la separabilidad entre clases.
    print("Aplicando LDA para aumentar la separabilidad entre clases...")
    lda = LDA(n_components=2)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)

    ##scatter plot de los datos transformados por LDA. Mapa de colores por clase
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_train_lda[:, 0], X_train_lda[:, 1], c=y_train, cmap='Set1', edgecolors='k', s=100, alpha=0.7)
    if apply_pca:
        plt.title('Datos de Entrenamiento transformados por PCA + LDA')
    else:
        plt.title('Datos de Entrenamiento transformados por LDA (sin PCA)')
    plt.xlabel('Componente LDA 1')
    plt.ylabel('Componente LDA 2')
    plt.colorbar(scatter, label='Categoría de Producto')
    if show_plots:
        plt.show()

    if return_data:
        return X_train_lda, y_train, X_test_lda, y_test

# remove_columns = ["product_length_cm","packaging_width_cm"]
# processSegmenDataset(n_components_pca=2, apply_pca=True, seed=42, show_plots=True, remove_columns=[])
# processSegmentedDataset_v2(seed=42, show_plots=True)



## ************************************************************************************************************************
## PARTE 3

def loadSegmentedData(test_size=0.3, seed=42):
    try:
        df = pd.read_csv('cdan//datos//Producto_Dataset_Segmentado.csv')
    except Exception as e:
        df = pd.read_csv('datos\\Producto_Dataset_Segmentado.csv')

    df = df[df['product_category'] != "Outlier"]

    ##quito columna product_id
    if 'product_id' in df.columns:
        df = df.drop(columns=['product_id'])

    encoder = OneHotEncoder(sparse_output=False, drop=None)

    categorias = df['product_category']
    encoded = encoder.fit_transform(df[["product_category"]])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["product_category"]))

    df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1)
    
    # Dividir en características y etiquetas
    X = df.drop(columns=['sales_velocity']).values
    y = df['sales_velocity'].values
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=seed,
                                                        stratify=categorias)
    
    feature_names = df.drop(columns=['sales_velocity']).columns

    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_train['sales_velocity'] = y_train
    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test['sales_velocity'] = y_test

    ##drop columna product_id
    if 'product_id' in df_train.columns:
        df_train = df_train.drop(columns=['product_id'])
        df_test = df_test.drop(columns=['product_id'])

    return df_train, df_test

def basicAnalysisSegmentedDataset(test_size = 0.2, pairplot_height=2):
    df_train, df_test = loadSegmentedData(test_size=test_size)
    ##quito filas Outlier de la columna product_category
    df_train = df_train[df_train['product_category'] != "Outlier"]
    df_test = df_test[df_test['product_category'] != "Outlier"]

    feature_names = df_train.drop(columns=['product_category','sales_velocity']).columns #product_category_A
    cols_to_show_pair = df_train.drop(columns=['sales_velocity',
                                               'product_category_A',
                                               'product_category_B',
                                               'product_category_C']).columns
    
    #convierto a numéricas las columnas que se puedan
    for col in feature_names:
        df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
        df_test[col] = pd.to_numeric(df_test[col], errors='coerce')

    num_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df_train.columns if c not in num_cols]

    print("************************************************************************")
    print("Categorías de productos en el dataset segmentado:")
    ##cantidad por clase
    class_counts = df_train['product_category'].value_counts(dropna=False).rename_axis('product_category').reset_index(name='count')
    print(class_counts)
    print("************************************************************************",end="\n\n")

    print("************************************************************************")
    print("Estadísticas descriptivas del dataset segmentado:")
    print(df_train[num_cols].describe().T)
    print("************************************************************************",end="\n\n")

    # Skewness/Kurtosis
    sk_kurt = pd.DataFrame({
        'skew': df_train[num_cols].skew(numeric_only=True),
        'kurtosis': df_train[num_cols].kurtosis(numeric_only=True)
    }).sort_values('skew', key=lambda s: s.abs(), ascending=False)

    print("************************************************************************")
    print("Skewness y Kurtosis del dataset segmentado:")
    print(sk_kurt)
    print("************************************************************************",end="\n\n")

    ##heatmap de la correlación entre las variables numéricas
    plt.figure(figsize=(8, 6))
    corr = df_train[num_cols].corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", square=True, cbar_kws={"shrink": .8})
    plt.title("Matriz de correlación del dataset segmentado")
    #ponto etiquetas del eje x en 45 grados
    plt.xticks(rotation=45, ha='right')
    # plt.show()
    df_pair = cols_to_show_pair = df_train.drop(columns=['product_category_A',
                                                         'product_category_B',
                                                         'product_category_C'])

    p = sns.pairplot(df_pair, hue='product_category',
                     diag_kind='kde', markers=["o", "s", "D"], palette='Set1',height=pairplot_height, aspect=1.2)
    # Rotar solo las labels (nombres de variables) en los ejes
    for ax in p.axes[-1, :]:  # última fila → etiquetas del eje x
        ax.set_xlabel(ax.get_xlabel(),fontsize=9)

    for ax in p.axes[:, 0]:  # primera columna → etiquetas del eje y
        ax.set_ylabel(ax.get_ylabel(), fontsize=9)

    plt.suptitle('Gráfico de dispersión y KDE del conjunto de datos segmentado', y=1.02)
    # plt.tight_layout()
    plt.show()

def processSegmentedDataset(test_size = 0.2, pairplot_height=2, features_selected = None,
                            apply_pca = True, n_components_pca = 2, use_product_category = True,
                            show_stats = False, return_data = False):
    df_train, df_test = loadSegmentedData(test_size=test_size)
    ##quito filas Outlier de la columna product_category
    df_train = df_train[df_train['product_category'] != "Outlier"]
    df_test = df_test[df_test['product_category'] != "Outlier"]

    feature_names = df_train.drop(columns=['product_category','sales_velocity']).columns #product_category_A

    #convierto a numéricas las columnas que se puedan
    for col in feature_names:
        df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
        df_test[col] = pd.to_numeric(df_test[col], errors='coerce')

    num_cols = ["product_length_cm","shelf_presence_score", "packaging_width_cm",
                "price_usd","customer_satisfaction","sales_velocity"]
    
    cat_cols = ["product_category_A", "product_category_B", "product_category_C"]

    print("Aplicando StandardScaler a las variables numéricas...")
    scaler = StandardScaler()
    df_train_escaled = df_train.copy()
    df_test_escaled = df_test.copy()
    df_train_escaled[num_cols] = scaler.fit_transform(df_train[num_cols])
    df_test_escaled[num_cols] = scaler.transform(df_test[num_cols])

    sales_scaler = StandardScaler()
    sales_scaler.fit(df_train[['sales_velocity']])

    if apply_pca:
        interested_features = ["product_length_cm","shelf_presence_score","price_usd","customer_satisfaction"]
        print("Aplicando PCA a las variables numéricas altamente correlacionadas...")
        print("Características seleccionadas para PCA:", interested_features)
        pca = PCA(n_components=n_components_pca)
        X_train_pca = pca.fit_transform(df_train_escaled[interested_features])
        X_test_pca = pca.transform(df_test_escaled[interested_features])

        ##agrego las columnas pca1 y pca2 a los dataframes
        df_train_pca = pd.DataFrame(X_train_pca, columns=[f'pca_{i+1}' for i in range(n_components_pca)])
        df_test_pca = pd.DataFrame(X_test_pca, columns=[f'pca_{i+1}' for i in range(n_components_pca)])

        ##genero nuevos dataframes llamados df_train_processed y df_test_processed con las columnas pca y las categóricas
        df_train_processing = pd.concat([df_train_pca.reset_index(drop=True),
                                  df_train_escaled[["packaging_width_cm","sales_velocity"] + cat_cols].reset_index(drop=True)], axis=1)
        df_test_processing = pd.concat([df_test_pca.reset_index(drop=True),
                                 df_test_escaled[["packaging_width_cm","sales_velocity"] + cat_cols].reset_index(drop=True)], axis=1)
        
        if not use_product_category:
            df_train_processing = df_train_processing.drop(columns=cat_cols)
            df_test_processing = df_test_processing.drop(columns=cat_cols)
        
    else:
        print("No se aplica PCA.")
        if not features_selected:
            print("No se ha especificado una lista de features a usar, se usarán todas las features numéricas y categóricas.")

            features_selected = num_cols + cat_cols
            features_selected.remove("sales_velocity") #la quito porque es el target

        print("Características seleccionadas:", features_selected)
        df_train_processing = df_train_escaled[features_selected + ["sales_velocity"]]
        df_test_processing = df_test_escaled[features_selected + ["sales_velocity"]]

    if show_stats:
        print("************************************************************************")
        print("Estadísticas descriptivas del dataset segmentado procesado:")
        print(df_train_processing.describe().T)
        print("************************************************************************",end="\n\n")

    X_train = df_train_processing.drop(columns=['sales_velocity']).values
    y_train = df_train_processing['sales_velocity'].values
    X_test = df_test_processing.drop(columns=['sales_velocity']).values
    y_test = df_test_processing['sales_velocity'].values

    if return_data:
        return X_train, y_train, X_test, y_test, sales_scaler

# X_train, y_train, X_test, y_test = processSegmentedDataset(apply_pca=True,return_data=True, show_stats=True, use_product_category=False)

def predictSalesVelocity(test_size = 0.2, features_selected = None,
                            apply_pca = True, n_components_pca = 2,
                            use_product_category = True):

    X_train, y_train, X_test, y_test, sales_scaler = processSegmentedDataset(test_size = test_size,
                                                                             features_selected = features_selected,
                                                                             apply_pca = apply_pca,
                                                                             n_components_pca = n_components_pca,
                                                                             use_product_category = use_product_category,
                                                                             show_stats = False, return_data = True)

    models = {
    "Linear Regression": LinearRegression(),
    # "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42)
    }

    for name, model in models.items():
        print("="*60)
        print(f"Entrenando modelo: {name}")
        
        # Entrenamiento
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Métricas
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"Cantidad de datos para set de testeo y entrenamiento: {len(y_test)} y {len(X_train)}, respectivamente")
        print("R²:", round(r2,4))
        print("RMSE:", round(rmse,4))
        print("MAE:", round(mae,4))

        ##imprimo algunos valores reales vs predichos
        ##revierto la escala original de y_test e y_pred
        y_test = sales_scaler.inverse_transform(np.concatenate((np.zeros((len(y_test), X_test.shape[1])), y_test.reshape(-1,1)), axis=1))[:,-1]
        y_pred = sales_scaler.inverse_transform(np.concatenate((np.zeros((len(y_pred), X_test.shape[1])), y_pred.reshape(-1,1)), axis=1))[:,-1]
        comparison_df = pd.DataFrame({'Real': y_test, 'Predicho': y_pred})
        ##agrego columna de diferencia absoluta y porcentaje de error
        comparison_df['Diferencia Absoluta'] = np.abs(comparison_df['Real'] - comparison_df['Predicho'])
        comparison_df['Porcentaje de Error'] = comparison_df['Diferencia Absoluta'] / comparison_df['Real'] * 100
        print("Primeros 5 valores reales vs predichos:")
        print(comparison_df.head(5))

        # Scatterplot real vs predicho
        plt.figure(figsize=(6,6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, label='Predicciones',
                        color="#b41f38", s=100, edgecolor='k')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal')
        plt.xlabel("Valores Reales")
        plt.ylabel("Valores Predichos")
        plt.title(f"Real vs Predicho - {name}")
        plt.legend()
        plt.show()

# features_selected = ["product_length_cm","shelf_presence_score"]
# predictSalesVelocity(apply_pca=False, use_product_category=False, n_components_pca=2, features_selected=features_selected)