import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os
import shutil
import tarfile
from sklearn.datasets import fetch_lfw_people, fetch_olivetti_faces, get_data_home
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import plotly.express as px 
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def plotPCAExamples(n_points = 10, std1 = 1, std2 = 0.2, figsize = (14,6), title = "Ejemplo de PCA", seed = 42,
                    show_labels = False, cmap = "winter", scalepc1 = 1, scalepc2 = 2):

    # Genero un set de datos.
    np.random.seed(seed)
    x1 = np.random.normal(5, std1, n_points)
    x2 = 0.5 * x1 + np.random.normal(0, std2, n_points)
    X = np.vstack((x1, x2)).T
    #genero etiquetas
    labels = np.array([0] * (n_points // 2) + [1] * (n_points // 2))
    #asigno aleatoriamente
    np.random.shuffle(labels)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Datos originales
    if show_labels:
        axes[0].scatter(X[:,0], X[:,1], c=labels, cmap=cmap, alpha=0.7, s=50)
    else:
        axes[0].scatter(X[:,0], X[:,1], alpha=0.7, color="#2574F4", s=50)
    axes[0].set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")
    axes[0].set_title("Datos originales en 2D")
    axes[0].axis("equal")

    # Caclulo PCA
    pca = PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.transform(X)

    # Componentes principales
    pc1 = pca.components_[0]
    pc2 = pca.components_[1]
    mean = pca.mean_

    # Datos con ejes principales
    ##std de los datos
    stdx1 = np.std(X[:,0])
    stdx2 = np.std(X[:,1])
    std_max = max(stdx1, stdx2)

    if show_labels:
        axes[1].scatter(X[:,0], X[:,1], c=labels, cmap=cmap, alpha=0.7, s=50)
    else:
        axes[1].scatter(X[:,0], X[:,1], alpha=0.7, color="#2574F4", s=50)
    axes[1].quiver(mean[0], mean[1], pc1[0], pc1[1], 
            angles="xy", scale_units="xy", scale=scalepc1, color="#c71e34", label="PC1", width=0.015)
    axes[1].quiver(mean[0], mean[1], pc2[0], pc2[1], 
            angles="xy", scale_units="xy", scale=scalepc2, color="#67dc96", label="PC2", width=0.015)
    axes[1].set_xlabel("$x_1$")
    axes[1].set_ylabel("$x_2$")
    axes[1].legend()
    axes[1].set_title("Ejes principales (PC1 y PC2)")
    axes[1].axis("equal")

    # Grafico datos en el espacio PCA
    if show_labels:
        axes[2].scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap=cmap, alpha=0.7, s=50)
    else:
        axes[2].scatter(X_pca[:,0], X_pca[:,1], alpha=0.7, color="#2574F4", s=50)
    #línea horizontal
    axes[2].axhline(0, color="#c71e34", lw=2, label="PC1") 
    axes[2].axvline(0, color="#67dc96", lw=2, label="PC2")
    axes[2].set_xlabel("PC1")
    axes[2].set_ylabel("PC2")
    axes[2].set_title("Datos proyectados en el espacio PCA")
    axes[2].axis("equal")
    axes[2].legend()

    plt.suptitle(title, fontsize=16)
    plt.show()

def plotDigit(digit, cmap = "Blues", show = False):
    """Grafica un dígito de la base de datos MNIST"""

    digit = digit.reshape(28, 28)
    plt.imshow(digit, cmap=cmap)
    plt.axis('off')
    if show:
        plt.show()

def loadMNIST():
    try:
        X = np.load("cdan//datos//mnist_X_1000.npy", allow_pickle=True)
        y = np.load("cdan//datos//mnist_y_1000.npy", allow_pickle=True)
    except FileNotFoundError:
        X = np.load("datos\\mnist_X_1000.npy", allow_pickle=True)
        y = np.load("datos\\mnist_y_1000.npy", allow_pickle=True)

    return X, y

def plotSomeNumbers(n_numbers=10, figsize=(10,10), seed=42, cmap = "Blues"):
    np.random.seed(seed)

    X, y = loadMNIST()

    data = np.array([X[y == str(i)][:5] for i in range(n_numbers)])
    data = data.reshape(data.shape[0] * data.shape[1], -1)
    plt.figure(figsize=(10,5))
    for i, digit in enumerate(data):
        plt.subplot(5, 10, i+1)
        plotDigit(digit, cmap = "Blues")
    plt.tight_layout()
    plt.show()

# plotSomeNumbers()


def makeZoom(numero=0, cmap="gray", figsize=(10, 5), 
             r_start=10, r_end=15, c_start=10, c_end=15):
    """
    Muestra una imagen MNIST con un recuadro indicando la región seleccionada,
    junto a un zoom de esa región con valores de píxel superpuestos.
    """

    # --- Cargar datos MNIST (asegúrate de definir esta función o usar Keras directamente)

    X, y = loadMNIST()
    img=X[y==str(numero)][0].reshape(28,28)

    # --- Extraer región de zoom
    zoom_region = img[r_start:r_end, c_start:c_end]

    # --- Crear figura con dos subplots: original y zoom
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    # === Imagen original con recuadro ===
    axs[0].imshow(img, cmap=cmap, interpolation="nearest")
    axs[0].set_title(f"Dígito {numero} completo")
    axs[0].axis("off")

    # recuadro en rojo
    rect_w = c_end - c_start
    rect_h = r_end - r_start
    rect = plt.Rectangle((c_start, r_start), rect_w, rect_h, 
                         linewidth=2, edgecolor='red', facecolor='none')
    axs[0].add_patch(rect)

    axs[1].imshow(zoom_region, cmap=cmap, interpolation="nearest")
    axs[1].set_title("Zoom con valores de píxel")
    axs[1].axis("off")

    for i in range(zoom_region.shape[0]):
        for j in range(zoom_region.shape[1]):
            val = zoom_region[i, j]
            axs[1].text(j, i, f"{int(val)}", ha="center", va="center",
                        color="red" if val < 200 else "black", fontsize=12)

    plt.tight_layout()
    plt.show()


def extract_mean_features(X):
    features = []
    for x in X:
        img = x.reshape(28, 28)
        row_mean = img.mean(axis=1).mean()  # promedio de promedios de filas
        col_mean = img.mean(axis=0).mean()  # promedio de promedios de columnas
        features.append([col_mean, row_mean])  # x=col, y=row
    print(np.array(features).shape)
    return np.array(features)

def plotMNIST2DUsingMeans(figsize=(8,4)):
    X, y = loadMNIST()
    y = y.astype(int)

    # Extraer características 2D: (mean_col, mean_row)
    X_features = extract_mean_features(X)  # shape: (60000, 2)

    # Visualizar
    plt.figure(figsize=figsize)
    scatter = plt.scatter(X_features[:, 0], X_features[:, 1], c=y, cmap="tab10", alpha=0.5, s=10)
    plt.title("Distribución 2D usando promedio por filas y columnas")
    plt.xlabel("Promedio columnas (horizontal)")
    plt.ylabel("Promedio filas (vertical)")
    plt.colorbar(scatter, ticks=range(10), label='Dígito')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plotMNIST2DUsingMeans_px(figsize=(8,6), sample=None, point_size=10, opacity=0.5, use_webgl=True, return_fig=False):
    """
    Versión interactiva con plotly.express.
    
    Parámetros
    ----------
    figsize : tuple (ancho, alto) en pulgadas (para compatibilidad con tu firma).
    sample : int or None
        Si no es None, muestrea esa cantidad de puntos al azar para acelerar el render.
    point_size : tamaño del marcador.
    opacity : opacidad de los puntos.
    use_webgl : bool
        Si True usa render WebGL (muy recomendado para muchos puntos).
    return_fig : bool
        Si True devuelve la figura en lugar de mostrarla.
    """
    # Cargar datos
    X, y = loadMNIST()
    y = y.astype(int)

    # Extraer características 2D: (mean_col, mean_row)
    X_features = extract_mean_features(X)  # shape: (N, 2)

    # Muestreo opcional para rendimiento
    N = X_features.shape[0]
    if sample is not None and sample < N:
        idx = np.random.default_rng().choice(N, size=sample, replace=False)
        X_features = X_features[idx]
        y = y[idx]

    # Armar DataFrame para plotly
    df = pd.DataFrame({
        "Promedio columnas (horizontal)": X_features[:, 0],
        "Promedio filas (vertical)": X_features[:, 1],
        "Dígito": y.astype(str)  # string => paleta cualitativa (discreta)
    })

    # Tamaño en píxeles aproximado (pulgadas * 100 px)
    width = int(figsize[0] * 100)
    height = int(figsize[1] * 100)

    fig = px.scatter(
        df,
        x="Promedio columnas (horizontal)",
        y="Promedio filas (vertical)",
        color="Dígito",
        category_orders={"Dígito": [str(i) for i in range(10)]},
        color_discrete_sequence=px.colors.qualitative.Set3,  # similar a matplotlib tab10
        hover_data={
            "Dígito": True,
            "Promedio columnas (horizontal)": ":.3f",
            "Promedio filas (vertical)": ":.3f",
        },
        width=width,
        height=height,
        render_mode="webgl" if use_webgl else "auto",
        title="Distribución 2D usando promedio por filas y columnas",
    )

    # Estética y marcadores
    fig.update_traces(marker=dict(size=point_size, opacity=opacity))
    fig.update_layout(
        template="simple_white",
        legend_title_text="Dígito",
    )

    if return_fig:
        return fig
    fig.show()

def plotMINST2dUsingPCA(figsize=(8,4)):
    X, y = loadMNIST()
    y = y.astype(int)

    # Aplicar PCA para reducir a 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Visualizar
    plt.figure(figsize=figsize)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="tab10", alpha=0.3, s=20)
    plt.title("Distribución 2D usando PCA")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.colorbar(scatter, ticks=range(10), label='Dígito')
    plt.grid(True)
    plt.tight_layout()
    plt.axis("off")
    plt.show()


def plotMNIST2DUsingPCA_px(
    figsize=(12, 8),
    sample=None,
    random_state=0,
    point_size=10,
    opacity=0.6,
    use_webgl=True,
    whiten=False,
    svd_solver="auto",  # "auto" | "full" | "randomized"
    show_axes=False,
    return_fig=False,
    return_pca=False
):
    """
    Visualización interactiva 2D de MNIST con PCA usando plotly.express.

    Parámetros
    ----------
    figsize : tuple
        Tamaño (pulgadas) para aproximar width/height del canvas.
    sample : int or None
        Si se especifica, muestrea esa cantidad de ejemplos para acelerar el render.
    random_state : int
        Semilla para muestreo y PCA (si 'svd_solver' == 'randomized').
    point_size : int
        Tamaño de los marcadores.
    opacity : float
        Opacidad de los puntos (0–1).
    use_webgl : bool
        Si True usa render WebGL (recomendado para >10k puntos).
    whiten : bool
        Si True, aplica 'whitening' en PCA (normaliza varianza de cada PC).
    svd_solver : str
        'auto' | 'full' | 'randomized'. 'randomized' acelera para alta dimensión.
    show_axes : bool
        Si False, oculta ejes (equivalente a plt.axis('off')).
    return_fig : bool
        Si True, devuelve la figura en lugar de mostrarla.
    return_pca : bool
        Si True, devuelve también el objeto PCA ajustado.

    Retorna
    -------
    fig (opcional), pca (opcional)
    """
    # Cargar datos
    X, y = loadMNIST()
    y = y.astype(int)

    # Muestreo opcional
    N = X.shape[0]
    if sample is not None and sample < N:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(N, size=sample, replace=False)
        X = X[idx]
        y = y[idx]

    # PCA -> 2D
    pca = PCA(
        n_components=2,
        whiten=whiten,
        svd_solver=svd_solver,
        random_state=(random_state if svd_solver == "randomized" else None)
    )
    X_pca = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_

    # DataFrame para plotly
    df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Dígito": y.astype(str),
        "idx": np.arange(len(y))
    })

    # Tamaño aproximado en píxeles
    width = int(figsize[0] * 100)
    height = int(figsize[1] * 100)

    title = (
        "Distribución 2D usando PCA"
        f"<br><sup>Var. explicada: PC1={evr[0]*100:.1f}%, "
        f"PC2={evr[1]*100:.1f}% (total={evr.sum()*100:.1f}%)</sup>"
    )

    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color="Dígito",
        category_orders={"Dígito": [str(i) for i in range(10)]},
        color_discrete_sequence=px.colors.qualitative.T10,  # similar a 'tab10'
        hover_data={
            "idx": True,
            "Dígito": True,
            "PC1": ":.3f",
            "PC2": ":.3f",
        },
        width=width,
        height=height,
        render_mode=("webgl" if use_webgl else "auto"),
        title=title,
    )

    # Estilo de los puntos y layout
    fig.update_traces(marker=dict(size=point_size, opacity=opacity))
    fig.update_layout(template="simple_white", legend_title_text="Dígito")

    # Emular plt.axis('off') si se desea
    if not show_axes:
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)

    if return_fig and return_pca:
        return fig, pca
    if return_fig:
        return fig
    if return_pca:
        return pca

    fig.show()

def plotDigitReconstruction(num_components=None, digit=5, ndigits_to_show = 5, cmap="Blues", figsize=(12,6),
                            seed=42):
    np.random.seed(seed)
    X, y = loadMNIST()
    y = y.astype(int)
    N = X.shape[1]  # 784

    if num_components is None:
        num_components = N

    pca = PCA(n_components=num_components)
    X_proyectado = pca.fit_transform(X)

    X_reconstructed = pca.inverse_transform(X_proyectado)

    ##elijo ndigits_to_show dígitos aleatorios del dígito especificado
    digit_indices = np.where(y == digit)[0]
    selected_indices = np.random.choice(digit_indices, size=ndigits_to_show, replace=False)
    selected_original = X[selected_indices]
    selected_reconstructed = X_reconstructed[selected_indices]

    fig, axes = plt.subplots(2, ndigits_to_show, figsize=figsize)
    for i in range(ndigits_to_show):
        axes[0, i].imshow(selected_original[i].reshape(28, 28), cmap=cmap)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        if i == 0:
            axes[0, i].set_ylabel(f"Original\nN=({N})", fontsize=14)

        axes[1, i].imshow(selected_reconstructed[i].reshape(28, 28), cmap=cmap)
        # if i != 0:
            #borro labels de eje x e y
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        if i == 0:
            #agrego etiqeueta en el eje Y
            axes[1, i].set_ylabel(f"Reconstruido\nK=({num_components} PCs)", fontsize=14)
    plt.suptitle(f"Reconstrucción de dígito '{digit}' usando {num_components} componentes principales", fontsize=16)
    plt.tight_layout()
    plt.show()

def openRealEstateData():
    try:
        df = pd.read_csv('cdan//datos//real_estate.csv', encoding='utf-8')
    except FileNotFoundError:
        df = pd.read_csv('datos\\real_estate.csv', encoding='utf-8')
    return df


def plotRealEstatePCA(figsize=(10,6),point_size=3, opacity=0.7):
    df = openRealEstateData()

    df['Categoria'] = pd.qcut(df['price of unit area'], 3, labels=['1.Económica', '2.Rango-Medio', '3.Cara'])
    df['Categoria'].value_counts().sort_index()

    enc=OrdinalEncoder()
    df['Categoria enc']=enc.fit_transform(df[['Categoria']])

    fig = px.scatter_3d(df, 
                        x=df['X1 transaction date'], y=df['X2 house age'], z=df['X3 distance to the nearest MRT station'],
                        color=df['Categoria'],
                        color_discrete_sequence=['#636EFA','#EF553B','#00CC96'], 
                        hover_data=['X3 distance to the nearest MRT station', 'price of unit area'],
                        height=900, width=900
                    )

    fig.update_layout(#title_text="Scatter 3D Plot",
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5),
                    scene_camera=dict(up=dict(x=0, y=0, z=1), 
                                            center=dict(x=0, y=0, z=-0.2),
                                            eye=dict(x=-1.5, y=1.5, z=0.5)),
                                            margin=dict(l=0, r=0, b=0, t=0),
                    scene = dict(xaxis=dict(backgroundcolor='white',
                                            color='black',
                                            gridcolor='#f0f0f0',
                                            title_font=dict(size=10),
                                            tickfont=dict(size=10),
                                            ),
                                yaxis=dict(backgroundcolor='white',
                                            color='black',
                                            gridcolor='#f0f0f0',
                                            title_font=dict(size=10),
                                            tickfont=dict(size=10),
                                            ),
                                zaxis=dict(backgroundcolor='lightgrey',
                                            color='black', 
                                            gridcolor='#f0f0f0',
                                            title_font=dict(size=10),
                                            tickfont=dict(size=10),
                                            )))

    fig.update_traces(marker=dict(size=point_size, opacity=opacity))

    fig.show()

def varianceExplainedPlot(max_components=100, figsize=(10,6), pasos=30):
 
    X, y = loadMNIST()
    y = y.astype(int)
    N = X.shape[1]  # 784
    if max_components is None:
        max_components = N

    k = int(min(max_components, N))

    X_std = StandardScaler().fit_transform(X)

    #PCA
    pca = PCA(n_components=k)
    pca.fit(X_std)

    #Varianza explicada
    var_exp = pca.explained_variance_ratio_
    cum_var_exp = np.cumsum(var_exp)

    x = np.arange(1, k + 1)

    plt.figure(figsize=(8,5))
    plt.bar(x, var_exp, alpha=0.6, align="center",
            label="Varianza explicada por componente")
    plt.step(x, cum_var_exp, where="mid",
            label="Varianza acumulada", color="red")

    plt.ylabel("Proporción de varianza explicada")
    plt.xlabel("Número de componentes principales")
    plt.title(f"PCA en MNIST - $Varianza\ explicada$ - Cantidad de componentes: {k}")
    plt.xlim(0, k)  # incluye el 0 (aunque no haya barra en 0) y el máximo k
    
    ticks = np.arange(0, k + 1, pasos)
    if ticks[-1] != k:
        ticks = np.r_[ticks, k]
    plt.xticks(ticks)
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()


def houserWithPCA(n_components=2, figsize=(8,6), point_size=8, opacity=0.7):
    df = openRealEstateData()

    df['Categoria'] = pd.qcut(df['price of unit area'], 3, labels=['1.Económica', '2.Rango-Medio', '3.Cara'])
    # Check distribution
    df['Categoria'].value_counts().sort_index()

    # Print dataframe
    enc=OrdinalEncoder()
    # Encode categorical values
    df['Categoria enc']=enc.fit_transform(df[['Categoria']])

    features = df.columns[:-2]  # Exclude 'price of unit area' and 'Categoria' columns
    x = df[['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station']].values
    y = df.loc[:, ['Categoria enc']].values

    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents, columns=[f'PC{i+1}' for i in range(n_components)])
    finalDf = pd.concat([principalDf, df[['Categoria']]], axis=1)

    if n_components == 2:
        fig = px.scatter(finalDf, 
                        x='PC1', y='PC2',
                        color='Categoria',
                        color_discrete_sequence=['#636EFA','#EF553B','#00CC96'], 
                        hover_data=['PC1', 'PC2'],
                        height=600, width=800
                    )

        # Update chart looks
        fig.update_layout(#title_text="Scatter 2D Plot",
                        showlegend=True,
                        title=f"PCA en Set de datos de casa - Varianza explicada: {np.sum(pca.explained_variance_ratio_)*100:.2f}%",
                        legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5),
                        margin=dict(l=0, r=0, b=0, t=0),
                        scene = dict(xaxis=dict(backgroundcolor='white',
                                                color='black',
                                                gridcolor='#f0f0f0',
                                                title_font=dict(size=10),
                                                tickfont=dict(size=10),
                                                ),
                                    yaxis=dict(backgroundcolor='white',
                                                color='black',
                                                gridcolor='#f0f0f0',
                                                title_font=dict(size=10),
                                                tickfont=dict(size=10),
                                                ),
                                    ))

        fig.update_traces(marker=dict(size=point_size, opacity=opacity))

        ## ****************** RESUMEN PCA ******************
        print("****************** RESUMEN PCA ******************")
        print(f"Número de características: {x.shape[1]}")
        print(f"Número de muestras: {x.shape[0]}")
        print(f"Número de componentes principales: {n_components}")
        print(f"Varianza explicada: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
        print("**************************************************")

        fig.show()

def houserWithLDA(n_components=2, figsize=(8,6), point_size=8, opacity=0.7):
    df = openRealEstateData()

    df['Categoria'] = pd.qcut(df['price of unit area'], 3, labels=['1.Económica', '2.Rango-Medio', '3.Cara'])
    # Check distribution
    df['Categoria'].value_counts().sort_index()

    # Print dataframe
    enc=OrdinalEncoder()
    # Encode categorical values
    df['Categoria enc']=enc.fit_transform(df[['Categoria']])

    features = df.columns[:-2]  # Exclude 'price of unit area' and 'Categoria' columns
    X = df[['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station']].values
    y = df.loc[:, ['Categoria enc']].values.ravel()

    X = StandardScaler().fit_transform(X)

    lda = LDA(n_components=n_components, solver='eigen')
    X_lda = lda.fit_transform(X, y)
    lda_df = pd.DataFrame(data=X_lda, columns=[f'LD{i+1}' for i in range(n_components)])
    finalDf = pd.concat([lda_df, df[['Categoria']]], axis=1)

    if n_components == 2:
        fig = px.scatter(finalDf, 
                        x='LD1', y='LD2',
                        color='Categoria',
                        color_discrete_sequence=['#636EFA','#EF553B','#00CC96'], 
                        hover_data=['LD1', 'LD2'],
                        height=600, width=800
                    )

        # Update chart looks
        fig.update_layout(#title_text="Scatter 2D Plot",
                        showlegend=True,
                        title=f"LDA en Set de datos de casa - Varianza explicada: {np.sum(lda.explained_variance_ratio_)*100:.2f}%",
                        legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5),
                        margin=dict(l=0, r=0, b=0, t=0),
                        scene = dict(xaxis=dict(backgroundcolor='white',
                                                color='black',
                                                gridcolor='#f0f0f0',
                                                title_font=dict(size=10),
                                                tickfont=dict(size=10),
                                                ),
                                    yaxis=dict(backgroundcolor='white',
                                                color='black',
                                                gridcolor='#f0f0f0',
                                                title_font=dict(size=10),
                                                tickfont=dict(size=10),
                                                ),
                                    ))

        fig.update_traces(marker=dict(size=point_size, opacity=opacity))
        fig.show()
        
def classifiedUsingPCA(clasificador='SVM', num_comps=2, test_size=0.3, random_state=42,
                       show_info = True, show_cm = True, return_cm = False):
    df = openRealEstateData()

    df['Categoria'] = pd.qcut(df['price of unit area'], 3, labels=['1.Económica', '2.Rango-Medio', '3.Cara'])
    # Check distribution
    df['Categoria'].value_counts().sort_index()

    # Print dataframe
    enc=OrdinalEncoder()
    # Encode categorical values
    df['Categoria enc']=enc.fit_transform(df[['Categoria']])

    features = df.columns[:-2]  # Exclude 'price of unit area' and 'Categoria' columns
    X = df[['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station']].values
    y = df.loc[:, ['Categoria enc']].values.ravel()

    X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=num_comps, whiten=True)
    X_pca = pca.fit_transform(X)

    # Split data into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=test_size, random_state=random_state)

    if clasificador == 'SVM':
        model = SVC(kernel='linear', random_state=random_state)
    elif clasificador == 'DecisionTree':
        model = DecisionTreeClassifier(random_state=random_state)
    else:
        raise ValueError("clasificador not recognized. Use 'SVM' or 'DecisionTree'.")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    if show_info:
        print(f"clasificador: {clasificador}")
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)

    ##grafico matriz de confusión
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=enc.categories_[0], yticklabels=enc.categories_[0])
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.title(f'Matriz de Confusión - Clasificador: {clasificador} + PCA ({num_comps} componentes)')
    if show_cm:
        plt.show()
    else:
        plt.close()

    if return_cm:
        return cm


def classifiedUsingLDA(clasificador='SVM', num_comps=2, test_size=0.3, random_state=42,
                       show_info = True, show_cm = True, return_cm = False):
    df = openRealEstateData()

    df['Categoria'] = pd.qcut(df['price of unit area'], 3, labels=['1.Económica', '2.Rango-Medio', '3.Cara'])
    # Check distribution
    df['Categoria'].value_counts().sort_index()

    # Print dataframe
    enc=OrdinalEncoder()
    # Encode categorical values
    df['Categoria enc']=enc.fit_transform(df[['Categoria']])

    features = df.columns[:-2]  # Exclude 'price of unit area' and 'Categoria' columns
    X = df[['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station']].values
    y = df.loc[:, ['Categoria enc']].values.ravel()

    X = StandardScaler().fit_transform(X)

    lda = LDA(n_components=num_comps, solver='eigen')
    X_lda = lda.fit_transform(X, y)

    # Split data into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=test_size, random_state=random_state)

    if clasificador == 'SVM':
        model = SVC(kernel='linear', random_state=random_state)
    elif clasificador == 'DecisionTree':
        model = DecisionTreeClassifier(random_state=random_state)
    else:
        raise ValueError("clasificador not recognized. Use 'SVM' or 'DecisionTree'.")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    if show_info:
        print(f"clasificador: {clasificador}")
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)

    ##grafico matriz de confusión
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=enc.categories_[0], yticklabels=enc.categories_[0])
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.title(f'Matriz de Confusión - Clasificador: {clasificador} + LDA ({num_comps} componentes)')
    if show_cm:
        plt.show()
    else:
        plt.close()

    if return_cm:
        return cm
    
def compararCMs(clasificador="SVM", figsize=(12,5)):
    cm_pca = classifiedUsingPCA(clasificador=clasificador, num_comps=2, test_size=0.3, random_state=42,
                       show_info = False, show_cm = False, return_cm = True)
    cm_lda = classifiedUsingLDA(clasificador=clasificador, num_comps=2, test_size=0.3, random_state=42,
                       show_info = False, show_cm = False, return_cm = True)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    sns.heatmap(cm_pca, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['1.Económica', '2.Rango-Medio', '3.Cara'], 
                yticklabels=['1.Económica', '2.Rango-Medio', '3.Cara'],
                ax=axes[0])
    axes[0].set_xlabel('Predicho')
    axes[0].set_ylabel('Real')
    if clasificador == "SVM":
        axes[0].set_title(f'Matriz de Confusión - Clasificador: SVM + PCA (2 componentes)')
    else:
        axes[0].set_title(f'Matriz de Confusión - Clasificador: DecisionTree + PCA (2 componentes)')

    sns.heatmap(cm_lda, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['1.Económica', '2.Rango-Medio', '3.Cara'], 
                yticklabels=['1.Económica', '2.Rango-Medio', '3.Cara'],
                ax=axes[1])
    axes[1].set_xlabel('Predicho')
    axes[1].set_ylabel('Real')
    if clasificador == "SVM":
        axes[1].set_title(f'Matriz de Confusión - Clasificador: SVM + LDA (2 componentes)')
    else:
        axes[1].set_title(f'Matriz de Confusión - Clasificador: DecisionTree + LDA (2 componentes)')

    plt.tight_layout()
    plt.show()

def gettingFeaturesImportanceFromPCA(n_components=2, figsize=(9, 3.6),cmap="Reds"):
    """
    Usaremos un set de datos con las siguientes características:

    - product_id: identificador único de cada producto (código o SKU).
    - product_length_cm: longitud del producto en centímetros (dimensión física del ítem).
    - packaging_width_cm: ancho del empaque en centímetros (dimensión del packaging).
    - shelf_presence_score: puntaje de presencia en góndola/estantería en escala 0–10; valores mayores implican mayor visibilidad o espacio asignado.
    - sales_velocity: tasa de ventas (unidades por unidad de tiempo definida por el negocio); mide qué tan rápido se vende el producto.
    - price_usd: precio del producto en dólares estadounidenses.
    - product_category: categoría o segmento del producto (p. ej., A, B, C); Outlier indica ítems atípicos fuera de los segmentos principales.
    - customer_satisfaction: calificación de satisfacción de clientes en escala 1–5 (mayor = más satisfecho).

    Para el análisis, quitaremos las columnas product_id y product_category.
    """

    try:
        df = pd.read_csv('cdan//datos//Producto_Dataset_Segmentado.csv', encoding='utf-8')
    except FileNotFoundError:
        df = pd.read_csv('datos\\Producto_Dataset_Segmentado.csv', encoding='utf-8')

    # df = df[df["product_category"] != "Outlier"]
    ##drop columnas product_id y product_category
    df.drop(['product_id', 'product_category'], axis=1, inplace=True)
    features = df.columns
    ##quito las filas correspondientes a category == outliers

    data = df.values
    num_cols = len(features)

    scaler=StandardScaler()
    X_scaled = scaler.fit_transform(data)

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    columnas = [f"PCA{i}" for i in range(1,n_components+1)]

    pca_df = pd.DataFrame(X_pca, columns=columnas)

    loadings = pd.DataFrame(
    pca.components_.T,
    index=features,
    columns=["Cargas PC1", "Cargas PC2"])

    #magnitud absoluta
    loadings["|PC1|"] = loadings["Cargas PC1"].abs()
    loadings["|PC2|"] = loadings["Cargas PC2"].abs()

    rank_pc1 = loadings.sort_values("|PC1|", ascending=False)[["Cargas PC1", "|PC1|"]]
    rank_pc2 = loadings.sort_values("|PC2|", ascending=False)[["Cargas PC2", "|PC2|"]]

    plt.figure(figsize=figsize)
    vals = pca.components_
    im2 = plt.imshow(vals, aspect="auto", cmap=cmap)
    plt.colorbar(im2, fraction=0.046, pad=0.04)
    plt.yticks(ticks=[0, 1], labels=["PC1", "PC2"])
    plt.xticks(ticks=np.arange(num_cols), labels=features, rotation=45, ha="right")
    plt.title("Cargas (loadings) de PCA por variable")
    # anotar
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            plt.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=12)
    plt.tight_layout()
    plt.show()

def _limpiarCacheLFW(data_home=None, verbose=True):
    """
    Borra la carpeta ``lfw_home`` de la caché de scikit-learn.

    Se usa cuando la descarga del dataset quedó incompleta (el archivo .tgz truncado
    hace fallar la descompresión con un EOFError). Después de limpiar, scikit-learn
    vuelve a descargar el dataset desde cero.
    """

    lfw_home = os.path.join(get_data_home(data_home=data_home), "lfw_home")
    if os.path.isdir(lfw_home):
        shutil.rmtree(lfw_home, ignore_errors=True)
        if verbose:
            print(f"Caché de LFW eliminada: {lfw_home}")
    elif verbose:
        print(f"No había caché de LFW para eliminar en: {lfw_home}")


def loadFaces(dataset="olivetti", seed=0, shuffle=True, min_faces_per_person=70,
              resize=0.4, funneled=True, color=False, data_home=None, verbose=True,
              force_download=False):
    """
    Descarga (o carga desde la caché local) un dataset de caras.

    Hay dos opciones:

    - ``"olivetti"`` (por defecto): *Olivetti faces* (AT&T), 400 imágenes de 64x64 de
      40 personas. Pesa ~15MB y es el dataset que usa el ejemplo de scikit-learn
      "Faces dataset decompositions". Es el recomendado para la clase: descarga rápida.
    - ``"lfw"``: *Labeled Faces in the Wild*, caras reales tomadas de noticias.
      Es mucho más pesado (~233MB la primera vez) pero más "realista".

    Parámetros
    ----------
    dataset : str
        "olivetti" o "lfw".
    seed : int
        Semilla para el mezclado de las imágenes (el ejemplo de scikit-learn usa 0).
    shuffle : bool
        Mezcla las imágenes al cargarlas, para que las caras no queden ordenadas
        por persona.
    min_faces_per_person : int
        Sólo para LFW: se conservan las personas con al menos esta cantidad de fotos.
        Valores más chicos => más personas y más imágenes (dataset más pesado).
    resize : float
        Sólo para LFW: factor de escalado. 0.4 => imágenes de 50x37 píxeles.
    funneled : bool
        Sólo para LFW: usa la versión alineada ("funneled") de las imágenes.
    color : bool
        Sólo para LFW: si es True devuelve las imágenes en RGB, si no en escala de grises.
    data_home : str o None
        Carpeta donde guardar/buscar los datos descargados.
    verbose : bool
        Imprime un resumen del dataset cargado.
    force_download : bool
        Sólo para LFW: borra la caché y vuelve a descargar todo desde cero.
        Útil cuando una descarga previa quedó incompleta.

    Retorna
    -------
    dict con las claves:
        X            : matriz (n_muestras, n_features) con cada cara "aplanada".
        X_centered   : X con centrado global (por píxel) y local (por imagen),
                       tal como se hace en el ejemplo de scikit-learn.
        y            : etiquetas (índice de la persona).
        target_names : nombres de las personas.
        images       : imágenes con su forma original (n_muestras, alto, ancho).
        image_shape  : tupla (alto, ancho) de cada imagen.
        n_samples, n_features, n_classes : tamaños del dataset.
        nombre       : nombre del dataset cargado.
    """

    dataset = dataset.lower()

    if dataset == "olivetti":
        rng = np.random.RandomState(seed)
        faces = fetch_olivetti_faces(shuffle=shuffle, random_state=rng, data_home=data_home)
        # Olivetti no trae nombres: las personas se identifican por un número.
        target_names = np.array([f"Persona {i}" for i in range(faces.target.max() + 1)])
        nombre = "Olivetti faces"

    elif dataset == "lfw":
        if force_download:
            _limpiarCacheLFW(data_home, verbose=verbose)

        try:
            faces = fetch_lfw_people(min_faces_per_person=min_faces_per_person, resize=resize,
                                     funneled=funneled, color=color, data_home=data_home)
        except (EOFError, tarfile.ReadError, OSError) as error:
            # Típicamente pasa cuando la descarga (~233MB) se corta y el .tgz queda truncado.
            print(f"La descarga anterior de LFW quedó incompleta o corrupta ({type(error).__name__}: {error}).")
            print("Limpiando la caché y reintentando la descarga...")
            _limpiarCacheLFW(data_home, verbose=verbose)
            faces = fetch_lfw_people(min_faces_per_person=min_faces_per_person, resize=resize,
                                     funneled=funneled, color=color, data_home=data_home)

        target_names = faces.target_names
        nombre = "Labeled Faces in the Wild (LFW)"

    else:
        raise ValueError(f"dataset debe ser 'olivetti' o 'lfw', se recibió '{dataset}'")

    X = faces.data.astype(np.float64)
    y = faces.target
    images = faces.images
    n_samples, n_features = X.shape
    image_shape = images.shape[1:]

    if dataset == "lfw" and shuffle:
        rng = np.random.RandomState(seed)
        orden = rng.permutation(n_samples)
        X, y, images = X[orden], y[orden], images[orden]

    # Centrado global: a cada píxel le resto su valor promedio en todo el dataset.
    X_centered = X - X.mean(axis=0)
    # Centrado local: a cada imagen le resto su propio brillo promedio.
    X_centered -= X_centered.mean(axis=1).reshape(n_samples, -1)

    data = {"X": X,
            "X_centered": X_centered,
            "y": y,
            "target_names": target_names,
            "images": images,
            "image_shape": image_shape,
            "n_samples": n_samples,
            "n_features": n_features,
            "n_classes": len(target_names),
            "nombre": nombre}

    if verbose:
        print(f"Dataset '{nombre}' cargado:")
        print(f" - Imágenes: {n_samples}")
        print(f" - Tamaño de cada imagen: {image_shape[0]}x{image_shape[1]} = {n_features} features")
        print(f" - Personas distintas: {data['n_classes']}")

    return data


def _facesGallery(images, image_shape, titles=None, n_row=2, n_col=3, cmap="gray",
                  figsize=None, title=None, fontsize=16, titles_fontsize=9,
                  symmetric_scale=True, show_colorbar=True):
    """
    Helper interno: dibuja una grilla de caras.

    Replica la función ``plot_gallery()`` del ejemplo de scikit-learn
    "Faces dataset decompositions": escala de grises simétrica alrededor de 0,
    ejes apagados y una barra de color horizontal debajo de la grilla.
    """

    if figsize is None:
        figsize = (2.0 * n_col, 2.3 * n_row)

    fig, axs = plt.subplots(nrows=n_row, ncols=n_col, figsize=figsize,
                            facecolor="white", constrained_layout=True)
    axs = np.atleast_1d(axs)
    try:
        fig.get_layout_engine().set(w_pad=0.01, h_pad=0.02, hspace=0, wspace=0)
    except AttributeError:
        # matplotlib < 3.6 usa otra API para ajustar el constrained_layout
        fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02, hspace=0, wspace=0)
    fig.set_edgecolor("black")
    if title is not None:
        fig.suptitle(title, size=fontsize)

    im = None
    for i, ax in enumerate(axs.flat):
        if i >= len(images):
            ax.axis("off")
            continue
        vec = images[i]
        if symmetric_scale:
            # escala simétrica alrededor de 0: así se ven las caras en el ejemplo
            vmax = max(vec.max(), -vec.min())
            im = ax.imshow(vec.reshape(image_shape), cmap=cmap, interpolation="nearest",
                           vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(vec.reshape(image_shape), cmap=cmap, interpolation="nearest")
        ax.axis("off")
        if titles is not None and i < len(titles):
            ax.set_title(titles[i], fontsize=titles_fontsize)

    if show_colorbar and im is not None:
        fig.colorbar(im, ax=axs, orientation="horizontal", shrink=0.99, aspect=40, pad=0.01)

    plt.show()


def plotSomeFaces(faces, n_row=2, n_col=3, cmap="gray", figsize=None,
                  title="Caras del dataset", seed=42, random=False,
                  use_centered=True, show_names=False, show_colorbar=True,
                  fontsize=16, titles_fontsize=9):
    """
    Grafica una grilla de caras del dataset devuelto por ``loadFaces()``.

    Con los valores por defecto reproduce la figura "Faces from dataset" del ejemplo
    de scikit-learn: las primeras 6 caras ya centradas, en escala de grises simétrica
    y con barra de color.

    Parámetros
    ----------
    faces : dict
        Dataset devuelto por ``loadFaces()``.
    n_row, n_col : int
        Cantidad de filas y columnas de la grilla (se muestran n_row*n_col caras).
    cmap : str
        Mapa de colores de matplotlib ("gray", "Blues", "magma", etc.).
    figsize : tupla o None
        Tamaño de la figura. Si es None se calcula a partir de n_row y n_col.
    title : str o None
        Título general de la figura.
    seed : int
        Semilla para la selección aleatoria de caras (sólo si random=True).
    random : bool
        Si es True elige caras al azar, si es False toma las primeras del dataset
        (como en el ejemplo de scikit-learn).
    use_centered : bool
        Si es True grafica las imágenes centradas (X_centered), como en el ejemplo.
        Si es False grafica las imágenes originales.
    show_names : bool
        Muestra el nombre de la persona arriba de cada cara (útil con LFW).
    show_colorbar : bool
        Agrega la barra de color horizontal debajo de la grilla.
    fontsize, titles_fontsize : int
        Tamaño de fuente del título general y de los títulos de cada cara.
    """

    np.random.seed(seed)

    X = faces["X_centered"] if use_centered else faces["X"]
    n_faces = min(n_row * n_col, faces["n_samples"])

    if random:
        indices = np.random.choice(faces["n_samples"], size=n_faces, replace=False)
    else:
        indices = np.arange(n_faces)

    titles = None
    if show_names:
        titles = [faces["target_names"][faces["y"][i]] for i in indices]

    _facesGallery(X[indices], faces["image_shape"], titles=titles, n_row=n_row, n_col=n_col,
                  cmap=cmap, figsize=figsize, title=title, fontsize=fontsize,
                  titles_fontsize=titles_fontsize, symmetric_scale=use_centered,
                  show_colorbar=show_colorbar)


def plotEigenfaces(faces, n_row=2, n_col=3, cmap="gray", figsize=None,
                   title="Eigenfaces - PCA usando SVD randomizado", whiten=True,
                   seed=0, fontsize=16, show_colorbar=True, return_pca=False):
    """
    Ajusta PCA sobre las caras centradas y grafica las primeras componentes
    principales como imágenes: las famosas *eigenfaces*.

    Reproduce la figura "Eigenfaces - PCA using randomized SVD" del ejemplo de
    scikit-learn. Cada eigenface es una "dirección" del espacio de caras: sumándolas
    con distintos pesos se puede reconstruir cualquier cara del dataset.

    Parámetros
    ----------
    faces : dict
        Dataset devuelto por ``loadFaces()``.
    n_row, n_col : int
        Grilla de eigenfaces a mostrar (se ajustan n_row*n_col componentes).
    cmap, figsize, title, fontsize, show_colorbar
        Igual que en ``plotSomeFaces()``.
    whiten : bool
        Parámetro ``whiten`` de PCA (el ejemplo de scikit-learn usa True).
    seed : int
        Semilla del solver randomizado.
    return_pca : bool
        Si es True devuelve el objeto PCA ya ajustado.
    """

    n_components = n_row * n_col

    pca = PCA(n_components=n_components, svd_solver="randomized", whiten=whiten,
              random_state=seed)
    pca.fit(faces["X_centered"])

    _facesGallery(pca.components_[:n_components], faces["image_shape"],
                  titles=[f"PC{i+1}" for i in range(n_components)],
                  n_row=n_row, n_col=n_col, cmap=cmap, figsize=figsize, title=title,
                  fontsize=fontsize, symmetric_scale=True, show_colorbar=show_colorbar)

    if return_pca:
        return pca


def plotFaceReconstruction(faces, num_components=50, nfaces_to_show=5, cmap="gray",
                           figsize=None, seed=42, title=None, show_eigenfaces=False,
                           show_names=False, verbose=True, return_pca=False):
    """
    Aplica PCA sobre el dataset de caras y compara las caras originales contra su
    reconstrucción usando sólo ``num_components`` componentes principales.

    Es el análogo de ``plotDigitReconstruction()`` pero con caras, y sirve para mostrar
    cómo PCA comprime la información: con unos pocos cientos de números se puede
    reconstruir una imagen de miles de píxeles.

    Parámetros
    ----------
    faces : dict
        Dataset devuelto por ``loadFaces()``.
    num_components : int o None
        Cantidad de componentes principales a retener. Si es None usa todas las posibles.
    nfaces_to_show : int
        Cantidad de caras a mostrar (columnas de la figura).
    cmap : str
        Mapa de colores de matplotlib.
    figsize : tupla o None
        Tamaño de la figura. Si es None se calcula según la cantidad de caras y filas.
    seed : int
        Semilla para elegir las caras a mostrar.
    title : str o None
        Título general. Si es None se arma uno automáticamente.
    show_eigenfaces : bool
        Agrega una tercera fila con las primeras "eigenfaces" (componentes principales).
    show_names : bool
        Muestra el nombre de la persona sobre cada cara original (útil con LFW).
    verbose : bool
        Imprime la varianza explicada y el factor de compresión logrado.
    return_pca : bool
        Si es True devuelve el objeto PCA ya ajustado.
    """

    np.random.seed(seed)

    X = faces["X"]
    image_shape = faces["image_shape"]
    n_samples, n_features = X.shape

    if num_components is None:
        num_components = min(n_samples, n_features)
    num_components = int(min(num_components, n_samples, n_features))

    pca = PCA(n_components=num_components, svd_solver="randomized", random_state=seed)
    X_proyectado = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_proyectado)

    nfaces_to_show = min(nfaces_to_show, n_samples)
    if show_eigenfaces:
        nfaces_to_show = min(nfaces_to_show, num_components)
    selected_indices = np.random.choice(n_samples, size=nfaces_to_show, replace=False)

    n_row = 3 if show_eigenfaces else 2
    if figsize is None:
        figsize = (2.2 * nfaces_to_show, 2.6 * n_row)
    fig, axes = plt.subplots(n_row, nfaces_to_show, figsize=figsize, facecolor="white")
    axes = np.atleast_2d(axes)

    for i, idx in enumerate(selected_indices):
        axes[0, i].imshow(X[idx].reshape(image_shape), cmap=cmap, interpolation="nearest")
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        if show_names:
            axes[0, i].set_title(faces["target_names"][faces["y"][idx]], fontsize=9)
        if i == 0:
            axes[0, i].set_ylabel(f"Original\nN=({n_features})", fontsize=13)

        axes[1, i].imshow(X_reconstructed[idx].reshape(image_shape), cmap=cmap,
                          interpolation="nearest")
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        if i == 0:
            axes[1, i].set_ylabel(f"Reconstruido\nK=({num_components} PCs)", fontsize=13)

    if show_eigenfaces:
        for i in range(nfaces_to_show):
            eigenface = pca.components_[i]
            vmax = max(eigenface.max(), -eigenface.min())
            axes[2, i].imshow(eigenface.reshape(image_shape), cmap=cmap,
                              interpolation="nearest", vmin=-vmax, vmax=vmax)
            axes[2, i].set_xticks([])
            axes[2, i].set_yticks([])
            axes[2, i].set_title(f"PC{i+1}", fontsize=9)
            if i == 0:
                axes[2, i].set_ylabel("Eigenfaces", fontsize=13)

    if title is None:
        title = f"Reconstrucción de caras usando {num_components} componentes principales"
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

    if verbose:
        var_explicada = pca.explained_variance_ratio_.sum() * 100
        print(f"Componentes usadas: {num_components} de {n_features} features originales")
        print(f"Varianza explicada acumulada: {var_explicada:.2f}%")
        print(f"Compresión: {n_features / num_components:.1f}x menos números por imagen")

    if return_pca:
        return pca

# data = loadFaces()
# plotSomeFaces(data)
# plotEigenfaces(data)
# plotFaceReconstruction(data, num_components=200)
