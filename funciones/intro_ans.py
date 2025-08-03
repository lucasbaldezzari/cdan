import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_housingdata():
    try:
        return pd.read_csv("cdan//datos//housing.csv")
    except FileNotFoundError:
        # print("El archivo no se encontró, al parecer no estamos en colab. Se carga desde la carpeta local.")
        return pd.read_csv("datos//housing.csv")
    

def plot_histograms(housing, figsize=(12, 6)):

    housing.hist(bins=50, figsize=figsize,color='#fab700')
    plt.suptitle("Histogramas de las características de las casas", fontsize=12)
    # plt.tight_layout()
    
    plt.show()

def makeSimpleScatterHousing(housing, figsize=(12, 6)):
    housing.plot(kind="scatter", x="longitude", y="latitude", grid=True)
    ##agrego título y etiquetas
    plt.title("Distribución de las casas en California", fontsize=12)
    plt.xlabel("Longitud", fontsize=10)
    plt.ylabel("Latitud", fontsize=10)
    plt.show()

def makeBetterScatterHousing(housing, figsize=(12, 6)):
    housing_renamed = housing.rename(columns={
    "latitude": "Latitud", "longitude": "Longitud",
    "population": "Población",
    "median_house_value": "Valor de la casa mediana (USD)"})
    housing_renamed.plot(
                kind="scatter", x="Longitud", y="Latitud",
                s=housing_renamed["Población"] / 100, label="Población",
                c="Valor de la casa mediana (USD)", cmap="jet", colorbar=True,
                legend=True, sharex=False, figsize=(10, 7))
    plt.title("Distribución de las casas en California", fontsize=12)
    try:
        california_img = plt.imread("cdan/imagenes/california.png")
    except FileNotFoundError:
        california_img = plt.imread("imagenes\\california.png")
    axis = -124.55, -113.95, 32.45, 42.05
    plt.axis(axis)
    plt.imshow(california_img, extent=axis)

    plt.show()