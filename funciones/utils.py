import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
#importo kmeans
from sklearn.cluster import KMeans
#importo dbscan
from sklearn.cluster import DBSCAN
#importo para poder crear data moons y data circles
from sklearn.datasets import make_moons, make_circles

def transform_and_get_iris():
    """
    Columna original	Nombre simulado	            Descripción de negocio
    sepal length (cm)	media_visitas_diarias	    Cantidad promedio de visitas diarias al producto (popularidad)
    sepal width (cm)	precio_unitario	            Precio en USD del producto
    petal length (cm)	unidades_vendidas_mensual	Unidades vendidas por mes
    petal width (cm)	valoracion_media	        Valoración media de usuarios (0 a )

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

    return df