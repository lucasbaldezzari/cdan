import seaborn as sns
import matplotlib.pyplot as plt

def plotExample1(data, labels, centers, figsize=(10, 6)):
    """
    Muestra un gráfico de los datos de fake iris antes y después de aplicar KMeans.
    """

    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=data['media_visitas_diarias'], y=data['unidades_vendidas_mensuales'], color="#3ea25c")
    plt.title('Datos originales')
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=data['media_visitas_diarias'], y=data['unidades_vendidas_mensuales'],
                    hue=labels, palette='Set2', style=labels)
    plt.title('Puntos agrupados por KMeans')
    plt.legend()
    plt.tight_layout()
    plt.show()