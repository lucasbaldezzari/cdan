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


def plotExample2(data, labels, centers, figsize=(10, 6)):
    """
    Muestra un gráfico interactivo de los datos de fake iris antes y después de aplicar KMeans.
    """

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError("plotExample2 requiere plotly. Instalalo con `pip install plotly`.") from exc

    x_col = 'media_visitas_diarias'
    y_col = 'unidades_vendidas_mensuales'

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=('Datos originales', 'Puntos agrupados por KMeans')
    )

    fig.add_trace(
        go.Scatter(
            x=data[x_col],
            y=data[y_col],
            mode='markers',
            marker=dict(color="#3ea25c"),
            name='Datos originales',
            showlegend=False
        ),
        row=1,
        col=1
    )

    plot_data = data.copy()
    plot_data['_plotExample2_label'] = labels
    palette = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3"]
    symbols = ["circle", "diamond", "square", "cross", "x", "triangle-up", "triangle-down", "star"]

    for index, (label, group) in enumerate(plot_data.groupby('_plotExample2_label', sort=False)):
        fig.add_trace(
            go.Scatter(
                x=group[x_col],
                y=group[y_col],
                mode='markers',
                marker=dict(
                    color=palette[index % len(palette)],
                    symbol=symbols[index % len(symbols)]
                ),
                name=str(label)
            ),
            row=1,
            col=2
        )

    width = int(figsize[0] * 100)
    height = int(figsize[1] * 100)
    fig.update_xaxes(title_text=x_col, row=1, col=1)
    fig.update_yaxes(title_text=y_col, row=1, col=1)
    fig.update_xaxes(title_text=x_col, row=1, col=2)
    fig.update_yaxes(title_text=y_col, row=1, col=2)
    fig.update_layout(width=width, height=height)

    try:
        import google.colab  # noqa: F401
        fig.show(renderer="colab")
    except ImportError:
        fig.show()
