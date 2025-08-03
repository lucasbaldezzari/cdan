import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression


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
                legend=True, sharex=False, figsize=figsize)
    plt.title("Distribución de las casas en California", fontsize=12)
    try:
        california_img = plt.imread("cdan/imagenes/california.png")
    except FileNotFoundError:
        california_img = plt.imread("imagenes\\california.png")
    axis = -124.55, -113.95, 32.45, 42.05
    plt.axis(axis)
    plt.imshow(california_img, extent=axis)

    plt.show()

from sklearn.utils.validation import check_array, check_is_fitted

class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True):  # no *args or **kwargs!
        self.with_mean = with_mean

    def fit(self, X, y=None):  # y is required even though we don't use it
        X = check_array(X)  # checks that X is an array with finite float values
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1]  # every estimator stores this in fit()
        return self  # always return self!

    def transform(self, X):
        check_is_fitted(self)  # looks for learned attributes (with trailing _)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10,
                              random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]
    
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

def makeLinearRegressionPipeline(housing):
    housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
    housing_num = housing.select_dtypes(include=[np.number])

    num_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("standardize", StandardScaler()),
    ])

    num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
               "total_bedrooms", "population", "households", "median_income"]
    cat_attribs = ["ocean_proximity"]

    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"))

    preprocessing = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])

    preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object)),)

    log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())
    cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
    default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                        StandardScaler())
    preprocessing = ColumnTransformer([
            ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
            ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
            ("people_per_house", ratio_pipeline(), ["population", "households"]),
            ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                                "households", "median_income"]),
            ("geo", cluster_simil, ["latitude", "longitude"]),
            ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
        ],
        remainder=default_num_pipeline)  # one column remaining: housing_median_age
    
    lin_reg = make_pipeline(preprocessing, LinearRegression())

    return lin_reg