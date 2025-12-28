# Manejo de archivos y sistema
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

## Clustering
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score

# Funciones y configuraciones propias:
import src.utils as uts
from parameters.config import *

def main():
    validation_periods = uts.get_validation_periods(TEST_PERIOD, 1)

    for validation_period in validation_periods:
        path_validation = os.path.join(f"./data/output/validation/{CATEGORY}/{CHANNEL}/{validation_period}")
        dataset_predict = uts.load_pickle(os.path.join(path_validation, "scoring_lgbm.pkl"))

        X = get_X_for_clustering(dataset_predict)

        # Selección de k óptimo:
        best_k, k_ranking = select_k(X, k_min=2, k_max=10)
        best_k = 5

        print(f'Número óptimo de clusters (k) seleccionado: {best_k}')
        # Entrenamiento de KMeans con k óptimo:
        labels = fit_kmeans(X, n_clusters=best_k)
        # Asignación de labels al dataset original:
        dataset_predict['cluster'] = labels
        cluster_vars = ['crecimiento_fut_est', 'size_base']

        boxplots_by_cluster(
            dataset_predict,
            vars_to_plot=cluster_vars,
            title_prefix=f"KMeans (k={best_k}) – "
        )

        # persistencia de resultados:
        uts.save_pickle(dataset_predict, os.path.join(path_validation, 'clusters_kmeans.pkl'))









def filter_join_data(dataset_predict, dataset_processing) -> pd.DataFrame:
    '''
    Filtra y une los datasets necesarios para el proceso de clustering.
    inpunts
        dataset_processing: pd.DataFrame, contiene el dataset generado en el pre-processing
        dataset_predict: pd.DataFrame, contiene las predicciones generadas por el modelo LightGBM
    output:
        pd.DataFrame, dataset filtrado y unido para clustering
    '''

    # 1. Filtrado de columnas relevantes para set general:
    cols_processing = ['id_cliente','id_periodo', # identificadores y llave primaria.
                       'id_barrio','id_comuna','indice_gse', # información geográfica y socioeconómica del cliente.
                       'canal','segmento', # segmentación actual.
                       'compra', 'frecuency', 'recency' # RFMScore variables.
                       ]

    df_processing_filtered = dataset_processing[cols_processing].copy()
    # 2. Filtrado de columnas relevantes para set de predicciones:

    cols_predict = ['id_cliente','id_periodo', # identificadores y llave primaria.
                    'volumen_sem_fut_est' # variable objetivo estimada, completa el RFMSet
                    ]
    df_predict_filtered = dataset_predict[cols_predict].copy()

    # 3. Unión de ambos datasets por llave primaria:
    df_merged = df_predict_filtered.merge(df_processing_filtered,
                                         on=['id_cliente','id_periodo'],
                                         how='inner')

    return df_merged

def get_X_for_clustering(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Prepara el dataset para el proceso de clustering, seleccionando y escalando las variables relevantes.
    inputs:
        df: pd.DataFrame, dataset filtrado y unido para clustering
    output:
        pd.DataFrame, dataset preparado para clustering
    '''

    #cols_to_drop = ['id_cliente','id_periodo','id_barrio', 'id_comuna','canal','segmento','indice_gse', 'volumen_sem_fut_est']
    #X = df.drop(cols_to_drop, axis=1)
    X = df[['crecimiento_fut_est', 'size_base']]

    ## Regularización y escalamiento de variables:

    # Escalamiento de variables:
    list_num = []
    list_bool = []

    for col in X.columns:
        if X[col].dtype in [np.float64, np.int64]:
            list_num.append(col)
        elif X[col].dtype == bool:
            list_bool.append(col)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), list_num),
            ('bool', 'passthrough', list_bool)
        ])
    X_scaled = preprocessor.fit_transform(X)
    return X_scaled

# -----------------------------
# Entrenar KMeans y elegir k
# -----------------------------

def select_k_silhouette(X: np.ndarray, k_min: int, k_max: int) -> int:
    '''
    Selecciona el número óptimo de clusters (k) utilizando el coeficiente de silhouette.
    Para lo anterior, se selecciona el k que genera el coef silhouette más alto entre un rango definido
    inputs:
        X: np.ndarray, datos estandarizados para clustering
        k_min: int, número mínimo de clusters a evaluar
        k_max: int, número máximo de clusters a evaluar
    output:
        int, número óptimo de clusters (k)
    '''
    best_k = k_min
    best_silhouette = -1

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k,
                        random_state= int(RANDOM_STATE),
                        n_init=10,
                        init='k-means++',
                        algorithm='lloyd')
        labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, labels)

        if silhouette_avg > best_silhouette:
            best_silhouette = silhouette_avg
            best_k = k

    return best_k

def select_k(X: np.ndarray, k_min: int = 2, k_max: int = 10, random_state: int = 42,
    n_init: int = 10,
):
    """
    Evalúa KMeans para distintos k y devuelve:

    - best_k: k con mayor silhouette
    - metrics: DataFrame con ranking completo de k
               ordenado por silhouette (desc)
               incluye inertia para elbow plots
    """

    rows = []

    for k in range(k_min, k_max + 1):
        km = KMeans(
            n_clusters=k,
            random_state=int(random_state),
            n_init=n_init,
            init="k-means++",
            algorithm="lloyd",
        )

        labels = km.fit_predict(X)

        # Silhouette solo es válido si hay >1 cluster y no colapsa
        sil = silhouette_score(X, labels)

        rows.append({
            "k": k,
            "silhouette": sil,
            "inertia": km.inertia_,
        })

    metrics = pd.DataFrame(rows)

    # Ranking por silhouette
    metrics = metrics.sort_values(
        by="silhouette",
        ascending=False
    ).reset_index(drop=True)

    best_k = int(metrics.loc[0, "k"])

    return best_k, metrics

def fit_kmeans(X: np.ndarray, n_clusters: int) -> KMeans:
    '''
    Ajusta el modelo KMeans con el número de clusters especificado.
    inputs:
        X: np.ndarray, datos estandarizados para clustering
        n_clusters: int, número de clusters
    output:
        KMeans, modelo ajustado, del clustering perteneciente.
    '''
    kmeans = KMeans(n_clusters=n_clusters,
                    random_state= int(RANDOM_STATE),
                    n_init=10,
                    init='k-means++',
                    algorithm='lloyd')
    kmeans.fit(X)
    labels = kmeans.predict(X)

    return labels

def diagnose_clustering_data(df: pd.DataFrame):
    """
    Analiza el dataframe para detectar problemas que causan inestabilidad en KMeans.
    """
    print("=== Diagnóstico de Datos para Clustering ===")

    # 1. Identificar columnas numéricas
    cols_num = df.select_dtypes(include=[np.number]).columns.tolist()

    # 2. Verificar NaNs e Infinitos
    has_nan = df[cols_num].isna().any().any()
    has_inf = np.isinf(df[cols_num].values).any()

    print(f"¿Existen valores NaN?: {has_nan}")
    print(f"¿Existen valores Infinitos?: {has_inf}")

    if has_nan or has_inf:
        print("--> RECOMENDACIÓN: Limpiar NaNs e Infinitos antes de escalar.")

    # 3. Análisis de Outliers y Rangos Extremos
    print("\n--- Análisis de Rangos (Posibles Outliers) ---")
    stats = []
    for col in cols_num:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

        stats.append({
            'Variable': col,
            'Min': df[col].min(),
            'Max': df[col].max(),
            'Media': df[col].mean(),
            'Outliers (IQR)': outliers_count,
            'Max/Min Ratio': 'Inf' if df[col].min() == 0 else df[col].max() / (df[col].min() + 1e-9)
        })

    stats_df = pd.DataFrame(stats)
    print(stats_df.to_string(index=False))

    # 4. Verificación de Varianza Cero
    constant_cols = [col for col in cols_num if df[col].nunique() <= 1]
    if constant_cols:
        print(f"\nADVERTENCIA: Columnas con varianza cero (constantes): {constant_cols}")
        print("--> RECOMENDACIÓN: Eliminar estas columnas, no aportan al clustering.")

    print("\n============================================")


def boxplots_by_cluster(
    df: pd.DataFrame,
    vars_to_plot: list[str],
    cluster_col: str = "cluster",
    title_prefix: str = "",
    figsize=(10, 5),
):
    """
    Boxplots por cluster para variables seleccionadas.
    Pensado para análisis post-clustering.
    """

    for var in vars_to_plot:
        plt.figure(figsize=figsize)

        sns.boxplot(
            data=df,
            x=cluster_col,
            y=var,
            showfliers=False  # evita que outliers te rompan la lectura
        )

        plt.title(
            f"{title_prefix}Distribución de {var} por cluster",
            fontsize=14,
            pad=12
        )
        plt.xlabel("Cluster")
        plt.ylabel(var)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()