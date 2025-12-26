# Manejo de archivos y sistema
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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
    ## importe de datos y periodos de clustering:
    clustering_periodos = [202406]
    dataset_predict = uts.load_pickle(f'./data/output/pickle/predictions_lgbm_{CATEGORY}_{TEST_PERIOD}_vf.pickle') #Predicciones del modelo de LightGBM
    dataset_processing = uts.load_pickle(f'./data/output/pickle/dataset_{CATEGORY}.pickle') #Dataset general con variables originales, del pre-processing
    # joins y filtrados previos (ver funciones adicionales más abajo):
    df = filter_join_data(dataset_predict, dataset_processing)

    ## incremento mayor a 0 en variable objetivo para evitar problemas con log(0):

    df = df[df['volumen_sem_fut_est'] > 0].copy()
    df_filtered = df[df['id_periodo'].isin(clustering_periodos)].copy()
    df_filtered['log_volumen_sem_fut_est'] = df_filtered['volumen_sem_fut_est'].apply(lambda x: np.log1p(x))

    # X_train y escalamiento de variables:
    X = get_X_for_clustering(df_filtered)

    # Selección de k óptimo:
    k_optimo = 5 ## Modificar si se desea re-ejecutar la selección de k
    print(f'Número óptimo de clusters (k) seleccionado: {k_optimo}')

    # Entrenamiento de KMeans con k óptimo:
    labels = fit_kmeans(X, n_clusters=k_optimo)
    # Asignación de labels al dataset original:
    df_filtered['cluster'] = labels

    # persistencia de resultados:
    uts.save_pickle(df_filtered, f'./data/output/pickle/clusters_kmeans_{CATEGORY}_{TEST_PERIOD}.pickle')


# -----------------------------
# Funciones adicionales para el proceso de clustering:
# -----------------------------

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

    cols_to_drop = ['id_cliente','id_periodo','id_barrio', 'id_comuna','canal','segmento','indice_gse', 'volumen_sem_fut_est']
    X = df.drop(cols_to_drop, axis=1)

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

if __name__ == "__main__":
    main()
