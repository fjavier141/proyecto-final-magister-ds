# Manejo de archivos y sistema
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import FuncFormatter
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

def set_plot_style():
    sns.set_theme(style="whitegrid", context="talk", palette="Set2")

def thousands_with_dot(x, pos=None):
    # 1234567 -> 1.234.567
    try:
        return f"{int(round(x)):,}".replace(",", ".")
    except Exception:
        return str(x)

PRETTY_LABELS = {
    "size_base": "Volumen Anual",
    "volumen_crecimiento_sem_fut": "Crecimiento Estimado"
}

def main():
    validation_periods = uts.get_validation_periods(TEST_PERIOD, 1)

    for validation_period in validation_periods:
        path_validation = os.path.join(f"./data/output/validation/{CATEGORY}/{CHANNEL}/{validation_period}")
        dataset_predict = uts.load_pickle(os.path.join(path_validation, "scoring_lgbm.pkl"))
        dataset_predict['volumen_crecimiento_sem_fut'] = dataset_predict['volumen_sem_fut_est'] - dataset_predict['volumen_sem']
        X = get_X_for_clustering(dataset_predict)

        # Selección de k óptimo:
        best_k, k_ranking = select_k(X, k_min=2, k_max=10)

        # Plot elbow (con línea en k elegido)
        plot_elbow_from_metrics(k_ranking, highlight_k=best_k)

        best_k = 5
        print(f'Número óptimo de clusters (k) seleccionado: {best_k}')

        # Entrenamiento de KMeans con k óptimo:
        labels = fit_kmeans(X, n_clusters=best_k)

        # Asignación de labels al dataset original:
        dataset_predict['cluster'] = labels
        dataset_predict["cluster_id"] = dataset_predict["cluster"] + 1

        # Boxplots (bonitos)
        cluster_vars = ['volumen_crecimiento_sem_fut', 'size_base']
        boxplots_by_cluster(
            dataset_predict,
            vars_to_plot=cluster_vars,
            title_prefix=f"KMeans (k={best_k}) – "
        )

        # Gráfico combinado (volumen total + #clientes)
        plot_cluster_volume_and_clients(dataset_predict)

        # Métricas descriptivas
        summary = (
            dataset_predict
            .groupby("cluster_id")
            .agg(
                mediana_crecimiento=("volumen_crecimiento_sem_fut", "median"),
                mediana_volumen_anual=("size_base", "median"),
                n_clientes=("cluster_id", "size")
            )
            .reset_index()
        )

        print(summary)

        alpha = 0.1  # 10% de eficiencia por visita (ajústalo)
        rep = cluster_impact_report(dataset_predict, alpha_eff=alpha)

        # persistencia de resultados:
        uts.save_pickle(dataset_predict, os.path.join(path_validation, 'clusters_kmeans.pkl'))
        rep.to_excel(os.path.join(path_validation, 'analisis_score_rescate.xlsx'))


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
    X = df[['volumen_crecimiento_sem_fut', 'size_base']]

    #X['size_base'] = np.log1p(X['size_base'])

    '''X['crecimiento_fut_est'] = np.log1p(
        X['crecimiento_fut_est'].clip(lower=0)
    )'''

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


'''def boxplots_by_cluster(
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
        plt.show()'''

def boxplots_by_cluster(
    df: pd.DataFrame,
    vars_to_plot: list[str],
    cluster_col: str = "cluster_id",
    title_prefix: str = "",
    figsize=(11, 6),
):
    """
    Boxplots por cluster para variables seleccionadas.
    Mejorado para presentación (labels, separador de miles, línea 0 en crecimiento).
    """
    set_plot_style()

    for var in vars_to_plot:
        pretty = PRETTY_LABELS.get(var, var)

        plt.figure(figsize=figsize)
        ax = sns.boxplot(
            data=df,
            x=cluster_col,
            y=var,
            showfliers=False
        )

        # Línea 0 solo para crecimiento
        if var == "volumen_crecimiento_sem_fut":
            ax.axhline(0, color="gray", linestyle="--", linewidth=1)

        ax.set_title(f"{title_prefix}Distribución de {pretty} por cluster", pad=12)
        ax.set_xlabel("Cluster")
        ax.set_ylabel(pretty)

        ax.yaxis.set_major_formatter(FuncFormatter(thousands_with_dot))
        plt.tight_layout()
        plt.show()

def plot_cluster_volume_and_clients(
    df: pd.DataFrame,
    cluster_col: str = "cluster_id",
    volume_col: str = "size_base",
    figsize=(12, 6),
    title="Tamaño de los Clusters: Volumen Anual Total y Número de Clientes",
):
    """
    Un solo gráfico:
    - Barras: suma de Volumen Anual por cluster
    - Línea/puntos (eje derecho): número de clientes por cluster
    """
    set_plot_style()

    agg = (
        df.groupby(cluster_col, as_index=False)
          .agg(
              volumen_total=(volume_col, "sum"),
              n_clientes=(cluster_col, "size")
          )
          .sort_values(cluster_col)
          .reset_index(drop=True)
    )

    x_pos = np.arange(len(agg))

    fig, ax1 = plt.subplots(figsize=figsize)

    # 🔵 Barras: Volumen anual
    ax1.bar(
        x_pos,
        agg["volumen_total"],
        color="#4C72B0",      # azul suave
        alpha=0.85
    )
    ax1.set_title(title, pad=12)
    ax1.set_xlabel("Cluster")
    ax1.set_ylabel("Volumen Anual Total")
    ax1.yaxis.set_major_formatter(FuncFormatter(thousands_with_dot))

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(agg[cluster_col])

    # 🟠 Línea: número de clientes
    ax2 = ax1.twinx()
    ax2.plot(
        x_pos,
        agg["n_clientes"],
        color="#DD8452",      # naranja contrastante
        marker="o",
        linewidth=2.5,
        markersize=7
    )
    ax2.set_ylabel("Número de Clientes")
    ax2.yaxis.set_major_formatter(FuncFormatter(thousands_with_dot))

    plt.tight_layout()
    plt.show()


def plot_elbow_from_metrics(
    metrics: pd.DataFrame,
    title="Método del Codo (Elbow) – KMeans",
    figsize=(10, 6),
    highlight_k: int | None = None,
):
    set_plot_style()

    metrics_sorted = metrics.sort_values("k")

    plt.figure(figsize=figsize)
    plt.plot(metrics_sorted["k"], metrics_sorted["inertia"], marker="o")
    plt.title(title, pad=12)
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Inercia")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(thousands_with_dot))

    if highlight_k is not None:
        plt.axvline(highlight_k, linestyle="--", linewidth=1)

    plt.tight_layout()
    plt.show()


def cluster_impact_report(
    df: pd.DataFrame,
    cluster_col: str = "cluster_id",
    impact_col: str = "volumen_crecimiento_sem_fut",
    volume_col: str = "size_base",
    visits_col: str | None = None,
    alpha_eff: float = 1.0,
) -> pd.DataFrame:
    """
    Reporte por cluster con:
    - Impacto neto: sum(impact_col)
    - Riesgo (solo caídas): sum(|impact|) para impact<0
    - Oportunidad (solo alzas): sum(impact) para impact>0
    - #clientes
    - Volumen anual total y mediano
    - Métricas trade-off con visitas:
        - rescate_potencial = alpha_eff * riesgo
        - score_rescate_por_visita = rescate_potencial / visitas

    alpha_eff:
      - 1.0 = asumes que "podrías" capturar/contener el 100% del riesgo si visitas
      - 0.3 = eficiencia 30% (más realista)
    """

    work = df.copy()

    # Visitas: si no existe columna, aproximamos visitas = 1 por cliente (una visita por cliente)
    if visits_col is None or visits_col not in work.columns:
        work["_visitas"] = 1
        visits_col_use = "_visitas"
    else:
        visits_col_use = visits_col

    # Componentes de impacto
    work["_neg"] = np.where(work[impact_col] < 0, -work[impact_col], 0.0)  # riesgo en valor absoluto
    work["_pos"] = np.where(work[impact_col] > 0,  work[impact_col], 0.0)  # oportunidad

    report = (
        work.groupby(cluster_col, as_index=False)
            .agg(
                n_clientes=("id_cliente", "nunique") if "id_cliente" in work.columns else (cluster_col, "size"),
                visitas=(visits_col_use, "sum"),

                impacto_neto=(impact_col, "sum"),
                riesgo_total=("_neg", "sum"),
                oportunidad_total=("_pos", "sum"),

                volumen_anual_total=(volume_col, "sum"),
                mediana_volumen_anual=(volume_col, "median"),
                mediana_crecimiento=(impact_col, "median"),
            )
            .sort_values(cluster_col)
            .reset_index(drop=True)
    )

    # Métricas trade-off
    report["rescate_potencial"] = alpha_eff * report["riesgo_total"]
    report["score_rescate_por_visita"] = report["rescate_potencial"] / report["visitas"].replace(0, np.nan)

    # % contribuciones (útil para storytelling)
    total_riesgo = report["riesgo_total"].sum()
    total_vol = report["volumen_anual_total"].sum()

    report["pct_riesgo_total"] = np.where(total_riesgo > 0, report["riesgo_total"] / total_riesgo, 0.0)
    report["pct_volumen_total"] = np.where(total_vol > 0, report["volumen_anual_total"] / total_vol, 0.0)

    # Orden recomendado para decidir dónde actuar (por rendimiento por visita)
    report = report.sort_values("score_rescate_por_visita", ascending=False).reset_index(drop=True)

    # Limpieza
    if "_visitas" in work.columns:
        pass  # no afecta afuera

    return report


if __name__ == "__main__":
    main()