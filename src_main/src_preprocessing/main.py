import pandas as pd
from ydata_profiling import ProfileReport

import os
import src.preprocessing as pre
import src.utils as uts
from dotenv import load_dotenv


# --- Config estática del dominio ---
DICT_MIX = {
    "cervezas": ["masivo"],
    "analcoholicos": ["gaseosas", "minerales"],
}

# Segmentos válidos (filtro de negocio)
SEGMENTS = [
    "AL", "BO", "AP", "KI", "BA", "EE", "ES", "FF", "FU", "CD",
    "IE", "DI", "RT", "RE", "BC", "RC", "GI", "FC", "FS", "RD",
]

# Cargar .env (ajusta ruta a tu entorno)

#env_path = "/Users/diegobascunan/PycharmProjects/proyecto-final-magister-ds/accesos_diego.env"
env_path = "D:\\Users\\fjavi\\Proyectos\\proyecto-final-magister-ds\\.env"
load_dotenv(dotenv_path = env_path)


def main():
    """
    Orquesta el preprocesamiento end-to-end para una categoría fija.
    Ajusta `category` y `test_date` según el escenario.
    """
    ## Parámetros (desde .env, como variable de entorno)
    category = os.getenv("CATEGORY") # 'cervezas' o 'analcoholicos', manejar desde .env la categoría que se usará.
    test_date = int(os.getenv("TEST_PERIOD")) # Formato AAAAMM, actualmente configurado como : 202412


    # 1) Extracción
    clients, barrios, macro_vars, sales = pre.extract_data(category)

    # 2) Intersección ventas-clientes y filtros de negocio
    df_sales = pre.intersect_sales_clients(clients, sales)
    df_sales = df_sales[df_sales['segmento'].isin(SEGMENTS)]

    # 3) Cálculo de volúmenes por tipo de mix
    df_sales = pre.volumen_mix(df_sales, category)

    # 4) Agregación por cliente-mes (y dimensiones estáticas relevantes)
    by = ["volumen", *[f"vol_{col}" for col in DICT_MIX[category]]]
    group_cols = [
        "id_categoria", "id_cliente", "id_periodo",
        "id_barrio", "id_comuna", "canal", "segmento", "descr_flag_patente",
    ]
    agg_sales = df_sales.groupby(group_cols)[by].sum().reset_index()

    # 5) Proporciones de mix (ej.: vol_masivo / volumen)
    agg_sales = pre.prop_vol_mix(agg_sales, category)

    # 6) Completar períodos faltantes por cliente
    agg_sales = pre.fill_missing_periods(agg_sales)

    # 7) Propagar dimensiones estáticas por cliente (ffill/bfill)
    static_cols = ['id_categoria', 'id_barrio', 'id_comuna', 'canal', 'segmento', 'descr_flag_patente']
    for c in static_cols:
        if c in agg_sales.columns:
            agg_sales[c] = (agg_sales.groupby("id_cliente")[c]
                       .ffill()
                       .bfill())

    # Evitar NaNs en volumen tras reindex
    agg_sales["volumen"] = agg_sales["volumen"].fillna(0)

    # 8) Features temporales (rolling, lags, AR, recency/frequency, target)
    df_fact = pre.calculate_rolling_lags(agg_sales, category)


    # 9) Enriquecer con barrios y macro
    df_fact = df_fact.merge(barrios, on='id_barrio', how='left')
    dataset = df_fact.merge(macro_vars, on='id_periodo', how='left')

    # 10) Limpieza de NAs (estrategias simples y controladas)
    cols_drop = ['id_barrio']
    cols_zeros = ['ar_mes0', 'ar0', 'ar_mes1', 'ar1', 'ar_mes2', 'ar2', 'ar_mes3', 'ar3']
    cols_advanced = ['indice_gse', 'densidad_hab', 'n_ptos_interes', 'superficie_km2', 'n_habitantes']

    dataset = pre.fill_nan_values(dataset, cols_zeros, cols_mean=[], cols_median=[], cols_advanced=cols_advanced,K_parametro=5)

    dataset = pre.drop_nan_values(dataset, cols_drop)

    # 11) Diagnóstico rápido (correlación con la y en train)
    per_train, per_test = uts.get_dates(test_date)

    filter_dataset = dataset[dataset['id_periodo'].isin(per_train)]

    target_corr = filter_dataset.corr(numeric_only=True)["volumen_sem_dif6_fut"].sort_values(ascending=False)

    print(target_corr)

    # 12) Resumen + EDA opcional
    review_dataset(dataset, f'dataset_{category}')

    # 13) Persistencia
    uts.save_pickle(dataset, f'./data/output/pickle/dataset_{category}.pickle')


def review_dataset(df, dataset_name):
    # Revisión general
    print("Dimensiones:", df.shape)
    print(df.info())
    print(df.describe().T)

    # Reglas básicas y duplicados
    print("Duplicados:", df.duplicated(subset=["id_cliente", "id_periodo"]).sum())
    print(df.isna().mean().sort_values(ascending=False).head(10))

    get_profile_report(df, dataset_name)




def get_profile_report(df, dataset_name):
    """
    Genera un informe de profiling en HTML (opcional pero útil para revisión).
    """
    profile = ProfileReport(df, title=f"EDA {dataset_name}", explorative=True)
    profile.to_file(f"./data/output/eda_{dataset_name}.html")


