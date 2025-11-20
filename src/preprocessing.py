import os

import numpy as np
import pandas as pd
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.impute import KNNImputer
from sqlalchemy import create_engine, text
from ydata_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler


from dotenv import load_dotenv


dict_mix = {
    "cervezas": ["masivo"],
    "analcoholicos": ["gaseosas", "minerales"]
}


def extract_data(category):
    """
    Conecta a Postgres (credenciales por .env) y extrae:
    - clientes (dimensión)
    - barrios (dimensión)
    - macro_vars (macro mensual)
    - sales (ventas históricas filtradas por categoría)

    Parameters
    ----------
    category : {"cervezas", "analcoholicos"}

    Returns
    -------
    (clientes, barrios, macro_vars, sales)
    """

    # Cargar .env (ajusta ruta a tu entorno)
    #env_path = "/Users/diegobascunan/iCloud Drive/Escritorio/Proyecto_Titulo/accesos.env"
    env_path = "D:\\Users\\fjavi\\Proyectos\\proyecto-final-magister-ds\\.env"
    load_dotenv(dotenv_path = env_path)

    # Parámetros de conexión (ajusta a tu entorno)
    USER = os.getenv("POSTGRES_USER")
    PASSWORD = os.getenv("POSTGRES_PASSWORD")
    HOST = os.getenv("POSTGRES_HOST")
    PORT = os.getenv("POSTGRES_PORT", "5432")
    DB = os.getenv("POSTGRES_DB")

    # Crea el motor SQLAlchemy
    engine = create_engine(f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}")

    # Test de conexión (informativo)
    try:
        with engine.connect() as conn:
            version = conn.execute(text("SELECT version();"))
            print("Conectado a:", list(version)[0][0])
    except Exception as e:
        print("Error al conectar:", e)

    # Dimensiones
    with engine.connect() as conn:
        clientes = pd.read_sql(
            text("SELECT id_cliente, id_barrio, id_comuna, segmento, canal, descr_flag_patente FROM stg.base_clientes;"),
            conn)

    with engine.connect() as conn:
        barrios = pd.read_sql(text(
            "SELECT id_barrio, indice_gse, n_habitantes, n_ptos_interes, superficie_km2, densidad_hab FROM stg.info_distritos;"),
                              conn)

    with engine.connect() as conn:
        macro_vars = pd.read_sql(
            text("SELECT id_periodo, uf, dolar, ipc, imacec, tpm, tasa_desempleo FROM stg.datos_macro;"), conn)

    # Ventas por categoría

    category_id = 1 if category == "cervezas" else 3

    with engine.connect() as conn:
        query = text("""
                     SELECT id_categoria, id_cliente, id_periodo, tipo_mix, id_sku_venta, liq_um
                     FROM stg.venta_historica
                     WHERE id_categoria = :category_id;
                     """)
        sales = pd.read_sql(query, conn, params={"category_id": category_id})

    return clientes, barrios, macro_vars, sales


def intersect_sales_clients(clients, sales):
    # ---------------------------------------------------------------------
    # Limpieza y cruces
    # ---------------------------------------------------------------------
    sales = sales[sales['liq_um'] > 0]

    df_sales = sales.merge(clients, on=['id_cliente'], how='inner')
    df_sales['id_periodo'] = df_sales['id_periodo'].astype(int)

    agg_df = (
        df_sales.groupby(['id_categoria', 'id_cliente', 'id_periodo', 'id_barrio', 'tipo_mix', 'id_comuna', 'canal', 'segmento', 'descr_flag_patente'],
                    as_index=False)
        .agg(volumen=('liq_um', 'sum'))
    )

    agg_df = agg_df.sort_values(["id_cliente", "id_periodo"])

    agg_df.rename(columns={"liq_um": "volumen"}, inplace=True)


    return agg_df

# ---------------------------------------------------------------------
# Mix y proporciones
# ---------------------------------------------------------------------

def volumen_mix(df, category):
    """
    Crea columnas 'vol_{col}' por cada mix de la categoría,
    con el volumen correspondiente cuando el tipo coincide.
    """
    df1 = df.copy()
    for col in dict_mix[category]:
        df1['vol_' + col] = 0.0
        df1.loc[df1['tipo_mix'] == col.upper(), 'vol_' + col] = df1.loc[df1['tipo_mix'] == col.upper(), 'volumen']

    return df1


def prop_vol_mix(df, category):
    """
    Reemplaza 'vol_{col}' por 'porc_{col}' = vol_{col} / volumen.
    Evita divisiones por cero/inf reemplazando por 0 cuando corresponda.
    """
    df1 = df.copy()
    for col in dict_mix[category]:
        df1['porc_' + col] = df1['vol_' + col] / df1['volumen']
        df1.loc[df1['porc_' + col].isin([np.inf, -np.inf]), 'porc_' + col] = 0
        df1.drop(columns=['vol_' + col], inplace=True)
    return df1


def fill_missing_periods(sales):
    """
    Construye el producto cartesiano (cliente x período) usando los períodos presentes.
    OJO: si faltan meses en el universo global, no se inventan; se usa lo observado.

    Tip: para un llenado "denso" entre min y max por cliente, habría que generar
    el rango completo por cliente; aquí se mantiene el comportamiento original.
    """
    clientes = sales["id_cliente"].unique()
    periodos = sales["id_periodo"].unique()

    # Crear producto cartesiano cliente × periodo
    full_index = pd.MultiIndex.from_product([clientes, periodos], names=["id_cliente", "id_periodo"])
    sales = sales.set_index(["id_cliente", "id_periodo"]).reindex(full_index).reset_index()

    return sales


def calculate_rolling_lags(agg_sales, category):
    """
    Genera:
    - volumen_sem: rolling 6m de 'volumen'
    - prop_vol_{mix}: promedio móvil 6m
    - Lags AR sobre volumen_sem (6 y 12)
    - Diferencias (volumen, volumen_sem) y lags de diferencias
    - Target: volumen_sem_dif6_fut (shift -6)
    - Señales RFM básicas: compra, frequency(12m), recency(12 cap)

    Mantiene el orden por (id_cliente, id_periodo).
    """
    sales = agg_sales.copy()
    sales = sales.sort_values(["id_cliente", "id_periodo"]).reset_index(drop=True)

    # Rolling 6m del volumen
    sales["volumen_sem"] = (
        sales.groupby("id_cliente")["volumen"]
        .transform(lambda x: x.rolling(window=6, min_periods=1).sum())
    )

    # Promedio móvil 6m de proporciones de mix
    for column in dict_mix[category]:
        sales[f'prop_vol_{column}'] = sales.groupby(['id_cliente'])[f'porc_{column}'].rolling(window=6).mean().to_list()

    # Lags de volumen_sem
    sales.sort_values(by=['id_cliente', 'id_periodo'], inplace=True, ignore_index=True)
    sales['volumen_sem_ar1'] = sales.groupby(['id_cliente'])['volumen_sem'].shift(6)
    sales['volumen_sem_ar2'] = sales.groupby(['id_cliente'])['volumen_sem'].shift(12)
    sales['volumen_sem_fut'] = sales.groupby(['id_cliente'])['volumen_sem'].shift(-6)

    # Diferencias
    sales.sort_values(by=['id_cliente', 'id_periodo'], inplace=True, ignore_index=True)
    sales['volumen_dif1'] = sales.groupby(['id_cliente'])['volumen'].diff()
    sales['volumen_dif1_dif12'] = sales.groupby(['id_cliente'])['volumen_dif1'].diff(12)
    sales['volumen_sem_dif6'] = sales.groupby(['id_cliente'])['volumen_sem'].diff(6)

    # AR sobre diferencias (mes a mes y cada 6 meses)
    sales.sort_values(by=['id_cliente', 'id_periodo'], inplace=True, ignore_index=True)
    for i in range(0, 4):
        sales['ar_mes{}'.format(i)] = sales.groupby(['id_cliente'])['volumen_dif1_dif12'].shift(i)
        sales['ar{}'.format(i)] = sales.groupby(['id_cliente'])['volumen_sem_dif6'].shift(i * 6)

    # Target (futuro a 6 meses)
    sales = sales.sort_values(["id_cliente", "id_periodo"])
    sales['volumen_sem_dif6_fut'] = sales.groupby(['id_cliente'])['volumen_sem_dif6'].shift(-6)

    # RFM básico
    sales["compra"] = (sales["volumen"] > 0).astype(int)

    # Frecuencia: compras últimos 12 meses
    sales["frecuency"] = (
        sales.groupby("id_cliente")["compra"]
        .transform(lambda x: x.rolling(window=12, min_periods=1).sum())
    )

    # Recency: meses desde la última compra (cap 12)
    sales["recency"] = sales.groupby("id_cliente")["compra"].transform(recency_12_cap)

    return sales


def recency_12_cap(x: pd.Series) -> pd.Series:
    """
    Calcula recency "capped" a 12:
    - Si no hay compra previa, 12
    - Si hay compra en el mes i, 0
    - En meses sin compra, distancia en meses a la última compra, con tope 12
    """
    rec = np.zeros(len(x), dtype=int)
    last_purchase = None

    for i, v in enumerate(x):
        if v == 1:
            rec[i] = 0
            last_purchase = i
        else:
            if last_purchase is None:
                rec[i] = 12
            else:
                rec[i] = min(i - last_purchase, 12)

    return pd.Series(rec, index=x.index)


def fill_nan_values(df: pd.DataFrame, cols_zeros: list, cols_mean: list, cols_median: list, cols_advanced: list, K_parametro=5):
    """
    Reemplaza NaNs:
    - En `cols_zeros`, por 0
    - En `cols_mean`, por la media de la columna (calculada sobre df no vacío)
    - En 'cols_median', por la mediana de la columna (calculada sobre df no vacío)
    - En 'col_advanced', usando Knnimputer (k=5) para imputación basada en vecinos
    """
    df1 = df.copy()

    # Reemplazar los NA de las columnas con ceros
    df1[cols_zeros] = df1[cols_zeros].fillna(0)

    # Reemplazar los NA de las columnas con la media de la columna
    for col in cols_mean:
        df1[col] = df1[col].fillna(df1[col].mean())

    # Reemplazar los NA de las columnas con la mediana de la columna
    for col in cols_median:
        df1[col] = df1[col].fillna(df1[col].median())

    if cols_advanced:
        data_to_impute = df1[cols_advanced]
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_to_impute)
        imputer = KNNImputer(n_neighbors=K_parametro)
        data_imputed_scaled = imputer.fit_transform(data_scaled)
        data_imputed = scaler.inverse_transform(data_imputed_scaled)
        df1[cols_advanced] = data_imputed

    return df1



def drop_nan_values(df: pd.DataFrame, cols_to_drop: list):
    """
    Elimina filas con NaN en las columnas listadas (útil cuando el NA implica inconsistencia).
    """
    df1 = df.copy()
    for col in cols_to_drop:
        df1.drop(df1[df1[col].isna()].index, inplace=True)
    return df1

# ---------------------------------------------------------------------
# Helpers para splits de modelado
# (se conserva la interfaz original)
# ---------------------------------------------------------------------


def get_train_data(df: pd.DataFrame, train_date_set: list, category):
    """
    Devuelve X de train: filtra por fechas, descarta NaN en y, y remueve columnas no entrenables.
    """
    cols_to_drop = [
        'id_categoria', 'id_periodo', 'id_cliente', 'id_canal',
        'volumen', 'volumen_sem', 'superficie_km2', 'n_habitantes', 'prop_vol_masivo', 'canal', 'descr_flag_patente',
        'volumen_sem_ar1', 'volumen_sem_ar2', 'volumen_sem_fut',
        'volumen_dif1', 'volumen_dif1_dif12', 'volumen_sem_dif6',
        'volumen_sem_dif6_fut'
    ]

    for col in dict_mix[category]:
        cols_to_drop.append(f'porc_{col}')
    df1 = df.copy()
    df1.drop(df1[~df1['id_periodo'].isin(train_date_set)].index, inplace=True)
    df1.drop(df1[df1['volumen_sem_dif6_fut'].isna()].index, inplace=True)
    df1.reset_index(drop=True, inplace=True)
    df1.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    return df1


def get_train_target(df: pd.DataFrame, train_date_set):
    """
    Devuelve y de train (volumen_sem_dif6_fut) alineado con get_train_data.
    """
    df1 = df.copy()
    df1.drop(df1[~df1['id_periodo'].isin(train_date_set)].index, inplace=True)
    df1.drop(df1[df1['volumen_sem_dif6_fut'].isna()].index, inplace=True)
    df1.reset_index(drop=True, inplace=True)
    return df1['volumen_sem_dif6_fut']


def get_val_data(df: pd.DataFrame, test_date_set, category):
    """
    Devuelve X de validación/test: mismo filtrado y columnas que train.
    """
    cols_to_drop = [
        'id_categoria', 'id_periodo', 'id_cliente', 'id_canal',
        'volumen', 'volumen_sem', 'superficie_km2', 'n_habitantes', 'prop_vol_masivo', 'canal', 'descr_flag_patente',
        'volumen_sem_ar1', 'volumen_sem_ar2', 'volumen_sem_fut',
        'volumen_dif1', 'volumen_dif1_dif12', 'volumen_sem_dif6',
        'volumen_sem_dif6_fut'
    ]
    for col in dict_mix[category]:
        cols_to_drop.append(f'porc_{col}')
    df1 = df.copy()
    df1.drop(df1[~df1['id_periodo'].isin(test_date_set)].index, inplace=True)
    df1.drop(df1[df1['volumen_sem_dif6_fut'].isna()].index, inplace=True)
    df1.reset_index(drop=True, inplace=True)
    df1.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    return df1


def get_val_target(df: pd.DataFrame, test_date_set):
    """
    Devuelve y para validación/test (se nulifica para evitar fuga por accidente).
    """
    df1 = df.copy()
    df1.drop(df1[~df1['id_periodo'].isin(test_date_set)].index, inplace=True)
    df1.drop(df1[df1['volumen_sem_dif6_fut'].isna()].index, inplace=True)
    df1.reset_index(drop=True, inplace=True)
    df1['volumen_sem_dif6_fut'] = np.nan
    return df1['volumen_sem_dif6_fut']


def get_pred_set(df: pd.DataFrame, test_date_set):
    """
    Devuelve un set con identificadores y target real (para evaluación posterior),
    pero con la columna objetivo anulada (para predicción).
    """
    cols_to_copy = [
        'id_categoria', 'id_periodo', 'id_cliente',
        'volumen_sem_ar1', 'volumen_sem', 'volumen_sem_dif6_fut'
    ]
  
    df1 = df[cols_to_copy].copy()
    df1.drop(df1[~df1['id_periodo'].isin(test_date_set)].index, inplace=True)
    df1.drop(df1[df1['volumen_sem_dif6_fut'].isna()].index, inplace=True)

    df1['volumen_sem_dif6_fut_real'] = df1['volumen_sem_dif6_fut']
    df1.reset_index(drop=True, inplace=True)
    df1['volumen_sem_dif6_fut'] = np.nan
    return df1
