import os

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sqlalchemy import create_engine, text
from ydata_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

from parameters.config import *


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


def fill_missing_periods_by_client(sales: pd.DataFrame) -> pd.DataFrame:
    sales = sales.copy()
    sales["id_periodo"] = sales["id_periodo"].astype(int)

    # Periodo mensual real
    sales["_per"] = _to_period(sales["id_periodo"])
    if sales["_per"].isna().any():
        bad = sales.loc[sales["_per"].isna(), "id_periodo"].unique()
        raise ValueError(f"id_periodo inválidos (esperado YYYYMM con MM 01..12): {bad[:20]}")

    out = []
    for id_cli, df_c in sales.groupby("id_cliente", sort=False):
        df_c = df_c.sort_values("id_periodo").copy()
        min_per = df_c["_per"].min()
        max_per = df_c["_per"].max()

        full_per = pd.period_range(min_per, max_per, freq="M")
        full_p = pd.DataFrame({
            "_per": full_per,
            "id_periodo": (full_per.year * 100 + full_per.month).astype(int)
        })

        df_c2 = full_p.merge(df_c.drop(columns=["id_cliente"], errors="ignore"),
                             on=["id_periodo", "_per"], how="left")
        df_c2["id_cliente"] = id_cli
        out.append(df_c2)

    sales_full = pd.concat(out, ignore_index=True)

    # Limpieza
    sales_full.drop(columns=["_per"], inplace=True, errors="ignore")
    return sales_full


def _to_period(s: pd.Series) -> pd.PeriodIndex:
    # id_periodo: YYYYMM (mes 01..12)
    # Convertimos a Period[M] para poder generar rangos mensuales reales.
    dt = pd.to_datetime(s.astype(str), format="%Y%m", errors="coerce")
    return dt.dt.to_period("M")


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
        .transform(lambda x: x.rolling(window=6, min_periods=6).sum())
    )

    # Promedio móvil 6m de proporciones de mix
    for column in dict_mix[category]:
        sales[f'prop_vol_{column}'] = sales.groupby(['id_cliente'])[f'porc_{column}'].rolling(window=6).mean().to_list()

    # Lags de volumen_sem
    sales.sort_values(by=['id_cliente', 'id_periodo'], inplace=True, ignore_index=True)
    sales['volumen_sem_ar1'] = sales.groupby(['id_cliente'])['volumen_sem'].shift(6).fillna(0)
    sales['volumen_sem_ar2'] = sales.groupby(['id_cliente'])['volumen_sem'].shift(12).fillna(0)
    sales['volumen_sem_fut'] = sales.groupby(['id_cliente'])['volumen_sem'].shift(-6).fillna(0)

    # Diferencias
    sales.sort_values(by=['id_cliente', 'id_periodo'], inplace=True, ignore_index=True)
    sales['volumen_dif1'] = sales.groupby(['id_cliente'])['volumen'].diff()
    sales['volumen_dif1_dif12'] = sales.groupby(['id_cliente'])['volumen_dif1'].diff(12)
    sales['volumen_sem_dif6'] = sales.groupby(['id_cliente'])['volumen_sem'].diff(6)

    # Cambio relativo a 6m (MUY potente para crecimiento)
    eps = 1e-3
    sales["vol_sem_rel_dif6"] = sales["volumen_sem_dif6"] / (sales["volumen_sem_ar1"] + eps)

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


def calculate_rolling_lags_2(agg_sales: pd.DataFrame, category: str) -> pd.DataFrame:
    """
    Features temporales (mensuales) por cliente.

    Genera (principales):
    - volumen_sem: suma móvil 6 meses (min_periods=6)
    - prop_vol_{mix}: promedio móvil 6 meses de proporciones de mix (min_periods=6)

    - Lags de nivel:
        vol_sem_lag6, vol_sem_lag12
    - Cambios:
        vol_sem_dif6 (abs), vol_sem_rel_dif6 (relativo)
    - Lags de cambios:
        vol_sem_dif6_lag6, vol_sem_rel_dif6_lag6, vol_sem_rel_dif6_lag12
    - Tendencia y volatilidad:
        trend_6m_vol (pendiente últimos 6 meses de volumen mensual)
        vol_std_6m (desv std últimos 6 meses de volumen mensual)

    - Target:
        volumen_sem_dif6_fut = shift(-6) de vol_sem_dif6

    - RFM básico:
        compra, frequency_12m, recency (cap 12)

    Importante:
    - NO rellena NaN con 0 en lags/targets: NaN = “no hay historia suficiente”.
      Se filtra después en train/test.
    """
    sales = agg_sales.copy()
    sales = sales.sort_values(["id_cliente", "id_periodo"]).reset_index(drop=True)

    g = sales.groupby("id_cliente", sort=False)

    # =========================
    # 1) Rolling 6M del volumen (nivel semestral)
    # =========================
    sales["volumen_sem"] = g["volumen"].transform(lambda x: x.rolling(6, min_periods=6).sum())

    # =========================
    # 2) Rolling 6M de proporciones de mix
    # =========================
    for mix in dict_mix[category]:
        col = f"porc_{mix}"
        if col in sales.columns:
            sales[f"prop_vol_{mix}"] = g[col].transform(lambda x: x.rolling(6, min_periods=6).mean())

    # =========================
    # 3) Lags del nivel semestral
    # =========================
    sales["vol_sem_lag6"]  = g["volumen_sem"].shift(6)
    sales["vol_sem_lag12"] = g["volumen_sem"].shift(12)

    # Si quieres conservar nombres antiguos para compatibilidad:
    sales["volumen_sem_ar1"] = sales["vol_sem_lag6"]
    sales["volumen_sem_ar2"] = sales["vol_sem_lag12"]
    sales["volumen_sem_fut"] = g["volumen_sem"].shift(-6)

    # =========================
    # 4) Diferencias y cambios relativos
    # =========================
    sales["volumen_dif1"] = g["volumen"].diff(1)  # mes a mes

    # Cambio semestral del nivel semestral (abs)
    sales["vol_sem_dif6"] = g["volumen_sem"].diff(6)
    sales["volumen_sem_dif6"] = sales["vol_sem_dif6"]  # compat

    # Cambio relativo semestral (MUY útil para crecimiento)
    eps = 1e-3
    sales["vol_sem_rel_dif6"] = sales["vol_sem_dif6"] / (sales["vol_sem_lag6"] + eps)

    # Lags de cambios (contexto: cómo venía cambiando)
    sales["vol_sem_dif6_lag6"] = g["vol_sem_dif6"].shift(6)
    sales["vol_sem_rel_dif6_lag6"]  = g["vol_sem_rel_dif6"].shift(6)
    sales["vol_sem_rel_dif6_lag12"] = g["vol_sem_rel_dif6"].shift(12)

    # Si quieres mantener tus columnas ar0..ar3 (pero coherentes y simples)
    # ar0 = cambio actual, ar1 = cambio 6m atrás, etc.
    for i in range(0, 4):
        sales[f"ar{i}"] = g["vol_sem_dif6"].shift(i * 6)

    # (Opcional) Mantener ar_mes* si lo usas; si no lo usas, bórralo
    sales["volumen_dif1_dif12"] = g["volumen_dif1"].diff(12)
    for i in range(0, 4):
        sales[f"ar_mes{i}"] = g["volumen_dif1_dif12"].shift(i)

    # =========================
    # 5) Tendencia 6M y volatilidad (señales de estabilidad)
    # =========================
    def _slope_6m(x: pd.Series) -> pd.Series:
        # slope sobre ventanas de 6 puntos (volumen mensual)
        # devuelve NaN si no hay 6 obs
        arr = x.to_numpy(dtype=float)
        out = np.full(len(arr), np.nan, dtype=float)
        t = np.arange(6, dtype=float)
        t_mean = t.mean()
        denom = ((t - t_mean) ** 2).sum()

        for i in range(5, len(arr)):
            y = arr[i-5:i+1]
            if np.any(~np.isfinite(y)):
                continue
            y_mean = y.mean()
            num = ((t - t_mean) * (y - y_mean)).sum()
            out[i] = num / denom
        return pd.Series(out, index=x.index)

    sales["trend_6m_vol"] = g["volumen"].transform(_slope_6m)
    sales["vol_std_6m"] = g["volumen"].transform(lambda x: x.rolling(6, min_periods=6).std())

    # =========================
    # 6) Target (futuro 6 meses del cambio semestral)
    # =========================
    sales["volumen_sem_dif6_fut"] = g["vol_sem_dif6"].shift(-6)

    # =========================
    # 7) RFM básico
    # =========================
    sales["compra"] = (sales["volumen"] > 0).astype(int)

    sales["frecuency"] = g["compra"].transform(lambda x: x.rolling(12, min_periods=1).sum())
    sales["recency"] = g["compra"].transform(recency_12_cap)

    return sales


def recency_12_cap(x: pd.Series) -> pd.Series:
    """
    Recency capped a 12 SOLO para clientes con al menos una compra previa.

    - recency = 0  → compra en el mes actual
    - recency = k  → k meses desde la última compra
    - recency = NaN → nunca ha comprado (se filtra antes o después)
    """
    rec = np.full(len(x), np.nan, dtype=float)
    last_purchase = None

    for i, v in enumerate(x):
        if v == 1:
            rec[i] = 0
            last_purchase = i
        else:
            if last_purchase is not None:
                rec[i] = min(i - last_purchase, 12)

    return pd.Series(rec, index=x.index)


def fill_nan_values(df: pd.DataFrame, cols_zeros: list, cols_mean: list, cols_median: list):
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

    return df1


# ---------------------------------------------------------------------
# Función auxiliar para KNN Imputer
# ---------------------------------------------------------------------

def nan_knn_imputer(df: pd.DataFrame, cols: list, k = 9) -> pd.DataFrame:
    """Función para imputar valores NaN en columnas específicas usando KNN Imputer para variables dimensionales asociadas a barrios.
    Args:
        df (pd.DataFrame): DataFrame original con posibles valores NaN.
        cols (list): Lista de columnas a imputar.
        k (int): Número de vecinos a considerar en KNN, se prueba que k = 9 es un buen valor (ver EDA)
    out:
        pd.DataFrame: DataFrame con valores NaN imputados en las columnas especificadas.
    """
    imputer = KNNImputer(n_neighbors=k)
    df_imputed = df.copy()
    df_imputed[cols] = imputer.fit_transform(df[cols])
    return df_imputed


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

    cols_to_drop = [
        'id_categoria', 'id_periodo', 'id_cliente', 'id_canal', 'volumen', 'volumen_sem', 'superficie_km2',
        'n_habitantes', 'prop_vol_masivo', 'canal', 'descr_flag_patente', 'volumen_sem_ar1', 'volumen_sem_ar2',
        'volumen_sem_fut', 'volumen_dif1', 'volumen_dif1_dif12', 'volumen_sem_dif6', 'volumen_sem_dif6_fut',
        'imacec', 'uf', 'tpm', 'ipc', 'tasa_desempleo', 'vol_sem_rel_dif6_lag6', 'vol_sem_rel_dif6_lag12', 'compra'
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

    cols_to_drop = [
        'id_categoria', 'id_periodo', 'id_cliente', 'id_canal', 'volumen', 'volumen_sem', 'superficie_km2',
        'n_habitantes', 'prop_vol_masivo', 'canal', 'descr_flag_patente', 'volumen_sem_ar1', 'volumen_sem_ar2',
        'volumen_sem_fut', 'volumen_dif1', 'volumen_dif1_dif12', 'volumen_sem_dif6', 'volumen_sem_dif6_fut',
        'imacec', 'uf', 'tpm', 'ipc', 'tasa_desempleo', 'vol_sem_rel_dif6_lag6', 'vol_sem_rel_dif6_lag12', 'compra'
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

def clean_for_ar_simple(
    df: pd.DataFrame,
    *,
    max_recency: int = 12,
    min_ar1: float = 1.0,
    clip_rel: tuple = (0.01, 0.99),
    clip_y: tuple = (0.01, 0.99),
    rel_col: str = "vol_sem_rel_dif6",
    y_col: str = "volumen_sem_dif6_fut",
    ar1_col: str = "vol_sem_lag6",   # o "volumen_sem_ar1" si prefieres
) -> pd.DataFrame:
    """
    Limpieza mínima post-features:
    1) recency: deja clientes activos (si existe la col)
    2) evita divisiones/ratios locos: ar1 mínimo
    3) clip robusto de outliers en rel y target
    4) deja todo finito
    """
    d = df.copy()

    # 1) Clientes vivos
    if "recency" in d.columns and max_recency is not None:
        d = d[d["recency"].notna() & (d["recency"] <= max_recency)]

    # 2) Evitar ratios explosivos por denominador chico
    if ar1_col in d.columns and min_ar1 is not None:
        d = d[d[ar1_col].notna() & (d[ar1_col] >= min_ar1)]

    # 3) Clipping robusto
    def _clip(col, q):
        if col in d.columns and d[col].notna().sum() > 100:
            lo, hi = d[col].quantile(q[0]), d[col].quantile(q[1])
            d[col] = d[col].clip(lo, hi)

    _clip(rel_col, clip_rel)
    _clip(y_col, clip_y)

    # 4) No finitos fuera
    num_cols = [c for c in [rel_col, y_col, ar1_col] if c in d.columns]
    for c in num_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=num_cols)

    return d.reset_index(drop=True)
