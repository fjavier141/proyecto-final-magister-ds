import numpy as np
import pandas as pd


def main(df_macro_vars):

    macro_cols = ["imacec", "uf", "tasa_desempleo", "tpm", "ipc"]

    for col in macro_cols:
        df_macro_vars = add_macro_trends(df_macro_vars, col, window=6)

    # Dummy COVID
    df_macro_vars["covid"] = 0
    df_macro_vars.loc[df_macro_vars["id_periodo"].between(202003, 202106), "covid"] = 1

    return df_macro_vars


def add_macro_trends(df: pd.DataFrame, column: str, window: int = 6):
    """
    Calcula tendencias macro BACKWARD-LOOKING (sin fuga temporal).

    Genera:
    - {col}_sem        : media móvil
    - {col}_chg_{w}m   : cambio absoluto
    - {col}_pct_{w}m   : cambio relativo (si aplica)
    - {col}_trend_{w}m : pendiente OLS local
    - {col}_std_{w}m   : volatilidad
    """

    s = df[column]

    # Nivel suavizado
    df[f"{column}_sem"] = s.rolling(window, min_periods=window).mean()

    # Cambio absoluto
    df[f"{column}_chg_{window}m"] = s - s.shift(window)

    # Cambio relativo (solo si no explota)
    eps = 1e-6
    df[f"{column}_pct_{window}m"] = (s - s.shift(window)) / (s.shift(window) + eps)

    # Volatilidad
    df[f"{column}_std_{window}m"] = s.rolling(window, min_periods=window).std()

    # Tendencia local (slope OLS)
    def _slope(x):
        if np.any(~np.isfinite(x)):
            return np.nan
        t = np.arange(len(x))
        return np.polyfit(t, x, 1)[0]

    df[f"{column}_trend_{window}m"] = (
        s.rolling(window, min_periods=window)
         .apply(_slope, raw=False)
    )

    return df
