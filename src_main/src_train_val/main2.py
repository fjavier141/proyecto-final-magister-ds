# -*- coding: utf-8 -*-
import datetime
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
import xgboost as xgb

import src.preprocessing as pre
import src.utils as uts

# =========================
# Configuración de corrida
# =========================
TEST_PERIOD = 202412
CATEGORY = "cervezas"
DATA_PICKLE = f"./data/output/pickle/dataset_{CATEGORY}.pickle"
SEG_PREFIX = "seg_"
RANDOM_STATE = 42


# =========================
# Flujo principal
# =========================
def main():
    # 1) Particiones temporales (tus helpers)
    per_train, per_test = uts.get_dates(TEST_PERIOD)

    # 2) Carga y OHE de segmento
    dataset = uts.load_pickle(DATA_PICKLE)
    df = dataset.copy()
    df = pd.get_dummies(df, columns=["segmento"], prefix=SEG_PREFIX)

    # 3) Armar train/test (tus helpers)
    X_train = pre.get_train_data(df, per_train, CATEGORY)
    y_train = pre.get_train_target(df, per_train)
    X_test  = pre.get_val_data(df, per_test, CATEGORY)
    y_test  = pre.get_val_target(df, per_test)
    df_pred_like = pre.get_pred_set(df, per_test)  # debe traer 'volumen_sem' y 'volumen_sem_dif6_fut_real'

    # 5) Entrenar y evaluar cada modelo
    MODEL_ZOO = build_model_zoo()
    results = []

    for name, model in MODEL_ZOO.items():
        print(f"\n=== {name} ===")
        if model is None:
            # Baseline: predecir 0 cambio → futuro = volumen_sem actual
            yhat_diff6 = np.zeros(len(X_test), dtype=float)
        else:
            print("Comienzo entrenamiento:", datetime.datetime.now())
            model.fit(X_train, y_train)
            print("Fin entrenamiento:", datetime.datetime.now())
            yhat_diff6 = model.predict(X_test)

        df_out = reconstruct_predictions(df_pred_like, yhat_diff6)
        metrics = eval_metrics(df_out)
        results.append({"Modelo": name, **metrics})

        # Muestra rápida
        print(f"Corr: {metrics['Corr']:.3f} | WAPE: {metrics['WAPE']*100:.2f}% | R2: {metrics['R2']:.3f} | "
              f"MAE: {metrics['MAE']:.2f} | RMSE: {metrics['RMSE']:.2f} | sMAPE: {metrics['sMAPE']*100:.2f}% | n={metrics['n']}")

    # 6) Tabla ordenada por WAPE (menor es mejor)
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(by=["WAPE", "RMSE"]).reset_index(drop=True)
    print("\n===== Resultados comparados =====")
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return res_df


# =========================
# Reconstrucción de target
# =========================
def reconstruct_predictions(df_pred_like: pd.DataFrame, yhat_diff6: np.ndarray) -> pd.DataFrame:
    """
    df_pred_like DEBE contener al menos:
      - 'volumen_sem'                 (nivel del semestre actual)
      - 'volumen_sem_dif6_fut_real'   (para calcular métricas vs real)
    """
    out = df_pred_like.copy()
    out["volumen_sem_dif6_fut"] = yhat_diff6
    out["volumen_sem_fut_est"] = out["volumen_sem"] + out["volumen_sem_dif6_fut"]
    out["volumen_sem_fut_real"] = out["volumen_sem"] + out["volumen_sem_dif6_fut_real"]
    # Negativos a cero (regla de negocio)
    out.loc[out["volumen_sem_fut_est"] < 0, "volumen_sem_fut_est"] = 0
    # Limpieza mínima
    out.drop(columns=["volumen_sem_dif6_fut"], inplace=True)
    return out

# =========================
# Modelos a comparar
# =========================
def build_model_zoo():
    """
    Suma o quita modelos aquí.
    Todos predicen 'volumen_sem_dif6_fut' (el cambio semestral).
    """
    return {
        "Baseline_no_change": None,  # yhat_diff6 = 0
        "XGBoost": xgb.XGBRegressor(
            learning_rate=0.05,
            n_estimators=800,
            max_depth=3,
            min_child_weight=7,
            gamma=0.0,
            subsample=0.9,
            colsample_bytree=1.0,
            reg_alpha=10.0,
            reg_lambda=10.0,
            objective="reg:squarederror",
            n_jobs=10,
            random_state=RANDOM_STATE,
            tree_method="hist",
            eval_metric="rmse"
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),
        "Ridge": Ridge(alpha=1.0, random_state=None),
        "LightGBM": LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            max_depth=-1,        # sin límite, lo controla num_leaves
            num_leaves=63,       # controla complejidad
            min_child_samples=20,
            subsample=0.9,       # bagging_fraction si usas el API nativo
            colsample_bytree=0.9,# feature_fraction
            reg_alpha=5.0,
            reg_lambda=10.0,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            objective="regression",
            metric="rmse",
        )
    }


# =========================
# Utilidades de limpieza
# =========================
def make_finite(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    if isinstance(X, pd.DataFrame):
        X = X.apply(pd.to_numeric, errors="coerce")
    else:
        X = pd.DataFrame(X)
    X = X.replace([np.inf, -np.inf], np.nan)
    # Política simple y reproducible (ajústala si prefieres mediana):
    X = X.fillna(0)
    assert np.isfinite(X.to_numpy()).all(), "Siguen habiendo no finitos en X"
    return X

def make_finite_y(y: pd.Series) -> pd.Series:
    y = pd.to_numeric(pd.Series(y), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    assert np.isfinite(y.to_numpy()).all(), "Siguen habiendo no finitos en y"
    return y

# =========================
# Métricas
# =========================
def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    out = np.zeros_like(y_true, dtype=float)
    mask = denom != 0
    out[mask] = 2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]
    return np.mean(out)

def eval_metrics(df_out: pd.DataFrame) -> dict:
    y_true = df_out["volumen_sem_fut_real"]
    y_pred = df_out["volumen_sem_fut_est"]

    corr = y_true.corr(y_pred)
    wape = (np.abs(y_true - y_pred).sum() / (np.abs(y_true).sum() if np.abs(y_true).sum() != 0 else np.nan))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)
    sm = smape(y_true, y_pred)

    return {
        "Corr": corr,
        "WAPE": wape,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "sMAPE": sm,
        "n": len(df_out)
    }


if __name__ == "__main__":
    _ = main()
