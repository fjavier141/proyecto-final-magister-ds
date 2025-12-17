# -*- coding: utf-8 -*-
import datetime
import json
import os
from dotenv import load_dotenv


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
import xgboost as xgb

import src.preprocessing as pre
import src.utils as uts
from src_main.src_search_hyperparams.main import *
from src_main.src_train_val.iax_preliminar import log_iax_metrics
from src_main.src_train_val.metrics import eval_metrics

# Cargar .env (ajusta ruta a tu entorno)
env_path = "/Users/diegobascunan/PycharmProjects/proyecto-final-magister-ds/accesos_diego.env"
#env_path = "D:\\Users\\fjavi\\Proyectos\\proyecto-final-magister-ds\\.env"
load_dotenv(dotenv_path = env_path)

# =========================
# Configuración de corrida
# =========================
TEST_PERIOD = int(os.getenv("TEST_PERIOD")) # Formato AAAAMM, fijado env como : 202412
CATEGORY = os.getenv("CATEGORY") # 'cervezas' o 'analcoholicos' según .env
DATA_PICKLE = f"./data/output/pickle/dataset_{CATEGORY}_knn.pickle"
SEG_PREFIX = "seg_"
RANDOM_STATE = int(os.getenv("RANDOM_STATE") )# para reproducibilidad fijado en .env como 42
# Flag global: usar hiperparámetros guardados o defaults del código
USE_SAVED_HYPERPARAMS = True  # pon False si quieres usar siempre los estándar
HYPERPARAMS_DIR = "./data/output/hyperparams"

# =========================
# Flujo principal
# =========================
def main():
    # 1) Particiones temporales (tus helpers)
    per_train, per_test = uts.get_dates(TEST_PERIOD)
    test_date = per_test[0]

    # 2) Carga y OHE de segmento
    dataset = uts.load_pickle(DATA_PICKLE)
    df = dataset.copy()
    df_encoded = pd.get_dummies(df, columns=["segmento"], prefix=SEG_PREFIX)

    # 3) Armar train/test (tus helpers)
    X_train = pre.get_train_data(df_encoded, per_train, CATEGORY)
    y_train = pre.get_train_target(df_encoded, per_train)
    X_test  = pre.get_val_data(df_encoded, per_test, CATEGORY)
    y_test  = pre.get_val_target(df_encoded, per_test)
    df_pred_like = pre.get_pred_set(df, per_test)  # debe traer 'volumen_sem' y 'volumen_sem_dif6_fut_real'

    #xgboost_cross_validation(X_train, y_train, df, per_train, test_date, CATEGORY)
    #lightgbm_cross_validation(X_train, y_train, df, per_train, test_date, CATEGORY)
    #random_forest_cross_validation(X_train, y_train, df, per_train, test_date, CATEGORY)
    #ridge_cross_validation(X_train, y_train, df, per_train, test_date, CATEGORY)


    # 5) Entrenar y evaluar cada modelo
    MODEL_ZOO = build_model_zoo(use_saved_hyperparams=USE_SAVED_HYPERPARAMS, category=CATEGORY, test_period=TEST_PERIOD)
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
        log_iax_metrics(name, model, X_test, df_out)

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
def build_model_zoo(use_saved_hyperparams: bool = False,
                    category: str = CATEGORY,
                    test_period: int = TEST_PERIOD):
    """
    Suma o quita modelos aquí.
    Todos predicen 'volumen_sem_dif6_fut' (el cambio semestral).

    Si use_saved_hyperparams=True, intenta leer hiperparámetros desde disco
    y sobreescribir los defaults definidos aquí.
    """
    # =============================
    # XGBoost
    # =============================
    xgb_params = {
        "learning_rate": 0.05,
        "n_estimators": 800,
        "max_depth": 3,
        "min_child_weight": 7,
        "gamma": 0.0,
        "subsample": 0.9,
        "colsample_bytree": 1.0,
        "reg_alpha": 10.0,
        "reg_lambda": 10.0,
        "objective": "reg:squarederror",
        "n_jobs": 10,              # controlas threads desde acá
        "random_state": RANDOM_STATE,
        "tree_method": "hist",
        "eval_metric": "rmse",
    }

    if use_saved_hyperparams:
        saved = load_hyperparams_from_disk("XGBoost", category, test_period)
        if saved:
            # Solo actualizamos claves que existan en el dict del modelo
            for k, v in saved.items():
                if k in xgb_params:
                    xgb_params[k] = v

    xgb_model = xgb.XGBRegressor(**xgb_params)

    # =============================
    # RandomForest
    # =============================
    rf_params = {
        "n_estimators": 600,
        "max_depth": None,
        "min_samples_leaf": 1,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }

    if use_saved_hyperparams:
        saved = load_hyperparams_from_disk("RandomForest", category, test_period)
        if saved:
            for k, v in saved.items():
                if k in rf_params:
                    rf_params[k] = v

    rf_model = RandomForestRegressor(**rf_params)

    # =============================
    # Ridge
    # =============================
    ridge_params = {
        "alpha": 1.0,
        "fit_intercept": True,
        # random_state Ridge sólo lo usa con algunos solvers, lo dejamos fijo o None
    }

    if use_saved_hyperparams:
        saved = load_hyperparams_from_disk("Ridge", category, test_period)
        if saved:
            for k, v in saved.items():
                if k in ridge_params:
                    ridge_params[k] = v

    ridge_model = Ridge(**ridge_params)

    # =============================
    # LightGBM
    # =============================
    lgbm_params = {
        "n_estimators": 1200,
        "learning_rate": 0.03,
        "max_depth": -1,          # sin límite, lo controla num_leaves
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 5.0,
        "reg_lambda": 10.0,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        "objective": "regression",
        "metric": "rmse",
    }

    if use_saved_hyperparams:
        saved = load_hyperparams_from_disk("LightGBM", category, test_period)
        if saved:
            for k, v in saved.items():
                if k in lgbm_params:
                    lgbm_params[k] = v

    lgbm_model = LGBMRegressor(**lgbm_params)

    # =============================
    # Armamos el zoo
    # =============================
    return {
        "Baseline_no_change": None,   # yhat_diff6 = 0
        "XGBoost": xgb_model,
        "RandomForest": rf_model,
        "Ridge": ridge_model,
        "LightGBM": lgbm_model,
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

def load_hyperparams_from_disk(model_name: str,
                               category: str,
                               test_period: int,
                               base_dir: str = HYPERPARAMS_DIR) -> dict:
    """
    Lee hiperparámetros desde disco para un modelo dado.

    Convención de nombres de archivo:
      - XGBoost:      hyperparams_{category}_{test_period}.json
      - LightGBM:     hyperparams_lgbm_{category}_{test_period}.json
      - RandomForest: hyperparams_rf_{category}_{test_period}.json
      - Ridge:        hyperparams_ridge_{category}_{test_period}.json
    """

    if model_name == "XGBoost":
        filename = f"hyperparams_xgb_{category}_{test_period}.json"
    elif model_name == "LightGBM":
        filename = f"hyperparams_lgbm_{category}_{test_period}.json"
    elif model_name == "RandomForest":
        filename = f"hyperparams_rf_{category}_{test_period}.json"
    elif model_name == "Ridge":
        filename = f"hyperparams_ridge_{category}_{test_period}.json"
    else:
        return {}

    path = os.path.join(base_dir, filename)

    if not os.path.exists(path):
        print(f"[WARN] No se encontró archivo de hiperparámetros para {model_name}: {path}")
        return {}

    try:
        with open(path, "r") as f:
            params = json.load(f)
        print(f"[INFO] Hiperparámetros cargados para {model_name} desde {path}")
        return params
    except Exception as e:
        print(f"[WARN] Error leyendo hiperparámetros de {path}: {e}")
        return {}



if __name__ == "__main__":
    _ = main()
