import os
import json
import datetime
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import xgboost as xgb
import pandas as pd
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor

import src.preprocessing as pre
from src_main.src_search_hyperparams.main import lightgbm_cross_validation
import src.utils as uts

# Cargar .env (ajusta ruta a tu entorno)
env_path = ".env"
load_dotenv(dotenv_path = env_path)
USE_SAVED_HYPERPARAMS = True  #pon False si quieres usar siempre los estándar
HYPERPARAMS_DIR = "./data/output/hyperparams"


segments = ["AL", "BO", "AP", "KI", "BA", "EE", "ES", "FF", "FU", "CD", "IE", "DI", "RT", "RE", "BC", "RC", "GI",
            "FC", "FS", "RD"]


def main():
    test_period = int(os.getenv("TEST_PERIOD")) # Formato AAAAMM, actualmente configurado como : 202412
    category = os.getenv("CATEGORY") # 'cervezas' o 'analcoholicos', manejar desde .env la categoría que se usará.
    random_state = os.getenv("RANDOM_STATE")

    per_train, per_test = uts.get_dates(test_period)

    dataset = uts.load_pickle(f'./data/output/pickle/dataset_{category}.pickle')
    df = dataset.copy()
    df_encoded = pd.get_dummies(df, columns=['segmento'], prefix='seg')

    train_x = pre.get_train_data(df_encoded, per_train, category)
    train_y = pre.get_train_target(df_encoded, per_train)
    test_x = pre.get_val_data(df_encoded, per_test, category)
    test_y = pre.get_val_target(df_encoded, per_test)
    pred = pre.get_pred_set(df_encoded, per_test)

    lightgbm_cross_validation(train_x, train_y, df, per_train, test_period, category)

    model = train_ligthgbm(train_x, train_y, category, test_period, random_state)

    yhat_diff6 = model.predict(test_x)

    df_out = reconstruct_predictions(pred, yhat_diff6)

    # Correlación de Pearson
    corr = df_out['volumen_sem_fut_real'].corr(df_out['volumen_sem_fut_est'])

    # WAPE (Weighted Absolute Percentage Error)
    wape = (
            abs(df_out['volumen_sem_fut_real'] - df_out['volumen_sem_fut_est']).sum()
            / df_out['volumen_sem_fut_real'].sum()
    )

    # R² (Coeficiente de determinación)
    r2 = r2_score(df_out['volumen_sem_fut_real'], df_out['volumen_sem_fut_est'])

    print(f"Correlación: {corr:.3f}")
    print(f"WAPE: {wape:.3%}")
    print(f"R²: {r2:.3f}")


def train_ligthgbm(train_x, train_y, category, test_period, random_state):
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
        "random_state": random_state,
        "objective": "regression",
        "metric": "rmse",
    }

    if USE_SAVED_HYPERPARAMS:
        saved = load_hyperparams_from_disk("LightGBM", category, test_period)
        if saved:
            for k, v in saved.items():
                if k in lgbm_params:
                    lgbm_params[k] = v

    lgbm_model = LGBMRegressor(**lgbm_params)
    lgbm_model.fit(train_x, train_y)
    return lgbm_model


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


def make_finite(X):
    X = X.copy()
    # fuerza numéricos
    if isinstance(X, pd.DataFrame):
        X = X.apply(pd.to_numeric, errors="coerce")
    else:
        X = pd.DataFrame(X)  # por si te pasan ndarray
    # reemplaza inf/-inf por NaN y luego imputa
    X = X.replace([np.inf, -np.inf], np.nan)
    # imputación simple: 0 o mediana (elige tu política)
    X = X.fillna(0)  # o X.fillna(X.median(numeric_only=True))
    # asegura finitos
    assert np.isfinite(X.to_numpy()).all(), "Siguen habiendo no finitos en X"
    return X


def make_finite_y(y):
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