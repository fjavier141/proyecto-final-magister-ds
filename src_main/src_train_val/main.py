import json

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
import category_encoders as ce
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import shap

import src.preprocessing as pre
from src_main.src_search_hyperparams.main import lightgbm_cross_validation
import src.utils as uts
from src.metrics import eval_metrics, growth_quartile_metrics, plot_confusion_matrix_percent, growth_binary_metrics
from parameters.config import *


SEGMENTS_TO_USE = SEGMENTS_CHANNEL[CHANNEL]

def main():
    validation_periods = uts.get_validation_periods(TEST_PERIOD, 2)
    dataset = uts.load_pickle(f'./data/output/pickle/dataset_{CATEGORY}.pickle')

    df = dataset.copy()
    df = df[df['segmento'].isin(SEGMENTS_TO_USE)]

    for validation_period in validation_periods:

        path_validation = os.path.join(f"./data/output/validation/{CATEGORY}/{CHANNEL}/{validation_period}")
        uts.create_directory_if_not_exists([path_validation])

        per_train, per_test = uts.get_dates(validation_period)

        df_encoded = pd.get_dummies(df, columns=['segmento'], prefix='seg')

        # ====== Split TRAIN (requiere y) ======
        train_x, train_y, train_like = prepare_split(df_encoded, per_train, CATEGORY, mode="train")

        # Limpieza SOLO en train para evitar colas que LightGBM puede sobre-aprender
        train_pack = train_x.copy()
        train_pack["__y__"] = train_y.values
        train_pack = clip_by_quantiles(train_pack, cols=["__y__"], q=(0.01, 0.99))
        train_y = train_pack["__y__"]
        train_x = train_pack.drop(columns=["__y__"])

        train_x = train_x.reset_index(drop=True)
        train_y = train_y.reset_index(drop=True)
        train_like = train_like.reset_index(drop=True)

        # ====== Split VAL (para evaluar requiere y_real en pred_like) ======
        # Para evaluar sí necesitamos y; pero no lo uses en X.
        test_x, test_y, pred_like = prepare_split(df_encoded, per_test, CATEGORY, mode="val")
        pred_like = pred_like.dropna(subset=["volumen_sem_dif6_fut_real"])

        # ====== Hyperparams (opcional) ======
        lightgbm_cross_validation(train_x, train_y, train_like, per_train, validation_period, CATEGORY, CHANNEL)

        # ====== Train final ======
        model = train_ligthgbm(train_x, train_y, CATEGORY, validation_period, RANDOM_STATE)

        # ====== Predict ======
        yhat_diff6 = model.predict(test_x)
        yhat_diff6_base = np.zeros(len(test_x), dtype=float)

        df_out = reconstruct_predictions(pred_like, yhat_diff6)
        df_out_base = reconstruct_predictions(pred_like, yhat_diff6_base)

        # ====== Métricas volumen ======
        metrics_volume = eval_metrics(df_out, 'volumen_sem_fut_real', 'volumen_sem_fut_est')
        print(f"\n Métricas de Volumen {validation_period}:")
        for k, v in metrics_volume.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, (int, float)) else f"  {k}: {v}")
        uts.save_json(metrics_volume, os.path.join(path_validation, "metrics_volume.json"))

        metrics_volume_base = eval_metrics(df_out_base, 'volumen_sem_fut_real', 'volumen_sem_fut_est')
        print(f"\n Métricas de Volumen Base {validation_period}:")
        for k, v in metrics_volume_base.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, (int, float)) else f"  {k}: {v}")
        uts.save_json(metrics_volume_base, os.path.join(path_validation, "metrics_volume_base.json"))

        # ====== Crecimiento ======
        df_out_grow = calculate_grow(df_out)
        df_out_grow_base = calculate_grow(df_out_base)

        # ====== Cuartiles de crecimiento LigthGBM ======
        class_metrics_grow = growth_quartile_metrics(df_out_grow, 'crecimiento_fut_real', 'crecimiento_fut_est')
        plot_confusion_matrix_percent(class_metrics_grow["confusion_matrix"], labels=class_metrics_grow["labels"],
                                      title=f'Cuartiles Crecimiento ligthgbm {validation_period}', path_save=os.path.join(path_validation, "confusion_matrix_lithgbm.png"))
        print(f'Accuracy cuartiles crecimiento modelo ligthgbm {validation_period}: {class_metrics_grow['accuracy']:.3f}')
        print(f'F1 ponderado crecimiento modelo ligthgbm {validation_period}: {class_metrics_grow['f1_weighted']:.3f}')
        uts.save_json(class_metrics_grow, os.path.join(path_validation, "class_metrics_grow.json"))

        # ====== Cuartiles de crecimiento Baseline ======
        class_metrics_grow_base = growth_quartile_metrics(df_out_grow_base, 'crecimiento_fut_real', 'crecimiento_fut_est')
        plot_confusion_matrix_percent(class_metrics_grow_base["confusion_matrix"], labels=class_metrics_grow_base["labels"],
                                      title=f'Cuartiles Crecimiento base {validation_period}', path_save=os.path.join(path_validation, "confusion_matrix_base.png"))
        print(f'Accuracy cuartiles crecimiento modelo baseline {validation_period}: {class_metrics_grow_base['accuracy']:.3f}')
        print(f'F1 ponderado crecimiento modelo baseline {validation_period}: {class_metrics_grow_base['f1_weighted']:.3f}')
        uts.save_json(class_metrics_grow_base, os.path.join(path_validation, "class_metrics_grow_base.json"))

        # ====== Binario (p50) crecimiento LightGBM ======
        bin_metrics_grow = growth_binary_metrics(df_out_grow, 'crecimiento_fut_real', 'crecimiento_fut_est', q_cut=0.50)
        plot_confusion_matrix_percent(
            np.array(bin_metrics_grow["confusion_matrix_percent"]),
            labels=bin_metrics_grow["labels"],
            title=f'Bajo/Alto (p50) Crecimiento ligthgbm {validation_period}',
            path_save=os.path.join(path_validation, "confusion_matrix_binary_lithgbm.png")
        )
        print(f'Cut p50 (real) ligthgbm {validation_period}: {bin_metrics_grow["cut_value"]:.6f}')
        print(f'Accuracy binario ligthgbm {validation_period}: {bin_metrics_grow["accuracy"]:.3f}')
        print(f'F1 weighted binario ligthgbm {validation_period}: {bin_metrics_grow["f1_weighted"]:.3f}')
        uts.save_json(bin_metrics_grow, os.path.join(path_validation, "class_metrics_grow_binary.json"))

        # ====== Binario (p50) crecimiento Baseline ======
        bin_metrics_grow_base = growth_binary_metrics(df_out_grow_base, 'crecimiento_fut_real', 'crecimiento_fut_est',
                                                      q_cut=0.50)
        plot_confusion_matrix_percent(
            np.array(bin_metrics_grow_base["confusion_matrix_percent"]),
            labels=bin_metrics_grow_base["labels"],
            title=f'Bajo/Alto (p50) Crecimiento base {validation_period}',
            path_save=os.path.join(path_validation, "confusion_matrix_binary_base.png")
        )
        print(f'Cut p50 (real) baseline {validation_period}: {bin_metrics_grow_base["cut_value"]:.6f}')
        print(f'Accuracy binario baseline {validation_period}: {bin_metrics_grow_base["accuracy"]:.3f}')
        print(f'F1 weighted binario baseline {validation_period}: {bin_metrics_grow_base["f1_weighted"]:.3f}')
        uts.save_json(bin_metrics_grow_base, os.path.join(path_validation, "class_metrics_grow_binary_base.json"))

        # ====== Importancia de las variables conjunto de Train ======
        explain_model(model, train_x, CATEGORY, CHANNEL, validation_period, path_validation, mode='train')
        plot_shap_dependence(model, train_x, 'ar0', CATEGORY, CHANNEL, validation_period, path_validation,'train')

        # ====== Importancia de las variables conjunto de Validación ======
        explain_model(model, test_x, CATEGORY, CHANNEL, validation_period, path_validation, mode='test')
        plot_shap_dependence(model, test_x, 'ar0', CATEGORY, CHANNEL, validation_period, path_validation,'test')

        save_validation_dataset(df, df_out_grow, df_out_grow_base, path_validation,
                                validation_period)


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
        "n_jobs": 8,
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

    te = ce.TargetEncoder(cols=[c for c in ["id_barrio","id_comuna"] if c in train_x.columns])

    lgbm_model = Pipeline([
        ("te", te),
        ("lgbm", LGBMRegressor(**lgbm_params)),
    ])
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

def calculate_grow(df):
    df1 = df.copy()
    denominator = df1["volumen_sem_ar1"] + df1["volumen_sem"]

    #Filtrar filas donde el denominador es distinto de cero
    mask = denominator > 1e-6
    df_new = df1[mask]
    den = (df_new["volumen_sem_ar1"] + df_new["volumen_sem"])
    df_new['crecimiento_fut_real'] = (df_new["volumen_sem_fut_real"] + df_new["volumen_sem"]) / den
    df_new['crecimiento_fut_est'] = (df_new["volumen_sem_fut_est"] + df_new["volumen_sem"]) / den

    return df_new


def clip_by_quantiles(df: pd.DataFrame, cols: list[str], q=(0.01, 0.99)) -> pd.DataFrame:
    """Clipping robusto, pensado para TRAIN (evitar que el modelo se coma colas infinitas)."""
    d = df.copy()
    for c in cols:
        if c in d.columns:
            s = pd.to_numeric(d[c], errors="coerce")
            if s.notna().sum() > 200:
                lo, hi = s.quantile(q[0]), s.quantile(q[1])
                d[c] = s.clip(lo, hi)
    return d


def prepare_split(df, periods, category, *, mode: str):
    """
    Devuelve:
    - X listo (sin columnas no entrenables)
    - y (si require_y=True, alineado) o None
    - pred_like (id + volumen_sem + ar1 + y_real si existe) para reconstrucción
    """
    d = df[df["id_periodo"].isin(periods)].copy()

    req_feats = ["volumen_sem", "vol_sem_lag6", "vol_sem_dif6"]
    req_feats = [c for c in req_feats if c in d.columns]
    d = d.dropna(subset=req_feats)

    # y real existe o no
    d = d.dropna(subset=["volumen_sem_dif6_fut"])

    pred_cols = ["id_categoria","id_periodo","id_cliente","volumen_sem","volumen_sem_ar1","volumen_sem_dif6_fut"]
    pred_cols = [c for c in pred_cols if c in d.columns]
    pred_like = d[pred_cols].copy()
    pred_like["volumen_sem_dif6_fut_real"] = pred_like["volumen_sem_dif6_fut"]

    if mode == "train":
        X = pre.get_train_data(d, periods, category)
    elif mode == "val":
        X = pre.get_val_data(d, periods, category)
    else:
        raise ValueError("mode debe ser 'train' o 'val'")

    y = pre.get_train_target(d, periods)  # el real, siempre igual

    return X.reset_index(drop=True), y.reset_index(drop=True), pred_like.reset_index(drop=True)


def save_validation_dataset(dataset, df_validation_ligthgbm, df_validation_baseline, path_validation, validation_period):
    df = dataset.copy()
    df = df[df['id_periodo'].isin([validation_period])].reset_index(drop=True)
    df = df[['id_cliente', 'volumen_sem', 'volumen_sem_ar1', 'segmento', 'id_barrio', 'id_comuna', 'indice_gse',
             'frecuency', 'recency']]

    df["size_base"] = (
            df["volumen_sem"].fillna(0)
            + df["volumen_sem_ar1"].fillna(0)
    )

    df_seg = df[['id_cliente', 'segmento', 'id_barrio', 'id_comuna', 'indice_gse', 'frecuency', 'recency',
                 'size_base']].drop_duplicates(subset='id_cliente')

    df_scoring_lgbm = df_validation_ligthgbm.merge(df_seg, on="id_cliente", how="left")
    df_scoring_base = df_validation_baseline.merge(df_seg, on="id_cliente", how="left")
    df_scoring_lgbm["tam_vol"] = pd.qcut(
        df_scoring_lgbm["size_base"],
        q=[0, 0.333, 0.666, 1.0],
        labels=["Chico", "Mediano", "Grande"],
        duplicates="drop"
    ).astype(str)

    df_scoring_base["tam_vol"] = pd.qcut(
        df_scoring_base["size_base"],
        q=[0, 0.333, 0.666, 1.0],
        labels=["Chico", "Mediano", "Grande"],
        duplicates="drop"
    ).astype(str)


    uts.save_pickle(df_scoring_lgbm, os.path.join(path_validation, "scoring_lgbm.pkl"))
    uts.save_pickle(df_scoring_base, os.path.join(path_validation, "scoring_base.pkl"))


### IAX ###

def explain_model(model, X, category, channel, validation_period, path_validation, mode="train"):
    """
    Explica un Pipeline (TargetEncoder + LGBM) usando SHAP TreeExplainer.
    - Extrae el estimador de árbol del pipeline
    - Transforma X con el encoder antes de SHAP
    """

    # 1) Extraer steps
    te = model.named_steps.get("te", None)
    lgbm = model.named_steps.get("lgbm", None)

    if lgbm is None:
        raise ValueError("No encontré el step 'lgbm' dentro del Pipeline. Revisa el nombre del step.")

    # 2) Transformar X con el encoder (si existe)
    if te is not None:
        X_enc = te.transform(X)   # TargetEncoder entrega DataFrame/ndarray
    else:
        X_enc = X

    # 3) Asegurar DataFrame con nombres (shap.summary_plot lo agradece)
    if not isinstance(X_enc, pd.DataFrame):
        X_enc = pd.DataFrame(X_enc, columns=getattr(X, "columns", None))

    # 4) SHAP sobre el estimador tree (NO el pipeline)
    explainer = shap.TreeExplainer(lgbm)
    shap_values = explainer.shap_values(X_enc)

    # 5) Plots
    plt.figure(figsize=(14, 8))
    shap.summary_plot(shap_values, X_enc, show=False)
    plt.title(f"Impacto de Variables (SHAP) - {category} {channel} {validation_period} ({mode})")
    plt.savefig(os.path.join(path_validation, f"shap_summary_{mode}.png"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(14, 8))
    shap.summary_plot(shap_values, X_enc, plot_type="bar", show=False)
    plt.title(f"Importancia de Variables (SHAP) - {category} {channel} {validation_period} ({mode})")
    plt.savefig(os.path.join(path_validation, f"shap_importance_{mode}.png"), bbox_inches="tight")
    plt.close()

    return shap_values



def plot_shap_dependence(model, X, feature_name, category, channel, validation_period, path_validation, mode="train",
    interaction_feature="auto"):
    """
    Dependence plot SHAP para un Pipeline (TargetEncoder + LGBM).
    Explica SOLO el estimador LGBM y transforma X con el TargetEncoder.
    """

    # 1) Extraer steps del pipeline
    te = model.named_steps.get("te", None)
    lgbm = model.named_steps.get("lgbm", None)

    if lgbm is None:
        raise ValueError("No se encontró el step 'lgbm' dentro del Pipeline.")

    # 2) Transformar X con TargetEncoder (si existe)
    if te is not None:
        X_enc = te.transform(X)
    else:
        X_enc = X.copy()

    # 3) Asegurar DataFrame con nombres de columnas
    if not isinstance(X_enc, pd.DataFrame):
        X_enc = pd.DataFrame(X_enc, columns=X.columns)

    # 4) Validar que la feature exista post-encoding
    if feature_name not in X_enc.columns:
        raise ValueError(
            f"feature '{feature_name}' no existe en X post-encoding. "
            f"Columnas disponibles: {list(X_enc.columns)[:15]}..."
        )

    # 5) SHAP sobre el estimador de árboles
    explainer = shap.TreeExplainer(lgbm)
    shap_values = explainer.shap_values(X_enc)

    # 6) Plot
    plt.figure(figsize=(14, 8))
    shap.dependence_plot(
        feature_name,
        shap_values,
        X_enc,
        interaction_index=interaction_feature,
        show=False
    )

    plt.title(
        f"Dependencia SHAP: {feature_name} "
        f"({category} {channel} {validation_period}) [{mode}]"
    )

    out_path = os.path.join(
        path_validation, f"shap_dependence_{feature_name}_{mode}.png"
    )
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    return out_path
