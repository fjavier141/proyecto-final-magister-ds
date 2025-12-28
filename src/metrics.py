import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    out = np.zeros_like(y_true, dtype=float)
    mask = denom != 0
    out[mask] = 2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]
    return np.mean(out)


def eval_metrics(df_out: pd.DataFrame, y_true_label, y_pred_label) -> dict:
    y_true = df_out[y_true_label]
    y_pred = df_out[y_pred_label]

    corr = y_true.corr(y_pred)
    wape = (np.abs(y_true - y_pred).sum() / (np.abs(y_true).sum() if np.abs(y_true).sum() != 0 else np.nan))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
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


def growth_quartile_metrics(df_out: pd.DataFrame, col_real: str, col_pred: str) -> dict:
    """
    Calcula métricas de clasificación en cuartiles de crecimiento.

    - Define cuartiles en base a `col_real`
    - Asigna cuartil a col_real y col_pred
    - Devuelve:
        * accuracy
        * F1 macro
        * F1 weighted (ponderado por soporte)
        * confusion_matrix (4x4, para clases 0,1,2,3)
        * labels (orden de las clases en la matriz)
    """
    df = df_out.copy()

    # Limpiar NA
    df.dropna(subset=[col_real, col_pred], inplace=True)

    if df.empty:
        return {
            "accuracy": np.nan,
            "f1_macro": np.nan,
            "f1_weighted": np.nan,
            "confusion_matrix": None,
            "labels": [0, 1, 2, 3],
        }

    # Cuartiles sobre el crecimiento REAL
    q1, q2, q3 = df[col_real].quantile([0.25, 0.5, 0.75])

    def bucket(s: pd.Series) -> pd.Series:
        return np.where(
            s <= q1, 0,
            np.where(
                s <= q2, 1,
                np.where(
                    s <= q3, 2, 3
                )
            )
        )

    df["q_real"] = bucket(df[col_real])
    df["q_pred"] = bucket(df[col_pred])

    # Accuracy simple
    accuracy = (df["q_real"] == df["q_pred"]).mean()

    # F1 por clases (4 clases: 0,1,2,3)
    f1_macro = f1_score(df["q_real"], df["q_pred"],
                        average="macro", zero_division=0)
    f1_weighted = f1_score(df["q_real"], df["q_pred"],
                           average="weighted", zero_division=0)

    # Matriz de confusión (filas = reales, columnas = predichos)
    labels = [0, 1, 2, 3]
    cm = confusion_matrix(df["q_real"], df["q_pred"], labels=labels)

    return {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm.tolist(),  # para poder serializar
        "labels": labels,
    }

def plot_confusion_matrix_percent(cm, labels=[0, 1, 2, 3], title='xgboost', path_save=None):
    """
    Genera un heatmap de la matriz de confusión normalizada por fila (porcentajes).

    - cm: matriz 2D (list of lists o np.array)
    - labels: nombres de las clases (0-3)
    """
    cm = np.array(cm)

    # Normalizar por fila (sum(axis=1))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = cm / np.where(row_sums == 0, 1, row_sums)  # evitar div. por cero

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5
    )

    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title(f"Matriz de confusión (porcentajes) — {title}")
    plt.tight_layout()
    plt.savefig(path_save)
    plt.show()


def growth_binary_metrics(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    *,
    q_cut: float = 0.50,
    labels=("Bajo", "Alto"),
) -> dict:
    """
    Clasificación binaria de crecimiento:
    - Bajo: <= p50 de y_true
    - Alto:  > p50 de y_true

    Devuelve confusion_matrix (conteos), confusion_matrix_percent, accuracy, f1 (macro/weighted), etc.
    """
    d = df[[y_true_col, y_pred_col]].copy()
    d[y_true_col] = pd.to_numeric(d[y_true_col], errors="coerce")
    d[y_pred_col] = pd.to_numeric(d[y_pred_col], errors="coerce")
    d = d.dropna()

    if len(d) == 0:
        return {
            "labels": list(labels),
            "cut_value": np.nan,
            "n": 0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "confusion_matrix_percent": [[0.0, 0.0], [0.0, 0.0]],
            "accuracy": np.nan,
            "f1_macro": np.nan,
            "f1_weighted": np.nan,
            "precision_macro": np.nan,
            "recall_macro": np.nan,
        }

    cut = d[y_true_col].quantile(q_cut)

    y_true_bin = (d[y_true_col] > cut).astype(int)  # 0=Bajo, 1=Alto
    y_pred_bin = (d[y_pred_col] > cut).astype(int)

    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    cm_percent = (cm / cm.sum(axis=1, keepdims=True)) * 100.0
    cm_percent = np.nan_to_num(cm_percent, nan=0.0)

    return {
        "labels": list(labels),
        "cut_value": float(cut),
        "n": int(len(d)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_percent": cm_percent.tolist(),
        "accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
        "f1_macro": float(f1_score(y_true_bin, y_pred_bin, average="macro")),
        "f1_weighted": float(f1_score(y_true_bin, y_pred_bin, average="weighted")),
        "precision_macro": float(precision_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)),
    }
