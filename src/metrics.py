import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


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
