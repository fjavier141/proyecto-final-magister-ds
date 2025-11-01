import datetime
import os

import numpy as np
import pandas as pd
import xgboost as xgb
import pandas as pd
from sklearn.metrics import r2_score

import src.preprocessing as pre
import src.utils as uts

segments = ["AL", "BO", "AP", "KI", "BA", "EE", "ES", "FF", "FU", "CD", "IE", "DI", "RT", "RE", "BC", "RC", "GI",
            "FC", "FS", "RD"]


def main():
    test_period = 202412
    category = 'cervezas'
    per_train, per_test = uts.get_dates(test_period)

    dataset = uts.load_pickle(f'./data/output/pickle/dataset_{category}.pickle')
    df = dataset.copy()
    df_encoded = pd.get_dummies(df, columns=['segmento'], prefix='seg')

    train_x = pre.get_train_data(df_encoded, per_train, category)
    train_y = pre.get_train_target(df_encoded, per_train)
    train_x = make_finite(train_x)
    train_y = make_finite_y(train_y)

    #seg_columns = [col for col in segments if col in df.columns]
    #df = dataset[dataset[seg_columns].eq(1).any(axis=1)]
    test_x = pre.get_val_data(df_encoded, per_test, category)
    test_y = pre.get_val_target(df_encoded, per_test)
    pred = pre.get_pred_set(df_encoded, per_test)

    reg = entrenar_xgboost(train_x, train_y)
    df_out = get_df_out(pred, reg, test_x)

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




def entrenar_xgboost(train_x, train_y):
    """
    Entrena un modelo XGBoost con early stopping utilizando test_x y test_y como conjunto de validación.
    """
    reg = xgb.XGBRegressor(
        learning_rate=0.05,
        n_estimators=800,
        max_depth=3,
        min_child_weight=7,
        gamma=0.0,
        subsample= 0.9,
        colsample_bytree=1,
        reg_alpha=10,
        reg_lambda=10,
        objective="reg:squarederror",
        nthread=10
    )

    print('Comienzo entrenamiento: ', datetime.datetime.now())
    reg.fit(
        train_x,
        train_y,
        #eval_set=[(test_x, test_y)],
        #early_stopping_rounds=50,
        verbose=True
    )
    print('Fin entrenamiento: ', datetime.datetime.now())

    return reg


def get_df_out(df, reg, test_x):
    """
    Genera las predicciones finales y ajusta el DataFrame original con las proyecciones desescaladas.
    """
    df1 = df.copy()
    df1['volumen_sem_dif6_fut'] = reg.predict(test_x)
    df1['volumen_sem_fut_est'] = df1['volumen_sem'] + df1['volumen_sem_dif6_fut']
    df1['volumen_sem_fut_real'] = df1['volumen_sem'] + df1['volumen_sem_dif6_fut_real']
    df1.loc[df1['volumen_sem_fut_est'] < 0, 'volumen_sem_fut_est'] = 0
    df1['volumen_sem_ar1'] = df1['volumen_sem_ar1'].fillna(0)
    df1['mape'] = abs((df1['volumen_sem_fut_real'] - df1['volumen_sem_fut_est']) / df1['volumen_sem_fut_real'])
    df1.drop(df1[df1['volumen_sem_fut_est'].isna()].index, inplace=True)
    cols_to_drop = ['volumen_sem_dif6_fut', 'volumen_sem_dif6_fut_real']
    df1.drop(columns=cols_to_drop, inplace=True)
    return df1



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