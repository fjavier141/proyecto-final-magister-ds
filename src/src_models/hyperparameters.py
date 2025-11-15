import datetime

from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pandas as pd
import xgboost as xgb
import category_encoders as ce


def crear_grid_search_cv(param_test: dict, hyperparams: dict, gs_cv: dict):
    """
    Crea un objeto para realizar grid search cross validation.

    :param param_test: hiperparámetros óptimos que se quiere encontrar
    :param hyperparams: posee los hiperparámetros que se utilizarán en el xgboost
    :param gs_cv: posee los parámetros que se utilizarán en el grid search cross validation.
    :return:
    """
    ce_te = ce.TargetEncoder(cols=['ID_BARRIO'])
    reg = xgb.XGBRegressor(
        learning_rate=hyperparams['learning_rate'],
        n_estimators=hyperparams['n_estimators'],
        max_depth=hyperparams['max_depth'],
        min_child_weight=hyperparams['min_child_weight'],
        gamma=hyperparams['gamma'],
        subsample=hyperparams['subsample'],
        colsample_bytree=hyperparams['colsample_bytree'],
        reg_alpha=hyperparams.get('reg_alpha', 0),
        reg_lambda=hyperparams.get('reg_lambda', 1),
        objective=hyperparams['objective'],
        nthread=hyperparams['nthread']
    )

    pipe = Pipeline([
        ('cat_scaler', ce_te),
        ('xgb_model', reg)]
    )

    gsearch = GridSearchCV(estimator=pipe,
                           param_grid=param_test,
                           scoring=gs_cv['scoring'],
                           n_jobs=gs_cv['n_jobs'],
                           cv=gs_cv['cv'],
                           return_train_score=True)
    return gsearch


def reporte_gsearch(gsearch):
    print('Mean test score: ', gsearch.cv_results_['mean_test_score'])
    print('Std test score: ', gsearch.cv_results_['std_test_score'])
    print('Mean validation score: ', gsearch.cv_results_['mean_train_score'])
    print('Std validation score: ', gsearch.cv_results_['std_train_score'])
    print('Parámetros: ', gsearch.cv_results_['params'])
    print('Mejor hyperparametro: ', gsearch.best_params_)
    print('Mejor score: ', gsearch.best_score_)



def hyp_max_depth(train_x: pd.DataFrame, train_y: pd.DataFrame, hyperparams: dict, gs_cv: dict):
    """
    Buscar max_depth y min_child_weight

    :param train_x: features del conjunto de entrenamiento
    :param train_y: variable dependiente del conjunto de entrenamiento
    :param hyperparams: conjunto de hyperparámetros del xgboost
    :param gs_cv: posee los parámetros que se utilizarán en el grid search cross validation.
    :return: Diccionario con el max_depth y min_child_weight óptimo.
    """

    param_test = {
        'xgb_model__max_depth': [3, 5, 7, 9, 11, 13],
        'xgb_model__min_child_weight': [1, 3, 5, 7, 9]
    }
    print('Hora de inicio: ', datetime.datetime.now())
    gsearch = crear_grid_search_cv(param_test, hyperparams, gs_cv)
    gsearch.fit(train_x, train_y)
    reporte_gsearch(gsearch)
    hyperparams['max_depth'] = gsearch.best_params_['xgb_model__max_depth']
    hyperparams['min_child_weight'] = gsearch.best_params_['xgb_model__min_child_weight']
    return hyperparams


def hyp_gamma(train_x: pd.DataFrame, train_y: pd.DataFrame, hyperparams: dict, gs_cv: dict):
    """
    Buscar hiperparámetro gamma. Un nodo se divide solo cuando la división resultante da una reducción positiva en
    la función de pérdida. Gamma especifica la reducción mínima requerida en la función de pérdida para hacer
    una división.

    :param train_x: features del conjunto de entrenamiento
    :param train_y: variable dependiente del conjunto de entrenamiento
    :param hyperparams: conjunto de hyperparámetros del xgboost
    :param gs_cv: posee los parámetros que se utilizarán en el grid search cross validation.
    :return: (dict) Diccionario con gamma.
    """

    param_test = {
        'xgb_model__gamma': [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1]
    }
    print('Hora de inicio: ', datetime.datetime.now())
    gsearch = crear_grid_search_cv(param_test, hyperparams, gs_cv)
    gsearch.fit(train_x, train_y)
    reporte_gsearch(gsearch)
    hyperparams['gamma'] = gsearch.best_params_['xgb_model__gamma']
    return hyperparams


def hyp_subsample(train_x: pd.DataFrame, train_y: pd.DataFrame, hyperparams: dict, gs_cv: dict):
    """
    Busca hiperparámetros subsample y colsample_bytree, que son el porcentaje de filas y predictores
    respectivamente que se consideran para cada árbol.

    :param train_x: features del conjunto de entrenamiento
    :param train_y: variable dependiente del conjunto de entrenamiento
    :param hyperparams: conjunto de hyperparámetros del xgboost
    :param gs_cv: posee los parámetros que se utilizarán en el grid search cross validation.
    :return: (dict) Diccionario con subsample y colsample_bytree.
    """

    param_test = {
        'xgb_model__subsample': [0.5, 0.7, 0.9, 1],
        'xgb_model__colsample_bytree': [0.5, 0.7, 0.9, 1]
    }
    print('Hora de inicio: ', datetime.datetime.now())
    gsearch = crear_grid_search_cv(param_test, hyperparams, gs_cv)
    gsearch.fit(train_x, train_y)
    reporte_gsearch(gsearch)
    hyperparams['subsample'] = gsearch.best_params_['xgb_model__subsample']
    hyperparams['colsample_bytree'] = gsearch.best_params_['xgb_model__colsample_bytree']
    return hyperparams


def hyp_regularization(train_x: pd.DataFrame, train_y: pd.DataFrame, hyperparams: dict, gs_cv: dict):
    """
    Busca hiperparámetro reg_alpha que es una regularización tipo 1, la cual es cuadrática.

    :param train_x: features del conjunto de entrenamiento
    :param train_y: variable dependiente del conjunto de entrenamiento
    :param hyperparams: conjunto de hyperparámetros del xgboost
    :param gs_cv: posee los parámetros que se utilizarán en el grid search cross validation.
    :return: (dict) Diccionario con reg_alpha.
    """

    param_test = {
        'xgb_model__reg_alpha': [0, 0.01, 0.1, 1, 10, 50],
        'xgb_model__reg_lambda': [0, 0.01, 0.1, 1, 10, 100, 500]
    }
    print('Hora de inicio: ', datetime.datetime.now())
    gsearch = crear_grid_search_cv(param_test, hyperparams, gs_cv)
    gsearch.fit(train_x, train_y)
    reporte_gsearch(gsearch)
    hyperparams['reg_alpha'] = gsearch.best_params_['xgb_model__reg_alpha']
    hyperparams['reg_lambda'] = gsearch.best_params_['xgb_model__reg_lambda']
    return hyperparams


def hyp_finetune(train_x: pd.DataFrame, train_y: pd.DataFrame, hyperparams: dict, gs_cv: dict):
    """
    Busca hiperparámetro reg_alpha que es una regularización tipo 1, la cual es cuadrática.

    :param train_x: features del conjunto de entrenamiento
    :param train_y: variable dependiente del conjunto de entrenamiento
    :param hyperparams: conjunto de hyperparámetros del xgboost
    :param gs_cv: posee los parámetros que se utilizarán en el grid search cross validation.
    :return: (dict) Diccionario con reg_alpha.
    """
    param_test = {
        'xgb_model__learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2],
        'xgb_model__n_estimators': [100, 300, 500, 800, 1000, 1500]
    }
    print('Hora de inicio (fine tuning): ', datetime.datetime.now())
    gsearch = crear_grid_search_cv(param_test, hyperparams, gs_cv)
    gsearch.fit(train_x, train_y)
    reporte_gsearch(gsearch)
    hyperparams['learning_rate'] = gsearch.best_params_['xgb_model__learning_rate']
    hyperparams['n_estimators'] = gsearch.best_params_['xgb_model__n_estimators']
    return hyperparams