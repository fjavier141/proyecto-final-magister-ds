import os
import json

from src.src_models import train_val, hyperparameters as hyp
from src.utils import create_directory_if_not_exists


def xgboost_cross_validation(train_x, train_y, preprocessed_data, train_date_set, test_date, category):
    '''
    Validación cruzada para la búsqueda de hiperparámetros, entrega un xml con los
    valores óptimos para cada hiperparametro.
    Returns
    -------

    '''
    # Fechas para entrenamiento y validació

    # Parámetros base
    hyperparams = {
        'learning_rate': 0.1,
        'n_estimators': 500,
        'max_depth': 6,
        'min_child_weight': 1,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'objective': 'reg:squarederror',
        'nthread': 3
    }

    # Configuración de GridSearchCV
    gs_cv = {
        'scoring': 'neg_mean_squared_error',
        'n_jobs': 8,
        'cv': train_val.get_val_set_index(train_date_set, preprocessed_data, category)
    }

    # Pipeline jerárquico para búsqueda de hiperparámetros
    print("Inicio de búsqueda jerárquica de hiperparámetros")
    hyperparams = hyp.hyp_max_depth(train_x, train_y, hyperparams, gs_cv)
    hyperparams = hyp.hyp_gamma(train_x, train_y, hyperparams, gs_cv)
    hyperparams = hyp.hyp_regularization(train_x, train_y, hyperparams, gs_cv)
    hyperparams = hyp.hyp_subsample(train_x, train_y, hyperparams, gs_cv)

    # Etapa final: fine tuning
    print("Etapa final: refinamiento de hiperparámetros")
    hyperparams = hyp.hyp_finetune(train_x, train_y, hyperparams, gs_cv)

    print(f'Hiperparámetros encontrados corresponden a {json.dumps(hyperparams)}.')

    path_hyperparams = os.path.join(f"./data/output/hyperparams")
    create_directory_if_not_exists([path_hyperparams])
    path_json = os.path.join(path_hyperparams, f'hyperparams_xgb_{category}_{test_date}.json')
    with open(path_json, 'w') as json_file:
        json.dump(hyperparams, json_file)
    print(f'Ejecución Finalizada de búsqueda de hiperparámetros.')



def lightgbm_cross_validation(train_x, train_y, train_date_set, test_date, category):
    """
    Búsqueda de hiperparámetros para LightGBM usando validación cruzada temporal.
    Guarda un JSON con los hiperparámetros óptimos.
    """

    # Parámetros base
    hyperparams = {
        'objective': 'regression',
        'learning_rate': 0.05,
        'n_estimators': 500,
        'num_leaves': 31,
        'max_depth': -1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0,
        'n_jobs': -1,
        'random_state': 42
    }

    gs_cv = {
        'scoring': 'neg_mean_squared_error',
        'n_jobs': -1,
        'cv': train_val.get_val_set_index(train_date_set, train_x)
    }

    print("Inicio de búsqueda de hiperparámetros LightGBM")
    hyperparams = hyp.hyp_lgbm(train_x, train_y, hyperparams, gs_cv)

    print(f'Hiperparámetros LightGBM encontrados: {json.dumps(hyperparams)}.')

    path_hyperparams = os.path.join("./data/output/hyperparams")
    create_directory_if_not_exists([path_hyperparams])
    path_json = os.path.join(path_hyperparams, f'hyperparams_lgbm_{category}_{test_date}.json')

    with open(path_json, 'w') as json_file:
        json.dump(hyperparams, json_file)

    print('Ejecución finalizada de búsqueda de hiperparámetros LightGBM.')


def random_forest_cross_validation(train_x, train_y, train_date_set, test_date, category):
    """
    Búsqueda de hiperparámetros para RandomForestRegressor usando validación cruzada temporal.
    Guarda un JSON con los hiperparámetros óptimos.
    """

    hyperparams = {
        'n_estimators': 500,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'auto',
        'n_jobs': -1,
        'random_state': 42
    }

    gs_cv = {
        'scoring': 'neg_mean_squared_error',
        'n_jobs': -1,
        'cv': train_val.get_val_set_index(train_date_set, train_x)
    }

    print("Inicio de búsqueda de hiperparámetros RandomForest")
    hyperparams = hyp.hyp_random_forest(train_x, train_y, hyperparams, gs_cv)

    print(f'Hiperparámetros RandomForest encontrados: {json.dumps(hyperparams)}.')

    path_hyperparams = os.path.join("./data/output/hyperparams")
    create_directory_if_not_exists([path_hyperparams])
    path_json = os.path.join(path_hyperparams, f'hyperparams_rf_{category}_{test_date}.json')

    with open(path_json, 'w') as json_file:
        json.dump(hyperparams, json_file)

    print('Ejecución finalizada de búsqueda de hiperparámetros RandomForest.')


def ridge_cross_validation(train_x, train_y, train_date_set, test_date, category):
    """
    Búsqueda de hiperparámetros para Ridge Regression usando validación cruzada temporal.
    Guarda un JSON con los hiperparámetros óptimos.
    """

    hyperparams = {
        'alpha': 1.0,
        'fit_intercept': True
    }

    gs_cv = {
        'scoring': 'neg_mean_squared_error',
        'n_jobs': -1,
        'cv': train_val.get_val_set_index(train_date_set, train_x)
    }

    print("Inicio de búsqueda de hiperparámetros Ridge")
    hyperparams = hyp.hyp_ridge(train_x, train_y, hyperparams, gs_cv)

    print(f'Hiperparámetros Ridge encontrados: {json.dumps(hyperparams)}.')

    path_hyperparams = os.path.join("./data/output/hyperparams/")
    create_directory_if_not_exists([path_hyperparams])
    path_json = os.path.join(path_hyperparams, f'hyperparams_ridge_{category}_{test_date}.json')

    with open(path_json, 'w') as json_file:
        json.dump(hyperparams, json_file)

    print('Ejecución finalizada de búsqueda de hiperparámetros Ridge.')


