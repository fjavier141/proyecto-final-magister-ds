import os
import json

from src.src_models import train_val, hyperparameters as hyp
from src.utils import create_directory_if_not_exists


def main(train_x, train_y, train_date_set, test_date, category):
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
        'n_jobs': -1,
        'cv': train_val.get_val_set_index(train_date_set, train_x)
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

    path_hyperparams = os.path.join(f"./data/output/hyperparams/")
    create_directory_if_not_exists([path_hyperparams])
    path_json = os.path.join(path_hyperparams, f'hyperparams_{category}_{test_date}.json')
    with open(path_json, 'w') as json_file:
        json.dump(hyperparams, json_file)
    print(f'Ejecución Finalizada de búsqueda de hiperparámetros.')
