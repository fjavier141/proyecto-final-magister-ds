import os
import json
from pickle import load, dump

import pandas as pd


def get_dates(test_date: int):
    """
    Función usada para obtener los periodos de entrenamiento semestrales. En este caso son los 8 últimos semestres
    anteriores.
    :param test_date: Corresponde al periodo con el formato %Y%m el cual coincide con el mes anterior al semestre a
    predecir.
    :return:
    """
    per_train = []
    per_test = [test_date]
    for i in range(1, 9):
        fecha = pd.to_datetime(str(test_date), format='%Y%m')
        train_date = fecha - pd.DateOffset(months=i * 6)
        train_date_int = int(train_date.strftime('%Y%m'))
        if train_date_int >= 201707:
            per_train.append(train_date_int)
    per_train.sort()
    return per_train, per_test


def get_validation_periods(test_date: int, n_periods: int):
    per_val = []
    for i in range(1, n_periods + 1):
        fecha = pd.to_datetime(str(test_date), format='%Y%m')
        fecha = pd.to_datetime(str(test_date), format='%Y%m')
        train_date = fecha - pd.DateOffset(months=i * 6)
        train_date_int = int(train_date.strftime('%Y%m'))
        if train_date_int >= 201707:
            per_val.append(train_date_int)
    per_val.sort()
    return per_val


def create_directory_if_not_exists(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f'Se crea directorio {path}')
        else:
            print(f'Ruta {path} ya existe')


def concat_dfs_hor(list_dfs):
    concatenated_df = pd.DataFrame()
    for df in list_dfs:
        concatenated_df = pd.concat([concatenated_df, df], axis=1)
    return concatenated_df


def load_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)


def save_json(data, filepath):
    with open(filepath, 'w') as file:
        json.dump(data, file)


def load_pickle(filepath):
    with open(filepath, 'rb') as file:
        return load(file)


def save_pickle(data, filepath):
    with open(filepath, 'wb') as file:
        dump(data, file)
