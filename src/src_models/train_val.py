import datetime
from datetime import datetime as dt
from dateutil.relativedelta import *

import pandas as pd


dict_mix = {
    "cervezas": ["masivo"],
    "analcoholicos": ["gaseosas", "minerales"]
}



def get_val_set_index(train_date_set, preprocessed_data, category):
    """
    Devuelve el conjunto de entrenamiento y de validación para un periodo seleccionado

    :param train_date_set: fechas en la cual se va a entrenar
    :return: conjuntos de entrenamiento y validación
    """

    val_split_period = max(train_date_set) - 200
    val_split_date = (pd.to_datetime(str(val_split_period), format='%Y%m') - pd.DateOffset(1)).date()

    df = (
        preprocessed_data
        .pipe(get_train_val_data, train_date_set, category)
    )

    tscv = TimeBasedCV(
        train_period=1465,  # Number of days that training set considers
        test_period=180,     # Number of days that test set considers
        freq='days'
    )

    index_output = tscv.split(
        df,
        validation_split_date=val_split_date,   # Date that separate validation and test set
        date_column='periodo_date'
    )

    return index_output


def get_train_val_data(df: pd.DataFrame, train_date_set, category):
    """
    Devuelve el conjunto de entrenamiento

    :param df: data de entrada
    :param train_date_set: fechas en la cual se va a entrenar
    :return:
    """
    cols_to_drop = [
        'id_categoria', 'id_periodo', 'id_cliente', 'id_canal', 'volumen', 'volumen_sem', 'superficie_km2',
        'n_habitantes', 'prop_vol_masivo', 'canal', 'descr_flag_patente', 'volumen_sem_ar1', 'volumen_sem_ar2',
        'volumen_sem_fut', 'volumen_dif1', 'volumen_dif1_dif12', 'volumen_sem_dif6', 'volumen_sem_dif6_fut',
        'imacec', 'uf', 'tpm', 'ipc', 'tasa_desempleo', 'vol_sem_rel_dif6_lag6', 'vol_sem_rel_dif6_lag12', 'compra',
        'vol_sem_dif6', 'vol_sem_dif6_lag6', 'dolar', 'id_comuna', 'recency', 'n_ptos_interes', 'tpm_sem', 'tpm_chg_6m',
        'tpm_pct_6m', 'tpm_std_6m', 'tpm_trend_6m', 'covid', 'tasa_desempleo_sem'
    ]

    for col in dict_mix[category]:
        cols_to_drop.append(f'porc_{col}')
    df1 = df.copy()
    df1.drop(df1[~df1['id_periodo'].isin(train_date_set)].index, inplace=True)
    df1.drop(df1[df1['volumen_sem_dif6_fut'].isna()].index, inplace=True)
    df1.reset_index(drop=True, inplace=True)
    df1['periodo_date'] = pd.to_datetime(df1['id_periodo'].astype(str), format='%Y%m')
    df1.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    return df1


class TimeBasedCV(object):
    """
    Parameters
    ----------
    train_period: int
        number of time units to include in each validation set
        default is 30
    test_period: int
        number of time units to include in each test set
        default is 7
    freq: string
        frequency of input parameters. possible values are: days, months, years, weeks, hours, minutes, seconds
        possible values designed to be used by dateutil.relativedelta class
        deafault is days
    """

    def __init__(self, train_period=30, test_period=7, freq='days'):
        self.train_period = train_period
        self.test_period = test_period
        self.freq = freq

    def split(self, data, validation_split_date=None, date_column='record_date', gap=0):
        """
        Generate indices to split data into training and test set

        Parameters
        ----------
        data: pandas DataFrame
            your data, contain one column for the record date
        validation_split_date: datetime.date()
            first date to perform the splitting on.
            if not provided will set to be the minimum date in the data after the first training set
        date_column: string, deafult='record_date'
            date of each record
        gap: int, default=0
            for cases the test set does not come right after the validation set,
            *gap* days are left between validation and test sets

        Returns
        -------
        train_index ,test_index:
            list of tuples (validation index, test index) similar to sklearn model selection
        """

        # check that date_column exist in the data:
        try:
            data[date_column]
        except:
            raise KeyError(date_column)

        train_indices_list = []
        test_indices_list = []

        if validation_split_date is None:
            validation_split_date = data[date_column].min().date() + eval(
                'relativedelta(' + self.freq + '=self.train_period)')

        start_train = validation_split_date - eval('relativedelta(' + self.freq + '=self.train_period)')
        end_train = start_train + eval('relativedelta(' + self.freq + '=self.train_period)')
        start_test = end_train + eval('relativedelta(' + self.freq + '=gap)')
        end_test = start_test + eval('relativedelta(' + self.freq + '=self.test_period)')

        while end_test < data[date_column].max().date():
            # validation indices:
            cur_train_indices = list(data[(data[date_column].dt.date >= start_train) &
                                          (data[date_column].dt.date < end_train)].index)

            # test indices:
            cur_test_indices = list(data[(data[date_column].dt.date >= start_test) &
                                         (data[date_column].dt.date < end_test)].index)

            print("Train period:", start_train, "-", end_train, ", Test period", start_test, "-", end_test,
                  "# validation records", len(cur_train_indices), ", # test records", len(cur_test_indices))

            train_indices_list.append(cur_train_indices)
            test_indices_list.append(cur_test_indices)

            # update dates:
            start_train = start_train + eval('relativedelta(' + self.freq + '=self.test_period)')
            end_train = start_train + eval('relativedelta(' + self.freq + '=self.train_period)')
            start_test = end_train + eval('relativedelta(' + self.freq + '=gap)')
            end_test = start_test + eval('relativedelta(' + self.freq + '=self.test_period)')

        # mimic sklearn output
        index_output = [(train, test) for train, test in zip(train_indices_list, test_indices_list)]

        self.n_splits = len(index_output)

        return index_output

    def get_n_splits(self):
        """Returns the number of splitting iterations in the cross-validator
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits
