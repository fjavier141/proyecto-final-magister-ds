
def get_val_set_index(train_date_set: list):
    """
    Devuelve el conjunto de entrenamiento y de validación para un periodo seleccionado

    :param train_date_set: fechas en la cual se va a entrenar
    :return: conjuntos de entrenamiento y validación
    """

    val_split_period = max(train_date_set) - 200
    val_split_date = (pd.to_datetime(str(val_split_period), format='%Y%m') - pd.DateOffset(1)).date()

    df = (
        data_input.leer_data_train()
        .pipe(get_train_val_data, train_date_set)
    )

    tscv = time_split.TimeBasedCV(
        train_period=1465,  # Number of days that training set considers
        test_period=180,     # Number of days that test set considers
        freq='days'
    )

    index_output = tscv.split(
        df,
        validation_split_date=val_split_date,   # Date that separate validation and test set
        date_column='PERIODO_DATE'
    )

    return index_output