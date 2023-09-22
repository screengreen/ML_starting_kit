import yaml

import pandas as pd

import src.data_preprocessing as dp
from src.utils import load_data, get_options

#opening config file
options = get_options()

def additional_preprocessing_block(data, experiment, type='train'):
    """
    Here you can add some unusial things 
    """





    return data


def data_preprocessing_block(data, experiment, type='train', columns_to_drop=None):
    #ЗАПОЛНЕНИЕ ПРОПУСКОВ

    fill_missing_values = options['fill_missing_values']
    # Заполнение пропущенных значений средним
    if fill_missing_values != 'no':
        if fill_missing_values =='mean':
            dp.fill_missing_with_mean(df=data)
        elif fill_missing_values == 'mode':
            dp.fill_missing_with_mode(df=data)
        elif fill_missing_values == 'value':
            dp.fill_missing_with_value(df=data, fill_value=0)
        elif fill_missing_values == 'knn':
            dp.fill_missing_with_knn(df=data, k_neighbors=5)
        elif fill_missing_values == 'ml':
            dp.fill_missing_with_ml(df=data)

    balance_target = options['balance_target']
    # Балансировка целевой переменной (oversampling)
    if type == 'train' and balance_target != "no":
        if balance_target == 'oversample':
            data = dp.oversample_minority_class(data)
        elif balance_target == 'undersample':
            data = dp.undersample_majority_class(data)

    if options['normalization']:
        # Нормализация данных
        data = dp.normalize_data(data, target_name=options['target_name'])

    if type == 'train' and options['remove_outliers']:
        # Удаление выбросов
        data = dp.remove_outliers_iqr(data)

    encode_object_type = options['encode_object_type']
    if encode_object_type != 'no':
        if encode_object_type == 'one_hot':
            data = dp.one_hot_encode(data)
        elif encode_object_type == 'label':
            data = dp.label_encode(data)
        elif encode_object_type == 'target':
            data = dp.target_encode(data)

    if options['encode_datetime']:
        data = dp.encode_datetime_features(data)

    if type == 'train' and options['drop_duplicates']:
        # Удаление дубликатов
        data = dp.drop_duplicates(data)

    # Удаление ненужных колонок
    if columns_to_drop != None and options['drop_columns']:
        data = dp.drop_columns(data, columns_to_drop=columns_to_drop)

    if options['data_engineering']:
    # Применение логарифмического преобразования к признакам
        data = dp.apply_log_transform(data, columns=['numeric_column1', 'numeric_column2'])

 
    if options['logging'] and not options['using_model']:
        parameters = {'fill_missing_values': fill_missing_values, 'balance_target': balance_target, 
                      'normalization': options["normalization"], 'remove_outliers': options['remove_outliers'],
                        'encode_object_type': encode_object_type, 'encode_datetime': options['encode_datetime'], 
                        'drop_columns': options['drop_columns'], 'drop_duplicates': options['drop_duplicates'],
                        'data_engineering': options['data_engineering'], 'columns': data.columns.tolist()}
        experiment.log_parameters(parameters)

    return data

def make_split(data, type = 'train'):
    # Разделение данных на обучающую и тестовую выборки
    if type == 'train':
        return dp.split_data(data, options['target_name'], test_size=0.2)
    elif type == 'test':
        return data
    
def full_prerocessing(data, experiment, type='train', columns_to_drop=None):
    data = data_preprocessing_block(data, experiment = experiment, type = type, columns_to_drop=None)

    data = additional_preprocessing_block(data , experiment=experiment, type=type)

    return make_split(data, type = type)

    
