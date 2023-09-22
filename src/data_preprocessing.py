import os
import yaml
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


from src.utils import find_minority_and_majority_classes, load_data, get_options



#opening config file
options = get_options()


#ЗАПОЛНЕНИЕ ПРОПУСКОВ
# Заполнение средним
def fill_missing_with_mean(df, column_names=None):
    if column_names == None:
        column_names = df.columns[df.isnull().any()].tolist()

    for column_name in column_names:
        mean_value = df[column_name].mean()
        df[column_name].fillna(mean_value, inplace=True)


#Заполнение модой
def fill_missing_with_mode(df, column_names=None):
    if column_names == None:
        column_names = df.columns[df.isnull().any()].tolist()

    for column_name in column_names:
        mode_value = df[column_name].mode()[0]
        df[column_name].fillna(mode_value, inplace=True)


# Заполнение каким то числом
def fill_missing_with_value(df, column_names=None, fill_value=0):
    if column_names == None:
        column_names = df.columns[df.isnull().any()].tolist()

    for column_name in column_names:
        df[column_name].fillna(fill_value, inplace=True)


# Заполнение с помощью knn
def fill_missing_with_knn(df, column_names=None, k_neighbors=5):
    if column_names == None:
        column_names = df.columns[df.isnull().any()].tolist()

    # Создаем копию DataFrame для обработки
    df_copy = df.copy()
    
    for column_name in column_names:
        # Разделяем данные на две части: строки с пропущенными значениями и строки без
        df_missing = df_copy[df_copy[column_name].isnull()]
        df_not_missing = df_copy[df_copy[column_name].notnull()]
        
        # Выбираем признаки для обучения K-NN модели
        features = df.columns[df.notnull().all()].tolist()  # Список признаков, на основе которых будет проводиться заполнение
        
        # Создаем K-NN модель
        knn_model = KNeighborsRegressor(n_neighbors=k_neighbors)
        
        # Обучаем модель на данных без пропусков
        knn_model.fit(df_not_missing[features], df_not_missing[column_name])
        
        # Предсказываем пропущенные значения
        predicted_values = knn_model.predict(df_missing[features])
        
        # Заполняем пропуски в исходном DataFrame предсказанными значениями
        df_copy.loc[df_copy[column_name].isnull(), column_name] = predicted_values
    
    return df_copy


# Заполнение с помощью мл 
def fill_missing_with_ml(df, column_names=None, features_to_predict=None):
    if column_names == None:
        column_names = df.columns[df.isnull().any()].tolist()

    # Создаем копию DataFrame для обработки
    df_copy = df.copy()
    if features_to_predict == None:
        features_to_predict = df.columns[~df.isnull().any()].tolist()
    
    for column_name in column_names:
        # Разделяем данные на две части: строки с пропущенными значениями и строки без
        df_missing = df_copy[df_copy[column_name].isnull()]
        df_not_missing = df_copy[df_copy[column_name].notnull()]
        
        # Создаем модель (например, случайный лес)
        model = RandomForestRegressor()
        
        # Обучаем модель на данных без пропусков
        model.fit(df_not_missing[features_to_predict], df_not_missing[column_name])
        
        # Предсказываем пропущенные значения
        predicted_values = model.predict(df_missing[features_to_predict])
        
        # Заполняем пропуски в исходном DataFrame предсказанными значениями
        df_copy.loc[df_copy[column_name].isnull(), column_name] = predicted_values
    
    return df_copy




#Балансировка целевой переменной
def oversample_minority_class(data, target_column='target', random_state=None):
    """
    Oversamples the minority class by replicating random samples.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing both features and the target variable.
    target_column (str): The name of the target variable column.
    random_state (int, optional): Seed for random number generator for reproducibility.

    Returns:
    pd.DataFrame: New DataFrame with balanced classes.
    """
    y = data[target_column]
    X = data.drop(target_column, axis=1)
    
    minority_class, majority_class = find_minority_and_majority_classes(y)
    
    data_minority = data[data[target_column] == minority_class]
    
    # Oversampling minority class
    data_oversampled = resample(data_minority,
                                replace=True,
                                n_samples=data[data[target_column] == majority_class].shape[0],
                                random_state=random_state)
    
    # Combining majority class with oversampled minority class
    data_balanced = pd.concat([data[data[target_column] == majority_class], data_oversampled])
    
    return data_balanced



def undersample_majority_class(data, target_column, random_state=None):
    """
    Reduces the number of samples for the majority class to balance the classes.

    Parameters:
    data (pd.DataFrame): The DataFrame containing both features and the target variable.
    target_column (str): The name of the target variable column.
    random_state (int, optional): Seed for random number generator for reproducibility.

    Returns:
    pd.DataFrame: New DataFrame with balanced classes.
    """
    y = data[target_column]
    X = data.drop(target_column, axis=1)

    minority_class, majority_class = find_minority_and_majority_classes(y)
    
    data_majority = data[data[target_column] == majority_class]
    
    # Reducing the number of samples for the majority class
    data_undersampled = resample(data_majority,
                                 replace=False,
                                 n_samples=data[data[target_column] == minority_class].shape[0],
                                 random_state=random_state)
    
    # Combining undersampled majority and minority classes
    data_balanced = pd.concat([data_undersampled, data[data[target_column] == minority_class]])
    
    return data_balanced


#Нормализация и или стандартизация данных
def normalize_data(df, target_name, columns=None):
    """
    Нормализует данные в DataFrame.

    Parameters:
    df (pd.DataFrame): Исходный DataFrame.
    columns (list, optional): Список колонок, которые нужно нормализовать. Если None, нормализуются все числовые колонки.

    Returns:
    pd.DataFrame: DataFrame с нормализованными данными.
    """
    # Если не указан список колонок для нормализации, нормализуем все числовые колонки
    if columns is None:
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    else:
        numeric_columns = columns
    numeric_columns.remove(target_name)
    # Создаем копию DataFrame для обработки
    df_normalized = df.copy()
    


    # Применяем нормализацию к выбранным колонкам
    scaler = StandardScaler()
    df_normalized[numeric_columns] = scaler.fit_transform(df_normalized[numeric_columns])

    return df_normalized



#Работа с выбросами
def remove_outliers_iqr(df, columns=None, threshold=1.5):
    """
    Удаляет выбросы из данных на основе интерквартильного размаха (IQR).

    Parameters:
    df (pd.DataFrame): Исходный DataFrame.
    columns (list, optional): Список колонок, в которых нужно удалить выбросы. Если None, удаляются выбросы из всех числовых колонок.
    threshold (float, optional): Параметр множителя IQR для определения выбросов. Значение по умолчанию - 1.5.

    Returns:
    pd.DataFrame: DataFrame без выбросов.
    """
    # Если не указан список колонок, удаляем выбросы из всех числовых колонок
    if columns is None:
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    else:
        numeric_columns = columns

    # Создаем копию DataFrame для обработки
    df_no_outliers = df.copy()

    # Удаляем выбросы для каждой выбранной колонки
    for column in numeric_columns:
        Q1 = df_no_outliers[column].quantile(0.25)
        Q3 = df_no_outliers[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df_no_outliers = df_no_outliers[(df_no_outliers[column] >= lower_bound) & (df_no_outliers[column] <= upper_bound)]

    return df_no_outliers



# кодирование категориальных признаков
def one_hot_encode(df, columns=None):
    """
    Выполняет бинарное кодирование (One-Hot Encoding) выбранных категориальных колонок.

    Parameters:
    df (pd.DataFrame): Исходный DataFrame.
    columns (list): Список названий колонок для кодирования.

    Returns:
    pd.DataFrame: DataFrame с примененным One-Hot Encoding к указанным колонкам.
    """
    if columns == None:
        columns = df.select_dtypes(include=['object']).columns.tolist()

    df_encoded = pd.get_dummies(df, columns=columns, prefix=columns)
    return df_encoded

def label_encode(df, columns=None):
    """
    Выполняет кодирование метками (Label Encoding) выбранных категориальных колонок.

    Parameters:
    df (pd.DataFrame): Исходный DataFrame.
    columns (list): Список названий колонок для кодирования.

    Returns:
    pd.DataFrame: DataFrame с примененным Label Encoding к указанным колонкам.
    """
    if columns == None:
        columns = df.select_dtypes(include=['object']).columns.tolist()

    df_encoded = df.copy()
    label_encoder = LabelEncoder()
    
    for column in columns:
        df_encoded[column] = label_encoder.fit_transform(df_encoded[column])
    
    return df_encoded

def target_encode(df, categorical_column= None, target_column='target'):
    """
    Выполняет кодирование целью (Target Encoding) для выбранной категориальной колонки.

    Parameters:
    df (pd.DataFrame): Исходный DataFrame.
    categorical_column (str): Название категориальной колонки, которую нужно закодировать.
    target_column (str): Название целевой переменной, на основе которой будет выполнено кодирование.

    Returns:
    pd.DataFrame: DataFrame с примененным Target Encoding к указанной колонке.
    """
    if columns == None:
        columns = df.select_dtypes(include=['object']).columns.tolist()

    encoding_map = df.groupby(categorical_column)[target_column].mean().to_dict()
    df_encoded = df.copy()
    df_encoded[categorical_column] = df_encoded[categorical_column].map(encoding_map)
    
    return df_encoded



# трансформирование даты 
def encode_datetime_features(df, datetime_column=None):
    """
    Кодирует временную колонку, извлекая из неё признаки года, месяца и дня.

    Parameters:
    df (pd.DataFrame): Исходный DataFrame.
    datetime_column (str): Название колонки с данными о времени или дате.

    Returns:
    pd.DataFrame: DataFrame с добавленными признаками года, месяца и дня.
    """
    if datetime_column == None:
        datetime_column = df.select_dtypes(include=['datetime64']).columns.tolist()
    if datetime_column:
        datetime_column = datetime_column[0]
        df_encoded = df.copy()
        datetime_series = pd.to_datetime(df_encoded[datetime_column])

        df_encoded['Year'] = datetime_series.dt.year
        df_encoded['Month'] = datetime_series.dt.month
        df_encoded['Day'] = datetime_series.dt.day

        # Вы можете добавить другие признаки, такие как час, минуты, день недели и т. д., по аналогии

        # Удалите исходную колонку с датой или временем, если необходимо
        df_encoded.drop(columns=[datetime_column], inplace=True)

        return df_encoded
    else: 
        return df



#Базовые штуки
## удаление дупликатов
def drop_duplicates(data):
    data.drop_duplicates(inplace=True)
    return data


# train test split
def split_data(df, target_column ='target', test_size=0.2, random_state=None):
    """
    Разделяет данные на обучающую и тестовую выборки.

    Parameters:
    df (pd.DataFrame): Исходный DataFrame с данными.
    target_column (str): Название целевой переменной.
    test_size (float, optional): Доля данных, которая будет отведена для тестовой выборки (по умолчанию 0.2).
    random_state (int, optional): Зерно генератора случайных чисел для воспроизводимости (по умолчанию None).

    Returns:
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series: Обучающая выборка, тестовая выборка, обучающая целевая переменная, тестовая целевая переменная.
    """
    target_column = options['target_name']
    y = df[target_column]
    
    X = df.drop(columns=[target_column])
    
    return  train_test_split(X, y, test_size=test_size)

   



#Удаление ненужных признаков
def drop_columns(df, columns_to_drop):
    """
    Удаляет указанные колонки из DataFrame.

    Parameters:
    df (pd.DataFrame): Исходный DataFrame.
    columns_to_drop (list): Список названий колонок для удаления.

    Returns:
    pd.DataFrame: DataFrame без указанных колонок.
    """
    df_filtered = df.drop(columns=columns_to_drop, axis=1)
    return df_filtered



# FEATURE ENGINEERING
def apply_log_transform(df, columns):
    """
    Применяет логарифмическое преобразование (например, натуральный логарифм) к указанным колонкам DataFrame.

    Parameters:
    df (pd.DataFrame): Исходный DataFrame.
    columns (list): Список названий колонок, к которым будет применено логарифмическое преобразование.

    Returns:
    pd.DataFrame: DataFrame с примененным логарифмическим преобразованием к указанным колонкам.
    """
    df_transformed = df.copy()
    
    for column in columns:
        if column in df.columns:
            df_transformed[column] = np.log(df_transformed[column])
    
    return df_transformed


