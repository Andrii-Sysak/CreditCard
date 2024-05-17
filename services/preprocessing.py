import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from category_encoders import LeaveOneOutEncoder

from sklearn.preprocessing import StandardScaler
from data_split import data_split_and_save

from columns import (
    delete_columns, 
    scaling_columns,
    mean_impute_columns,
    mode_impute_columns,
    outlier_columns
)


def impute_na(df, variable, value) -> pd.DataFrame:
    return df[variable].fillna(value)


def input_missing_values(df: pd.DataFrame):
    mean_impute_values = dict()
    for column in mean_impute_columns:
        mean_impute_values[column] = df[column].mean()
        df[column] = impute_na(df, column, mean_impute_values[column])

    mode_impute_values = dict()
    for column in mode_impute_columns:
        mode_impute_values[column] = df[column].mode()[0]
        df[column] = impute_na(df, column, mode_impute_values[column])
    
    return df


def find_skewed_boundaries(df, variable, distance):
    df[variable] = pd.to_numeric(df[variable],errors='coerce')
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)
    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
    upper_boundary = df[variable].quantile(0.75) + (IQR * distance)
    return upper_boundary, lower_boundary


def fix_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    upper_lower_limits = dict()
    for column in outlier_columns:
        upper_lower_limits[column+'_upper_limit'], upper_lower_limits[column+'_lower_limit'] = find_skewed_boundaries(df, column, 5)
    for column in outlier_columns:
        df = df[~ np.where(df[column] > upper_lower_limits[column+'_upper_limit'], True,
                        np.where(df[column] < upper_lower_limits[column+'_lower_limit'], True, False))]
    
    return df


def categorical_encoding(df: pd.DataFrame) -> pd.DataFrame:
    le = LabelEncoder
    looe = LeaveOneOutEncoder()

    columns_le = ['Attrition_Flag', 'Gender']
    columns_looe = ['Education_Level', 'Income_Category']
    columns_fe = ['Marital_Status', 'Card_Category']
    y = 'Avg_Utilization_Ratio'

    for x in columns_le:
        df[x] = le().fit_transform(df[x])
    for x in columns_looe:
        df[x] = looe.fit_transform(df[x], df[y])
    for x in columns_fe:
        freq_encoding = df[x].value_counts().to_dict()
        df[x] = df[x].map(freq_encoding)
    
    return df


def data_scarling(df: pd.DataFrame) -> pd.DataFrame:
    # set up the scaler
    scaler = StandardScaler()

    # fit the scaler to the train set, it will learn the parameters
    scaler.fit(df[scaling_columns])

    # transform train and test sets
    df_scaled = scaler.transform(df[scaling_columns])

    df_scaled = pd.DataFrame(df_scaled, columns=scaling_columns)
    
    df.update(df_scaled)

    return df


def preparing_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=delete_columns)
    df = input_missing_values(df=df)
    df = categorical_encoding(df=df)
    df = data_scarling(df=df)

    df.to_csv('./data/corrected_BankChurners_.csv', index=False)

    data_split_and_save('corrected_BankChurners_.csv')

    return df
    

    
