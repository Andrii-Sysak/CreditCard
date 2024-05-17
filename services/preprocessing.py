import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from category_encoders import LeaveOneOutEncoder

from sklearn.preprocessing import StandardScaler
from data_split import data_split_and_save

from columns import delete_columns, scaling_data

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
    scaler.fit(df[scaling_data])

    # transform train and test sets
    df_scaled = scaler.transform(df[scaling_data])

    df_scaled = pd.DataFrame(df_scaled, columns=scaling_data)
    
    df.update(df_scaled)

    return df


def preparing_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=delete_columns)
    
    df = categorical_encoding(df=df)
    df = data_scarling(df=df)

    df.to_csv('./data/corrected_BankChurners_.csv', index=False)

    data_split_and_save('corrected_BankChurners_.csv')

    return df
    

    
