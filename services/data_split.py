import pandas as pd
from sklearn.model_selection import train_test_split
from columns import X_columns

def data_split_and_save(file_name: str) -> None:

    df = pd.read_csv(f"./data/{file_name}")
    
    X_train, X_test = train_test_split(df, train_size=0.8, random_state=42)
    X_train.to_csv('./data/train_data.csv', index=False)
    
    X_test[X_columns].to_csv('./data/new_data.csv', index=False)
    X_test.to_csv('./data/compare_data.csv', index=False)