import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesRegressor

from preprocessing import preparing_data
from columns import X_columns, y_column
from hyper_parameters import best_params



def train(file_name: str, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(f'./data/{file_name}')

    df = preparing_data(df=df)

    X = df[X_columns]

    y = df[y_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    etr = ExtraTreesRegressor()
    etr.set_params(**best_params)

    model = etr.fit(X_train,y_train)

    filename = f'./models/{model_name}'
    pickle.dump(model, open(filename, 'wb'))


train(file_name='BankChurners.csv',model_name='finalized_model_.sav')
