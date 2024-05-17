import pickle
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score
)

from columns import X_columns


def predict(model_name: str):
    df = pd.read_csv("./data/new_data.csv")

    X = df[X_columns]

    etr = pickle.load(open(f'./models/{model_name}', 'rb'))

    y_pred = etr.predict(X)

    df_ = pd.read_csv('./data/compare_data.csv')
    y_test = df_["Avg_Utilization_Ratio"]

    df['Avg_Utilization_Ratio'] = etr.predict(X)
    df.to_csv('./data/prediction_results_.csv', index=False)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R-squared:", r2)
    print("Explained Variance Score:", evs)

predict('finalized_model_.sav')