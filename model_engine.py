# model_engine.py

# Common python package imports.
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Import from model_api/app/app.py.
# from app.features import FEATURES

# data = datasets.load_boston()
# df = pd.DataFrame(data.data, columns=data.feature_names)
# df['target'] = data.target


def model_pipeline():
    df = pd.read_csv('BankNote_Authentication.csv')

    """Get the data, train the model, and save it."""
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    model = RandomForestClassifier(n_estimators=700)
    model.fit(X, y)
    pickle.dump(model, open('app/model.pkl', 'wb'))
    print('Success!')


def get_model_data(feature_names):
    """Load and prepare data for modeling."""
    df = build_dataframe()
    df = clean_data(df)

    # Limit the feature set for simplicity.
    X, y = df[feature_names].values, df['target'].values
    return X, y


def build_dataframe():
    """Build dataframe to facilitate cleaning."""
    df = pd.read_csv('BankNote_Authentication.csv')
    return df


def clean_data(df):
    """Clean data in preparation for modeling."""
    return df


if __name__ == '__main__':
    model_pipeline()
