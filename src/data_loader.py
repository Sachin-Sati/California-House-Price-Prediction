import pandas as pd
from sklearn.datasets import fetch_california_housing

def load_housing_data():
    '''
    Loads the California housing dataset.
    Returns:
        X (pd.DataFrame): Features
        y (pd.Series): Target
    '''
    housing = fetch_california_housing(as_frame=True)
    X = housing.data.copy()
    y = housing.target.copy()
    return X, y