import pandas as pd
import joblib
from src.data_loader import load_housing_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data():
    '''
    Load data -> Feature Engineering -> Train-Test Split -> Scaling
    Returns: X_train_scaled, X_test_scaled, y_train, y_test, scaler
    '''
    # Load data
    X, y = load_housing_data()

    # Feature Engineering
    X = X.copy()
    X['RoomsPerHouse'] = X['AveRooms'] / X['AveOccup']
    X['BedroomsRatio'] = X['AveBedrms'] / X['AveRooms']
    X['PopulationPerHouse'] = X['Population'] / X['AveOccup']
    X['IncomePerPerson'] = X['MedInc'] / X['Population']

    # Train-Test-Split
    X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(x_test)

    # save the scaler deployment
    joblib.dump(scaler, 'models/scaler.pkl')

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler