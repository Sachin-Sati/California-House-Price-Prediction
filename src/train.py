from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib


def train_and_evaluate(X_train, y_train, X_test, y_test):
    ''''
    Trains multiple regression models and evaulates their performance.
    Returns:
        best_model: model that performes best on testing set
    '''
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    }

    result = {}

    for name, model in models.items():
        # cross validation
        cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        # train the model
        model.fit(X_train, y_train)
        # predict on test set
        y_pred = model.predict(X_test)
        # evaluate performance
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        result[name] = {
            'MSE': mse,
            'R2': r2
        }
        print(f"{name}: MSE={mse:.4f}, R2={r2:.4f}")

    # train the best model
    best_model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
    best_model.fit(X_train, y_train)

    # save the best model
    joblib.dump(best_model, 'models/best_model.pkl')

    return best_model