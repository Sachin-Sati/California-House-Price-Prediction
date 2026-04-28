from src.data_preprocessing import preprocess_data
from src.train import train_and_evaluate

X_train_s, X_test_s, y_train, y_test, scaler = preprocess_data()

# train models
model = train_and_evaluate(X_train_s, y_train, X_test_s, y_test)

print("ML Pipeline Executed Successfully!")