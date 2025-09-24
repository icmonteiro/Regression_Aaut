import numpy as np
import joblib

# Load the saved model only once
model_data = joblib.load("regression_model.pkl")
model = model_data['model']

def predict(X_test):

    """
    Predict function for regression problem.
    
    Args:
        X_test: numpy array of shape (n_samples, 6) containing test features
    
    Returns:
        numpy array of shape (n_samples,) containing predictions
    """

    # Model is a full pipeline (scaler + polynomial features + regressor)
    return model.predict(X_test)
