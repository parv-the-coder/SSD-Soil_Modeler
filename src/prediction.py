import pickle
import pandas as pd
import numpy as np
from src.data_preprocessing import preprocess_data

def load_model(path):
    """Load model with error handling."""
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    with open(path, 'rb') as f:
        return pickle.load(f)

def predict_soil_property(model_path, new_data_path, target_column):
    # Load model
    model = load_model(model_path)
    
    # Load new data
    if new_data_path.endswith('.xls'):
        new_data = pd.read_excel(new_data_path)
    else:
        new_data = pd.read_csv(new_data_path)
    
    # Preprocess (assuming same preprocessing as training)
    X, _, _ = preprocess_data(new_data, target_column)  # Ignore y and preprocessing dict
    
    # Make predictions
    predictions = model.predict(X)
    
    return predictions

def predict_with_model(model_path, user_df):
    """
    Loads the model from model_path and predicts on user_df (expects preprocessed features).
    Returns predictions as a numpy array with post-processing to avoid negative values.
    """
    import os
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Please train models first.")
    
    model = load_model(model_path)
    # Make predictions and return as array
    preds = model.predict(user_df)
    
    # Post-process to avoid negative predictions for soil properties
    # Clip negative values to small positive value
    preds = np.clip(preds, 0.001, None)
    
    return preds
