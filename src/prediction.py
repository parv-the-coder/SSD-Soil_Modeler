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
