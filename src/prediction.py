import pickle
import pandas as pd
import numpy as np
from src.data_preprocessing import preprocess_data

def load_model(path):
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
    Returns a DataFrame with predictions.
    """
    model = load_model(model_path)
    # If user_df needs preprocessing, add here (currently assumes preprocessed)
    preds = model.predict(user_df)
    return pd.DataFrame({'Prediction': preds})
