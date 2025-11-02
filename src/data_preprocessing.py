import pandas as pd
import numpy as np


def preprocess_data(data: pd.DataFrame, target_column: str):
    """Extract spectral features and create preprocessing functions.

    Args:
        data: input DataFrame containing spectral bands and target.
        target_column: name of the target column to predict.

    Returns:
        X (DataFrame): spectral features only
        y (Series): target values
        preprocessing (dict): mapping of preprocessing name -> callable that accepts X and returns transformed X
    """
    # Choose numeric columns as spectral columns but explicitly exclude the target column
    numeric_cols = list(data.select_dtypes(include=[np.number]).columns)
    spectral_cols = [c for c in numeric_cols if c != target_column]

    if not spectral_cols:
        raise ValueError("No numeric spectral columns found (after excluding the target).")

    X = data[spectral_cols].copy()
    y = data[target_column]

    def reflectance(x: pd.DataFrame) -> pd.DataFrame:
        return x.copy()

    def absorbance(x: pd.DataFrame) -> pd.DataFrame:
        # absorbance = log(1 / reflectance). Avoid division by zero.
        xr = x.replace(0, 1e-6)
        return np.log(1.0 / xr)

    def continuum_removal(x: pd.DataFrame) -> pd.DataFrame:
        # Continuum removal (simple proxy): divide each row by its max value
        row_max = x.max(axis=1).replace(0, 1e-6)
        return x.div(row_max, axis=0)

    preprocessing = {
        "Reflectance": reflectance,
        "Absorbance": absorbance,
        "Continuum Removal": continuum_removal,
    }

    return X, y, preprocessing
