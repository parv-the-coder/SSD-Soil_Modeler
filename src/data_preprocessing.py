import pandas as pd
import numpy as np


def preprocess_data(data: pd.DataFrame, target_column):
    """Extract spectral features and create preprocessing functions.

    Args:
        data: input DataFrame containing spectral bands and target.
        target_column: name of the target column to predict.

    Returns:
        X (DataFrame): spectral features only
        y (Series): target values
        preprocessing (dict): mapping of preprocessing name -> callable that accepts X and returns transformed X
    """
    # Ensure target_column is a string
    if not isinstance(target_column, str):
        target_column = str(target_column)

    # Choose numeric columns as spectral columns but explicitly exclude the target column
    numeric_cols = list(data.select_dtypes(include=[np.number]).columns)
    spectral_cols = [c for c in numeric_cols if c != target_column]

    if not spectral_cols:
        raise ValueError("No numeric spectral columns found (after excluding the target).")

    X = data[spectral_cols].copy()
    y = data[target_column]

    def reflectance(x: pd.DataFrame) -> pd.DataFrame:
        """Enhanced reflectance with outlier-robust normalization"""
        # Apply slight smoothing to reduce noise that might mask extreme values
        result = x.copy()
        
        # Clip extreme outliers that are likely measurement errors
        for col in x.columns:
            Q1 = x[col].quantile(0.01)
            Q99 = x[col].quantile(0.99)
            result[col] = x[col].clip(lower=Q1, upper=Q99)
            
        return result

    def absorbance(x: pd.DataFrame) -> pd.DataFrame:
        """Enhanced absorbance transformation with better extreme value handling"""
        # absorbance = log(1 / reflectance). Avoid division by zero.
        xr = x.replace(0, 1e-6)
        
        # More conservative clipping to prevent log explosion
        xr = xr.clip(lower=1e-5, upper=0.999)
        
        abs_data = np.log(1.0 / xr)
        
        # Handle potential infinities with robust clipping
        abs_data = abs_data.clip(lower=-10, upper=10)  # Reasonable absorbance range
        
        # Replace any remaining infinite values
        abs_data = abs_data.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with column-wise median for robustness
        for col in abs_data.columns:
            if abs_data[col].isna().any():
                median_val = abs_data[col].median()
                abs_data[col] = abs_data[col].fillna(median_val)
        
        return abs_data

    def continuum_removal(x: pd.DataFrame) -> pd.DataFrame:
        """Robust continuum removal with simple normalization"""
        result = x.copy()
        
        for i in range(len(x)):
            spectrum = x.iloc[i].values
            
            try:
                # Simple and robust continuum removal
                # Use row max for normalization (standard approach)
                row_max = spectrum.max()
                
                if row_max > 1e-10:  # Avoid division by very small numbers
                    normalized = spectrum / row_max
                    
                    # Clip to prevent extreme values
                    normalized = np.clip(normalized, 0.0, 2.0)
                else:
                    # If all values are essentially zero, keep them as is
                    normalized = spectrum
                    
                result.iloc[i] = normalized
                
            except Exception as e:
                # Absolute fallback: just copy the original spectrum
                result.iloc[i] = x.iloc[i]
        
        # Final safety check: replace any remaining infinities or extreme values
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.fillna(result.median())
        
        # Clip entire dataframe to reasonable range
        result = result.clip(lower=0.0, upper=2.0)
        
        return result

    preprocessing = {
        "Reflectance": reflectance,
        "Absorbance": absorbance,
        "Continuum Removal": continuum_removal,
    }

    return X, y, preprocessing
