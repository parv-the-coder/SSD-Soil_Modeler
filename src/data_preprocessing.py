import pandas as pd
import numpy as np


# ---------------------------------------------------------
# 1. REFLECTANCE (NO RANGE CHANGE)
# ---------------------------------------------------------
def reflectance_transform(x: pd.DataFrame) -> pd.DataFrame:
    """
    Reflectance preprocessing without clipping or normalization.
    Only fixes zeros/NaN values.
    """
    result = x.copy()

    # Replace zeros (log unsafe) with small epsilon, but keep original scale
    result = result.replace(0, np.nan)
    result = result.fillna(method="ffill").fillna(method="bfill")

    return result


# ---------------------------------------------------------
# 2. ABSORBANCE (NO EXTRA CLIPPING, PURE TRANSFORM)
# ---------------------------------------------------------
def absorbance_transform(x: pd.DataFrame) -> pd.DataFrame:
    """
    Absorbance = -log10(reflectance).
    No clipping, no normalization.
    """
    xr = x.replace(0, 1e-6)       # avoid log(0)
    abs_data = -np.log10(xr)

    # Replace inf/nan only where needed
    abs_data = abs_data.replace([np.inf, -np.inf], np.nan)

    for col in abs_data.columns:
        abs_data[col] = abs_data[col].fillna(abs_data[col].median())

    return abs_data


# ---------------------------------------------------------
# 3. CONTINUUM REMOVAL (SCALE-PRESERVING VERSION)
#    Uses hull subtraction instead of dividing by max
# ---------------------------------------------------------
def continuum_removal_transform(x: pd.DataFrame) -> pd.DataFrame:
    """
    Continuum removal without normalization.
    Subtracts convex hull instead of dividing by it.
    Keeps original data range intact.
    """
    result = x.copy()
    wavelengths = np.arange(x.shape[1])

    for i in range(len(x)):
        spectrum = x.iloc[i].values

        # Simple two-endpoint hull (you can extend to full convex hull later)
        hull = np.interp(wavelengths,
                         [0, len(wavelengths) - 1],
                         [spectrum[0], spectrum[-1]])

        removed = spectrum - hull
        result.iloc[i] = removed

    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.fillna(result.median())

    return result


# ---------------------------------------------------------
# 4. PREPROCESS WRAPPER
# ---------------------------------------------------------
def preprocess_data(data: pd.DataFrame, target_column):
    """
    Extract spectral features and return preprocessing functions.
    """
    if not isinstance(target_column, str):
        target_column = str(target_column)

    numeric_cols = list(data.select_dtypes(include=[np.number]).columns)
    spectral_cols = [c for c in numeric_cols if c != target_column]

    if not spectral_cols:
        raise ValueError("No numeric spectral columns found (after excluding the target).")

    X = data[spectral_cols].copy()
    y = data[target_column]

    preprocessing = {
        "Reflectance": reflectance_transform,
        "Absorbance": absorbance_transform,
        "Continuum Removal": continuum_removal_transform,
    }

    return X, y, preprocessing
