import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import cross_val_predict, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR

try:
    from cubist import Cubist
    CUBIST_AVAILABLE = True
except ImportError:
    CUBIST_AVAILABLE = False

def load_data(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        return pd.read_csv(file_path)
    elif ext in ('.xls', '.xlsx'):
        try:
            if ext == '.xls':
                return pd.read_excel(file_path, engine='xlrd')
            else:
                return pd.read_excel(file_path, engine='openpyxl')
        except:
            return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def preprocess_data(data, target_column):
    if not isinstance(target_column, str):
        target_column = str(target_column)
    numeric_cols = list(data.select_dtypes(include=[np.number]).columns)
    spectral_cols = [c for c in numeric_cols if c != target_column]
    if not spectral_cols:
        raise ValueError("No numeric spectral columns found (after excluding the target).")
    X = data[spectral_cols].copy()
    y = data[target_column]

    def reflectance(x):
        result = x.copy()
        for col in x.columns:
            Q1 = x[col].quantile(0.01)
            Q99 = x[col].quantile(0.99)
            result[col] = x[col].clip(lower=Q1, upper=Q99)
        return result

    def absorbance(x):
        xr = x.replace(0, 1e-6)
        xr = xr.clip(lower=1e-5, upper=0.999)
        abs_data = np.log(1.0 / xr).clip(lower=-10, upper=10)
        abs_data = abs_data.replace([np.inf, -np.inf], np.nan)
        for col in abs_data.columns:
            if abs_data[col].isna().any():
                abs_data[col] = abs_data[col].fillna(abs_data[col].median())
        return abs_data

    def continuum_removal(x):
        result = x.copy()
        for i in range(len(x)):
            spectrum = x.iloc[i].values
            try:
                row_max = spectrum.max()
                normalized = spectrum / row_max if row_max > 1e-10 else spectrum
                result.iloc[i] = np.clip(normalized, 0.0, 2.0)
            except:
                result.iloc[i] = x.iloc[i]
        result = result.replace([np.inf, -np.inf], np.nan).fillna(result.median()).clip(lower=0.0, upper=2.0)
        return result

    preprocessing = {
        "Reflectance": reflectance,
        "Absorbance": absorbance,
        "Continuum Removal": continuum_removal
    }

    return X, y, preprocessing

def run_all_pipelines(X, y, preprocessing):
    results = {}
    best_score = -np.inf
    best_model = None
    best_pipeline = None

    for prep_name, prep_func in preprocessing.items():
        try:
            X_prep = prep_func(X)
            results[prep_name] = {}
            for model_name in ['cubist', 'rf', 'gbr', 'svr', 'krr', 'pls']:
                if model_name == 'cubist':
                    model = Cubist() if CUBIST_AVAILABLE else RandomForestRegressor(n_estimators=100)
                    params = {'committees': [50, 100]} if CUBIST_AVAILABLE else {'n_estimators': [50, 100]}
                elif model_name == 'rf':
                    model = RandomForestRegressor()
                    params = {'n_estimators': [50, 100]}
                elif model_name == 'gbr':
                    model = GradientBoostingRegressor()
                    params = {'n_estimators': [100], 'learning_rate': [0.1]}
                elif model_name == 'svr':
                    model = SVR()
                    params = {'C': [1.0], 'epsilon': [0.1]}
                elif model_name == 'krr':
                    model = KernelRidge()
                    params = {'alpha': [1.0]}
                elif model_name == 'pls':
                    model = PLSRegression()
                    params = {'n_components': [5]}

                valid_idx = ~(X_prep.isnull().any(axis=1) | y.isnull())
                X_valid = X_prep[valid_idx].replace([np.inf, -np.inf], np.nan).fillna(X_prep.median())
                y_valid = y[valid_idx]

                try:
                    search = RandomizedSearchCV(model, params, cv=3, n_iter=2, n_jobs=-1)
                    search.fit(X_valid, y_valid)
                    best_est = search.best_estimator_
                except Exception as e:
                    print(f"  {prep_name}-{model_name} failed: {e}")
                    continue

                try:
                    y_pred = cross_val_predict(best_est, X_valid, y_valid, cv=3)
                    r2 = r2_score(y_valid, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
                    results[prep_name][model_name] = {'r2': r2, 'rmse': rmse}
                    if r2 > best_score:
                        best_score = r2
                        best_model = best_est
                        best_pipeline = f"{prep_name}-{model_name}"
                except Exception as e:
                    print(f"  Scoring failed for {prep_name}-{model_name}: {e}")
        except Exception as outer_e:
            print(f"Preprocessing failed for {prep_name}: {outer_e}")
    return results, best_model, best_pipeline

if __name__ == '__main__':
    data_dir = 'data'
    filenames = [f'spectra_with_target_T{i}.xls' for i in range(1, 6)]
    for file in filenames:
        path = os.path.join(data_dir, file)
        try:
            df = load_data(path)
            X, y, prep = preprocess_data(df, 'target')
            results, model, name = run_all_pipelines(X, y, prep)
            if model:
                os.makedirs('model', exist_ok=True)
                fname = os.path.splitext(file)[0]
                with open(f'model/{fname}.pkl', 'wb') as f:
                    pickle.dump(model, f)
                print(f"Trained and saved: {name} → model/{fname}.pkl")
        except Exception as e:
            print(f"Error training {file}: {e}")