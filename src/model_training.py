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
        # Try reading as Excel first; if that fails (file mislabelled), try CSV/text fallbacks.
        try:
            if ext == '.xls':
                return pd.read_excel(file_path, engine='xlrd')
            else:
                return pd.read_excel(file_path, engine='openpyxl')
        except Exception:
            # Some files may be CSVs with an .xls extension. Try CSV first, then a loose table read.
            try:
                return pd.read_csv(file_path)
            except Exception:
                try:
                    return pd.read_table(file_path, sep=None, engine='python')
                except Exception as e:
                    # Re-raise a helpful error
                    raise ValueError(f"Failed to read file {file_path} as Excel or CSV: {e}")
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
    improvement_log = []
    feature_importances = {}

    for prep_name, prep_func in preprocessing.items():
        try:
            X_prep = prep_func(X)
            for model_name in ['cubist', 'rf', 'gbr', 'svr', 'krr', 'pls']:
                if model_name == 'cubist':
                    # Use Cubist when available; otherwise fall back to RandomForest.
                    if CUBIST_AVAILABLE:
                        model = Cubist()
                        # Cubist Python wrapper uses parameter names like 'n_committees' and 'n_rules'.
                        params = {
                            'n_committees': [5, 10, 25],
                            'n_rules': [1, 3, 5]
                        }
                    else:
                        model = RandomForestRegressor(n_estimators=100)
                        params = {'n_estimators': [50, 100]}
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
                    # Use single-threaded search to avoid joblib Parallel executor shutdown
                    # issues that can occur when many nested parallel jobs are used.
                    search = RandomizedSearchCV(model, params, cv=3, n_iter=3, n_jobs=1)
                    search.fit(X_valid, y_valid)
                    best_est = search.best_estimator_
                except Exception as e:
                    print(f"  {prep_name}-{model_name} failed: {e}")
                    continue

                try:
                    y_pred = cross_val_predict(best_est, X_valid, y_valid, cv=3)
                    r2 = r2_score(y_valid, y_pred)
                    mse = mean_squared_error(y_valid, y_pred)
                    rmse = np.sqrt(mse)
                    result_key = f"{prep_name}_{model_name}"
                    results[result_key] = {'r2': r2, 'rmse': rmse, 'mse': mse, 'y_true': y_valid.values, 'y_pred': y_pred}

                    # Log improvement info
                    improvement_log.append(f"{prep_name}_{model_name}: R2={r2:.4f}, RMSE={rmse:.4f}")

                    # Capture feature importances or coefficients if available
                    try:
                        if hasattr(best_est, 'feature_importances_'):
                            feature_importances[f"{prep_name}_{model_name}"] = best_est.feature_importances_.tolist()
                        elif hasattr(best_est, 'coef_'):
                            coef = np.ravel(best_est.coef_)
                            feature_importances[f"{prep_name}_{model_name}"] = coef.tolist()
                    except Exception:
                        # Ignore introspection errors
                        pass

                    if r2 > best_score:
                        best_score = r2
                        best_model = best_est
                        best_pipeline = f"{prep_name}-{model_name}"
                except Exception as e:
                    print(f"  Scoring failed for {prep_name}-{model_name}: {e}")
        except Exception as outer_e:
            print(f"Preprocessing failed for {prep_name}: {outer_e}")
    return results, best_model, best_score, best_pipeline, improvement_log, feature_importances

if __name__ == '__main__':
    data_dir = 'data'
    filenames = [f'spectra_with_target_T{i}.xls' for i in range(1, 6)]
    for file in filenames:
        path = os.path.join(data_dir, file)
        try:
            df = load_data(path)
            X, y, prep = preprocess_data(df, 'target')
            results, model, best_score, name, improvement_log, feature_importances = run_all_pipelines(X, y, prep)
            if model:
                os.makedirs('model', exist_ok=True)
                fname = os.path.splitext(file)[0]
                with open(f'model/{fname}.pkl', 'wb') as f:
                    pickle.dump(model, f)
                print(f"Trained and saved: {name} → model/{fname}.pkl")
        except Exception as e:
            print(f"Error training {file}: {e}")