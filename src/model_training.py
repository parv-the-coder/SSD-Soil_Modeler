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

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression
import time

def run_all_pipelines(X, y, preprocessing, random_state=42):
    """
    Enhanced version that combines the best features from both codes:
    - Systematic pipeline comparison with proper nested cross-validation
    - Robust hyperparameter tuning with GridSearchCV
    - Detailed per-fold results and comprehensive metrics
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        preprocessing (dict): Dictionary of preprocessing functions {name: function}
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (results, best_model, best_score, best_pipeline, improvement_log, feature_importances, summary_table)
    """
    results = {}
    detailed_results = []
    summaries = []
    best_score = -np.inf
    best_model = None
    best_pipeline = None
    improvement_log = []
    feature_importances = {}
    
    # Initialize KFold for outer CV
    kf_outer = KFold(n_splits=5, shuffle=True, random_state=random_state)
    
    for prep_name, prep_func in preprocessing.items():
        try:
            print(f"\nApplying preprocessing: {prep_name}")
            X_prep = prep_func(X)
            X_prep.columns = X_prep.columns.astype(str)  # Ensure string columns
            
            for model_name in ['cubist', 'rf', 'gbr', 'svr', 'krr', 'pls']:
                print(f"  Training model: {model_name} with {prep_name} preprocessing")
                start_time = time.time()
                fold_results = []
                
                # Get model and parameters
                model, params = _get_model_and_params(model_name)
                
                # Handle missing values
                valid_idx = ~(X_prep.isnull().any(axis=1) | y.isnull())
                X_valid = X_prep[valid_idx].replace([np.inf, -np.inf], np.nan).fillna(X_prep.median())
                y_valid = y[valid_idx]
                
                if model_name == 'cubist':
                    # Use manual grid search for Cubist (from second code)
                    cubist_results = _manual_cubist_search(X_valid, y_valid, params, kf_outer, random_state)
                    fold_results.extend(cubist_results)
                    
                    # Extract best Cubist model from successful folds
                    successful_folds = [r for r in cubist_results if not np.isnan(r['y_pred']).all()]
                    if successful_folds:
                        best_cubist_params = successful_folds[0]['best_params']
                        best_cubist_model = Cubist(**best_cubist_params) if CUBIST_AVAILABLE else RandomForestRegressor(n_estimators=100)
                        best_cubist_model.fit(X_valid, y_valid)
                        
                else:
                    # Use GridSearchCV for other models (from second code)
                    for fold, (train_index, val_index) in enumerate(kf_outer.split(X_valid)):
                        X_train, X_val = X_valid.iloc[train_index], X_valid.iloc[val_index]
                        y_train, y_val = y_valid.iloc[train_index], y_valid.iloc[val_index]
                        
                        try:
                            grid_search = GridSearchCV(model, params, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=0)
                            grid_search.fit(X_train, y_train)
                            
                            best_estimator = grid_search.best_estimator_
                            best_params = grid_search.best_params_
                            y_pred = best_estimator.predict(X_val)
                            
                            # Store detailed fold results
                            fold_result = {
                                'preprocessing': prep_name,
                                'model': model_name,
                                'fold': fold,
                                'best_params': best_params,
                                'y_true': y_val.tolist(),
                                'y_pred': y_pred.tolist()
                            }
                            fold_results.append(fold_result)
                            detailed_results.append(fold_result)
                            
                        except Exception as e:
                            print(f"    Fold {fold} failed: {e}")
                            fold_result = {
                                'preprocessing': prep_name,
                                'model': model_name,
                                'fold': fold,
                                'best_params': None,
                                'y_true': y_val.tolist(),
                                'y_pred': [np.nan] * len(y_val)
                            }
                            fold_results.append(fold_result)
                            detailed_results.append(fold_result)
                
                # Calculate metrics for this combination
                end_time = time.time()
                total_time = end_time - start_time
                
                # Compute aggregate metrics across folds
                all_y_true = []
                all_y_pred = []
                fold_r2_scores = []
                fold_rmse_scores = []
                
                for fold_result in fold_results:
                    if not np.isnan(fold_result['y_pred']).all():
                        y_t = np.array(fold_result['y_true'])
                        y_p = np.array(fold_result['y_pred'])
                        all_y_true.extend(y_t)
                        all_y_pred.extend(y_p)
                        fold_r2_scores.append(r2_score(y_t, y_p))
                        fold_rmse_scores.append(np.sqrt(mean_squared_error(y_t, y_p)))
                
                if all_y_true:
                    # Overall metrics
                    overall_r2 = r2_score(all_y_true, all_y_pred)
                    overall_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
                    overall_mse = mean_squared_error(all_y_true, all_y_pred)
                    
                    # Store in results (compatible with first code)
                    result_key = f"{prep_name}_{model_name}"
                    results[result_key] = {
                        'r2': overall_r2,
                        'rmse': overall_rmse, 
                        'mse': overall_mse,
                        'y_true': all_y_true,
                        'y_pred': all_y_pred
                    }
                    
                    # Store summary (from second code)
                    summaries.append({
                        'preprocessing': prep_name,
                        'model': model_name,
                        'total_time': total_time,
                        'mean_rmse': np.mean(fold_rmse_scores),
                        'std_rmse': np.std(fold_rmse_scores),
                        'mean_r2': np.mean(fold_r2_scores),
                        'std_r2': np.std(fold_r2_scores),
                        'best_params': str(fold_results[0]['best_params']) if fold_results else '{}'
                    })
                    
                    # Update best model tracking
                    if overall_r2 > best_score:
                        best_score = overall_r2
                        best_pipeline = f"{prep_name}-{model_name}"
                        
                        # Retrain best model on full data
                        if model_name == 'cubist' and CUBIST_AVAILABLE:
                            best_params = fold_results[0]['best_params']
                            best_model = Cubist(**best_params)
                        else:
                            grid_search = GridSearchCV(model, params, cv=3, scoring='neg_root_mean_squared_error')
                            grid_search.fit(X_valid, y_valid)
                            best_model = grid_search.best_estimator_
                        
                        best_model.fit(X_valid, y_valid)
                    
                    # Feature importance extraction (from first code)
                    try:
                        if hasattr(best_model, 'feature_importances_'):
                            feature_importances[result_key] = best_model.feature_importances_.tolist()
                        elif hasattr(best_model, 'coef_'):
                            coef = np.ravel(best_model.coef_)
                            feature_importances[result_key] = coef.tolist()
                    except Exception:
                        pass
                    
                    improvement_log.append(f"{prep_name}_{model_name}: R2={overall_r2:.4f}, RMSE={overall_rmse:.4f}")
                    
        except Exception as outer_e:
            print(f"Preprocessing failed for {prep_name}: {outer_e}")
    
    ## results, model, best_score, name, improvement_log, feature_importances
    print("Going to Return !!!")
    return results, best_model, best_score, best_pipeline, improvement_log, feature_importances


def _get_model_and_params(model_name):
    """Helper function to get model instance and parameter grid"""
    if model_name == 'cubist':
        if CUBIST_AVAILABLE:
            model = Cubist()
            params = {'n_committees': [5, 10, 25]}
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
    
    return model, params


def _manual_cubist_search(X, y, params, kf_outer, random_state):
    """Manual grid search for Cubist (from second code)"""
    results = []
    
    if not CUBIST_AVAILABLE:
        # Fallback to simple approach if Cubist not available
        for fold, (train_index, val_index) in enumerate(kf_outer.split(X)):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            
            model = RandomForestRegressor(n_estimators=100)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            results.append({
                'preprocessing': 'cubist_fallback',
                'model': 'cubist',
                'fold': fold,
                'best_params': {'n_estimators': 100},
                'y_true': y_val.tolist(),
                'y_pred': y_pred.tolist()
            })
        return results
    
    # Manual grid search for Cubist
    best_cubist_score = -np.inf
    best_cubist_params = {}
    
    for n_committees_val in params['n_committees']:
        fold_scores = []
        inner_kf = KFold(n_splits=3, shuffle=True, random_state=random_state)
        
        for inner_train_index, inner_val_index in inner_kf.split(X):
            X_inner_train, X_inner_val = X.iloc[inner_train_index], X.iloc[inner_val_index]
            y_inner_train, y_inner_val = y.iloc[inner_train_index], y.iloc[inner_val_index]
            
            try:
                current_model = Cubist(n_committees=n_committees_val)
                current_model.fit(X_inner_train, y_inner_train)
                y_inner_pred = current_model.predict(X_inner_val)
                fold_scores.append(r2_score(y_inner_val, y_inner_pred))
            except Exception as e:
                print(f"    Cubist inner CV failed for n_committees={n_committees_val}: {e}")
                fold_scores.append(np.nan)
        
        mean_score = np.nanmean(fold_scores)
        if mean_score > best_cubist_score:
            best_cubist_score = mean_score
            best_cubist_params = {'n_committees': n_committees_val}
    
    if not best_cubist_params:
        best_cubist_params = {'n_committees': 1}
    
    # Use best params for outer CV
    for fold, (train_index, val_index) in enumerate(kf_outer.split(X)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        try:
            model = Cubist(**best_cubist_params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            results.append({
                'preprocessing': 'cubist',
                'model': 'cubist',
                'fold': fold,
                'best_params': best_cubist_params,
                'y_true': y_val.tolist(),
                'y_pred': y_pred.flatten().tolist() if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1 else y_pred.tolist()
            })
        except Exception as e:
            print(f"    Cubist fold {fold} failed: {e}")
            results.append({
                'preprocessing': 'cubist',
                'model': 'cubist',
                'fold': fold,
                'best_params': best_cubist_params,
                'y_true': y_val.tolist(),
                'y_pred': [np.nan] * len(y_val)
            })
    
    return results

if __name__ == '__main__':
    data_dir = 'data'
    filenames = [f'spectra_with_target_T{i}.xls' for i in range(1, 6)]
    for file in filenames:
        path = os.path.join(data_dir, file)
        try:
            df = load_data(path)
            X, y, prep = preprocess_data(df, 'target')
            print("Here we go!")
            results, model, best_score, name, improvement_log, feature_importances = run_all_pipelines(X, y, prep)
            print("Broke Here")
            if model:
                os.makedirs('model', exist_ok=True)
                fname = os.path.splitext(file)[0]
                with open(f'model/{fname}.pkl', 'wb') as f:
                    pickle.dump(model, f)
                print(f"Trained and saved: {name} → model/{fname}.pkl")
        except Exception as e:
            print(f"Error training {file}: {e}")