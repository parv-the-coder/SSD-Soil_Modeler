from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
import numpy as np
import pandas as pd

def run_all_pipelines(X, y, preprocessing, log_improvements=False, return_feature_importances=False):
    models = {
        'PLS': PLSRegression(n_components=10),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GBRT': GradientBoostingRegressor(),
        'KRR': KernelRidge(),
        'SVR': SVR()
    }
    results = {}
    best_score = -np.inf
    best_model = None
    best_pipeline = None
    improvement_log = []
    feature_importances = None
    try:
        for prep_name, prep_func in preprocessing.items():
            X_prep = prep_func(X)
            # Drop rows with NaNs in features or target
            valid_idx = ~(X_prep.isnull().any(axis=1) | y.isnull())
            X_prep_valid = X_prep[valid_idx]
            y_valid = y[valid_idx]
            for model_name, model in models.items():
                pipeline_name = f'{prep_name}_{model_name}'
                y_pred = cross_val_predict(model, X_prep_valid, y_valid, cv=5)
                r2 = r2_score(y_valid, y_pred)
                mse = mean_squared_error(y_valid, y_pred)
                results[pipeline_name] = {
                    'model': model.fit(X_prep_valid, y_valid),
                    'y_pred': y_pred,
                    'y_true': y_valid,
                    'r2': r2,
                    'mse': mse
                }
                if log_improvements:
                    improvement_log.append(f"{pipeline_name}: R2={r2:.4f}, MSE={mse:.4f}")
                if r2 > best_score:
                    best_score = r2
                    best_model = results[pipeline_name]['model']
                    best_pipeline = pipeline_name
                    if return_feature_importances and hasattr(best_model, 'feature_importances_'):
                        feature_importances = pd.Series(best_model.feature_importances_, index=X_prep_valid.columns)
        return results, best_model, best_score, best_pipeline, improvement_log, feature_importances
    except Exception as e:
        raise RuntimeError(f"Pipeline failed: {str(e)}")
