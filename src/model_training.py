from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
import numpy as np

def run_all_pipelines(X, y, preprocessing):
    models = {
        'PLS': PLSRegression(n_components=10),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GBRT': GradientBoostingRegressor(),
        'KRR': KernelRidge(),
        'SVR': SVR()
    }
    
    results = {}
    for prep_name, prep_func in preprocessing.items():
        X_prep = prep_func(X)
        for model_name, model in models.items():
            pipeline_name = f'{prep_name}_{model_name}'
            y_pred = cross_val_predict(model, X_prep, y, cv=5)
            results[pipeline_name] = {
                'model': model.fit(X_prep, y),  # Fit on full data for export
                'y_pred': y_pred,
                'y_true': y
            }
    return results
