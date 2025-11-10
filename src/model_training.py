from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
import numpy as np
import pandas as pd
import os

# Optimize for 16-core system
os.environ['OMP_NUM_THREADS'] = '16'        # OpenMP threads
os.environ['MKL_NUM_THREADS'] = '16'        # Intel MKL threads
os.environ['NUMEXPR_NUM_THREADS'] = '16'    # NumExpr threads
os.environ['OPENBLAS_NUM_THREADS'] = '16'   # OpenBLAS threads

def run_all_pipelines(X, y, preprocessing, log_improvements=False, return_feature_importances=False):
    # Models will be instantiated per-preprocessing step to ensure safe parameter choices
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
            # Clean feature matrix: replace infinities and fill NaNs with column medians
            X_prep_valid = X_prep_valid.replace([np.inf, -np.inf], np.nan)
            X_prep_valid = X_prep_valid.fillna(X_prep_valid.median())
            # Drop constant columns (zero variance) which can destabilize some models
            non_const_cols = X_prep_valid.columns[X_prep_valid.std(ddof=0) > 0]
            X_prep_valid = X_prep_valid.loc[:, non_const_cols]

            # Choose safe number of PLS components based on data shape
            max_pls = min(20, max(1, X_prep_valid.shape[1] // 3))
            n_comp = min(max_pls, max(1, X_prep_valid.shape[0] - 1), X_prep_valid.shape[1])

            # Balanced aggressive models for better extreme value prediction
            # Optimized for 16-core system - use all available cores
            models = {
                'PLS': PLSRegression(n_components=n_comp),
                'RandomForest': RandomForestRegressor(
                    n_estimators=150,           # Balanced tree count
                    max_depth=20,               # Deep but not unlimited
                    min_samples_split=2,        
                    min_samples_leaf=1,         
                    max_features=0.9,           # Use most features
                    bootstrap=True,
                    n_jobs=16,                  # Use all 16 cores
                    random_state=42
                ),
                'GBRT': GradientBoostingRegressor(
                    n_estimators=150,           
                    learning_rate=0.08,         # Faster learning
                    max_depth=10,               # Deeper trees
                    subsample=0.8,              
                    max_features=0.9,           
                    loss='huber',               # Robust to outliers
                    alpha=0.8,                  
                    random_state=42
                ),
                'KRR': KernelRidge(
                    kernel='rbf',
                    alpha=0.005,                # Less regularization
                    gamma=1 / max(1, X_prep_valid.shape[1])
                ),
                'SVR': SVR(
                    kernel='rbf',
                    C=200.0,                    # Higher C
                    epsilon=0.0005,             # Tighter fitting
                    gamma='scale'
                )
            }
            for model_name, model in models.items():
                pipeline_name = f'{prep_name}_{model_name}'
                
                # Enhanced cross-validation with extreme value focused strategy
                # Use all 16 cores for cross-validation
                from sklearn.model_selection import StratifiedKFold, KFold
                
                # Create stratified bins for target with emphasis on extremes
                # Use more bins to better preserve extreme values
                try:
                    y_bins = pd.qcut(y_valid, q=10, labels=False, duplicates='drop')
                    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    y_pred = cross_val_predict(model, X_prep_valid, y_valid, 
                                             cv=cv_strategy.split(X_prep_valid, y_bins),
                                             n_jobs=16)  # Use all 16 cores
                except:
                    # Enhanced fallback: use shuffle with multiple random seeds for diversity
                    cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
                    y_pred = cross_val_predict(model, X_prep_valid, y_valid, 
                                             cv=cv_strategy, n_jobs=16)  # Use all 16 cores
                
                r2 = r2_score(y_valid, y_pred)
                mse = mean_squared_error(y_valid, y_pred)
                
                # Calculate prediction range coverage for extreme value assessment
                pred_range = np.ptp(y_pred)  # Peak-to-peak range of predictions
                true_range = np.ptp(y_valid)  # Peak-to-peak range of true values
                
                # Robust range coverage calculation with bounds
                if true_range > 0 and pred_range > 0 and not np.isnan(pred_range) and not np.isnan(true_range):
                    range_coverage = min(pred_range / true_range, 2.0)  # Cap at 2.0 to prevent extreme values
                else:
                    range_coverage = 0.0
                
                # Calculate standard deviation ratios with bounds checking
                pred_std = np.std(y_pred)
                true_std = np.std(y_valid)
                
                if true_std > 0 and not np.isnan(pred_std) and not np.isnan(true_std):
                    std_ratio = min(pred_std / true_std, 3.0)  # Cap at 3.0 to prevent extreme ratios
                else:
                    std_ratio = 0.0
                
                # Calculate extreme value prediction accuracy
                # Find extreme values (top/bottom 10%)
                extreme_threshold_low = np.percentile(y_valid, 10)
                extreme_threshold_high = np.percentile(y_valid, 90)
                
                extreme_mask = (y_valid <= extreme_threshold_low) | (y_valid >= extreme_threshold_high)
                if np.sum(extreme_mask) > 0:
                    extreme_r2 = r2_score(y_valid[extreme_mask], y_pred[extreme_mask])
                else:
                    extreme_r2 = r2
                
                # Fit final model
                final_model = model.fit(X_prep_valid, y_valid)
                
                results[pipeline_name] = {
                    'model': final_model,
                    'y_pred': y_pred,
                    'y_true': y_valid,
                    'r2': r2,
                    'mse': mse,
                    'range_coverage': range_coverage,
                    'std_ratio': std_ratio,
                    'extreme_r2': extreme_r2
                }
                if log_improvements:
                    improvement_log.append(f"{pipeline_name}: R2={r2:.4f}, MSE={mse:.4f}, Range_Cov={range_coverage:.3f}, Std_Ratio={std_ratio:.3f}")
                
                # Enhanced scoring: heavily prioritize extreme value prediction
                # Comprehensive scoring that heavily rewards extreme value prediction
                combined_score = (
                    0.2 * r2 +                    # Reduced weight for base R²
                    0.4 * range_coverage +        # Higher weight for range coverage
                    0.3 * std_ratio +             # Higher weight for std preservation
                    0.1 * max(0, extreme_r2)      # Extreme value accuracy bonus
                )
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_model = results[pipeline_name]['model']
                    best_pipeline = pipeline_name
                    if return_feature_importances and hasattr(best_model, 'feature_importances_'):
                        feature_importances = pd.Series(best_model.feature_importances_, index=X_prep_valid.columns)
        return results, best_model, best_score, best_pipeline, improvement_log, feature_importances
    except Exception as e:
        raise RuntimeError(f"Pipeline failed: {str(e)}")
