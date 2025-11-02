import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
import numpy as np

def evaluate_models(results):
    metrics = []
    for name, res in results.items():
        r2 = r2_score(res["y_true"], res["y_pred"])
        rmse = np.sqrt(mean_squared_error(res["y_true"], res["y_pred"]))
        rpd = np.std(res["y_true"]) / rmse  # Ratio of Performance to Deviation
        metrics.append({"Pipeline": name, "R²": r2, "RMSE": rmse, "RPD": rpd})
    df = pd.DataFrame(metrics).sort_values("R²", ascending=False)
    return df

def plot_results(results, leaderboard):
    # Scatter plot for top model
    top_pipeline = leaderboard.iloc[0]["Pipeline"]
    res = results[top_pipeline]
    fig, ax = plt.subplots()
    ax.scatter(res["y_true"], res["y_pred"])
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Predictions: {top_pipeline}")
    st.pyplot(fig)
    
    # Feature importance (for tree-based models, e.g., GBRT)
    if "GBRT" in top_pipeline:
        model = res["model"]
        # Assuming X is available; in real code, pass X
        # imp = permutation_importance(model, X, y, n_repeats=10)
        # sns.barplot(x=imp.importances_mean, y=X.columns)
        st.info("Feature importance plot would go here (needs X passed in).")