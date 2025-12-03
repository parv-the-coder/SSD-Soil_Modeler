import streamlit as st
import pandas as pd
from src.data_preprocessing import (
    preprocess_data,
    reflectance_transform,
    absorbance_transform,
    continuum_removal_transform,
)
from src.model_training import run_all_pipelines
from src.evaluation import evaluate_models, plot_results
from src.prediction import predict_with_model
from src.export import export_best_model
from io import BytesIO
import os
import re
from pathlib import Path
from typing import Optional, Sequence
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import chardet
import numpy as np
import concurrent.futures
import plotly.express as px
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneOut
from sklearn.base import clone
import pickle
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from src.database import UserManager
import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager
import jwt
from datetime import datetime, timedelta
from streamlit.components.v1 import html as st_html

def smart_read(file):
    file.seek(0)
    sample = file.read(2048)
    file.seek(0)
    try:
        if file.name.endswith('.xlsx'):
            return pd.read_excel(file, engine='openpyxl')
        elif file.name.endswith('.xls'):
            try:
                return pd.read_excel(file, engine='xlrd')
            except Exception:
                pass
        if b',' in sample or file.name.endswith('.csv'):
            encoding = chardet.detect(sample)['encoding'] or 'utf-8'
            file.seek(0)
            return pd.read_csv(file, encoding=encoding)
        else:
            raise ValueError('Unsupported file type or format')
    except Exception as e:
        raise ValueError(f'Failed to read {file.name}: {e}')


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"


def extract_wavelength(label: str) -> float:
    """Convert column labels to numeric wavelength for sorting/plotting."""
    try:
        return float(label)
    except (TypeError, ValueError):
        cleaned = "".join(ch for ch in str(label) if ch.isdigit() or ch == ".")
        if cleaned:
            return float(cleaned)
        raise ValueError(f"Unable to parse wavelength from column '{label}'")


def compute_permutation_summary(model, X_df, y_series, best_pipeline, preprocessing_map):
    """Return sorted permutation importance dataframe for the given best pipeline."""
    if model is None or not best_pipeline:
        return None

    prep_name = best_pipeline.split("-")[0].strip() if isinstance(best_pipeline, str) else None
    prep_func = preprocessing_map.get(prep_name) if isinstance(preprocessing_map, dict) else None

    try:
        X_prepared = prep_func(X_df).copy() if callable(prep_func) else X_df.copy()
    except Exception:
        X_prepared = X_df.copy()

    valid_mask = ~(X_prepared.isnull().any(axis=1) | y_series.isnull())
    X_valid = X_prepared.loc[valid_mask]
    y_valid = y_series.loc[valid_mask]

    if X_valid.empty:
        return None

    X_valid = X_valid.replace([np.inf, -np.inf], np.nan)
    if X_valid.isnull().values.any():
        X_valid = X_valid.fillna(X_valid.median())

    try:
        perm_result = permutation_importance(
            model,
            X_valid,
            y_valid,
            n_repeats=15,
            random_state=42,
            n_jobs=-1,
        )
    except Exception:
        return None

    try:
        feature_correlation = X_valid.apply(lambda col: col.corr(y_valid), axis=0)
    except Exception:
        feature_correlation = pd.Series(np.nan, index=X_valid.columns)

    stats_df = pd.DataFrame(
        {
            "Feature": X_valid.columns.astype(str),
            "Mean": X_valid.mean().values,
            "Variance": X_valid.var(ddof=0).values,
            "Correlation": feature_correlation.reindex(X_valid.columns).values,
        }
    )

    importance_df = pd.DataFrame(
        {
            "Feature": X_valid.columns.astype(str),
            "Importance": perm_result.importances_mean,
            "Std": perm_result.importances_std,
        }
    )
    importance_df = importance_df.merge(stats_df, on="Feature", how="left")
    importance_df = importance_df.dropna(subset=["Importance"])
    if importance_df.empty:
        return None
    importance_df = importance_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)
    return importance_df


def run_leave_one_out_evaluation(best_model, X_df, y_series, best_pipeline, preprocessing_map):
    """Compute leave-one-out cross-validation metrics for the best pipeline."""
    if best_model is None or not best_pipeline or X_df is None or y_series is None:
        return None

    prep_name = best_pipeline.split("-")[0].strip() if isinstance(best_pipeline, str) else None
    prep_func = preprocessing_map.get(prep_name) if isinstance(preprocessing_map, dict) else None

    try:
        X_prepared = prep_func(X_df).copy() if callable(prep_func) else X_df.copy()
    except Exception:
        X_prepared = X_df.copy()

    valid_mask = ~(X_prepared.isnull().any(axis=1) | y_series.isnull())
    X_valid = X_prepared.loc[valid_mask]
    y_valid = y_series.loc[valid_mask]

    if len(y_valid) < 3:
        return None

    X_valid = X_valid.replace([np.inf, -np.inf], np.nan)
    if X_valid.isnull().values.any():
        X_valid = X_valid.fillna(X_valid.median())

    loo = LeaveOneOut()
    predictions = []
    truths = []

    for train_idx, test_idx in loo.split(X_valid):
        try:
            cloned_model = clone(best_model)
        except Exception:
            try:
                cloned_model = copy.deepcopy(best_model)
            except Exception:
                return None

        X_train = X_valid.iloc[train_idx]
        y_train = y_valid.iloc[train_idx]
        X_test = X_valid.iloc[test_idx]
        y_test = y_valid.iloc[test_idx]

        try:
            cloned_model.fit(X_train, y_train)
            y_pred = cloned_model.predict(X_test)
            pred_value = float(y_pred[0]) if isinstance(y_pred, (list, tuple, np.ndarray)) else float(y_pred)
        except Exception:
            pred_value = np.nan

        predictions.append(pred_value)
        truths.append(float(y_test.iloc[0]))

    loo_df = pd.DataFrame({"Actual": truths, "Predicted": predictions}).dropna()
    if loo_df.shape[0] < 3:
        return None

    actual = loo_df["Actual"].values
    predicted = loo_df["Predicted"].values

    try:
        loo_r2 = r2_score(actual, predicted)
        loo_rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
        loo_mae = float(np.mean(np.abs(predicted - actual)))
        loo_rpd = float(np.std(actual) / loo_rmse) if loo_rmse > 0 else np.nan
    except Exception:
        return None

    return {
        "r2": float(loo_r2),
        "rmse": float(loo_rmse),
        "mae": float(loo_mae),
        "rpd": loo_rpd,
        "y_true": loo_df["Actual"].tolist(),
        "y_pred": loo_df["Predicted"].tolist(),
        "n_samples": len(loo_df),
        "pipeline": best_pipeline,
    }


def build_feature_stat_figure(
    feature_df: pd.DataFrame,
    title: str,
    columns: Sequence[str],
    height: int = 360,
    chart: str = "line",
):
    """Return a Plotly figure for the requested feature statistics."""
    if feature_df is None or feature_df.empty:
        return None

    metrics = [col for col in columns if col in feature_df.columns]
    if not metrics:
        return None

    plot_df = feature_df[["Feature"] + metrics].copy()
    plot_df["Feature"] = plot_df["Feature"].astype(str)
    melted = plot_df.melt(id_vars="Feature", value_vars=metrics, var_name="Metric", value_name="Value")

    if chart == "bar":
        fig = px.bar(
            melted,
            x="Feature",
            y="Value",
            color="Metric",
            title=title,
        )
    else:
        fig = px.line(
            melted,
            x="Feature",
            y="Value",
            color="Metric",
            markers=True,
            title=title,
        )

    fig.update_layout(
        height=height,
        margin=dict(t=60, r=30, b=80, l=50),
        legend_title_text="Statistic",
    )
    fig.update_xaxes(tickangle=-45)
    return fig


def build_metric_line_chart(feature_df: pd.DataFrame, metric: str, title: str, height: int = 420):
    """Return a single-metric line chart covering all features."""
    if feature_df is None or feature_df.empty or metric not in feature_df.columns:
        return None

    plot_df = feature_df[["Feature", metric]].dropna().copy()
    if plot_df.empty:
        return None

    def _safe_wavelength(value):
        try:
            return extract_wavelength(value)
        except Exception:
            return np.nan

    plot_df["Feature"] = plot_df["Feature"].astype(str)
    plot_df["_order"] = plot_df["Feature"].apply(_safe_wavelength)
    plot_df = plot_df.sort_values("_order", kind="mergesort").drop(columns="_order")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["Feature"],
            y=plot_df[metric],
            mode="lines+markers",
            name=metric,
            line=dict(color="#1c6335", width=2.8, shape="spline"),
            marker=dict(
                size=7,
                color="#0f2917",
                line=dict(width=1.2, color="#f4fff7"),
            ),
            fill="tozeroy",
            fillcolor="rgba(28, 99, 53, 0.15)",
            hovertemplate="Feature %{x}<br>Value %{y:.4f}<extra></extra>",
        )
    )

    baseline = 0.0
    x_stems = []
    y_stems = []
    for feat_label, value in zip(plot_df["Feature"], plot_df[metric]):
        x_stems.extend([feat_label, feat_label, None])
        y_stems.extend([baseline, value, None])

    fig.add_trace(
        go.Scatter(
            x=x_stems,
            y=y_stems,
            mode="lines",
            line=dict(color="#9edbb3", width=1.3),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    y_tick_count = min(15, max(6, len(plot_df) // 4 if len(plot_df) > 8 else 8))
    axis_overrides = {
        "Importance": {"tick0": 0, "dtick": 20, "range": [0, 100], "tickformat": ".1f"},
        "Mean": {"tick0": 0, "dtick": 0.025},
        "Variance": {"tick0": 0, "dtick": 0.0005, "tickformat": ".4f"},
        "Correlation": {"tick0": -1, "dtick": 0.05, "range": [-1, 1]},
    }
    axis_config = axis_overrides.get(metric, {})
    fig.update_layout(
        height=height,
        margin=dict(t=50, r=30, b=80, l=50),
        plot_bgcolor="rgba(245, 251, 247, 1)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#ffffff", font=dict(color="#0c2b18")),
        title=title,
    )
    fig.update_xaxes(tickangle=-45, showgrid=False)
    fig.update_yaxes(
        nticks=y_tick_count,
        tickformat=axis_config.get("tickformat", ".3f"),
        zeroline=True,
        zerolinecolor="#c8e6d2",
        gridcolor="#dfeee7",
        **{k: v for k, v in axis_config.items() if k not in {"tickformat"}},
    )
    return fig


def render_full_feature_line_section(feature_df: pd.DataFrame, heading: Optional[str] = None):
    """Render separate line charts for importance, mean, variance, and correlation."""
    if feature_df is None or feature_df.empty:
        return 0

    metric_titles = [
        ("Importance", "Feature importance (all features)"),
        ("Mean", "Feature mean (all features)"),
        ("Variance", "Feature variance (all features)"),
        ("Correlation", "Feature correlation (all features)"),
    ]

    rendered = 0
    for metric, title in metric_titles:
        fig = build_metric_line_chart(feature_df, metric, title)
        if fig is None:
            continue
        if rendered == 0 and heading:
            st.markdown(heading)
            st.caption("Line charts show how each statistic evolves across every feature available to the model.")
        st.plotly_chart(fig, use_container_width=True)
        rendered += 1
    return rendered


def _find_training_file_for_model(model_key: str) -> Optional[Path]:
    prefix = f"spectra_with_target_{model_key}"
    for ext in (".xls", ".xlsx", ".csv"):
        candidate = DEFAULT_DATA_DIR / f"{prefix}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_prefixed_training_data(model_key: str):
    """Load default training file for a model and prefix columns to match training-time format."""
    dataset_path = _find_training_file_for_model(model_key)
    if not dataset_path:
        return None, None

    try:
        if dataset_path.suffix.lower() == ".csv":
            df = pd.read_csv(dataset_path)
        elif dataset_path.suffix.lower() == ".xls":
            try:
                df = pd.read_excel(dataset_path, engine="openpyxl")
            except Exception:
                df = pd.read_csv(dataset_path)
        else:
            df = pd.read_excel(dataset_path, engine="openpyxl")
    except Exception:
        return None, None

    prefix = f"spectra_with_target_{model_key}"
    renamed_cols = []
    for col in df.columns:
        col_str = str(col).strip()
        if col_str.startswith(f"{prefix}_"):
            renamed_cols.append(col_str)
        elif col_str.lower() == "target" or col_str.endswith("_target"):
            renamed_cols.append(f"{prefix}_target")
        else:
            renamed_cols.append(f"{prefix}_{col_str}")
    df.columns = renamed_cols

    target_col = next((c for c in renamed_cols if c.endswith("_target")), None)
    return df, target_col


def regenerate_feature_importance_from_saved_model(model_key, target_col, model_path, features_path, pipeline_label):
    """Compute permutation importance using saved model + default dataset when session cache is empty."""
    if not (os.path.exists(model_path) and os.path.exists(features_path) and pipeline_label):
        return None

    dataset_df, detected_target = load_prefixed_training_data(model_key)
    if dataset_df is None:
        return None

    resolved_target = target_col if target_col in dataset_df.columns else detected_target
    if resolved_target is None or resolved_target not in dataset_df.columns:
        return None

    try:
        with open(features_path, "r") as f:
            feature_names = [line.strip() for line in f if line.strip()]
    except Exception:
        return None

    if not feature_names:
        return None

    missing = [feat for feat in feature_names if feat not in dataset_df.columns]
    if missing:
        return None

    data_subset = dataset_df[feature_names + [resolved_target]].copy()
    X = data_subset[feature_names]
    y = data_subset[resolved_target]
    _, _, preprocessing_map = preprocess_data(data_subset, resolved_target)

    try:
        with open(model_path, "rb") as f:
            saved_model = pickle.load(f)
    except Exception:
        return None

    perm_df = compute_permutation_summary(saved_model, X, y, pipeline_label, preprocessing_map)
    return perm_df


@st.cache_data(show_spinner=False)
def discover_spectral_datasets(data_dir: str) -> dict:
    """Return mapping of dataset key (e.g., T1) to file path if available."""
    base = Path(data_dir)
    dataset_paths = {}
    if not base.exists():
        return dataset_paths

    patterns = ["spectra_with_target_*.xls", "spectra_with_target_*.xlsx", "spectra_with_target_*.csv"]
    for pattern in patterns:
        for path in sorted(base.glob(pattern)):
            key = path.stem.replace("spectra_with_target_", "").upper()
            dataset_paths[key] = str(path)
    return dataset_paths


@st.cache_data(show_spinner=False)
def load_spectral_dataset(dataset_key: str, dataset_path: str) -> dict:
    """Load spectral dataset and compute cached summaries for Streamlit UI."""
    path = Path(dataset_path)
    if path.suffix.lower() in {".xls", ".csv"}:
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, engine="openpyxl")

    if "target" not in df.columns:
        target_candidates = [col for col in df.columns if col.lower().endswith("_target")]
        if target_candidates:
            df = df.rename(columns={target_candidates[0]: "target"})
        else:
            raise ValueError(f"Dataset {dataset_key} does not include a 'target' column")

    feature_cols = [col for col in df.columns if col != "target"]
    if not feature_cols:
        raise ValueError(f"Dataset {dataset_key} has no spectral feature columns")

    sorted_feature_cols = sorted(feature_cols, key=extract_wavelength)
    features = df[sorted_feature_cols].copy()
    wavelengths = np.array([extract_wavelength(col) for col in sorted_feature_cols], dtype=float)

    correlations = features.corrwith(df["target"]).dropna()
    correlations = correlations.reindex(correlations.abs().sort_values(ascending=False).index)

    quartiles = features.quantile([0.25, 0.5, 0.75])

    summary = {
        "Samples": int(df.shape[0]),
        "Spectral bands": int(features.shape[1]),
        "Target mean": float(df["target"].mean()),
        "Target std": float(df["target"].std()),
        "Target min": float(df["target"].min()),
        "Target max": float(df["target"].max()),
    }

    return {
        "df": df,
        "features": features,
        "wavelengths": wavelengths,
        "correlations": correlations,
        "quartiles": quartiles,
        "summary": summary,
    }


LOG_LINE_PATTERN = re.compile(r"([A-Za-z ]+_[A-Za-z]+): R2=([0-9.-]+),\s*RMSE=([0-9.-]+)")
PREPROCESSING_FUNCTIONS = {
    "Reflectance": reflectance_transform,
    "Absorbance": absorbance_transform,
    "Continuum Removal": continuum_removal_transform,
}


def extract_best_pipeline_metrics(log_content: str):
    """Return the highest-R2 pipeline entry from a training log."""
    matches = LOG_LINE_PATTERN.findall(log_content or "")
    if not matches:
        return None
    best_entry = max(matches, key=lambda entry: float(entry[1]))
    return {
        "pipeline": best_entry[0],
        "r2": float(best_entry[1]),
        "rmse": float(best_entry[2]),
    }


def apply_preprocessing_for_pipeline(df: pd.DataFrame, pipeline_name: str | None) -> tuple[pd.DataFrame, str]:
    """Apply the preprocessing function that matches the saved pipeline name."""
    prep_label = None
    if pipeline_name:
        prep_label = pipeline_name.split("_")[0].strip()

    transform = PREPROCESSING_FUNCTIONS.get(prep_label, reflectance_transform)
    processed = transform(df.copy())
    processed = processed.replace([np.inf, -np.inf], np.nan)
    processed = processed.fillna(processed.median(numeric_only=True))

    missing_cols = [col for col in df.columns if col not in processed.columns]
    for col in missing_cols:
        processed[col] = 0.0
    processed = processed.reindex(columns=df.columns)
    return processed, (prep_label or "Reflectance")


def list_saved_model_keys() -> list[str]:
    """Return sorted identifiers for stored models that have exported artifacts."""
    models_dir = BASE_DIR / "models"
    if not models_dir.exists():
        return []

    keys: set[str] = set()
    for features_path in models_dir.glob("best_model_*_features.txt"):
        stem = features_path.stem
        key = stem.replace("best_model_", "").replace("_features", "").strip()
        if key:
            keys.add(key)

    return sorted(keys, key=lambda key: key.upper())


def run_leave_one_out_for_saved_model(model_key: str) -> tuple[dict | None, str | None]:
    """Load stored artifacts for a model and run leave-one-out evaluation."""
    if not model_key:
        return None, "Select a stored model before running leave-one-out."

    models_dir = BASE_DIR / "models"
    base_name = f"best_model_{model_key}"
    model_path = models_dir / f"{base_name}.pkl"
    features_path = models_dir / f"{base_name}_features.txt"
    log_path = models_dir / f"{base_name}_log.txt"

    if not model_path.exists():
        return None, f"Model file '{model_path}' not found."
    if not features_path.exists():
        return None, f"Feature list '{features_path}' not found."

    dataset_df, target_col = load_prefixed_training_data(model_key)
    if dataset_df is None or target_col is None:
        return None, (
            f"Could not locate default training data for model {model_key}. "
            "Place a corresponding 'spectra_with_target' file in the data/ folder."
        )

    try:
        with open(features_path, "r") as f:
            feature_names = [line.strip() for line in f if line.strip()]
    except Exception as exc:
        return None, f"Failed to read feature list: {exc}"

    if not feature_names:
        return None, "Stored feature list is empty. Retrain or regenerate the feature file."

    missing = [feat for feat in feature_names if feat not in dataset_df.columns]
    if missing:
        preview = ", ".join(missing[:5])
        if len(missing) > 5:
            preview += ", ..."
        return None, (
            f"Dataset for {model_key} is missing {len(missing)} required features (e.g., {preview}). "
            "Please ensure the default training file matches the saved model."
        )

    subset = dataset_df[feature_names + [target_col]].copy()
    subset = subset.dropna(subset=[target_col])
    if subset.shape[0] < 3:
        return None, "Need at least 3 samples with valid targets to run leave-one-out."

    X = subset[feature_names]
    y = subset[target_col]
    _, _, preprocessing_map = preprocess_data(subset, target_col)

    pipeline_label = None
    if log_path.exists():
        try:
            with open(log_path, "r") as f:
                log_content = f.read()
            best_metrics = extract_best_pipeline_metrics(log_content)
            if best_metrics:
                pipeline_label = best_metrics.get("pipeline")
        except Exception:
            pipeline_label = None
    if not pipeline_label:
        return None, "Unable to determine the best pipeline from the saved training log."

    try:
        with open(model_path, "rb") as f:
            saved_model = pickle.load(f)
    except Exception as exc:
        return None, f"Failed to load saved model: {exc}"

    loo_metrics = run_leave_one_out_evaluation(saved_model, X, y, pipeline_label, preprocessing_map)
    if not loo_metrics:
        return None, "Leave-one-out evaluation failed for the selected model."

    dataset_path = _find_training_file_for_model(model_key)
    loo_metrics["target_column"] = target_col
    loo_metrics["model_key"] = model_key
    loo_metrics["dataset_source"] = str(dataset_path) if dataset_path else None
    return loo_metrics, None


def render_leave_one_out_block(loo_metrics: dict):
    """Render metrics and scatter plot for a leave-one-out run."""
    if not loo_metrics:
        return

    st.markdown("#### Leave-One-Out Validation")
    metric_cols = st.columns(4)
    metric_cols[0].metric("LOO R²", f"{loo_metrics['r2']:.3f}")
    metric_cols[1].metric("LOO RMSE", f"{loo_metrics['rmse']:.3f}")
    metric_cols[2].metric("LOO MAE", f"{loo_metrics['mae']:.3f}")

    rpd_value = loo_metrics.get("rpd")
    rpd_display = "—"
    if rpd_value is not None and not pd.isna(rpd_value):
        try:
            rpd_display = f"{float(rpd_value):.3f}"
        except (TypeError, ValueError):
            rpd_display = "—"
    metric_cols[3].metric("LOO RPD", rpd_display)

    loo_df = pd.DataFrame({"Actual": loo_metrics["y_true"], "Predicted": loo_metrics["y_pred"]}).dropna()
    if len(loo_df) >= 2:
        fig_loo = px.scatter(
            loo_df,
            x="Actual",
            y="Predicted",
            color_discrete_sequence=["#2a7143"],
            title="Leave-One-Out predictions",
        )
        axis_min = float(min(loo_df["Actual"].min(), loo_df["Predicted"].min()))
        axis_max = float(max(loo_df["Actual"].max(), loo_df["Predicted"].max()))
        fig_loo.add_trace(
            go.Scatter(
                x=[axis_min, axis_max],
                y=[axis_min, axis_max],
                mode="lines",
                name="1:1 line",
                line=dict(color="#999999", dash="dash"),
            )
        )
        fig_loo.update_layout(height=420, margin=dict(t=60, r=30, b=40, l=50))
        st.plotly_chart(fig_loo, use_container_width=True)

    pipeline_label = loo_metrics.get("pipeline", "selected pipeline")
    st.caption(
        f"LOO evaluated on {loo_metrics['n_samples']} samples using the `{pipeline_label}` pipeline."
    )

def show_train_models():
    """Show Train Models section with elegant UI"""
    
    # Custom CSS for this section
    st.markdown("""
        <style>
            /* Section header styling */
            .train-header {
                background: linear-gradient(135deg, #2a7143 0%, #1e5030 100%);
                padding: 2rem;
                border-radius: 15px;
                color: white;
                margin-bottom: 2rem;
                box-shadow: 0 4px 15px rgba(42, 113, 67, 0.3);
            }
            .train-header h1 {
                color: white !important;
                margin: 0 0 0.5rem 0;
                font-size: 2.2rem;
            }
            .train-header p {
                color: rgba(255, 255, 255, 0.9);
                margin: 0;
                font-size: 1.05rem;
            }
            
            /* Upload section styling */
            .upload-section {
                background: white;
                padding: 2rem;
                border-radius: 12px;
                border: 2px dashed #2a7143;
                margin: 1.5rem 0;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            }
            
            /* Metric cards */
            .metric-card {
                background: linear-gradient(135deg, #f8fdf9 0%, #e8f5ed 100%);
                padding: 1.5rem;
                border-radius: 10px;
                border-left: 4px solid #2a7143;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
                transition: transform 0.2s ease;
            }
            .metric-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            }
            
            /* Training info box */
            .training-info {
                background: linear-gradient(135deg, #fff9e6 0%, #fff3d6 100%);
                padding: 1.5rem;
                border-radius: 12px;
                border-left: 4px solid #f39c12;
                margin: 1.5rem 0;
            }
            
            /* Results card */
            .results-card {
                background: white;
                padding: 1.5rem;
                border-radius: 12px;
                border: 1px solid #e0e0e0;
                margin: 1rem 0;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
            }
            
            /* Progress section */
            .progress-section {
                background: #f8f9fa;
                padding: 2rem;
                border-radius: 12px;
                margin: 1.5rem 0;
            }
            
            /* File info badge */
            .file-badge {
                display: inline-block;
                background: #e8f5ed;
                color: #2a7143;
                padding: 0.4rem 1rem;
                border-radius: 20px;
                font-size: 0.9rem;
                margin: 0.3rem;
                font-weight: 500;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Header section
    st.markdown("""
        <div class="train-header">
            <h1>Train ML Models</h1>
            <p>Upload your spectral training data to build and train machine learning models. 
            The system will automatically test multiple preprocessing methods and algorithms to find the best model for each target.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for left and right layout
    left_col, right_col = st.columns([1, 1], gap="large")
    
    dfs = []
    merged_df = None
    target_cols = []

    if "feature_importance_results" not in st.session_state:
        st.session_state["feature_importance_results"] = {}
    if "leave_one_out_results" not in st.session_state:
        st.session_state["leave_one_out_results"] = {}
    
    # LEFT SIDE - Upload Section
    with left_col:
        st.markdown("### Upload Training Data")
        st.info("Tip: Upload multiple Excel files containing spectral data with target columns ending in '_target'")
        
        uploaded_files = st.file_uploader(
            "Choose Excel files for training", 
            type=["xls", "xlsx"], 
            accept_multiple_files=True, 
            key="train_files",
            help="Select one or more Excel files containing your spectral data"
        )
        
        if uploaded_files:
            uploaded_files_list = list(uploaded_files)
            st.success(f"Successfully loaded **{len(uploaded_files_list)}** file(s)!")

            current_files = sorted(file.name for file in uploaded_files_list)
            if current_files != st.session_state.get("last_training_files", []):
                st.session_state["training_summaries"] = []
                st.session_state["feature_importance_results"] = {}
                st.session_state["last_training_files"] = current_files

            # Display loaded files in an elegant way
            st.markdown("#### Loaded Files")
            for file in uploaded_files_list:
                try:
                    df = smart_read(file)
                    prefix = file.name.split('.')[0]
                    df = df.add_prefix(prefix + '_')
                    dfs.append(df)
                    st.markdown(f"<span class='file-badge'>{file.name} — {df.shape[0]} samples × {df.shape[1]} features</span>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error loading {file.name}: {str(e)}")
                    continue
            
            if dfs:
                merged_df = pd.concat(dfs, axis=1)
                target_cols = [col for col in merged_df.columns if col.lower().endswith('_target')]
    
    # RIGHT SIDE - Data Overview and Training
    with right_col:
        if merged_df is not None:
            st.markdown("### Merged Data Overview")
            
            # Metrics in elegant cards (stacked vertically in right column)
            st.markdown("""
                <div class="metric-card">
                    <h3 style="color: #2a7143; margin: 0 0 0.5rem 0;">Total Samples</h3>
                    <p style="font-size: 2rem; font-weight: bold; color: #1e5030; margin: 0;">{}</p>
                </div>
            """.format(merged_df.shape[0]), unsafe_allow_html=True)
            
            st.markdown("""
                <div class="metric-card">
                    <h3 style="color: #2a7143; margin: 0 0 0.5rem 0;">Total Features</h3>
                    <p style="font-size: 2rem; font-weight: bold; color: #1e5030; margin: 0;">{}</p>
                </div>
            """.format(merged_df.shape[1]), unsafe_allow_html=True)
            
            st.markdown("""
                <div class="metric-card">
                    <h3 style="color: #2a7143; margin: 0 0 0.5rem 0;">Target Columns</h3>
                    <p style="font-size: 2rem; font-weight: bold; color: #1e5030; margin: 0;">{}</p>
                </div>
            """.format(len(target_cols)), unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            with st.expander("View Data Preview", expanded=False):
                st.dataframe(merged_df.head(10), use_container_width=True)
            
            target_display = ", ".join([f"`{col}`" for col in target_cols]) if target_cols else "None detected"
            st.markdown(f"**Detected target columns:** {target_display}")
        else:
            # Show placeholder when no data is loaded
            st.info("Upload training files on the left to see data overview and training options")
    
    # Training section - Full width below the two columns
    if merged_df is not None and target_cols:
        st.markdown("---")
        st.markdown("### Train ML Pipelines")
        
        st.markdown("""
            <div class="training-info">
                <h4 style="margin: 0 0 1rem 0; color: #856404;">Automated Training Process</h4>
            </div>
        """, unsafe_allow_html=True)
        
        # Center the training button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            train_button = st.button("Train All Models", type="primary", use_container_width=True)

        if train_button:
            if len(target_cols) == 0:
                st.error("No columns ending with '_target' found. Please verify your training files.")
                return

            st.session_state["training_summaries"] = []
            st.session_state["feature_importance_results"] = {}

            import concurrent.futures

            def train_target(target_col):
                try:
                    import re
                    match = re.search(r'T(\d+)_target$', target_col)
                    if match:
                        target_prefix = f"spectra_with_target_T{match.group(1)}_"
                    else:
                        target_prefix = None
                    
                    if target_prefix:
                        feature_cols = [col for col in merged_df.columns if col.startswith(target_prefix) and not col.endswith('_target')]
                    else:
                        feature_cols = [col for col in merged_df.columns if col != target_col]
                    
                    X = merged_df[feature_cols]
                    y = merged_df[target_col]
                    data = pd.concat([X, y], axis=1)
                    data = data.dropna(subset=[target_col])
                    X = data[feature_cols]
                    y = data[target_col]
                    data1 = pd.concat([X, y], axis=1)
                    preprocessing = preprocess_data(data1, target_col)[2]
                    results, best_model, best_score, best_pipeline, improvement_log, feature_importances = run_all_pipelines(X, y, preprocessing)
                    
                    summary = []
                    for name, res in results.items():
                        r2 = res['r2']
                        mse = res['mse']
                        rmse = np.sqrt(mse)
                        std_y = np.std(res['y_true'])
                        rpd = std_y / rmse if rmse != 0 else np.nan
                        summary.append({
                            'Pipeline': name,
                            'R2': r2,
                            'RMSE': rmse,
                            'RPD': rpd
                        })
                    summary_df = pd.DataFrame(summary)
                    
                    match = re.search(r'T(\d+)_target$', target_col)
                    if match:
                        model_id = f'T{match.group(1)}'
                    else:
                        model_id = target_col
                    
                    model_path = os.path.join("models", f"best_model_{model_id}.pkl")
                    feature_names_path = os.path.join("models", f"best_model_{model_id}_features.txt")
                    export_best_model(best_model, model_path)
                    with open(feature_names_path, "w") as f:
                        f.write("\n".join(X.columns))
                    
                    log_path = os.path.join("models", f"best_model_{model_id}_log.txt")
                    with open(log_path, "w") as f:
                        f.write("\n".join(improvement_log))

                    perm_df = compute_permutation_summary(best_model, X, y, best_pipeline, preprocessing)
                    perm_payload = perm_df.to_dict("records") if perm_df is not None else None
                    loo_metrics = run_leave_one_out_evaluation(best_model, X, y, best_pipeline, preprocessing)
                    
                    result_tuple = (target_col, summary_df, best_pipeline, best_score, improvement_log, feature_importances, perm_payload, loo_metrics)
                    if isinstance(result_tuple, tuple):
                        return result_tuple
                    else:
                        raise RuntimeError(f"train_target failed for {target_col}: Did not return tuple result.")
                except Exception as e:
                    raise RuntimeError(f"train_target failed for {target_col}: {str(e)}")

            # Progress section with elegant styling
            st.markdown("""
                <div class="progress-section">
                    <h3 style="color: #2a7143; margin: 0 0 1rem 0;">Training in Progress...</h3>
                </div>
            """, unsafe_allow_html=True)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_targets = len(target_cols)
            completed = 0

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(train_target, target_col): target_col for target_col in target_cols}
                for future in concurrent.futures.as_completed(futures):
                    target_col = futures[future]
                    completed += 1
                    progress_value = min(completed / total_targets, 1.0)
                    progress_bar.progress(progress_value)
                    status_text.markdown(f"**Completed {completed} of {total_targets} targets** · Last finished: `{target_col}`")
                    
                    try:
                        result = future.result()
                        if not isinstance(result, tuple):
                            raise RuntimeError(f"train_target for {target_col} did not return a tuple.")
                        target_col, summary_df, best_pipeline, best_score, improvement_log, feature_importances, perm_payload, loo_metrics = result
                        perm_df = pd.DataFrame(perm_payload) if perm_payload else None
                        
                        with st.expander(f"Results for **{target_col}**", expanded=True):
                            st.markdown("""
                                <div class="results-card">
                                    <h4 style="color: #2a7143; margin: 0 0 1rem 0;">Performance Metrics</h4>
                                </div>
                            """, unsafe_allow_html=True)
                            st.dataframe(summary_df, use_container_width=True)
                            st.success(f"**Best Model:** `{best_pipeline}` | **Score:** `{best_score:.4f}`")
                            
                            with st.expander("Step-by-step Model Improvement Log"):
                                for entry in improvement_log:
                                    st.markdown(f"• {entry}")

                            if perm_df is not None and not perm_df.empty:
                                top_perm = perm_df.head(20).copy().sort_values(by="Importance")
                                hover_fields = {
                                    field: ":.4f"
                                    for field in ["Importance", "Std", "Mean", "Variance", "Correlation"]
                                    if field in top_perm.columns
                                }
                                fig = px.bar(
                                    top_perm,
                                    x="Importance",
                                    y="Feature",
                                    orientation="h",
                                    error_x="Std" if "Std" in top_perm.columns else None,
                                    color="Importance",
                                    color_continuous_scale="YlGn",
                                    hover_data=hover_fields or None,
                                )
                                fig.update_layout(
                                    title="Permutation Feature Importance",
                                    height=420,
                                    margin=dict(t=40, r=20, b=20, l=0),
                                    coloraxis_showscale=False,
                                )
                                fig.update_yaxes(title_text="Spectral feature", tickfont=dict(size=11))
                                st.plotly_chart(fig, use_container_width=True)
                                st.caption("Permutation importance estimates computed from the training session's best pipeline.")
                                stats_fig = build_feature_stat_figure(
                                    top_perm,
                                    "Feature mean vs correlation",
                                    ["Mean", "Correlation"],
                                )
                                if stats_fig is not None:
                                    st.plotly_chart(stats_fig, use_container_width=True)
                                variance_fig = build_feature_stat_figure(
                                    top_perm,
                                    "Feature variance",
                                    ["Variance"],
                                    chart="bar",
                                )
                                if variance_fig is not None:
                                    st.plotly_chart(variance_fig, use_container_width=True)
                                render_full_feature_line_section(
                                    perm_df,
                                    f"#### {target_col}: all features (line view)",
                                )
                                preview_cols = perm_df.head(20).copy()
                                for col in ["Importance", "Std", "Mean", "Variance", "Correlation"]:
                                    if col in preview_cols.columns:
                                        preview_cols[col] = preview_cols[col].round(4)
                                column_config = {
                                    "Feature": st.column_config.TextColumn("Feature", width="large"),
                                }
                                if "Importance" in preview_cols.columns:
                                    column_config["Importance"] = st.column_config.NumberColumn("Importance", format="%.4f")
                                if "Std" in preview_cols.columns:
                                    column_config["Std"] = st.column_config.NumberColumn("Std", format="%.4f")
                                if "Mean" in preview_cols.columns:
                                    column_config["Mean"] = st.column_config.NumberColumn("Mean", format="%.4f")
                                if "Variance" in preview_cols.columns:
                                    column_config["Variance"] = st.column_config.NumberColumn("Variance", format="%.4f")
                                if "Correlation" in preview_cols.columns:
                                    column_config["Correlation"] = st.column_config.NumberColumn("Correlation", format="%.4f")
                                st.dataframe(
                                    preview_cols,
                                    use_container_width=True,
                                    hide_index=True,
                                    column_config=column_config,
                                )
                                st.caption("Higher bars indicate features that most influence the best model's predictions, along with their descriptive statistics.")
                                st.session_state["feature_importance_results"][target_col] = perm_payload
                            else:
                                st.session_state["feature_importance_results"].pop(target_col, None)
                                st.info("Permutation feature importance could not be computed for this target.")

                            if loo_metrics:
                                render_leave_one_out_block(loo_metrics)
                                st.session_state["leave_one_out_results"][target_col] = loo_metrics
                            else:
                                st.session_state["leave_one_out_results"].pop(target_col, None)
                            
                        st.success(f"Best model for **{target_col}** saved successfully!")

                        try:
                            best_row = summary_df.loc[summary_df['Pipeline'] == best_pipeline].iloc[0]
                            st.session_state["training_summaries"].append({
                                "Target": target_col,
                                "Pipeline": best_pipeline,
                                "R2": round(float(best_row['R2']), 4),
                                "RMSE": round(float(best_row['RMSE']), 4),
                                "RPD": round(float(best_row['RPD']), 3) if not pd.isna(best_row['RPD']) else None,
                            })
                        except Exception:
                            pass
                        
                    except Exception as e:
                        st.error(f"Error training {target_col}: {str(e)}")

            progress_bar.progress(1.0)
            status_text.success("**All targets training completed successfully!**")
    
        if st.session_state.get("training_summaries"):
            st.markdown("---")
            st.markdown("### Best Model Overview")
            st.markdown("""
                <div class="results-card">
                    <p style="margin: 0; color: #555;">Summary of the best performing models across all targets</p>
                </div>
            """, unsafe_allow_html=True)
            
            best_models_df = pd.DataFrame(st.session_state["training_summaries"])
            best_models_df = best_models_df.sort_values(by="Target").reset_index(drop=True)
            st.dataframe(best_models_df, use_container_width=True, height=300)

        if st.session_state.get("feature_importance_results"):
            st.markdown("---")
            st.markdown("### Feature Importance Explorer")
            st.caption("Permutation-based importances computed from each target's best pipeline.")

            targets_with_importance = sorted(st.session_state["feature_importance_results"].keys())
            if targets_with_importance:
                col_left, col_right = st.columns([1, 1])
                with col_left:
                    selected_target = st.selectbox(
                        "Select target",
                        targets_with_importance,
                        key="feature_importance_target_select",
                    )
                with col_right:
                    top_k = st.slider("Top features to display", min_value=5, max_value=30, value=15, step=5)

                target_payload = st.session_state["feature_importance_results"].get(selected_target)
                target_df = pd.DataFrame(target_payload) if target_payload else pd.DataFrame()

                if not target_df.empty:
                    top_df = target_df.head(top_k).copy().sort_values(by="Importance")
                    hover_fields = {
                        field: ":.4f"
                        for field in ["Importance", "Std", "Mean", "Variance", "Correlation"]
                        if field in top_df.columns
                    }
                    fig = px.bar(
                        top_df,
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        error_x="Std" if "Std" in top_df.columns else None,
                        color="Importance",
                        color_continuous_scale="YlGn",
                        hover_data=hover_fields or None,
                    )
                    fig.update_layout(
                        height=480,
                        margin=dict(t=40, r=30, b=30, l=0),
                        plot_bgcolor="rgba(245, 251, 247, 1)",
                    )
                    fig.update_yaxes(title_text="Spectral feature")
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Permutation importance estimates computed from the training session's best pipeline.")
                    stats_fig = build_feature_stat_figure(
                        top_df,
                        f"{selected_target}: Feature mean vs correlation",
                        ["Mean", "Correlation"],
                    )
                    if stats_fig is not None:
                        st.plotly_chart(stats_fig, use_container_width=True)
                    variance_fig = build_feature_stat_figure(
                        top_df,
                        f"{selected_target}: Feature variance",
                        ["Variance"],
                        chart="bar",
                    )
                    if variance_fig is not None:
                        st.plotly_chart(variance_fig, use_container_width=True)
                    render_full_feature_line_section(
                        target_df,
                        f"#### {selected_target}: all features (line view)",
                    )

                    preview_df = target_df.head(top_k).copy()
                    for col in ["Importance", "Std", "Mean", "Variance", "Correlation"]:
                        if col in preview_df.columns:
                            preview_df[col] = preview_df[col].round(4)
                    column_config = {
                        "Feature": st.column_config.TextColumn("Feature", width="large"),
                    }
                    if "Importance" in preview_df.columns:
                        column_config["Importance"] = st.column_config.NumberColumn("Importance", format="%.4f")
                    if "Std" in preview_df.columns:
                        column_config["Std"] = st.column_config.NumberColumn("Std", format="%.4f")
                    if "Mean" in preview_df.columns:
                        column_config["Mean"] = st.column_config.NumberColumn("Mean", format="%.4f")
                    if "Variance" in preview_df.columns:
                        column_config["Variance"] = st.column_config.NumberColumn("Variance", format="%.4f")
                    if "Correlation" in preview_df.columns:
                        column_config["Correlation"] = st.column_config.NumberColumn("Correlation", format="%.4f")
                    st.dataframe(
                        preview_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config=column_config,
                    )
                else:
                    st.info("No feature importance data available yet for the selected target.")

def show_make_predictions():
    """Show Make Predictions section"""
    st.header("Make Predictions")
    st.markdown("""
    Upload new spectral data to make predictions using your trained models.
    The system will automatically apply the same preprocessing used during training.
    """)

    col1, col2 = st.columns([1, 2])
    with col1:
        model_options = [f"T{i}" for i in range(1, 6)]
        selected_model = st.selectbox("Select Model", model_options, key="predict_model")
    with col2:
        user_file = st.file_uploader("Upload Spectral Data", type=["xls", "xlsx"], key="user_pred")

    model_path = os.path.join("models", f"best_model_{selected_model}.pkl")
    feature_names_path = os.path.join("models", f"best_model_{selected_model}_features.txt")
    log_path = os.path.join("models", f"best_model_{selected_model}_log.txt")

    if user_file:
        try:
            user_df = smart_read(user_file)
            if os.path.exists(feature_names_path) and os.path.exists(model_path):
                with open(feature_names_path) as f:
                    trained_feature_names = [line.strip() for line in f.readlines() if line.strip()]

                if not trained_feature_names:
                    st.error("Feature list for the selected model is empty. Please retrain the model.")
                    return

                first_feature = trained_feature_names[0]
                parts = first_feature.split('_')
                prefix = '_'.join(parts[:-1]) + '_' if len(parts) > 1 else ''

                st.info(f"Expected feature format: `{prefix}[wavelength]`")

                def rename_col(col):
                    col_str = str(col)
                    if col_str.isdigit():
                        return f"{prefix}{col_str}"
                    if prefix and 'spectra_with_target_' in col_str and f'spectra_with_target_{selected_model}_' not in col_str:
                        wavelength = col_str.split('_')[-1]
                        if wavelength.isdigit():
                            return f"{prefix}{wavelength}"
                    return col_str

                user_df.columns = [rename_col(col) for col in user_df.columns]

                missing = set(trained_feature_names) - set(user_df.columns)
                extra = set(user_df.columns) - set(trained_feature_names)

                if missing:
                    st.warning("Feature mismatch detected:")
                    st.write(f"**Missing columns:** {len(missing)} columns")
                    if len(missing) <= 10:
                        st.write(f"Missing: {sorted(list(missing))}")
                    else:
                        st.write(f"First 10 missing: {sorted(list(missing))[:10]}...")
                    if extra:
                        st.write(f"**Extra columns:** {len(extra)} columns")
                        if len(extra) <= 10:
                            st.write(f"Extra: {sorted(list(extra))}")
                        else:
                            st.write(f"First 10 extra: {sorted(list(extra))[:10]}...")
                    st.error("Column alignment failed. Please check your data format.")
                    st.info(f"""**Debugging Info:**
                    - **Expected prefix:** `{prefix}`
                    - **Your data should have:** Pure wavelength columns (410, 431, 452, ... 2490) OR already correctly formatted columns
                    - **Model expects:** {selected_model} model features

                    **File format tips:**
                    - Upload files with columns like: 410, 431, 452, ... (numbers only)
                    - Or with correct prefix: {prefix}410, {prefix}431, etc.
                    """)
                    return

                user_df_aligned = user_df[trained_feature_names]

                best_pipeline = None
                log_content = ""
                best_metrics = None
                if os.path.exists(log_path):
                    try:
                        with open(log_path, 'r') as f:
                            log_content = f.read()
                        best_metrics = extract_best_pipeline_metrics(log_content)
                        if best_metrics:
                            best_pipeline = best_metrics["pipeline"]
                    except Exception:
                        log_content = ""

                user_df_processed, applied_prep = apply_preprocessing_for_pipeline(user_df_aligned, best_pipeline)
                prep_message = {
                    "Absorbance": "Applying Absorbance preprocessing to match training...",
                    "Continuum Removal": "Applying Continuum Removal preprocessing to match training...",
                    "Reflectance": "Applying Reflectance clipping to match training...",
                }.get(applied_prep, "Applying Reflectance clipping to match training...")
                st.info(prep_message)
                
                
                try:
                    predictions = predict_with_model(model_path, user_df_processed)
                    st.success("Predictions completed successfully!")
                except FileNotFoundError:
                    st.error("Model file not found. Please train models first using the 'Train All Models' section.")
                    return
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
                    return

                model_accuracy = None
                model_r2 = best_metrics["r2"] if best_metrics else None
                model_rmse = best_metrics["rmse"] if best_metrics else None
                range_coverage = None

                if log_content:
                    try:
                        range_match = re.search(r'Range_Cov=([0-9.-]+)', log_content)
                        if range_match:
                            range_coverage = float(range_match.group(1))
                    except Exception as e:
                        st.warning(f"Could not read model performance metrics: {e}")

                if model_r2 is not None:
                    if model_r2 >= 0.8:
                        model_accuracy = 95
                    elif model_r2 >= 0.6:
                        model_accuracy = 85
                    elif model_r2 >= 0.4:
                        model_accuracy = 75
                    elif model_r2 >= 0.2:
                        model_accuracy = 65
                    else:
                        model_accuracy = max(30, model_r2 * 100)

                if model_accuracy is not None:
                    st.success(f"**Prediction completed successfully!** | **Model Accuracy: {model_accuracy:.1f}%** (R² = {model_r2:.3f})")
                else:
                    st.success("**Prediction completed successfully!**")

                st.subheader("Prediction Results & Model Performance")

                if model_accuracy is not None or model_rmse is not None or range_coverage is not None:
                    st.markdown("### Model Testing Accuracy")

                    performance_cols = st.columns(4)

                    with performance_cols[0]:
                        if model_accuracy is not None:
                            if model_accuracy >= 85:
                                accuracy_color = "good"
                            elif model_accuracy >= 70:
                                accuracy_color = "medium"
                            elif model_accuracy >= 60:
                                accuracy_color = "poor"
                            else:
                                accuracy_color = "bad"

                            st.metric(
                                "Testing Accuracy",
                                f"{model_accuracy:.1f}% ({accuracy_color})",
                                help=f"Based on R² score: {model_r2:.3f}"
                            )

                    with performance_cols[1]:
                        if model_r2 is not None:
                            st.metric("R² Score", f"{model_r2:.3f}")

                    with performance_cols[2]:
                        if model_rmse is not None:
                            st.metric("RMSE", f"{model_rmse:.3f}")

                    with performance_cols[3]:
                        if range_coverage is not None:
                            if range_coverage >= 0.7:
                                range_color = "good"
                            elif range_coverage >= 0.5:
                                range_color = "medium"
                            else:
                                range_color = "bad"

                            st.metric(
                                "Range Coverage",
                                f"{range_coverage:.2f} ({range_color})",
                                help="How well the model captures extreme values"
                            )

                    if model_accuracy is not None:
                        if model_accuracy >= 85:
                            st.info("**Excellent model performance** - Outstanding for spectral data!")
                        elif model_accuracy >= 70:
                            st.info("**Good model performance** - Very reliable predictions")
                        elif model_accuracy >= 60:
                            st.warning("**Fair model performance** - Reasonable predictions")
                        else:
                            st.warning("**Limited model performance** - Use predictions with caution")

                st.markdown("### Your Prediction Statistics")

                try:
                    if isinstance(predictions, pd.DataFrame):
                        predictions = predictions.values.flatten()
                    elif not isinstance(predictions, (np.ndarray, pd.Series)):
                        predictions = np.array(predictions)

                    pred_df = pd.DataFrame({f"{selected_model} Prediction": predictions})

                    actual_aliases = {
                        "target",
                        "actual",
                        "measured",
                        "reference",
                        "label",
                        "ground_truth",
                        "y",
                        "y_true",
                    }
                    actual_candidates = []
                    for col in user_df.columns:
                        col_lower = col.lower()
                        if col_lower.endswith('_target'):
                            actual_candidates.append(col)
                        elif col_lower in actual_aliases:
                            actual_candidates.append(col)
                        elif selected_model.lower() in col_lower and any(alias in col_lower for alias in actual_aliases):
                            actual_candidates.append(col)

                    actual_col = None
                    if actual_candidates:
                        prioritized = [col for col in actual_candidates if selected_model.lower() in col.lower()]
                        actual_col = prioritized[0] if prioritized else actual_candidates[0]

                    comparison_df = pd.DataFrame()
                    if actual_col:
                        actual_series = pd.to_numeric(user_df[actual_col], errors='coerce')
                        if not actual_series.dropna().empty:
                            aligned_actual = actual_series.reset_index(drop=True)
                            pred_df[f"{selected_model} Actual"] = aligned_actual
                            comparison_df = pd.DataFrame(
                                {
                                    "Actual": aligned_actual,
                                    "Predicted": pd.Series(predictions, name="Predicted"),
                                }
                            ).dropna()
                        else:
                            st.info("Actual values were detected but contained only missing data; displaying predictions only.")
                    elif not actual_candidates:
                        st.info(
                            "No actual or target column detected in your data. Include a column such as 'target', 'actual', or 'T1_target' to enable the comparison chart."
                        )

                    if not comparison_df.empty:
                        diff = comparison_df["Predicted"] - comparison_df["Actual"]
                        if len(comparison_df) >= 2:
                            corr = np.corrcoef(comparison_df["Actual"], comparison_df["Predicted"])[0, 1]

                        comparison_plot = comparison_df.copy()
                        comparison_plot["Residual"] = diff
                        comparison_plot["AbsResidual"] = np.abs(diff)
                        comparison_plot["Sample"] = np.arange(1, len(comparison_plot) + 1)

                        x_vals = comparison_plot["Actual"].astype(float)
                        y_vals = comparison_plot["Predicted"].astype(float)
                        axis_min = float(min(x_vals.min(), y_vals.min()))
                        axis_max = float(max(x_vals.max(), y_vals.max()))
                        pad = max((axis_max - axis_min) * 0.05, 1e-6)
                        x_range = [float(x_vals.min() - pad), float(x_vals.max() + pad)]
                        y_range = [float(y_vals.min() - pad), float(y_vals.max() + pad)]
                        line_min = min(x_range[0], y_range[0])
                        line_max = max(x_range[1], y_range[1])
                        line_space = np.array([line_min, line_max])

                        try:
                            slope, intercept = np.polyfit(x_vals, y_vals, 1)
                        except Exception:
                            slope, intercept = 1.0, 0.0

                        st.caption(f"Calibration fit: predicted ≈ {slope:.3f}·actual + {intercept:.3f}")

                        calibration_mode = st.radio(
                            "Calibration for display",
                            options=[
                                "None",
                                "Calibrate predictions to actual scale",
                                "Map actuals to prediction scale",
                            ],
                            index=0,
                            horizontal=True,
                            help=(
                                "Choose how to align scales for display/export. "
                                "'Calibrate predictions' maps predictions onto your actuals' units. "
                                "'Map actuals' shows your actuals on the prediction scale."
                            ),
                        )

                        calibrated_predictions = None
                        mapped_actuals = None
                        if calibration_mode == "Calibrate predictions to actual scale":
                            # Calibrate predictions by mapping model output to actual scale
                            # invert the fit (y_pred ≈ a*x_actual + b) => calibrated ≈ (y_pred - b)/a
                            if abs(slope) > 1e-9:
                                calibrated_predictions = (comparison_df["Predicted"] - intercept) / slope
                            else:
                                calibrated_predictions = comparison_df["Predicted"].copy()

                            # Recompute metrics on calibrated predictions
                            diff_cal = calibrated_predictions - comparison_df["Actual"]
                            mae_val = float(np.mean(np.abs(diff_cal)))
                            rmse_val = float(np.sqrt(np.mean(diff_cal ** 2)))
                            bias_val = float(np.mean(diff_cal))
                            if len(comparison_df) >= 2:
                                corr = np.corrcoef(comparison_df["Actual"], calibrated_predictions)[0, 1]
                                r2_local = float(corr ** 2) if not np.isnan(corr) else float("nan")
                            else:
                                r2_local = float("nan")
                        elif calibration_mode == "Map actuals to prediction scale":
                            # Forward mapping: y_pred ≈ a*x_actual + b ==> map actuals to prediction space
                            mapped_actuals = slope * comparison_df["Actual"] + intercept
                            diff_map = comparison_df["Predicted"] - mapped_actuals
                            mae_val = float(np.mean(np.abs(diff_map)))
                            rmse_val = float(np.sqrt(np.mean(diff_map ** 2)))
                            bias_val = float(np.mean(diff_map))
                            if len(comparison_df) >= 2:
                                corr = np.corrcoef(mapped_actuals, comparison_df["Predicted"])[0, 1]
                                r2_local = float(corr ** 2) if not np.isnan(corr) else float("nan")
                            else:
                                r2_local = float("nan")

                        hover_fmt = {
                            "Actual": ":.3f",
                            "Predicted": ":.3f",
                            "Residual": ":.3f",
                            "AbsResidual": ":.3f",
                            "Sample": True,
                        }

                        fig_compare = px.scatter(
                            comparison_plot,
                            x="Actual",
                            y="Predicted",
                            color="AbsResidual",
                            color_continuous_scale="Viridis",
                            labels={
                                "AbsResidual": "|Residual|",
                                "Sample": "Sample",
                            },
                            hover_data=hover_fmt,
                        )
                        fig_compare.update_traces(
                            marker=dict(size=10, line=dict(width=0.6, color="#ffffff")), opacity=0.9
                        )
                        fig_compare.add_trace(
                            go.Scatter(
                                x=line_space,
                                y=line_space,
                                mode="lines",
                                name="1:1 Line",
                                line=dict(color="#2a7143", dash="dash", width=2),
                            )
                        )
                        fig_compare.add_trace(
                            go.Scatter(
                                x=line_space,
                                y=slope * line_space + intercept,
                                mode="lines",
                                name="Best fit",
                                line=dict(color="#264653", width=2),
                            )
                        )
                        if calibration_mode == "Calibrate predictions to actual scale":
                            # Show calibrated line (should be close to 1:1 if calibration effective)
                            fig_compare.add_trace(
                                go.Scatter(
                                    x=line_space,
                                    y=line_space,
                                    mode="lines",
                                    name="Calibrated (aligned)",
                                    line=dict(color="#8a2be2", dash="dot", width=2),
                                )
                            )
                        fig_compare.update_layout(
                            height=640,
                            margin=dict(t=70, r=40, b=50, l=60),
                            legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
                            coloraxis_colorbar=dict(title="|Residual|"),
                            title=dict(text=f"{selected_model}: Predictions vs Actual", x=0.02),
                        )
                        fig_compare.update_xaxes(title="Actual", range=x_range)
                        fig_compare.update_yaxes(title="Predicted", range=y_range)

                        st.plotly_chart(fig_compare, use_container_width=True, config={"displayModeBar": True})

                    st.markdown("### Your Prediction Statistics")

                    mean_pred = float(np.mean(predictions))
                    std_pred = float(np.std(predictions))
                    min_pred = float(np.min(predictions))
                    max_pred = float(np.max(predictions))

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Mean Prediction", f"{mean_pred:.3f}")
                        st.metric("Std Deviation", f"{std_pred:.3f}")

                    with col2:
                        st.metric("Min Prediction", f"{min_pred:.3f}")
                        st.metric("Max Prediction", f"{max_pred:.3f}")

                    # If calibration applied and actuals available, include extra columns in table
                    if not comparison_df.empty and calibrated_predictions is not None:
                        pred_df[f"{selected_model} Calibrated"] = calibrated_predictions.values
                    if not comparison_df.empty and mapped_actuals is not None:
                        pred_df[f"{selected_model} Actual (Pred scale)"] = mapped_actuals.values

                    st.write("**Detailed Predictions:**")
                    st.dataframe(pred_df, width="stretch")

                    if len(predictions) > 1:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.hist(predictions, bins=min(20, len(predictions)//2), alpha=0.7, color='skyblue', edgecolor='black')
                        ax.set_xlabel(f'{selected_model} Prediction Value')
                        ax.set_ylabel('Frequency')
                        ax.set_title(f'Distribution of {selected_model} Predictions')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)

                    st.markdown("### Download Results")

                    pred_df['Sample_ID'] = range(1, len(predictions) + 1)

                    summary_stats = {
                        'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Count'],
                        'Value': [f"{mean_pred:.3f}", f"{std_pred:.3f}", f"{min_pred:.3f}", f"{max_pred:.3f}", len(predictions)]
                    }
                    summary_df = pd.DataFrame(summary_stats)

                    if model_accuracy is not None:
                        performance_info = {
                            'Performance_Metric': ['Testing_Accuracy_%', 'R²_Score', 'RMSE', 'Range_Coverage'],
                            'Value': [
                                f"{model_accuracy:.1f}%" if model_accuracy is not None else "N/A",
                                f"{model_r2:.3f}" if model_r2 is not None else "N/A",
                                f"{model_rmse:.3f}" if model_rmse is not None else "N/A",
                                f"{range_coverage:.3f}" if range_coverage is not None else "N/A"
                            ]
                        }
                        performance_df = pd.DataFrame(performance_info)
                    else:
                        performance_df = None

                    export_columns = ['Sample_ID', f"{selected_model} Prediction"]
                    if f"{selected_model} Calibrated" in pred_df.columns:
                        export_columns.append(f"{selected_model} Calibrated")
                    if f"{selected_model} Actual (Pred scale)" in pred_df.columns:
                        export_columns.append(f"{selected_model} Actual (Pred scale)")
                    actual_export_col = f"{selected_model} Actual"
                    if actual_export_col in pred_df.columns:
                        export_columns.append(actual_export_col)
                    final_export = pred_df[export_columns]

                    col1, col2 = st.columns(2)

                    with col1:
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            final_export.to_excel(writer, sheet_name='Predictions', index=False)
                            summary_df.to_excel(writer, sheet_name='Summary', index=False)
                            if performance_df is not None:
                                performance_df.to_excel(writer, sheet_name='Model_Performance', index=False)

                        st.download_button(
                            "Download Excel (with performance)",
                            data=output.getvalue(),
                            file_name=f"{selected_model}_predictions_with_accuracy.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                    with col2:
                        csv_data = final_export.to_csv(index=False)
                        st.download_button(
                            "Download as CSV",
                            data=csv_data,
                            file_name=f"{selected_model}_predictions.csv",
                            mime="text/csv"
                        )

                except Exception as metrics_error:
                    st.error(f"Error displaying prediction metrics: {metrics_error}")
                    st.write("**Raw Predictions:**")
                    st.write(predictions)
            else:
                st.error("Trained model or feature file not found. Please train models first.")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            return

def show_model_info():
    """Show Model Information section"""
    st.header("Model Information")
    st.markdown("""
    View information about your trained models, including performance metrics and feature importance.
    """)
    
    # Model selection for info
    info_model = st.selectbox("Select Model for Information", [f"T{i}" for i in range(1, 6)], key="info_model")
    
    model_path = os.path.join("models", f"best_model_{info_model}.pkl")
    log_path = os.path.join("models", f"best_model_{info_model}_log.txt")
    features_path = os.path.join("models", f"best_model_{info_model}_features.txt")
    
    if os.path.exists(model_path) and os.path.exists(log_path):
        log_content = ""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{info_model} Model Performance")
            try:
                with open(log_path, 'r') as f:
                    log_content = f.read()
                latest_metrics = extract_best_pipeline_metrics(log_content)
                if latest_metrics:
                    st.metric("Best R² Score", f"{latest_metrics['r2']:.4f}")
                    st.metric("Best RMSE", f"{latest_metrics['rmse']:.4f}")
                    st.caption(f"Pipeline: {latest_metrics['pipeline']}")
                else:
                    st.info("No R² / RMSE entries were found in the log file.")
            except Exception as e:
                st.error(f"Error reading model log: {e}")
        
        with col2:
            st.subheader(f"{info_model} Model Details")
            if os.path.exists(features_path):
                with open(features_path, 'r') as f:
                    features = [line.strip() for line in f.readlines()]
                st.metric("Number of Features", len(features))
                st.metric("Wavelength Range", f"{features[0].split('_')[-1]} - {features[-1].split('_')[-1]} nm")
        
        st.subheader("Training Log")
        with st.expander(f"View {info_model} Training Details", expanded=False):
            if log_content:
                st.text(log_content)
            else:
                st.info("No log details available for this model.")

        st.subheader("Feature Importance (Permutation)")
        if "feature_importance_results" not in st.session_state:
            st.session_state["feature_importance_results"] = {}
        feature_results = st.session_state["feature_importance_results"]

        prefix = f"spectra_with_target_{info_model}"
        canonical_target = f"{prefix}_target"

        if canonical_target not in feature_results:
            regenerated = regenerate_feature_importance_from_saved_model(
                info_model,
                canonical_target,
                model_path,
                features_path,
                latest_metrics["pipeline"] if latest_metrics else None,
            )
            if regenerated is not None:
                feature_results[canonical_target] = regenerated.to_dict("records")

        matching_targets = [key for key in feature_results if info_model.lower() in key.lower()]

        if matching_targets:
            default_target = canonical_target if canonical_target in matching_targets else matching_targets[0]
            target_key = st.selectbox(
                "Select target column",
                options=matching_targets,
                index=matching_targets.index(default_target),
                key=f"model_info_target_{info_model}",
                help="Choose which trained target column to inspect if multiple match this model.",
            ) if len(matching_targets) > 1 else default_target

            top_features = st.slider(
                "Top features to display",
                min_value=5,
                max_value=30,
                value=15,
                step=5,
                key=f"model_info_topk_{info_model}",
            )

            payload = feature_results.get(target_key)
            perm_df = pd.DataFrame(payload) if payload else pd.DataFrame()

            if perm_df.empty and target_key == canonical_target:
                regenerated = regenerate_feature_importance_from_saved_model(
                    info_model,
                    canonical_target,
                    model_path,
                    features_path,
                    latest_metrics["pipeline"] if latest_metrics else None,
                )
                if regenerated is not None:
                    feature_results[canonical_target] = regenerated.to_dict("records")
                    perm_df = regenerated

            if not perm_df.empty:
                top_df = perm_df.head(top_features).copy().sort_values(by="Importance")
                hover_fields = {
                    field: ":.4f"
                    for field in ["Importance", "Std", "Mean", "Variance", "Correlation"]
                    if field in top_df.columns
                }
                fig = px.bar(
                    top_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    error_x="Std" if "Std" in top_df.columns else None,
                    color="Importance",
                    color_continuous_scale="YlGn",
                    hover_data=hover_fields or None,
                )
                fig.update_layout(
                    height=450,
                    margin=dict(t=40, r=30, b=30, l=0),
                    coloraxis_showscale=False,
                )
                fig.update_yaxes(title_text="Spectral feature")
                st.plotly_chart(fig, use_container_width=True)
                stats_fig = build_feature_stat_figure(
                    top_df,
                    f"{target_key}: Feature mean vs correlation",
                    ["Mean", "Correlation"],
                )
                if stats_fig is not None:
                    st.plotly_chart(stats_fig, use_container_width=True)
                variance_fig = build_feature_stat_figure(
                    top_df,
                    f"{target_key}: Feature variance",
                    ["Variance"],
                    chart="bar",
                )
                if variance_fig is not None:
                    st.plotly_chart(variance_fig, use_container_width=True)

                preview = perm_df.head(top_features).copy()
                for col in ["Importance", "Std", "Mean", "Variance", "Correlation"]:
                    if col in preview.columns:
                        preview[col] = preview[col].round(4)
                column_config = {
                    "Feature": st.column_config.TextColumn("Feature", width="large"),
                }
                if "Importance" in preview.columns:
                    column_config["Importance"] = st.column_config.NumberColumn("Importance", format="%.4f")
                if "Std" in preview.columns:
                    column_config["Std"] = st.column_config.NumberColumn("Std", format="%.4f")
                if "Mean" in preview.columns:
                    column_config["Mean"] = st.column_config.NumberColumn("Mean", format="%.4f")
                if "Variance" in preview.columns:
                    column_config["Variance"] = st.column_config.NumberColumn("Variance", format="%.4f")
                if "Correlation" in preview.columns:
                    column_config["Correlation"] = st.column_config.NumberColumn("Correlation", format="%.4f")
                st.dataframe(
                    preview,
                    use_container_width=True,
                    hide_index=True,
                    column_config=column_config,
                )
                st.caption("Permutation importance estimates computed from the training session's best pipeline.")
                render_full_feature_line_section(
                    perm_df,
                    f"#### {target_key}: all features (line view)",
                )
            else:
                st.info("Feature importance data is not available yet for this target.")
        else:
            st.info(
                "Permutation feature importance could not be regenerated from the saved assets. Please verify that the corresponding training file still exists in the `data/` folder."
            )
    else:
        st.info(f"Model {info_model} has not been trained yet. Please train models first.")
    
def show_leave_one_out_section():
    """Show Leave-One-Out evaluation section for stored models."""
    st.header("Leave-One-Out Validator")
    st.markdown(
        """
        Re-run leave-one-out cross-validation on already trained models without repeating the full training pipeline.
        The app loads the saved model, feature list, and default dataset to reproduce LOO metrics on demand.
        """
    )

    if "leave_one_out_results" not in st.session_state:
        st.session_state["leave_one_out_results"] = {}

    saved_models = list_saved_model_keys()
    if not saved_models:
        st.info("No stored models were found in the models/ folder. Train models first to enable this view.")
        return

    selected_model = st.selectbox(
        "Select stored model",
        options=saved_models,
        index=0,
        key="stored_loo_model_select",
        help="Pick which exported model you would like to validate.",
    )

    models_dir = BASE_DIR / "models"
    base_name = f"best_model_{selected_model}"
    model_path = models_dir / f"{base_name}.pkl"
    features_path = models_dir / f"{base_name}_features.txt"
    log_path = models_dir / f"{base_name}_log.txt"
    dataset_path = _find_training_file_for_model(selected_model)

    def _format_path(path_obj):
        if not path_obj:
            return "—"
        try:
            return str(Path(path_obj).relative_to(BASE_DIR))
        except Exception:
            return str(path_obj)

    st.markdown("**Artifacts detected**")
    artifacts = [
        ("Model file", model_path, model_path.exists()),
        ("Feature list", features_path, features_path.exists()),
        ("Training log", log_path, log_path.exists()),
        ("Default dataset", dataset_path, dataset_path and dataset_path.exists()),
    ]
    for label, path_obj, exists_flag in artifacts:
        status = "Available" if exists_flag else "Missing"
        st.write(f"- {label}: {status} ({_format_path(path_obj)})")

    run_button = st.button(
        "Run Leave-One-Out",
        type="primary",
        help="Loads the stored artifacts and recomputes leave-one-out cross-validation metrics.",
    )

    current_result = None
    stored_results = st.session_state.get("leave_one_out_results", {})
    canonical_target = f"spectra_with_target_{selected_model}_target"
    if canonical_target in stored_results:
        current_result = stored_results[canonical_target]
    else:
        for key, value in stored_results.items():
            if selected_model.lower() in key.lower():
                current_result = value
                break

    if run_button:
        with st.spinner("Running leave-one-out on stored model..."):
            loo_result, error_msg = run_leave_one_out_for_saved_model(selected_model)
        if error_msg:
            st.error(error_msg)
        else:
            st.success(
                f"Leave-one-out completed for {loo_result['target_column']} ({loo_result['n_samples']} samples)."
            )
            st.session_state["leave_one_out_results"][loo_result["target_column"]] = loo_result
            current_result = loo_result

    if current_result:
        render_leave_one_out_block(current_result)
    else:
        st.info("Run the evaluation to see leave-one-out metrics for the selected model.")

    if stored_results:
        st.markdown("---")
        st.subheader("Cached Leave-One-Out Runs")
        summary_rows = []
        for target_key, metrics in stored_results.items():
            summary_rows.append(
                {
                    "Target": target_key,
                    "Model": metrics.get("model_key", "—"),
                    "R²": round(float(metrics.get("r2", np.nan)), 4),
                    "RMSE": round(float(metrics.get("rmse", np.nan)), 4),
                    "Samples": metrics.get("n_samples", 0),
                }
            )
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df, use_container_width=True)

def show_spectral_explorer():
    """Show Spectral Explorer section"""
    st.header("Spectral Explorer")
    st.markdown(
        """
        Review the bundled spectral datasets interactively. Inspect individual spectra, target distributions,
        and the wavelengths that correlate most strongly with the soil property of interest.
        """
    )

    dataset_paths = discover_spectral_datasets(str(DEFAULT_DATA_DIR))
    if not dataset_paths:
        st.info(
            "No spectral training files detected in the `data/` folder."
            " Upload datasets named like `spectra_with_target_T1.xls` to enable the explorer."
        )
    else:
        dataset_keys = sorted(dataset_paths.keys())

        def _format_dataset_label(key: str) -> str:
            return f"{key} ({Path(dataset_paths[key]).name})"

        selected_dataset = st.selectbox(
            "Dataset",
            options=dataset_keys,
            format_func=_format_dataset_label,
            key="spectral_explorer_dataset",
        )

        try:
            dataset = load_spectral_dataset(selected_dataset, dataset_paths[selected_dataset])
        except Exception as exc:
            st.error(f"Failed to load dataset {selected_dataset}: {exc}")
            dataset = None

        if dataset:
            df = dataset["df"]
            features = dataset["features"]
            wavelengths = dataset["wavelengths"]
            correlations = dataset["correlations"]
            quartiles = dataset["quartiles"]
            summary = dataset["summary"]

            summary_primary = ["Samples", "Spectral bands", "Target mean"]
            summary_secondary = ["Target std", "Target min", "Target max"]

            primary_cols = st.columns(len(summary_primary))
            for column, metric_name in zip(primary_cols, summary_primary):
                column.metric(metric_name, f"{summary[metric_name]:.3f}" if isinstance(summary[metric_name], float) else summary[metric_name])

            secondary_cols = st.columns(len(summary_secondary))
            for column, metric_name in zip(secondary_cols, summary_secondary):
                column.metric(metric_name, f"{summary[metric_name]:.3f}")

            st.divider()

            max_sample_index = max(len(df) - 1, 0)
            max_features = features.shape[1]

            # Top control - sample slider stays near spectral profile
            sample_index = st.slider(
                "Sample index",
                min_value=0,
                max_value=max_sample_index,
                value=min(0, max_sample_index),
                step=1,
            )

            def _format_wavelength_label(column_name: str) -> str:
                try:
                    value = extract_wavelength(column_name)
                    return f"{value:.0f} nm"
                except ValueError:
                    return column_name

            wavelength_options = list(features.columns)
            focus_default_index = min(len(wavelength_options) // 2, len(wavelength_options) - 1)

            top_n_default = min(10, max_features)

            st.markdown("#### Spectral profile")

            spectrum = features.iloc[sample_index]
            q1_spec = quartiles.loc[0.25]
            median_spec = quartiles.loc[0.5]
            q3_spec = quartiles.loc[0.75]

            spectrum_fig = go.Figure()
            spectrum_fig.add_trace(
                go.Scatter(
                    x=wavelengths,
                    y=q3_spec.values,
                    name="75th percentile",
                    line=dict(width=0),
                    hoverinfo="skip",
                )
            )
            spectrum_fig.add_trace(
                go.Scatter(
                    x=wavelengths,
                    y=q1_spec.values,
                    name="IQR envelope",
                    fill="tonexty",
                    fillcolor="rgba(36, 91, 52, 0.15)",
                    line=dict(width=0),
                    hoverinfo="skip",
                )
            )
            spectrum_fig.add_trace(
                go.Scatter(
                    x=wavelengths,
                    y=median_spec.values,
                    name="Median spectrum",
                    line=dict(color="#245b34", width=3),
                )
            )
            spectrum_fig.add_trace(
                go.Scatter(
                    x=wavelengths,
                    y=spectrum.values,
                    name=f"Sample {sample_index}",
                    line=dict(color="#ff7f0e", width=2),
                )
            )
            spectrum_fig.update_layout(
                height=420,
                margin=dict(t=50, r=30, b=50, l=60),
                legend=dict(orientation="h", x=0, y=1.05),
                xaxis_title="Wavelength (nm)",
                yaxis_title="Reflectance",
            )
            st.plotly_chart(spectrum_fig, use_container_width=True)

            st.markdown("#### Target distribution")
            hist_fig = px.histogram(
                df,
                x="target",
                nbins=30,
                opacity=0.85,
                color_discrete_sequence=["#2a7143"],
                title=None,
            )
            hist_fig.update_layout(height=360, margin=dict(t=20, r=30, b=50, l=60), xaxis_title="Target value", yaxis_title="Count")
            st.plotly_chart(hist_fig, use_container_width=True)

            # Top N wavelengths slider - placed above the correlation chart
            top_n = st.slider(
                "Top |corr| wavelengths",
                min_value=3,
                max_value=max(3, min(60, max_features)),
                value=max(3, top_n_default),
            )

            st.markdown(f"#### Top {top_n} wavelengths by |correlation|")
            top_corr = correlations.head(top_n)
            corr_fig = go.Figure(
                go.Bar(
                    x=top_corr.values[::-1],
                    y=[_format_wavelength_label(val) for val in top_corr.index[::-1]],
                    orientation="h",
                    marker=dict(color=np.abs(top_corr.values[::-1]), colorscale="YlGnBu", colorbar=dict(title="|corr|")),
                )
            )
            corr_fig.update_layout(height=420, margin=dict(t=30, r=30, b=50, l=120), xaxis_title="Correlation", yaxis_title="Wavelength")
            st.plotly_chart(corr_fig, use_container_width=True)

            st.markdown("#### Target vs reflectance")
            selected_wavelength = st.selectbox(
                "Focus wavelength",
                options=wavelength_options,
                index=focus_default_index,
                format_func=_format_wavelength_label,
                key="spectral_focus_wavelength",
                help="Choose which wavelength to highlight in the scatter plot",
            )
            x_vals = df[selected_wavelength].astype(float).values
            y_vals = df["target"].astype(float).values

            scatter_fig = go.Figure()
            scatter_fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="markers",
                    name="Samples",
                    marker=dict(size=8, color=y_vals, colorscale="Earth", showscale=False, opacity=0.8),
                )
            )

            if np.unique(x_vals).size > 1:
                try:
                    slope, intercept = np.polyfit(x_vals, y_vals, 1)
                    x_range = np.linspace(x_vals.min(), x_vals.max(), 120)
                    scatter_fig.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=slope * x_range + intercept,
                            mode="lines",
                            name="Linear fit",
                            line=dict(color="#d62728", width=2),
                        )
                    )
                except Exception:
                    pass

            scatter_fig.update_layout(
                height=420,
                margin=dict(t=40, r=30, b=50, l=60),
                xaxis_title=f"Reflectance at {_format_wavelength_label(selected_wavelength)}",
                yaxis_title="Target value",
            )
            st.plotly_chart(scatter_fig, use_container_width=True)

            st.markdown("#### Correlation table")
            # Create a proper dataframe with wavelength column
            corr_df = correlations.head(top_n).to_frame(name="Correlation")
            corr_df.index.name = "Wavelength (nm)"
            corr_df = corr_df.reset_index()
            corr_df["Wavelength (nm)"] = corr_df["Wavelength (nm)"].astype(str) + " nm"
            corr_df["Correlation"] = corr_df["Correlation"].round(3)
            
            st.dataframe(
                corr_df,
                use_container_width=False,
                hide_index=True,
                column_config={
                    "Wavelength (nm)": st.column_config.TextColumn("Wavelength", width="medium"),
                    "Correlation": st.column_config.NumberColumn("Correlation", format="%.3f", width="small"),
                }
            )

def old_main():
    st.set_page_config(
        page_title="The Spectral Soil Modeler",
        layout="wide",
    )

    if "training_summaries" not in st.session_state:
        st.session_state["training_summaries"] = []
    if "last_training_files" not in st.session_state:
        st.session_state["last_training_files"] = []
    if "active_section" not in st.session_state:
        st.session_state.active_section = "dashboard"

    # Global styling for a cleaner, branded experience
    st.markdown(
        """
        <style>
            [data-testid="stAppViewContainer"] > .main {
                background: linear-gradient(180deg, #f5fbf7 0%, #ffffff 30%);
                padding-top: 1rem;
            }
            .spectral-hero {
                text-align: center;
                padding: 2.5rem 0 1.5rem 0;
            }
            .spectral-hero-title {
                font-size: 2.8rem;
                font-weight: 700;
                color: #245b34;
                margin-bottom: 0.5rem;
            }
            .spectral-hero-tagline {
                font-size: 1.1rem;
                color: #3d4f3f;
                margin-bottom: 0.75rem;
            }
            .spectral-hero-meta {
                font-size: 0.95rem;
                color: #63736b;
                letter-spacing: 0.08rem;
                text-transform: uppercase;
            }
            .navigation-card {
                background-color: #ffffff;
                border: 1px solid #d6e7db;
                border-radius: 12px;
                padding: 1.5rem;
                box-shadow: 0 8px 24px rgba(30, 80, 50, 0.08);
                height: 280px;  /* Fixed height for all cards */
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                transition: all 0.3s ease;
                cursor: pointer;
                text-align: center;
            }
            .navigation-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 32px rgba(30, 80, 50, 0.15);
                border-color: #2a7143;
            }
            .navigation-card.active {
                border: 2px solid #2a7143;
                background-color: #f0f7f2;
            }
            .navigation-card h3 {
                margin: 0 0 0.8rem 0;
                font-size: 1.3rem;
                color: #2a7143;
            }
            .navigation-card p {
                color: #45554a;
                font-size: 0.95rem;
                line-height: 1.4rem;
                margin-bottom: 0;
                flex-grow: 1;  /* Makes description fill available space */
            }
            .navigation-icon {
                font-size: 2.5rem;
                margin-bottom: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Main dashboard header
    st.markdown(
        """
        <div class="spectral-hero">
            <div class="spectral-hero-title">The Spectral Soil Modeler</div>
            <div class="spectral-hero-tagline">An automated ML workflow turning soil spectra into agronomic intelligence.</div>
            <div class="spectral-hero-meta">Train · Evaluate · Predict · Export</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Show dashboard navigation or specific section
    if st.session_state.active_section == "dashboard":
        # Navigation cards
        navigation_items = [
            {
                "title": "Train Models",
                "description": "Upload spectral data and train machine learning models with automated preprocessing",
                "section": "train_models",
                "icon": ""
            },
            {
                "title": "Make Predictions",
                "description": "Use trained models to predict soil properties from new spectral data",
                "section": "make_predictions",
                "icon": ""
            },
            {
                "title": "Model Info",
                "description": "View model performance metrics, feature importance, and training logs",
                "section": "model_info",
                "icon": ""
            },
            {
                "title": "Leave-One-Out",
                "description": "Validate stored models with post-training leave-one-out analysis",
                "section": "leave_one_out",
                "icon": ""
            },
            {
                "title": "Spectral Explorer",
                "description": "Explore spectral datasets, correlations, and visualize wavelength relationships",
                "section": "spectral_explorer",
                "icon": ""
            },
        ]
        navigation_cols = st.columns(len(navigation_items), gap="large")
        
        for col, item in zip(navigation_cols, navigation_items):
            with col:
                is_active = st.session_state.active_section == item["section"]
                active_class = "active" if is_active else ""
                
                col.markdown(
                    f"""
                    <div class="navigation-card {active_class}" onclick="window.location.href='?section={item['section']}'">
                        <div class="navigation-icon">{item['icon']}</div>
                        <h3>{item['title']}</h3>
                        <p>{item['description']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Add click handler
                if st.button(f"Access {item['title']}", key=f"btn_{item['section']}", use_container_width=True):
                    st.session_state.active_section = item["section"]
                    st.rerun()
        
        # Quick stats or recent activity
        
    elif st.session_state.active_section == "train_models":
        # Back button
        if st.button("Back to Dashboard"):
            st.session_state.active_section = "dashboard"
            st.rerun()
        show_train_models()
        
    elif st.session_state.active_section == "make_predictions":
        # Back button
        if st.button("Back to Dashboard"):
            st.session_state.active_section = "dashboard"
            st.rerun()
        show_make_predictions()
        
    elif st.session_state.active_section == "model_info":
        # Back button
        if st.button("Back to Dashboard"):
            st.session_state.active_section = "dashboard"
            st.rerun()
        show_model_info()
        
    elif st.session_state.active_section == "leave_one_out":
        if st.button("Back to Dashboard"):
            st.session_state.active_section = "dashboard"
            st.rerun()
        show_leave_one_out_section()
        
    elif st.session_state.active_section == "spectral_explorer":
        # Back button
        if st.button("Back to Dashboard"):
            st.session_state.active_section = "dashboard"
            st.rerun()
        show_spectral_explorer()

JWT_SECRET = "abcdefghijklmnopqrstuvwxyz1234567890"
JWT_ALGORITHM = "HS256"
TOKEN_EXPIRY_DAYS = 7

cookies = EncryptedCookieManager(
    prefix="soil_modeller_",
    password="soil-modeller-project"
)

if not cookies.ready():
    st.stop()

def generate_jwt_token(username):
    payload = {
        'username': username,
        'exp': datetime.utcnow() + timedelta(days=TOKEN_EXPIRY_DAYS),
        'iat': datetime.utcnow()
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token

def verify_jwt_token(token):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload.get('username')
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def check_authentication():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None
    
    if 'just_logged_out' in st.session_state and st.session_state.just_logged_out:
        st.session_state.just_logged_out = False
        show_login_page()
        return False
    
    if not st.session_state.logged_in:
        auth_token = cookies.get('auth_token')
        if auth_token and auth_token != "":
            username = verify_jwt_token(auth_token)
            if username:
                st.session_state.logged_in = True
                st.session_state.username = username
                return True
        
        show_login_page()
        return False
    
    return True

def show_login_page():
    st.set_page_config(
        page_title="Spectral Soil Modeler | Login",
        page_icon="🌱",
        layout="centered"
    )
    
    # Clean modern CSS - Green gradient with proper layout
    st.markdown(
        """
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        /* Apply font globally */
        *, html, body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }
        
        /* Main app background - Fresh green gradient */
        .stApp {
            background: linear-gradient(145deg, #134e3a 0%, #166534 35%, #22863a 65%, #2ea44f 100%) !important;
            background-attachment: fixed !important;
        }
        
        /* Soft light overlay */
        .stApp::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                radial-gradient(ellipse at 20% 0%, rgba(255, 255, 255, 0.1) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 100%, rgba(255, 255, 255, 0.06) 0%, transparent 50%);
            pointer-events: none;
            z-index: 0;
        }
        
        /* Hide default Streamlit elements */
        #MainMenu, footer, header {visibility: hidden; display: none;}
        
        /* Hide ALL scrollbars */
        ::-webkit-scrollbar { display: none !important; width: 0 !important; }
        * { scrollbar-width: none !important; -ms-overflow-style: none !important; }
        html, body, .stApp { overflow: hidden !important; }
        
        /* Main content - centered with proper height */
        [data-testid="stAppViewContainer"] > .main {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            min-height: 100vh !important;
            padding: 1rem !important;
            overflow: hidden !important;
        }
        
        .block-container {
            max-width: 440px !important;
            width: 100% !important;
            padding: 0 1rem !important;
        }
        
        /* Brand header - compact */
        .brand-header {
            text-align: center;
            margin-bottom: 1.5rem;
        }
        
        .brand-logo {
            width: 72px;
            height: 72px;
            background: linear-gradient(135deg, #ffffff 0%, #dcfce7 100%);
            border-radius: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 1rem auto;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
            font-size: 2.2rem;
        }
        
        .brand-title {
            font-size: 2.4rem;
            font-weight: 800;
            color: #ffffff;
            margin: 0 0 0.4rem 0;
            letter-spacing: -0.5px;
            text-shadow: 0 2px 15px rgba(0, 0, 0, 0.2);
        }
        
        .brand-subtitle {
            font-size: 1.1rem;
            color: rgba(255, 255, 255, 0.85);
            margin: 0;
            font-weight: 500;
        }
        
        /* Form card - clean white */
        [data-testid="stForm"] {
            background: #ffffff !important;
            border-radius: 20px !important;
            padding: 1.75rem !important;
            box-shadow: 0 15px 50px rgba(0, 0, 0, 0.15) !important;
            border: none !important;
        }
        
        /* Tabs styling */
        [data-testid="stTabs"] > div:first-child {
            background: #f0fdf4;
            border-radius: 12px;
            padding: 4px;
            gap: 4px;
        }
        
        [data-testid="stTabs"] button {
            background: transparent !important;
            color: #166534 !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.7rem 1.25rem !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
        }
        
        [data-testid="stTabs"] button[aria-selected="true"] {
            background: linear-gradient(135deg, #22863a 0%, #166534 100%) !important;
            color: white !important;
            box-shadow: 0 4px 12px rgba(22, 101, 52, 0.3) !important;
        }
        
        [data-testid="stTabs"] > div:last-child {
            padding-top: 1.25rem;
        }
        
        /* Input fields - clean style */
        .stTextInput > div > div {
            background: #f8fdf9 !important;
            border: 2px solid #d1fae5 !important;
            border-radius: 12px !important;
        }
        
        .stTextInput > div > div:focus-within {
            border-color: #22863a !important;
            box-shadow: 0 0 0 3px rgba(34, 134, 58, 0.15) !important;
            background: #ffffff !important;
        }
        
        .stTextInput input {
            color: #134e3a !important;
            font-size: 1.05rem !important;
            padding: 0.85rem 1rem !important;
            font-weight: 500 !important;
        }
        
        .stTextInput input::placeholder {
            color: #6ee7b7 !important;
        }
        
        .stTextInput label {
            color: #134e3a !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
        }
        
        /* Password toggle - transparent */
        .stTextInput [data-testid="stTextInputRootElement"] button,
        .stTextInput button {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            color: #6ee7b7 !important;
        }
        
        .stTextInput button:hover { color: #22863a !important; }
        .stTextInput button svg { fill: #6ee7b7 !important; }
        .stTextInput button:hover svg { fill: #22863a !important; }
        
        /* Checkbox */
        .stCheckbox label {
            color: #134e3a !important;
            font-size: 0.95rem !important;
            font-weight: 500 !important;
        }
        
        .stCheckbox label span {
            color: #134e3a !important;
        }
        
        .stCheckbox p {
            color: #134e3a !important;
        }
        
        /* Submit button */
        .stFormSubmitButton button {
            background: linear-gradient(135deg, #22863a 0%, #166534 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.85rem 1.5rem !important;
            font-weight: 700 !important;
            font-size: 1.1rem !important;
            box-shadow: 0 4px 15px rgba(22, 101, 52, 0.35) !important;
            transition: all 0.2s ease !important;
        }
        
        .stFormSubmitButton button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(22, 101, 52, 0.45) !important;
        }
        
        /* Alerts */
        .stAlert { border-radius: 10px !important; }
        [data-testid="stAlert"] > div { font-size: 0.95rem !important; }
        
        /* Feature section - inline chips, proper alignment */
        .feature-row {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 1rem;
            margin-top: 1.5rem;
            flex-wrap: nowrap;
        }
        
        .feature-chip {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.4rem;
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            padding: 0.5rem 1rem;
            border-radius: 50px;
            border: 1px solid rgba(255, 255, 255, 0.25);
            white-space: nowrap;
        }
        
        .feature-chip span {
            font-size: 1rem;
            line-height: 1;
        }
        
        .feature-chip p {
            margin: 0;
            color: #ffffff;
            font-size: 0.85rem;
            font-weight: 600;
            line-height: 1;
            white-space: nowrap;
        }
        
        /* Footer */
        .login-footer {
            text-align: center;
            margin-top: 1rem;
            color: rgba(255, 255, 255, 0.7);
            font-size: 0.85rem;
        }
        
        /* Description text */
        .desc-text {
            color: #166534 !important;
            font-size: 0.95rem !important;
            margin-bottom: 1rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Brand header
    st.markdown(
        """
        <div class="brand-header">
            <div class="brand-logo">🌱</div>
            <h1 class="brand-title">Spectral Soil Modeler</h1>
            <p class="brand-subtitle">ML-Powered Soil Analysis Platform</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    user_manager = UserManager()
    
    # Login/Register tabs
    tab1, tab2 = st.tabs(["🔐 Sign In", "✨ Create Account"])
    
    with tab1:
        st.markdown("<p class='desc-text'>Welcome back! Enter your credentials.</p>", unsafe_allow_html=True)
        
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input(
                "Username",
                placeholder="Enter your username",
                key="login_username"
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
                key="login_password"
            )
            
            remember_me = st.checkbox("Remember me", value=True)
            
            login_btn = st.form_submit_button("Sign In", use_container_width=True)
            
            if login_btn:
                if username and password:
                    with st.spinner("Authenticating..."):
                        if user_manager.login(username, password):
                            token = generate_jwt_token(username)
                            if remember_me:
                                cookies['auth_token'] = token
                                cookies.save()
                            
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.session_state.active_page = "home"
                            st.session_state.active_section = "dashboard"
                            
                            st.success("✅ Welcome back!")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error("❌ Invalid credentials")
                else:
                    st.warning("⚠️ Please fill all fields")
    
    with tab2:
        st.markdown("<p class='desc-text'>Create an account to get started.</p>", unsafe_allow_html=True)
        
        with st.form("register_form", clear_on_submit=False):
            new_user = st.text_input(
                "Username",
                placeholder="Choose a unique username",
                key="reg_username"
            )
            new_pass = st.text_input(
                "Password",
                type="password",
                placeholder="Create a strong password",
                key="reg_password"
            )
            confirm_pass = st.text_input(
                "Confirm Password",
                type="password",
                placeholder="Re-enter your password",
                key="reg_confirm"
            )
            
            reg_btn = st.form_submit_button("Create Account", use_container_width=True)
            
            if reg_btn:
                if new_user and new_pass and confirm_pass:
                    if len(new_pass) < 6:
                        st.error("❌ Password must be 6+ characters")
                    elif new_pass != confirm_pass:
                        st.error("❌ Passwords do not match")
                    else:
                        with st.spinner("Creating account..."):
                            if user_manager.registerUser(new_user, new_pass):
                                st.success("🎉 Account created! Please sign in.")
                            else:
                                st.error("❌ Username already exists")
                else:
                    st.warning("⚠️ Please fill all fields")
    
    # Feature highlights - inline chips
    st.markdown(
        """
        <div class="feature-row">
            <div class="feature-chip">
                <span>🔬</span>
                <p>Spectral</p>
            </div>
            <div class="feature-chip">
                <span>🤖</span>
                <p>ML Models</p>
            </div>
            <div class="feature-chip">
                <span>📊</span>
                <p>Predict</p>
            </div>
        </div>
        <div class="login-footer">© 2025 Spectral Soil Modeler</div>
        """,
        unsafe_allow_html=True
    )
    
    st.stop()

def display_logout_section():
    with st.sidebar:
        st.title("Soil Modeler")
        st.subheader("Account")

        user_card_html = f"""
            <div style="
                padding: 15px;
                background: #f9f9f9;
                border-radius: 10px;
                border: 1px solid #eee;
                margin: 15px 0;
            ">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="
                        width: 38px;
                        height: 38px;
                        background: #2a7143;
                        border-radius: 50%;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        color: white;
                        font-size: 1.1rem;">
                    </div>
                    <div>
                        <div style="color: #777; font-size: 0.85rem;">Current User</div>
                        <div style="color: #2a7143; font-weight: 600;">{st.session_state.username}</div>
                    </div>
                </div>
            </div>
        """
        # Use components.html to avoid Markdown sanitization of inline styles
        st_html(user_card_html, height=110)
        
        if st.button("**Logout**", 
                    use_container_width=True,
                    type="primary"):
            cookies['auth_token'] = ""
            cookies.save()
            
            st.session_state.just_logged_out = True
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.active_page = "home"
            st.session_state.active_section = "dashboard"
            
            st.rerun()

        st.markdown("---")
        
        # Home button - goes to main dashboard
        if st.button("**Dashboard**", 
                    use_container_width=True,
                    help="Main dashboard with all features"):
            st.session_state.active_page = "home"
            st.session_state.active_section = "dashboard"
            st.rerun()
        
        # Model Library button
        if st.button("**Model Library**", 
                    use_container_width=True,
                    help="View and download all trained models"):
            st.session_state.active_page = "model_library"
            st.rerun()
        
        st.markdown("---")
        
        st.caption(f"Session: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

def main():
    if not check_authentication():
        return
    
    # Initialize active page if not set (default to home)
    if 'active_page' not in st.session_state:
        st.session_state.active_page = "home"
    if 'active_section' not in st.session_state:
        st.session_state.active_section = "dashboard"
    
    # Always show the common sidebar
    display_logout_section()
    
    # Route to the correct page based on active_page
    if st.session_state.active_page == "model_library":
        # Show Model Library page
        try:
            from src.model_library import show_model_library
            show_model_library()
        except ImportError as e:
            st.error(f"Cannot load Model Library: {e}")
            # Fall back to home page
            st.session_state.active_page = "home"
            st.session_state.active_section = "dashboard"
            st.rerun()
    
    elif st.session_state.active_page == "home":
        # Show the main app (home page with dashboard)
        old_main()
    
    else:
        # Default to home if invalid page
        st.session_state.active_page = "home"
        st.session_state.active_section = "dashboard"
        st.rerun()

if __name__ == "__main__":
    main()