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
import matplotlib.pyplot as plt
import seaborn as sns
import chardet
import numpy as np
import concurrent.futures
import plotly.express as px
import plotly.graph_objects as go
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from src.database import UserManager
import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager
import jwt
from datetime import datetime, timedelta

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
    processed = transform(df)
    processed = processed.replace([np.inf, -np.inf], np.nan)
    processed = processed.fillna(processed.median())
    non_const_cols = processed.columns[processed.std(ddof=0) > 0]
    processed = processed.loc[:, non_const_cols]
    return processed, (prep_label or "Reflectance")

def old_main():
    st.set_page_config(
        page_title="The Spectral Soil Modeler",
        page_icon=":bar_chart:",
        layout="wide",
    )

    if "training_summaries" not in st.session_state:
        st.session_state["training_summaries"] = []
    if "last_training_files" not in st.session_state:
        st.session_state["last_training_files"] = []

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
            .spectral-card {
                background-color: #ffffff;
                border: 1px solid #d6e7db;
                border-radius: 12px;
                padding: 1.2rem 1.4rem;
                box-shadow: 0 8px 24px rgba(30, 80, 50, 0.08);
                height: 100%;
            }
            .spectral-card h3 {
                margin: 0 0 0.4rem 0;
                font-size: 1.15rem;
                color: #2a7143;
            }
            .spectral-card p {
                color: #45554a;
                font-size: 0.95rem;
                line-height: 1.4rem;
            }
            .spectral-tab-container > div {
                border-radius: 0 0 12px 12px;
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

    # Feature cards
    step_cols = st.columns(3, gap="large")
    step_details = [
        (
            "Automated Training",
            "Upload one or more spectral datasets and let curated preprocessing and model grids run in parallel.",
        ),
        (
            "Performance Insights",
            "Track model accuracy, RMSE, and range coverage with structured logs and visual feedback.",
        ),
        (
            "Deployment Ready",
            "Export best models, reproduce preprocessing, and deliver predictions with aligned wavelengths.",
        ),
    ]
    for col, (title, description) in zip(step_cols, step_details):
        col.markdown(
            f"""
            <div class="spectral-card">
                <h3>{title}</h3>
                <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # Create tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs(["Train Models", "Make Predictions", "Model Info", "Spectral Explorer"])
    
    with tab1:
        st.header("Train ML Models")
        st.markdown("""
        Upload your spectral training data to build and train machine learning models.
        The system will automatically test multiple preprocessing methods and algorithms to find the best model for each target.
        """)
        
        st.subheader("Upload Training Data")
        st.info("Upload multiple Excel files containing spectral data with target columns ending in '_target'")
        uploaded_files = st.file_uploader(
            "Choose Excel files for training", 
            type=["xls", "xlsx"], 
            accept_multiple_files=True, 
            key="train_files"
        )
        
        dfs = []
        if uploaded_files:
            uploaded_files_list = list(uploaded_files)
            st.success(f"Loaded {len(uploaded_files_list)} files successfully!")

            current_files = sorted(file.name for file in uploaded_files_list)
            if current_files != st.session_state.get("last_training_files", []):
                st.session_state["training_summaries"] = []
                st.session_state["last_training_files"] = current_files

            for file in uploaded_files_list:
                try:
                    df = smart_read(file)
                    prefix = file.name.split('.')[0]
                    df = df.add_prefix(prefix + '_')
                    dfs.append(df)
                    st.write(f"**{file.name}**: {df.shape[0]} samples, {df.shape[1]} features")
                except Exception as e:
                    st.error(f"Error loading {file.name}: {str(e)}")
                    continue
            
            if dfs:
                merged_df = pd.concat(dfs, axis=1)
                st.markdown("**Merged Data Overview:**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Samples", merged_df.shape[0])
                with col2:
                    st.metric("Total Features", merged_df.shape[1])
                with col3:
                    target_cols = [col for col in merged_df.columns if col.lower().endswith('_target')]
                    st.metric("Target Columns", len(target_cols))
                
                with st.expander("View Data Preview"):
                    st.dataframe(merged_df.head(), width="stretch")
                
                target_display = ", ".join(target_cols) if target_cols else "None detected"
                st.markdown(f"**Detected target columns:** {target_display}")

                st.subheader("Train ML Pipelines")
                st.markdown("""
                Click the button below to automatically train and optimize ML models for all targets.
                The system will test multiple combinations of:
                - **Preprocessing**: Reflectance, Absorbance, Continuum Removal
                - **Algorithms**: PLS Regression, SVR, Gradient Boosting, Random Forest, Kernel Ridge
                """)
                
                train_button = st.button("Train All Models", type="primary")

                if train_button:
                    if len(target_cols) == 0:
                        st.error("No columns ending with '_target' found. Please verify your training files.")
                        return

                    st.session_state["training_summaries"] = []

                    import concurrent.futures

                    def train_target(target_col):
                        try:
                            # Use only columns relevant to this target (e.g., T1)
                            import re
                            match = re.search(r'T(\d+)_target$', target_col)
                            if match:
                                target_prefix = f"spectra_with_target_T{match.group(1)}_"
                            else:
                                target_prefix = None
                            # Select only columns with the correct prefix, except the target column
                            if target_prefix:
                                feature_cols = [col for col in merged_df.columns if col.startswith(target_prefix) and not col.endswith('_target')]
                            else:
                                feature_cols = [col for col in merged_df.columns if col != target_col]
                            X = merged_df[feature_cols]
                            y = merged_df[target_col]
                            # Drop rows with NaN in target
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
                            # Extract target name (e.g., T1, T2, ...) from target_col
                            match = re.search(r'T(\d+)_target$', target_col)
                            if match:
                                model_id = f'T{match.group(1)}'
                            else:
                                model_id = target_col  # fallback, should not happen
                            model_path = os.path.join("models", f"best_model_{model_id}.pkl")
                            feature_names_path = os.path.join("models", f"best_model_{model_id}_features.txt")
                            export_best_model(best_model, model_path)
                            with open(feature_names_path, "w") as f:
                                f.write("\n".join(X.columns))
                            # Save model improvement log to file
                            log_path = os.path.join("models", f"best_model_{model_id}_log.txt")
                            with open(log_path, "w") as f:
                                f.write("\n".join(improvement_log))
                            result_tuple = (target_col, summary_df, best_pipeline, best_score, improvement_log, feature_importances)
                            if isinstance(result_tuple, tuple):
                                return result_tuple
                            else:
                                raise RuntimeError(f"train_target failed for {target_col}: Did not return tuple result.")
                        except Exception as e:
                            raise RuntimeError(f"train_target failed for {target_col}: {str(e)}")

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
                            status_text.text(f"Completed {completed} of {total_targets} targets · Last finished: {target_col}")
                            try:
                                result = future.result()
                                if not isinstance(result, tuple):
                                    raise RuntimeError(f"train_target for {target_col} did not return a tuple. Got type: {type(result)}. Value: {repr(result)}")
                                target_col, summary_df, best_pipeline, best_score, improvement_log, feature_importances = result
                                
                                with st.expander(f"Results for {target_col}", expanded=True):
                                    st.dataframe(summary_df, width="stretch")
                                    st.success(f"**Best Model:** {best_pipeline} | **Score:** {best_score:.4f}")
                                    
                                    with st.expander("Step-by-step Model Improvement Log"):
                                        for entry in improvement_log:
                                            st.write(f"• {entry}")
                                    
                                st.success(f"Best model for {target_col} saved successfully!")

                                try:
                                    best_row = summary_df.loc[summary_df['Pipeline'] == best_pipeline].iloc[0]
                                    st.session_state["training_summaries"].append(
                                        {
                                            "Target": target_col,
                                            "Pipeline": best_pipeline,
                                            "R2": round(float(best_row['R2']), 4),
                                            "RMSE": round(float(best_row['RMSE']), 4),
                                            "RPD": round(float(best_row['RPD']), 3) if not pd.isna(best_row['RPD']) else None,
                                        }
                                    )
                                except Exception:
                                    # Skip summary aggregation if data missing
                                    pass
                                
                            except Exception as e:
                                st.error(f"Error training {target_col}: {str(e)}")

                    progress_bar.progress(1.0)
                    status_text.success("All targets training completed!")
        
                if st.session_state.get("training_summaries"):
                    best_models_df = pd.DataFrame(st.session_state["training_summaries"])
                    best_models_df = best_models_df.sort_values(by="Target").reset_index(drop=True)
                    st.subheader("Best Model Overview")
                    st.dataframe(best_models_df, width="stretch")

    with tab2:
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
                            st.markdown("### Predictions vs Actual")
                            comp_cols = st.columns(4)

                            diff = comparison_df["Predicted"] - comparison_df["Actual"]
                            mae_val = float(np.mean(np.abs(diff)))
                            rmse_val = float(np.sqrt(np.mean(diff ** 2)))
                            bias_val = float(np.mean(diff))
                            if len(comparison_df) >= 2:
                                corr = np.corrcoef(comparison_df["Actual"], comparison_df["Predicted"])[0, 1]
                                r2_local = float(corr ** 2) if not np.isnan(corr) else float("nan")
                            else:
                                r2_local = float("nan")

                            with comp_cols[0]:
                                st.metric("MAE", f"{mae_val:.3f}")
                            with comp_cols[1]:
                                st.metric("RMSE", f"{rmse_val:.3f}")
                            with comp_cols[2]:
                                st.metric("Bias", f"{bias_val:.3f}")
                            with comp_cols[3]:
                                st.metric("R²", f"{r2_local:.3f}" if not np.isnan(r2_local) else "N/A")

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
    
    with tab3:
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
        else:
            st.info(f"Model {info_model} has not been trained yet. Please train models first.")

    with tab4:
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

                controls_col1, controls_col2, controls_col3 = st.columns([1, 1, 1])

                sample_index = controls_col1.slider(
                    "Sample index",
                    min_value=0,
                    max_value=max_sample_index,
                    value=min(0, max_sample_index),
                    step=1,
                )

                top_n_default = min(10, max_features)
                top_n = controls_col2.slider(
                    "Top |corr| wavelengths",
                    min_value=3,
                    max_value=max(3, min(60, max_features)),
                    value=max(3, top_n_default),
                )

                def _format_wavelength_label(column_name: str) -> str:
                    try:
                        value = extract_wavelength(column_name)
                        return f"{value:.0f} nm"
                    except ValueError:
                        return column_name

                wavelength_options = list(features.columns)
                selected_wavelength = controls_col3.selectbox(
                    "Focus wavelength",
                    options=wavelength_options,
                    index=min(len(wavelength_options) // 2, len(wavelength_options) - 1),
                    format_func=_format_wavelength_label,
                )

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
                st.dataframe(
                    correlations.head(top_n).rename("Correlation").to_frame().style.format({"Correlation": "{:.3f}"}),
                    use_container_width=True,
                )

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
        page_title="Login Required",
        page_icon="🔐",
        layout="centered"
    )
    
    import base64
    
    def get_base64_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    
    try:
        img_base64 = get_base64_image("images/login_background.png")
        
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{img_base64}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            
            /* Make form containers semi-transparent with backdrop blur */
            [data-testid="stForm"] {{
                background-color: rgba(255, 255, 255, 0.9);
                padding: 2rem;
                border-radius: 10px;
                backdrop-filter: blur(10px);
            }}
            
            /* Style the tabs container */
            [data-testid="stTabs"] {{
                background-color: rgba(0, 0, 0, 0.85);
                padding: 1.5rem;
                border-radius: 10px;
                backdrop-filter: blur(10px);
            }}
            
            /* Style tab buttons */
            [data-testid="stTabs"] button {{
                color: white;
            }}
            
            /* Style active tab */
            [data-testid="stTabs"] button[aria-selected="true"] {{
                color: white;
                border-bottom-color: white;
            }}
            
            /* Style tab content text */
            [data-testid="stTabs"] h3 {{
                color: white;
            }}

            /* Style input labels to be black */
            [data-testid="stTabs"] label {{
                color: black !important;
            }}

            [data-testid="stCheckbox"] label {{
                color: black !important;
            }}
            
            /* Make title more visible */
            h1 {{
                color: black !important;
                text-align: center;
                -webkit-text-stroke: 1px #3D2B1F;
                text-stroke: 1px #3D2B1F;
                text-shadow: none !important;
                background: transparent !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning("⚠️ Background image not found at 'images/login_background.jpg'")
    
    st.title("Soil Modeller")
    st.markdown("---")
    
    user_manager = UserManager()
    
    tab1, tab2 = st.tabs(["🚪 Login", "📝 Register"])
    
    with tab1:
        st.subheader("Existing User Login")
        
        with st.form("login_form"):
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            remember_me = st.checkbox("Remember me", value=True)
            login_btn = st.form_submit_button("Login", use_container_width=True)
            
            if login_btn:
                if username and password:
                    with st.spinner("Logging in..."):
                        if user_manager.login(username, password):
                            token = generate_jwt_token(username)
                            if remember_me:
                                cookies['auth_token'] = token
                                cookies.save()
                            
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.session_state.active_page = "home"
                            
                            st.success("✅ Login successful!")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error("❌ Invalid username or password")
                else:
                    st.warning("⚠️ Please enter both username and password")
    
    with tab2:
        st.subheader("New User Registration")
        
        with st.form("register_form"):
            new_user = st.text_input("👤 Choose Username", placeholder="Enter a username")
            new_pass = st.text_input("🔒 Choose Password", type="password", placeholder="Enter a password")
            confirm_pass = st.text_input("🔒 Confirm Password", type="password", placeholder="Re-enter password")
            reg_btn = st.form_submit_button("Create Account", use_container_width=True)
            
            if reg_btn:
                if new_user and new_pass and confirm_pass:
                    if new_pass == confirm_pass:
                        with st.spinner("Creating account..."):
                            if user_manager.registerUser(new_user, new_pass):
                                st.success("✅ Account created successfully! Please login with your new credentials.")
                            else:
                                st.error("❌ Registration failed - username may already exist")
                    else:
                        st.error("❌ Passwords do not match")
                else:
                    st.warning("⚠️ Please fill in all fields")
    
    st.stop()

def display_logout_section():
    with st.sidebar:
        st.title("Soil Modeler")
        st.subheader("Account")

        st.markdown(f"""
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
                        font-size: 1.1rem;
                    ">
                        👤
                    </div>
                    <div>
                        <div style="color: #777; font-size: 0.85rem;">Current User</div>
                        <div style="color: #2a7143; font-weight: 600;">{st.session_state.username}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("🚪 **Logout**", 
                    use_container_width=True,
                    type="primary"):
            cookies['auth_token'] = ""
            cookies.save()
            
            st.session_state.just_logged_out = True
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.active_page = "home"
            
            st.rerun()

        st.markdown("---")
        
        # Home button - goes to main dashboard
        if st.button("🏠 **Dashboard**", 
                    use_container_width=True,
                    help="Main dashboard with all features"):
            st.session_state.active_page = "home"
            st.rerun()
        
        # Model Library button
        if st.button("📚 **Model Library**", 
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
            st.rerun()
    
    elif st.session_state.active_page == "home":
        # Show the main app (home page with all tabs)
        old_main()
    
    else:
        # Default to home if invalid page
        st.session_state.active_page = "home"
        st.rerun()

if __name__ == "__main__":
    main()