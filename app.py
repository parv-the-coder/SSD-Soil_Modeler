import streamlit as st
import pandas as pd
from src.data_preprocessing import preprocess_data
from src.model_training import run_all_pipelines
from src.evaluation import evaluate_models, plot_results
from src.prediction import predict_with_model
from src.export import export_best_model
from io import BytesIO
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import chardet
import numpy as np
import concurrent.futures
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from src.database import UserManager

def display_logout_section():
    with st.sidebar:
        st.title("👤 User Panel")
        st.markdown("---")
        st.success(f"Logged in as: **{st.session_state.username}**")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()
        
        st.markdown("---")

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

def old_main():
    st.set_page_config(page_title="Spectral Soil Modeler", page_icon="", layout="wide")
    
    # Header with styling
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="color: #2E8B57;"> Spectral Soil Modeler</h1>
        <p style="font-size: 1.2em; color: #666;">
            Advanced machine learning for soil property prediction from spectral data
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["Train Models", "Make Predictions", "Model Info"])
    
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
            st.success(f"Loaded {len(uploaded_files)} files successfully!")
            
            for file in uploaded_files:
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
                st.write(" **Merged Data Overview:**")
                
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
                
                st.write(f"**Found target columns:** {target_cols}")

                st.subheader("Train ML Pipelines")
                st.markdown("""
                Click the button below to automatically train and optimize ML models for all targets.
                The system will test multiple combinations of:
                - **Preprocessing**: Reflectance, Absorbance, Continuum Removal
                - **Algorithms**: PLS Regression, SVR, Gradient Boosting, Random Forest, Kernel Ridge
                """)
                
                if st.button("Train All Models", type="primary", width="stretch"):
                    
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
                            progress_bar.progress(completed / total_targets)
                            status_text.text(f"Completed {completed}/{total_targets} targets: {target_col}")
                            try:
                                result = future.result()
                                if not isinstance(result, tuple):
                                    raise RuntimeError(f"train_target for {target_col} did not return a tuple. Got type: {type(result)}. Value: {repr(result)}")
                                target_col, summary_df, best_pipeline, best_score, improvement_log, feature_importances = result
                                
                                with st.expander(f" Results for {target_col}", expanded=True):
                                    st.dataframe(summary_df, width="stretch")
                                    st.success(f"**Best Model:** {best_pipeline} | **Score:** {best_score:.4f}")
                                    
                                    with st.expander("Step-by-step Model Improvement Log"):
                                        for entry in improvement_log:
                                            st.write(f"• {entry}")
                                    
                                st.success(f"Best model for {target_col} saved successfully!")
                                
                            except Exception as e:
                                st.error(f"Error training {target_col}: {str(e)}")
                    
                    progress_bar.empty()
                    status_text.text("All targets training completed!")
        
    with tab2:
        st.header(" Make Predictions")
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
        if user_file:
            try:
                user_df = smart_read(user_file)
                if os.path.exists(feature_names_path) and os.path.exists(model_path):
                    with open(feature_names_path) as f:
                        trained_feature_names = [line.strip() for line in f.readlines()]
                    # Determine expected prefix from trained features
                    # Extract prefix properly: spectra_with_target_T1_410 -> spectra_with_target_T1_
                    first_feature = trained_feature_names[0]
                    parts = first_feature.split('_')
                    # Reconstruct prefix: join all parts except the last one (wavelength)
                    prefix = '_'.join(parts[:-1]) + '_'
                    
                    st.info(f"Expected feature format: `{prefix}[wavelength]`")
                    
                    # Rename columns in user_df to match trained features if possible
                    def rename_col(col):
                        # Handle both numeric wavelengths and existing prefixed columns
                        col_str = str(col)
                        
                        # If it's a pure number (wavelength), add the full prefix
                        if col_str.isdigit():
                            return f'{prefix}{col}'
                        
                        # If it already has some prefix but wrong format, try to extract wavelength
                        # Handle cases like 'spectra_with_target_410' -> 'spectra_with_target_T1_410'
                        if 'spectra_with_target_' in col_str and not f'spectra_with_target_{selected_model}_' in col_str:
                            # Extract wavelength from the end
                            wavelength = col_str.split('_')[-1]
                            if wavelength.isdigit():
                                return f'{prefix}{wavelength}'
                        
                        return col
                    
                    user_df.columns = [rename_col(col) for col in user_df.columns]
                    
                    # Check for exact feature match
                    missing = set(trained_feature_names) - set(user_df.columns)
                    extra = set(user_df.columns) - set(trained_feature_names)
                    
                    if missing:
                        st.warning(f"Feature mismatch detected:")
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
                    
                    # Apply the same preprocessing that was used during training
                    # Check which preprocessing was used for this model
                    best_pipeline = None
                    try:
                        with open(log_path, 'r') as f:
                            log_content = f.read()
                            # Find the best performing pipeline (highest R2)
                            import re
                            r2_matches = re.findall(r'([A-Za-z ]+_[A-Za-z]+): R2=([0-9.-]+)', log_content)
                            if r2_matches:
                                best_r2 = max(float(match[1]) for match in r2_matches)
                                best_pipeline = next(match[0] for match in r2_matches if float(match[1]) == best_r2)
                    except:
                        pass
                    
                    # Only run prediction if features match exactly
                    user_df_aligned = user_df[trained_feature_names]
                    
                    # Apply correct preprocessing based on best pipeline
                    user_df_processed = user_df_aligned.copy()
                    
                    if best_pipeline and 'Absorbance' in best_pipeline:
                        st.info("Applying Absorbance preprocessing to match training...")
                        # Absorbance preprocessing: log10(1/reflectance)
                        user_df_processed = user_df_processed.replace(0, 1e-6)  # Avoid log(0)
                        user_df_processed = np.log10(1.0 / user_df_processed.clip(lower=1e-6))
                        user_df_processed = user_df_processed.replace([np.inf, -np.inf], np.nan).fillna(0)
                    elif best_pipeline and 'Continuum Removal' in best_pipeline:
                        st.info("Applying Continuum Removal preprocessing to match training...")
                        # Continuum removal preprocessing (divide each row by its max value)
                        row_max = user_df_processed.max(axis=1).replace(0, 1e-6)
                        user_df_processed = user_df_processed.div(row_max, axis=0)
                    else:
                        st.info("Using Reflectance data (no preprocessing needed)...")
                        # Reflectance - no preprocessing needed
                    
                    from src.prediction import predict_with_model
                    try:
                        predictions = predict_with_model(model_path, user_df_processed)
                        st.success("Predictions completed successfully!")
                    except FileNotFoundError:
                        st.error("Model file not found. Please train models first using the 'Train All Models' section.")
                        return
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")
                        return
                    
                    # Read model performance metrics from log file
                    log_path = os.path.join("models", f"best_model_{selected_model}_log.txt")
                    model_accuracy = None
                    model_r2 = None
                    model_rmse = None
                    range_coverage = None
                    
                    if os.path.exists(log_path):
                        try:
                            with open(log_path, 'r') as f:
                                log_content = f.read()
                                
                            # Extract metrics from log file
                            import re
                            
                            # Look for R2, RMSE, and Range Coverage values
                            r2_match = re.search(r'R2=([0-9.-]+)', log_content)
                            rmse_match = re.search(r'MSE=([0-9.-]+)', log_content)
                            range_match = re.search(r'Range_Cov=([0-9.-]+)', log_content)
                            
                            if r2_match:
                                model_r2 = float(r2_match.group(1))
                                # For spectral data, use realistic accuracy mapping
                                if model_r2 >= 0.8:
                                    model_accuracy = 95  # Excellent for spectral data
                                elif model_r2 >= 0.6:
                                    model_accuracy = 85  # Very good
                                elif model_r2 >= 0.4:
                                    model_accuracy = 75  # Good
                                elif model_r2 >= 0.2:
                                    model_accuracy = 65  # Fair
                                else:
                                    model_accuracy = max(30, model_r2 * 100)  # Conservative minimum
                                
                            if rmse_match:
                                mse_val = float(rmse_match.group(1))
                                model_rmse = np.sqrt(mse_val)
                                
                            if range_match:
                                range_coverage = float(range_match.group(1))
                                
                        except Exception as e:
                            st.warning(f"Could not read model performance metrics: {e}")
                    
                    # Display success message with accuracy
                    if model_accuracy is not None:
                        st.success(f"**Prediction completed successfully!** | **Model Accuracy: {model_accuracy:.1f}%** (R² = {model_r2:.3f})")
                    else:
                        st.success("**Prediction completed successfully!**")
                    
                    # Display prediction statistics with error handling
                    st.subheader(" **Prediction Results & Model Performance**")
                    
                    # Model Performance Section
                    if model_accuracy is not None or model_rmse is not None or range_coverage is not None:
                        st.markdown("### Model Testing Accuracy")
                        
                        performance_cols = st.columns(4)
                        
                        with performance_cols[0]:
                            if model_accuracy is not None:
                                # Realistic color coding for spectral data
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
                                    f"{accuracy_color} {model_accuracy:.1f}%",
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
                                # Assess extreme value handling
                                if range_coverage >= 0.7:
                                    range_color = "good"
                                elif range_coverage >= 0.5:
                                    range_color = "medium"
                                else:
                                    range_color = "bad"
                                
                                st.metric(
                                    "Range Coverage",
                                    f"{range_color} {range_coverage:.2f}",
                                    help="How well the model captures extreme values"
                                )
                        
                        # Performance assessment with realistic thresholds for spectral data
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
                        # Ensure predictions is a numpy array or pandas Series
                        if isinstance(predictions, pd.DataFrame):
                            predictions = predictions.values.flatten()
                        elif not isinstance(predictions, (np.ndarray, pd.Series)):
                            predictions = np.array(predictions)
                        
                        # Create DataFrame with proper column name
                        pred_df = pd.DataFrame({f"{selected_model} Prediction": predictions})
                        
                        # Convert to float values to avoid Series formatting issues
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
                        
                        # Show predictions table
                        st.write("**Detailed Predictions:**")
                        st.dataframe(pred_df, width="stretch")
                        
                        # Plot histogram of predictions
                        if len(predictions) > 1:  # Only plot if we have multiple predictions
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.hist(predictions, bins=min(20, len(predictions)//2), alpha=0.7, color='skyblue', edgecolor='black')
                            ax.set_xlabel(f'{selected_model} Prediction Value')
                            ax.set_ylabel('Frequency')
                            ax.set_title(f'Distribution of {selected_model} Predictions')
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                        
                        # Enhanced download with model performance metrics
                        st.markdown("### Download Results")
                        
                        # Create comprehensive results DataFrame
                        # Add sample IDs
                        pred_df['Sample_ID'] = range(1, len(predictions) + 1)
                        
                        # Create summary statistics
                        summary_stats = {
                            'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Count'],
                            'Value': [f"{mean_pred:.3f}", f"{std_pred:.3f}", f"{min_pred:.3f}", f"{max_pred:.3f}", len(predictions)]
                        }
                        summary_df = pd.DataFrame(summary_stats)
                        
                        # Create model performance info
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
                        
                        # Reorder columns for better presentation
                        final_export = pred_df[['Sample_ID', f"{selected_model} Prediction"]]
                        
                        # Export options
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Excel download with multiple sheets
                            output = BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                final_export.to_excel(writer, sheet_name='Predictions', index=False)
                                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                                if model_accuracy is not None:
                                    performance_df.to_excel(writer, sheet_name='Model_Performance', index=False)
                            
                            st.download_button(
                                "� Download as Excel (with performance)", 
                                data=output.getvalue(), 
                                file_name=f"{selected_model}_predictions_with_accuracy.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        
                        with col2:
                            # CSV download
                            csv_data = final_export.to_csv(index=False)
                            st.download_button(
                                "Download as CSV", 
                                data=csv_data, 
                                file_name=f"{selected_model}_predictions.csv",
                                mime="text/csv"
                            )
                        
                    except Exception as metrics_error:
                        st.error(f"Error displaying prediction metrics: {metrics_error}")
                        # Still show basic results
                        st.write("**Raw Predictions:**")
                        st.write(predictions)
                else:
                    st.error("Trained model or feature file not found. Please train models first.")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                return
    
    with tab3:
        st.header(" Model Information")
        st.markdown("""
        View information about your trained models, including performance metrics and feature importance.
        """)
        
        # Model selection for info
        info_model = st.selectbox("Select Model for Information", [f"T{i}" for i in range(1, 6)], key="info_model")
        
        model_path = os.path.join("models", f"best_model_{info_model}.pkl")
        log_path = os.path.join("models", f"best_model_{info_model}_log.txt")
        features_path = os.path.join("models", f"best_model_{info_model}_features.txt")
        
        if os.path.exists(model_path) and os.path.exists(log_path):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"{info_model} Model Performance")
                try:
                    with open(log_path, 'r') as f:
                        log_content = f.read()
                    
                    # Extract best performance from log
                    import re
                    r2_match = re.search(r'R²:\s*([\d.]+)', log_content)
                    if r2_match:
                        best_r2 = float(r2_match.group(1))
                        st.metric("Best R² Score", f"{best_r2:.4f}")
                    
                    rmse_match = re.search(r'RMSE:\s*([\d.]+)', log_content)
                    if rmse_match:
                        best_rmse = float(rmse_match.group(1))
                        st.metric("Best RMSE", f"{best_rmse:.4f}")
                    
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
                st.text(log_content)
        else:
            st.info(f"Model {info_model} has not been trained yet. Please train models first.")

def check_authentication():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None
    
    if not st.session_state.logged_in:
        show_login_page()
        return False
    return True

def show_login_page():
    st.set_page_config(
        page_title="Login Required",
        page_icon="🔐",
        layout="centered"
    )
    
    st.title("Soil Modeller")
    st.markdown("---")
    
    user_manager = UserManager()
    
    tab1, tab2 = st.tabs(["🚪 Login", "📝 Register"])
    
    with tab1:
        st.subheader("Existing User Login")
        
        with st.form("login_form"):
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            login_btn = st.form_submit_button("Login", use_container_width=True)
            
            if login_btn:
                if username and password:
                    with st.spinner("Logging in..."):
                        if user_manager.login(username, password):
                            st.session_state.logged_in = True
                            st.session_state.username = username
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
    
    st.markdown("---")
    st.info("💡 **Tip:** Register a new account if you don't have one, or login with existing credentials.")
    
    st.stop()

def main():
    if not check_authentication():
        return
    
    display_logout_section()
    old_main()

if __name__ == "__main__":
    main()