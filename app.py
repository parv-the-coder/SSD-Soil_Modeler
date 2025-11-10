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

def main():
    st.title("Spectral Soil Modeler")
    st.write("Upload multiple Excel files containing spectral data. Select target column and run ML pipelines.")

    st.header("Step 1: Upload Training Data")
    st.write("Upload multiple Excel files to build and train your models for all targets.")
    uploaded_files = st.file_uploader("Upload Excel files for training", type=["xls", "xlsx"], accept_multiple_files=True, key="train_files")
    dfs = []
    if uploaded_files:
        for file in uploaded_files:
            try:
                df = smart_read(file)
                prefix = file.name.split('.')[0]
                df = df.add_prefix(prefix + '_')
            except Exception as e:
                st.error(str(e))
                continue
            dfs.append(df)
        merged_df = pd.concat(dfs, axis=1)
        st.write("Merged Data Preview:", merged_df.head())

        # Find all target columns ending with '_target'
        target_cols = [col for col in merged_df.columns if col.lower().endswith('_target')]
        st.write(f"Found target columns: {target_cols}")

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
                results, best_model, best_score, best_pipeline, improvement_log, feature_importances = run_all_pipelines(X, y, preprocessing, log_improvements=True, return_feature_importances=True)
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

        st.subheader("Step 2: Train ML Pipelines for All Targets (Parallel)")
        if st.button("Run Automated ML Pipelines for All Targets"):
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
                        st.write(f"Results for {target_col}:")
                        st.dataframe(summary_df)
                        st.write(f"Best Model: {best_pipeline} | Score: {best_score}")
                        st.subheader("Step-by-step Model Improvement Log")
                        for entry in improvement_log:
                            st.write(entry)
                        st.success(f"Best model for {target_col} saved for user prediction.")
                    except Exception as e:
                        st.error(f"Error training {target_col}: {str(e)}")
            progress_bar.empty()
            status_text.text("All targets completed.")

    st.header("Step 3: Predict with Your Own File")
    st.write("Upload a single Excel file to make predictions using the trained model.")
    # Add dropdown for model selection in prediction
    model_options = [f"T{i}" for i in range(1, 6)]
    selected_model = st.selectbox("Select trained model for prediction", model_options, key="predict_model")
    model_path = os.path.join("models", f"best_model_{selected_model}.pkl")
    feature_names_path = os.path.join("models", f"best_model_{selected_model}_features.txt")
    user_file = st.file_uploader("Upload Excel file for prediction", type=["xls", "xlsx"], key="user_pred")
    if user_file:
        try:
            user_df = smart_read(user_file)
            if os.path.exists(feature_names_path) and os.path.exists(model_path):
                with open(feature_names_path) as f:
                    trained_feature_names = [line.strip() for line in f.readlines()]
                # Determine expected prefix from trained features
                prefix = trained_feature_names[0].rsplit('_', 2)[0] + '_'
                # Rename columns in user_df to match trained features if possible
                def rename_col(col):
                    match = re.match(r'spectra_(\d+)$', col)
                    if match:
                        return f'{prefix}{match.group(1)}'
                    return col
                user_df.columns = [rename_col(col) for col in user_df.columns]
                # Check for exact feature match
                missing = set(trained_feature_names) - set(user_df.columns)
                extra = set(user_df.columns) - set(trained_feature_names)
                if missing:
                    st.error(f"Feature mismatch detected.\nMissing columns: {sorted(missing)}\nExtra columns: {sorted(extra)}\nPlease ensure your file matches the trained model's features exactly.")
                    return
                # Only run prediction if features match exactly
                user_df = user_df[trained_feature_names]
                from src.prediction import predict_with_model
                predictions = predict_with_model(model_path, user_df)
                st.write("Predictions:")
                st.dataframe(pd.DataFrame(predictions, columns=["Prediction"]))
                # Download button for predictions
                output = BytesIO()
                pd.DataFrame(predictions, columns=["Prediction"]).to_excel(output, index=False)
                st.download_button("Download Predictions", data=output.getvalue(), file_name="predictions.xlsx")
            else:
                st.error("Trained model or feature file not found. Please train models first.")
        except Exception as e:
            st.error(str(e))
            return

if __name__ == "__main__":
    main()