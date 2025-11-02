import streamlit as st
import pandas as pd
from src.data_preprocessing import preprocess_data
from src.model_training import run_all_pipelines
from src.evaluation import evaluate_models, plot_results
from src.export import export_best_model
import os
import re

st.title("The Spectral Soil Modeler: Automated ML Workflow")

# Sidebar for inputs

st.sidebar.header("Configuration")

# User test file uploader (move above model training block for scope)
st.sidebar.header("Test Your Own File")
user_test_file = st.sidebar.file_uploader("Upload Excel file for prediction", type=["xls", "xlsx", "csv"])
# List Excel files in data folder
data_folder = "data"
excel_files = [f for f in os.listdir(data_folder) if f.endswith(".xls")]
st.sidebar.write(f"Found {len(excel_files)} Excel files in data folder.")
target_column = st.sidebar.selectbox("Select Target Property", ["T1", "T2", "T3", "T4", "T5"])

# Detect wavelength columns from all files and use intersection
dfs = []
target_names = []

# --- NEW CONCATENATION LOGIC FOR PERFECTLY MATCHED FILES ---
dfs = []
target_names = []
wavelength_cols = None

for f in excel_files:
    file_path = os.path.join(data_folder, f)
    if f.lower().endswith('.xlsx'):
        df = pd.read_excel(file_path, engine='openpyxl')
    elif f.lower().endswith('.xls'):
        try:
            df = pd.read_excel(file_path, engine='xlrd')
        except Exception:
            df = pd.read_csv(file_path)
    else:
        df = pd.read_csv(file_path)

    match = re.search(r'T(\d)', f)
    if match:
        target_name = f'T{match.group(1)}'
        target_names.append(target_name)
        if 'target' in df.columns and target_name not in df.columns:
            df = df.rename(columns={'target': target_name})
        # Detect wavelength columns (all except target)
        if wavelength_cols is None:
            wavelength_cols = [col for col in df.columns if col != target_name]
        # Drop duplicate wavelength rows
        df = df.drop_duplicates(subset=wavelength_cols)
        # Set wavelength columns as index
        df = df.set_index(wavelength_cols)
        dfs.append(df[[target_name]])

# Concatenate all targets horizontally, using wavelength columns as index

# Debug: print row counts per file
print("Row counts per file:")
for f, df in zip(excel_files, dfs):
    print(f"{f}: {len(df)}")

combined_df = pd.concat(dfs, axis=1)
combined_df = combined_df.reset_index()  # bring wavelength columns back as columns
target_cols = target_names

# Debug: print combined dataframe row count
print(f"Combined dataframe rows: {len(combined_df)}")

# Debug: print missing values per target column
print("Missing values per target column:")
print(combined_df[target_cols].isnull().sum())

# Debug: print wavelengths missing T3
if 'T3' in combined_df.columns:
    missing_t3 = combined_df[combined_df['T3'].isnull()]
    print("Wavelengths missing T3:")
    print(missing_t3[wavelength_cols])

st.write("Combined Dataset Preview (perfect join):")
st.dataframe(combined_df.head())

# Validate target column
if target_column not in combined_df.columns:
    st.error(f"Selected target column '{target_column}' not found in combined data columns: {list(combined_df.columns)}")
    st.stop()

# Preprocess data

# Filter to rows where selected target is present
filtered_df = combined_df[combined_df[target_column].notnull()].copy()
print(f"Rows used for training {target_column}: {len(filtered_df)}")

# Drop rows with any NaN in wavelength columns

# Get actual spectral columns (numeric, not target)
spectral_cols = [c for c in filtered_df.select_dtypes(include='number').columns if c != target_column]
print(f"Spectral columns used for dropna: {spectral_cols}")

# Drop rows with any NaN in spectral columns
filtered_df = filtered_df.dropna(subset=spectral_cols)
print(f"Rows after dropping NaN in spectral columns: {len(filtered_df)}")

# Print columns with NaN after filtering
print("Columns with NaN after filtering:")
print(filtered_df.isnull().sum()[filtered_df.isnull().sum() > 0])

try:
    X, y, preprocessing = preprocess_data(filtered_df, target_column)
except Exception as exc:
    st.error(f"Preprocessing failed: {exc}")
    st.stop()

if st.button("Run Automated ML Pipelines"):

    with st.spinner("Training and evaluating models..."):
        results = run_all_pipelines(X, y, preprocessing)
        st.success("Pipelines completed!")

        # Real-time analysis: show metrics for each pipeline
        st.header("Model Leaderboard & Metrics")
        leaderboard = evaluate_models(results)
        st.dataframe(leaderboard)

        # Show metrics for top model
        top_pipeline = leaderboard.iloc[0]["Pipeline"]
        st.subheader(f"Top Model: {top_pipeline}")
        top_metrics = leaderboard.iloc[0]
        st.write({col: top_metrics[col] for col in leaderboard.columns if col != "Pipeline"})

        # Visualizations
        plot_results(results)

        # Export best model
        best_model = results[top_pipeline]["model"]
        export_filename = f"best_model_{target_column}.pkl"
        export_path = export_best_model(best_model, os.path.join("models", export_filename))
        with open(export_path, "rb") as fh:
            model_bytes = fh.read()
        st.download_button("Download Best Model", data=model_bytes, file_name=export_filename)

        # If user uploaded a test file, run predictions
        if user_test_file is not None:
            st.header("User Test File Prediction")
            try:
                if user_test_file.name.lower().endswith((".xls", ".xlsx")):
                    test_df = pd.read_excel(user_test_file)
                else:
                    test_df = pd.read_csv(user_test_file)
            except Exception as exc:
                st.error(f"Failed to read test file: {exc}")
                st.stop()

            # Use only wavelength columns for prediction
            test_X = test_df[wavelength_cols]
            preds = best_model.predict(test_X)
            st.write("Predictions for uploaded file:")
            st.dataframe(pd.DataFrame({f"Predicted {target_column}": preds}))
else:
    st.info("Please upload a dataset to begin.")