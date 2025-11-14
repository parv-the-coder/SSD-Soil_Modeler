# The Spectral Soil Modeler

Automated machine-learning workflows for predicting soil properties from spectral data.

## Overview
This repository contains a Streamlit app that automates model training and prediction for spectral soil datasets. The system:


---
## Team & Course Details
Course: Software Systems Development (Phase 2)
Project Title: The Spectral Soil Modeler: An Automated ML Workflow
Submission Date: November 13, 2025
Supervisor: Dr. Abhishek Singh

| Name | Roll No |
|------|---------|
| Akshat Kotadia | 2025201005 |
| Jewel Joseph | 2025201047 |
| Gaurav Patel | 2025201065 |
| Parv Shah | 2025201093 |
| Eshwar Pingili | 2025204030 |

---
## 1. Background & Motivation
Researchers at IIIT Hyderabad's Laboratory for Spatial Informatics (LSI) build predictive models for soil properties (e.g., clay, sand, TOC) from spectral data. Manually exploring every combination of preprocessing technique and algorithm is repetitive, error‑prone, and slow. This bottleneck limits hypothesis testing speed and research throughput.

### Problem Statement
Automate evaluation of multiple spectral preprocessing variants and ML algorithms (with proper cross‑validation and hyperparameter tuning) to accelerate soil property modeling.

### Solution Summary
The Spectral Soil Modeler converts a multi‑day manual workflow into a single automated session. A researcher uploads data, selects target(s), and triggers batch training. The app executes preprocessing, model training, tuning, evaluation, ranking, export, and prediction readiness end‑to‑end.

---
## 2. User Interface & Journey (Current Implementation)
Implemented as a multi‑tab Streamlit app:
1. Train Models: Upload multiple Excel files; targets auto‑detected via `_target` suffix.
2. Make Predictions: Load a trained model (T1–T5), upload new spectral data; feature alignment enforced.
3. Model Info (extensible): Intended for deeper diagnostics / documentation.

Real‑time feedback: Progress bars and status messages for each target’s pipeline. Errors surfaced as concise runtime messages (never raw pandas Series dumps).

---
## 3. Detailed Workflow & Architecture
High‑level flow:
1. Data Ingestion: `smart_read` loads XLS/XLSX/CSV, adds filename prefixes to columns.
2. Target Detection: Columns ending with `_target` become independent modeling tasks.
3. Feature Selection: For each target, only prefixed spectral columns (excluding the target) are taken as X.
4. Preprocessing Variants (from `src/data_preprocessing.py`): Reflectance, Absorbance (log transform), Continuum Removal (row‑wise normalization).
5. Model Set (`src/model_training.py`): PLSRegression, RandomForestRegressor, GradientBoostingRegressor, KernelRidge, SVR.
6. Cross‑Validation: 5‑fold `cross_val_predict` for each (preprocessing, model) combination.
7. Metrics: R², MSE, RMSE, RPD (std(y)/RMSE). Best pipeline selected by highest R².
8. Export: Best model pickle + feature names + improvement log written to `models/` per target.
9. Prediction: Column renaming and strict feature presence check; aborts if mismatch.

Simplified Architecture Layers:
- Presentation: Streamlit components (`app.py`).
- Processing: Preprocessing + model loops (`run_all_pipelines`).
- Persistence: Export of fitted model + metadata.
- Prediction: `predict_with_model` consuming trained artifacts.

---
## 4. Technology Stack
- Python 3.x
- Streamlit (UI & orchestration)
- Pandas / NumPy (data handling)
- scikit‑learn (models, metrics, CV)
- Matplotlib / Seaborn (visualization foundations; selectable for expansion)
- Git & GitHub (version control)

---
## 5. Model Training Details
For each target and preprocessing:
- Data rows with NaNs in target are removed before CV.
- Each estimator is fit on valid rows; predictions aggregated across folds.
- Feature importances captured when available (`feature_importances_`).
- Improvement log records metric progression across combinations for traceability.

Hyperparameter tuning placeholder: Current code employs direct model instantiation (Phase 2 baseline). GridSearchCV design described in report can be integrated next (see Future Work) ensuring per‑fold isolation to avoid leakage.

---
## 6. Exported Artifacts (Per Target)
- `models/best_model_Tn.pkl`: Serialized estimator.
- `models/best_model_Tn_features.txt`: Exact ordered feature names used during training.
- `models/best_model_Tn_log.txt`: Text log of pipeline performance entries.

Keep these three files together for reproducible prediction.

---
## 7. Revisions & Rationale (Phase 2)
1. Monolithic Streamlit Design: Simplified from proposed FastAPI + Streamlit split for faster iteration and single‑user research context.
2. Robust Error Handling: Replaced ambiguous returns with explicit exceptions; eliminated raw Series leakage in UI.
3. Feature Alignment Enforcement: Added strict matching + intelligent renaming to prevent silent zero‑filled inputs causing constant predictions.
4. Logging Improvements: Human‑readable per‑target improvement logs for audit / reproducibility.

Planned future revision: Reintegration of full hyperparameter tuning (GridSearchCV) per combination with careful execution semantics for heavier models.

---
## 8. Troubleshooting Guide
| Symptom | Likely Cause | Resolution |
|---------|--------------|------------|
| Constant identical predictions | Feature mismatch; columns auto‑renamed but missing | Compare with `*_features.txt`, fix input schema |
| Missing target detection | Column does not end with `_target` | Rename target column appropriately |
| RuntimeError during training | Invalid data (NaNs, non‑numeric spectral columns) | Inspect merged preview, clean source files |
| Feature mismatch error | Uploaded file schema differs | Adjust column names or regenerate features |

---
This extended README incorporates Phase 2 project report details, architecture, team roster, and operational guidance.
## Setup
1. Create and activate a Python environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the Streamlit app:

```powershell
streamlit run app.py
```

## Usage

### Step 1 — Upload Training Data
- In the **Train Models** tab, upload one or more Excel files (`.xls`/`.xlsx`) containing spectral bands and a target column whose name ends with `_target` (for example `spectra_with_target_T1_target`).
- Each file is merged and column names are prefixed with the filename. The app shows merged preview and lists detected target columns.

### Step 2 — Train Pipelines for All Targets
- Click **Run Automated ML Pipelines for All Targets** to train models in parallel. For each target the app:
	- Filters features relevant to that target,
	- Applies preprocessing variants,
	- Runs cross-validated training across multiple regressors (PLS, RandomForest, GBRT, KRR, SVR),
	- Selects and saves the best model and writes a feature list and a short training log to the `models/` folder.

Saved files for each trained target look like:
- `models/best_model_T1.pkl` — the serialized model (pickle)
- `models/best_model_T1_features.txt` — newline list of trained feature column names
- `models/best_model_T1_log.txt` — human-readable training/improvement log

### Step 3 — Predict with Your Own File
- In the **Make Predictions** tab, choose a trained model (T1..T5) and upload a single Excel file for prediction.
- The app attempts to align your file's columns to the model's expected feature names. It will automatically rename columns in the pattern `spectra_XXXX` to the expected `spectra_with_target_Tn_XXXX` format when possible.
- Prediction will only run if all required feature columns are present; otherwise the app lists missing/extra columns and stops to avoid producing meaningless constant predictions.

## File formats and reading
- Supported upload types: `.xls`, `.xlsx`, and CSV (`.csv`). The app uses a `smart_read` helper to auto-detect and read files.

## Troubleshooting
- If predictions are constant (same value for all rows), check `models/*_features.txt` and ensure your prediction file has matching column names and bands.
- If training shows an error, the app raises readable `RuntimeError` messages (no pandas Series printed as raw errors) and writes target-specific logs to `models/`.

## Development notes
- Preprocessing functions are defined in `src/data_preprocessing.py` and return a dictionary of callable preprocessors used by the training pipeline.
- Training logic lives in `src/model_training.py` and returns per-pipeline metrics and the selected best model.
- Prediction helpers are in `src/prediction.py` (there is a `predict_with_model` helper used by the app).

---
