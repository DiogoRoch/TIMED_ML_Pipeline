# TIMED ML Pipeline

Interactive Streamlit application and supporting scripts to explore the TIMED WP2 dataset, run automated regression model optimization (Optuna + PyCaret + custom logic), and visualize experiment results (performance metrics, SHAP explanations, residual diagnostics) through a built‑in dashboard.

---

## 🚀 Quick Start (TL;DR)

```bash
# 1. Clone
git clone https://github.com/DiogoRoch/TIMED_ML_Pipeline.git
cd TIMED_ML_Pipeline

# 2. (Option A - uv) Recommended if you have uv installed
curl -LsSf https://astral.sh/uv/install.sh | sh  # if you don't have it
uv sync  # creates .venv and installs locked deps from pyproject + uv.lock

# 2. (Option B - pip)
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Run Streamlit app
streamlit run main.py
```

Open the URL shown in the terminal (usually http://localhost:8501).

---

## 🧠 What This App Does

1. Analysis Page: Configure and launch automated multi‑model regression optimization over selected targets with feature selection, cross‑validation, Optuna hyperparameter tuning, and optional model persistence.
2. Dashboard Page: Browse completed experiments, inspect ranked model results, download filtered tables, view logs, load saved models, and generate SHAP summaries / scatter plots and residual diagnostics.
3. Reproducible Outputs: Each run produces a structured `output/ExpN` folder with results, logs, metadata, and saved models.

---

## 📦 Key Features

- Streamlit UI for experiment configuration & monitoring
- Supports multiple regressors (RF, ET, XGB, LGB, NN, SVR, LR) with Optuna‑driven hyperparameter search
- Optional feature selection (f_regression or mutual information)
- SHAP value computation & plotting (summary + per‑feature scatter)
- Residual diagnostics for saved models
- Experiment metadata JSON for later reproducibility
- Clean dashboard with ranking, KPIs, filtering, and CSV export

---

## 🗂 Project Structure (Essentials)

```
├── main.py                     # Streamlit application entrypoint
├── scripts/
│   ├── data_processing.py      # Preprocessing & feature selection
│   ├── regression_pipeline.py  # (If used) model/optimization logic
│   ├── analysis.py             # Orchestrates analysis (called by UI)
│   ├── utils.py                # Helpers: I/O, ranking, metadata, etc.
│   └── plotting.py             # Plot utilities (if used)
├── data/                       # Place your CSV datasets here
│   └── Demo_Data.csv           # Demo dataset loaded at startup
├── output/                     # Auto‑generated experiment folders
│   └── Exp1/Exp2/...           # Results, logs, models, metadata
├── pyproject.toml              # Project metadata & dependencies
├── uv.lock                     # (If using uv) lockfile for reproducibility
├── requirements.txt            # Alternative dependency list (pip)
└── README.md
```

---

## ✅ Prerequisites

- Python 3.11 (some libs do not work with 3.12)
- macOS / Windows (tested primarily on macOS)
- (Recommended) [uv](https://github.com/astral-sh/uv) for fast, reproducible installs
- Alternatively: `pip` + virtual environment
- Build tools may be required for packages like `lightgbm` / `xgboost` (macOS: `brew install cmake libomp` if you encounter compilation issues)

---

## 🔧 Environment Setup

### Option A: Using uv (fast & reproducible)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv if missing
cd TIMED_ML_Pipeline
uv sync    # creates and populates .venv from pyproject + uv.lock

# Activate (optional – uv can run commands directly)
source .venv/bin/activate
```

To run without activating:
```bash
uv run streamlit run main.py
```

### Option B: Using pip & venv
```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📊 Running the Streamlit App

From the project root (with environment active):

```bash
streamlit run main.py
```

If you change dependencies, restart the app. To clear Streamlit cache:
```bash
streamlit cache clear
```

---

## 🧪 Configuring & Launching an Analysis

1. Go to the Analysis page (left sidebar radio buttons).
2. Select a dataset from the `data/` directory.
3. Choose targets (one or multiple) (multiple targets may produce unexpected results if data leakage is not addressed).
4. (Optional) Drop features / mark categorical features (important to avoid data leakage).
5. (Optional) Enable feature selection (set >0 number of features & choose scoring) (0 = select all features).
6. Pick regressors and metrics to optimize (mse, mae, r2).
7. Set tuning parameters: iterations, k-fold, n_trials, n_jobs.
8. Click Start Analysis — a per‑target experiment is created sequentially.

Each target launches a threaded optimization loop. Logs stream live in the UI and are saved afterward.

---

## 🗃 Output Folder Anatomy (`output/ExpN/`)

Inside each experiment folder (example with Exp7):

```
Exp7/
	├── results_Exp7.csv          # Aggregated model iteration results
	├── metadata_Exp7.json        # Reproducibility metadata (data path, features, etc.) (important for dashboard)
	├── Log_Exp7.txt              # Full stdout log captured during run
	└── models/                   # Saved joblib model artifacts (trained models)
				XGB_Exp7_mse_1.joblib
				LGB_Exp7_mse_2.joblib
				...
```

File naming pattern for models:
```
<MODEL>_Exp<EXP_ID>_<PRIMARY_METRIC>_<ITERATION>.joblib
```

Metadata fields include: experiment id, timestamp, dataset path, random state, target, transform, dropped features, categorical features, feature count, k-fold value.

---

## 📈 Dashboard Features

- Summary KPIs (best model, averages, runtime aggregation)
- Interactive box/jitter plot of model family performance
- Sort & filter results table (download filtered CSV)
- Model loading: residuals, SHAP summary, per‑feature SHAP scatter
- Log viewer with tail & substring filter + full log download

---

## 🔍 Interpreting Results

- test_MSE / test_MAE / test_R2: Final evaluation metrics for each iteration / tuned model.
- best_* columns: Validation metrics during tuning for the best trial.
- optimization_time: Time spent (seconds) for the underlying tuning loop; aggregated across models.
- best_params: JSON blob of tuned hyperparameters (parsed in UI for the selected best row).

---

## 🧪 Re‑Evaluating Saved Models (Not implemented for LR, NN, and SVR)

On the Dashboard -> Models tab:

1. Select experiment.
2. Pick a model artifact (.joblib).
3. The app rebuilds preprocessing context & feature selection (based on metadata) and recomputes SHAP values on a fresh train/test split (20% test).
4. View residuals + SHAP visualizations.

Note: SHAP computation can be slow for large datasets or deep ensembles; LightGBM / XGBoost are optimized.

---

## 📄 Adding New Data

Place additional CSV files into `data/`. They will appear in the dataset selector automatically (refresh the app if needed).

Dataset expectations:
- A column matching your chosen target(s)
- Numeric / categorical columns (categoricals optionally specified in UI)
- Clean missing values (preprocessing script may need enhancement if heavy NaNs)

---

## 🧾 License

See `LICENSE` file.

---

## 🙌 Acknowledgements

- [Optuna](https://optuna.org/)
- [Streamlit](https://streamlit.io/)
- SHAP, LightGBM, XGBoost.
