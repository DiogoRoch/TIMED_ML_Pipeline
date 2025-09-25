### TODOS
# - [x] Implement feature importance output
# - [x] Implement shap summary output
# - [x] Implement shap scatter output

import os, sys, io, json, ast, glob, textwrap
import time
from datetime import timedelta
from io import StringIO
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
import psutil
import keyboard
import threading
from contextlib import redirect_stdout

import plotly.graph_objects as go
import plotly.colors as pc

from scripts.analysis import run_analysis
from scripts.utils import create_experiment_folders, get_experiments, metric_info, rank, sec_to_hms, safe_fmt, parse_params
from scripts.data_processing import data_processing


st.set_page_config(layout="wide")

# Define directories
current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, 'data')
output_dir = os.path.join(current_dir, 'output')

# Initizalize session state for dataframe
demo_path = os.path.join(data_dir, 'Demo_Data.csv')
df = pd.read_csv(demo_path)
if "df" not in st.session_state:
    st.session_state.df = df

# Load lists for multiselect options
models = ['RF', 'ET', 'XGB', 'LGB', 'NN', 'SVR', 'LR']
metrics = ['mse', 'mae', 'r2']
feat_scores = ['f_regression', 'mutual_info_regression']

# Load files
try:
    data_files = os.listdir(data_dir)
except FileNotFoundError:
    data_files = []

# Title of the Streamlit app
st.title("CSV Viewer & ML Analysis")


#####
# Create radio buttons in the sidebar for tab selection -- Sidebar options (tabs)
tab_selection = st.sidebar.radio("Select a page", ["Dashboard", "Analysis"])

# Button to close the Streamlit app
exit_app = st.sidebar.button("Shut Down")
if exit_app:
    time.sleep(1)
    pid = os.getpid()
    p = psutil.Process(pid)
    p.terminate()


#####
# Analysis Page
elif tab_selection == "Analysis":
    st.header("Configure the Machine Learning Analysis")

    if "show_config" not in st.session_state:
        st.session_state.show_config = False

    col1, col2 = st.columns([1.5, 1])

    with col1:
        with st.expander("1. DATASET"):
            selected_data = st.selectbox("Data", [""] + data_files, label_visibility="collapsed")
            selected_random_state = st.number_input("Random State", min_value=0, value=42)
            if selected_data:
                data_path = os.path.join(data_dir, selected_data)
                df_data = pd.read_csv(data_path)
                st.session_state.df = df_data

                # Show a preview of the dataframe
                with st.expander("Preview Dataset", expanded=False):
                    st.dataframe(st.session_state.df.head(), width="stretch")

        with st.expander("2. FEATURES"):
            target = st.multiselect("Target", list(st.session_state.df.columns) + ['DASS_Sum'])
            target_transform = st.selectbox("Target Transform", ["none", "log", "yeo-johnson"])
            features_to_drop = st.multiselect("Features to Drop", st.session_state.df.columns)
            categorical_features = st.multiselect("Categorical Features", st.session_state.df.columns)

        with st.expander("3. FEATURE SELECTION"):
            nb_features = st.number_input("Number of Features", min_value=0, value=0)
            feature_scoring = st.selectbox("Feature Selection Scoring", feat_scores, placeholder="Choose a metric")

        with st.expander("4. REGRESSION"):
            nb_iterations = st.number_input("Number of Iterations", min_value=1, value=3)
            regressors = st.multiselect("Regressors", models, placeholder="Choose the models to optimize")
            metrics = st.multiselect("Metrics", metrics, placeholder="Choose a metric")
            kfold = st.number_input("K-Fold", min_value=2, value=5)
            n_trials = st.number_input("Number of Trials", min_value=10, value=50)
            n_jobs = st.number_input("Number of Jobs", min_value=1, value=4)
            save_models = st.checkbox("Save Models", value=True)

    with col2:

        if st.button("Show/Hide Current Configuration"):
            st.session_state.show_config = not st.session_state.show_config

        if st.session_state.show_config:
            config_message = {
                "data_path":            data_path,
                "selected_data":        selected_data,
                "random_state":         selected_random_state,
                "output_dir":           output_dir,
                "target":               target,
                "target_transform":     target_transform,
                "features_to_drop":     features_to_drop,
                "categorical_features": categorical_features,
                "nb_features":          nb_features,
                "feature_scoring":      feature_scoring,
                "nb_iterations":        nb_iterations,
                "regressors":           regressors,
                "metrics":              metrics,
                "kfold":                kfold,
                "n_trials":             n_trials,
                "n_jobs":               n_jobs,
                "save_models":          save_models
            }
            st.write("### Current Configuration")
            st.json(config_message)

    if st.button("Start Analysis"):
        
        if not selected_data:
            st.error("Please select a dataset to start the analysis.")
        
        elif not target:
            st.error("Please select at least one target.")
        
        else:
            # Ensure output directory exists
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Get the number of existing experiments
            start_exp_num = len([f for f in os.listdir(output_dir) if f.startswith('Exp')])
            current_exp = int(str(start_exp_num).zfill(3)) # Add leading zeros

            # Load the dataframe once
            data_path = os.path.join(data_dir, selected_data)
            df_data = pd.read_csv(data_path)

            for tgt in target:

                # Increment experiment number
                current_exp += 1

                # Create experiment folder
                experiment_path, models_path = create_experiment_folders(output_dir=output_dir, current_exp=current_exp)

                st.subheader(f"🚀 Running analysis for target: **{tgt}**")

                # Prepare a fresh log buffer
                output_buffer = StringIO()
                log_display = st.empty()

                # Wrapper to call run_analysis for this single target
                def run_for_target():
                    with redirect_stdout(output_buffer):
                        run_analysis(
                            data_path             = data_path,
                            data                  = df_data,
                            random_state          = selected_random_state,
                            current_exp           = current_exp,
                            experiment_path       = experiment_path,
                            models_path           = models_path,
                            target                = tgt,
                            target_transform      = target_transform,
                            features_to_drop      = features_to_drop,
                            categorical_features  = categorical_features,
                            n_features            = nb_features,
                            feature_scoring       = feature_scoring,
                            models                = regressors,
                            n_iter                = nb_iterations,
                            k                     = kfold,
                            opti_scoring          = metrics,
                            n_trials              = n_trials,
                            n_jobs                = n_jobs,
                            save_models           = save_models
                        )

                # start and monitor the thread
                analysis_thread = threading.Thread(target=run_for_target, daemon=True)
                analysis_thread.start()
                while analysis_thread.is_alive():
                    log_display.text(output_buffer.getvalue())
                    time.sleep(0.1)

                # Final flush of logs
                final_output = output_buffer.getvalue()
                log_display.text(final_output)
                output_buffer.close()
                
                # Write out the per-experiment log inside the experiment folder
                log_name = f"Log_Exp{current_exp}.txt"
                log_path = os.path.join(output_dir, f"Exp{current_exp}", log_name)
                with open(log_path, "w", encoding="utf-8") as lf:
                    lf.write(final_output)

                st.success(f"✅ Saved log for experiment #{current_exp} at: `{log_path}`")


#####
# Dashboard Page
elif tab_selection == "Dashboard":

    st.header("Experiments Dashboard")

    # ---------- Helpers ----------
    # Defined here for the caching to work properly
    @st.cache_data(ttl=60, show_spinner=False)
    def load_results_csv(csv_path):
        return pd.read_csv(csv_path)

    @st.cache_data(ttl=60, show_spinner=False)
    def load_text_file(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    # ---------- UI: top controls ----------
    exp_folders = get_experiments(output_dir)

    col1, col2 = st.columns([3,1])
    with col1:
        selected_exp = st.selectbox(
            "Select an experiment",
            options=[""] + exp_folders,
            index=0,
            help=f"Experiments are read from: {output_dir}",
        )
    with col2:
        refresh = st.button("🔄 Refresh list")
        if refresh:
            st.cache_data.clear()

    if not selected_exp:
        if not exp_folders:
            st.info("No experiments found yet. Run an experiment to populate the `outputs/` directory.")
        else:
            st.caption("Pick an experiment to see summary, results, plots, models, and logs.")
        st.stop()

    exp_path = os.path.join(output_dir, selected_exp)
    results_file = os.path.join(exp_path, f"results_{selected_exp}.csv")
    log_file = os.path.join(exp_path, f"Log_{selected_exp}.txt")
    models_dir = os.path.join(exp_path, "models")
    plots_dir = os.path.join(exp_path, "plots")

    # ---------- File existence checks ----------
    if not os.path.exists(results_file):
        st.warning(f"Missing results CSV: `{os.path.basename(results_file)}`")
    if not os.path.exists(log_file):
        st.warning(f"Missing log file: `{os.path.basename(log_file)}`")
    if not (os.path.exists(results_file) and os.path.exists(log_file)):
        st.stop()

    # ---------- Load data ----------
    df_results = load_results_csv(results_file)
    log_content = load_text_file(log_file)

    # Normalize common columns if absent
    for col in ["test_MSE", "test_MAE", "test_R2", "optimization_time"]:
        if col not in df_results.columns:
            df_results[col] = np.nan

    # ---------- Tabs ----------
    t_summary, t_results, t_models, t_logs = st.tabs(
        ["📊 Summary", "📋 Results Table", "🧠 Models", "📜 Logs"]
    )

    # ======================== SUMMARY ========================
    with t_summary:
        st.subheader(f"Summary — {selected_exp}")

        # Metric selector
        metric_choice = st.radio(
            "Primary metric",
            options=["R²", "MAE", "MSE"],
            horizontal=True,
        )
        metric_col, higher_is_better, metric_label = metric_info(metric_choice)

        # Compute best row + ranked df
        best_row, ranked = rank(df_results.copy(), metric_col, higher_is_better)
        if best_row is None:
            st.warning(f"Column `{metric_col}` not found in results.")
        else:
            # KPIs
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Models evaluated", len(df_results))
            with c2:
                # Display the average and std of the chosen metric (test)
                try:
                    mean_val = np.nanmean(df_results[metric_col].astype(float))
                    std_val = np.nanstd(df_results[metric_col].astype(float))
                    if metric_choice == "R²":
                        st.metric(f"Avg Test {metric_label}", f"{mean_val:,.4f} ± {std_val:.4f}")
                    else:
                        st.metric(f"Avg Test {metric_label}", f"{mean_val:,.4f}")
                except Exception:
                    st.metric(f"Avg {metric_label}", "—")
            with c3:
                # total optimization time if available
                try:
                    total_t = float(df_results["optimization_time"].fillna(0).astype(float).sum())
                    st.metric("Total optimization time", sec_to_hms(total_t))
                except Exception:
                    st.metric("Total optimization time", "—")
            with c4:
                best_val = best_row.get(metric_col, None)
                best_str = f"{best_val:,.4f}" if isinstance(best_val, (int, float, np.floating)) else str(best_val)
                st.metric(f"Best Model Test {metric_label}", best_str)

            # Best model card
            with st.expander("🏆 Best model details", expanded=True):
                left, right = st.columns([1.2, 1])
                with left:
                    # Compact, scannable model info (still just text)
                    st.markdown(
                        f"""
            **Model:** `{best_row.get('model_name', '—')}`  
            **ID:** `{best_row.get('model_id', '—')}`  

            **Target:** `{best_row.get('target', '—')}`  
            **Transform:** `{best_row.get('target_transform', '—')}`  
            **Trials:** `{best_row.get('n_trials', '—')}`
                        """
                    )

                    # Metrics as a readable markdown table (no new components)
                    tr2  = safe_fmt(best_row.get("train_R2"))
                    tmae = safe_fmt(best_row.get("train_MAE"))
                    tmse = safe_fmt(best_row.get("train_MSE"))

                    vr2  = safe_fmt(best_row.get("best_R2"))
                    vmae = safe_fmt(best_row.get("best_MAE"))
                    vmse = safe_fmt(best_row.get("best_MSE"))

                    sr2  = safe_fmt(best_row.get("test_R2"))
                    smae = safe_fmt(best_row.get("test_MAE"))
                    smse = safe_fmt(best_row.get("test_MSE"))

                    st.markdown("**📊 Performance (by split)**")
                    st.markdown(
                        f"""
                        | Split | R² | MAE | MSE |
                        |:----:|:--:|:---:|:---:|
                        | **Train** | **{tr2}** | **{tmae}** | **{tmse}** |
                        | **Validation** | **{vr2}** | **{vmae}** | **{vmse}** |
                        | **Test** | **{sr2}** | **{smae}** | **{smse}** |
                        """
                    )
                    st.caption("R² ↑ higher is better · MAE/MSE ↓ lower is better")
                with right:
                    # Display the best hyperparameters as JSON
                    st.write("**Best Hyperparameters**")
                    params = parse_params(best_row.get("best_params", {}))
                    st.json(params)

            # Summary chart of model performances on the chosen metric with plotly
            df = ranked.copy()

            # precompute medians for overlay
            medians = df.groupby("model_name")[metric_col].median().sort_values(ascending=not higher_is_better)
            family_order = medians.index.tolist()

            # build figure: one box+jitter per family, colored consistently; overlay medians as diamonds
            fig = go.Figure()
            palette = pc.qualitative.Plotly

            for i, fam in enumerate(family_order):
                fam_df = df[df["model_name"] == fam]
                fig.add_trace(go.Box(
                    x=[fam] * len(fam_df),
                    y=fam_df[metric_col],
                    name=fam,
                    boxpoints="all",          # show individual iterations
                    jitter=0.35,              # spread the points
                    pointpos=0,               # center the points on the box
                    marker=dict(
                        color=palette[i % len(palette)],
                        size=6,
                        line=dict(width=0)
                    ),
                    line=dict(width=1),
                    hovertext=fam_df["model_name"],
                    hovertemplate="<b>%{hovertext}</b><br>" + metric_label + ": %{y:.4f}<extra></extra>",
                    showlegend=False,
                ))

            # median overlay
            fig.add_trace(go.Scatter(
                x=family_order,
                y=medians.values,
                mode="markers+text",
                name="Median",
                marker_symbol="diamond",
                marker_size=10,
                marker_line_width=0,
                text=[f"{v:,.4f}" for v in medians.values],
                textposition="top center",
                hovertemplate="<b>%{x}</b><br>Median " + metric_label + ": %{y:.4f}<extra></extra>",
            ))

            fig.update_layout(
                title=f"Model type performances by Test {metric_label} (iterations shown as jittered points)",
                xaxis_title="Model type",
                yaxis_title=metric_label,
                xaxis_tickangle=-30,
                xaxis=dict(categoryorder="array", categoryarray=family_order),
                height=max(420, 120 + 60 * len(family_order)),
                margin=dict(l=40, r=20, t=70, b=80),
            )

            st.plotly_chart(fig, width="stretch", theme="streamlit")


    # ======================== RESULTS TABLE ========================
    with t_results:
        st.subheader("Results")

        # Quick filters
        with st.expander("🔎 Filter & view options", expanded=False):
            colf1, colf2, colf3 = st.columns([2,2,1])
            with colf1:
                name_filter = st.text_input("Model name contains", "")
            with colf2:
                target_filter = st.text_input("Target equals (exact match)", "")
            with colf3:
                show_params = st.checkbox("Show params column", value=False)

            # Sort by chosen metric automatically
            sort_df = df_results.copy()
            if metric_col in sort_df.columns:
                sort_df = sort_df.sort_values(metric_col, ascending=not higher_is_better)

            if name_filter:
                sort_df = sort_df[sort_df["model_name"].astype(str).str.contains(name_filter, case=False, na=False)]
            if target_filter:
                sort_df = sort_df[sort_df["target"].astype(str) == target_filter]

            # Hide very long columns unless opted in
            base_cols = [
                "model_id","model_name","iteration","target","target_transform","n_trials",
                "train_R2","train_MAE","train_MSE",
                "test_R2","test_MAE","test_MSE",
                "best_R2","best_MAE","best_MSE",
                "optimization_time","score"
            ]
            extra_cols = ["best_params"] if show_params else []
            display_cols = [c for c in base_cols + extra_cols if c in sort_df.columns]
            view_df = sort_df[display_cols].copy()

            # Humanize optimization_time
            if "optimization_time" in view_df.columns:
                view_df["optimization_time"] = view_df["optimization_time"].apply(sec_to_hms)

            st.dataframe(view_df, width="stretch", hide_index=True)

            # Download filtered CSV
            csv_bytes = view_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download filtered results (CSV)",
                data=csv_bytes,
                file_name=f"{selected_exp}_results_filtered.csv",
                mime="text/csv",
            )

    # ======================== MODELS ========================
    with t_models:
        st.subheader("Model Evaluation & Interpretation")
        if os.path.isdir(models_dir):
            model_files = sorted(glob.glob(os.path.join(models_dir, "*.joblib")))
            if not model_files:
                st.info("No .joblib models saved.")
            else:
                # Selectbox to pick a model file
                selected_model_file = st.selectbox(
                    "Select a model file",
                    options=[""] + model_files,
                    format_func=lambda x: os.path.basename(x),
                    index=0,
                )

                # Find model type by splitting filename per underscore, e.g., LGB_Exp1_mse_1.joblib
                model_type = selected_model_file.split("/")[-1].split("_")[0]
                
                # Model evaluation and interpretation
                # - Residuals plot
                # - Shap summary plot
                # - Shap scatter plot for a selected feature
                if selected_model_file:
                    from joblib import load
                    import shap
                    import matplotlib.pyplot as plt
                    from sklearn.model_selection import train_test_split

                    # Load metadata from JSON file
                    metadata_file = os.path.join(exp_path, f"metadata_{selected_exp}.json")
                    if not os.path.exists(metadata_file):
                        st.error(f"Metadata file not found: `{metadata_file}`")
                        st.stop()
                    with open(metadata_file, "r", encoding="utf-8") as mf:
                        metadata = json.load(mf)
                    
                    data_path =             metadata.get("data_path", None)
                    random_state =          metadata.get("random_state", 42)
                    target_var =            metadata.get("target", None)                
                    features_to_drop =      metadata.get("features_to_drop", [])
                    categorical_features =  metadata.get("categorical_features", [])
                    n_features =            metadata.get("n_features", 0)
                    feature_scoring =       metadata.get("feature_scoring", "f_regression")
                    
                    # Load dataset
                    data_interp = pd.read_csv(data_path)
                    processed_data = data_processing(
                        data=data_interp,
                        target=target_var,
                        features_to_drop=features_to_drop,
                        categorical_features=categorical_features,
                        verbose=False
                    )
                    X = processed_data.drop(columns=[target_var])
                    y = processed_data[target_var]

                    # Feature selection if applied during training
                    if n_features != 0:
                        from scripts.data_processing import feature_selection
                        selected_features = feature_selection(X, y, n_features, score_func=feature_scoring)
                        X = X[selected_features]

                    # Split data into train and test for evaluation
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

                    # Load model
                    model = load(selected_model_file)
                    st.write(f"Loaded model: `{os.path.basename(selected_model_file)}`")

                    # Compute SHAP values
                    explainer = shap.TreeExplainer(model[model_type])
                    shap_values = explainer(X_test)

                    # Tabs for:
                    # - Residuals plot
                    # - Shap summary plot
                    # - Shap scatter plot for a selected feature
                    tab1, tab2, tab3 = st.tabs(["📈 Residuals Plot", "🔍 SHAP Summary Plot", "🔍 SHAP Scatter Plot"])

                    # Model residuals plot
                    with tab1:
                        st.write("### Residuals Plot")
                        y_pred = model.predict(X_test)
                        residuals = y_test - y_pred
                        
                        fig_res, ax = plt.subplots()
                        ax.scatter(y_pred, residuals, alpha=0.5)
                        ax.axhline(0, color='red', linestyle='--')
                        ax.set_xlabel("Predicted Values")
                        ax.set_ylabel("Residuals")
                        ax.set_title("Residuals vs Predicted Values")
                        st.pyplot(fig_res, clear_figure=True)
                        plt.close(fig_res)
                    
                    # Shap summary plot
                    with tab2:
                        st.write(f"### SHAP Summary Plot — Model type: **{model_type}**")

                        fig_shap, ax = plt.subplots()
                        shap.summary_plot(shap_values, X_test, show=False)
                        st.pyplot(fig_shap, clear_figure=True)
                        plt.close(fig_shap)
                    
                    # Shap scatter plot for a selected feature
                    with tab3:
                        selected_feature = st.selectbox(
                            "Select a feature for SHAP scatter plot",
                            options=[""] + list(X.columns),
                            index=0,
                        )
                        if selected_feature:
                            st.write(f"### SHAP Scatter Plot for feature: **{selected_feature}**")
                            feature_loc = X_test.columns.get_loc(selected_feature)
                            
                            fig_scatter, ax = plt.subplots()
                            shap.plots.scatter(shap_values[:, feature_loc], ax=ax, show=False)
                            st.pyplot(fig_scatter, clear_figure=True)
                            plt.close(fig_scatter)

        else:
            st.info("`models/` folder not found for this experiment.")

    # ======================== LOGS ========================
    with t_logs:
        st.subheader("Log file")
        # Controls
        c1, c2, c3 = st.columns([2,1,1])
        with c1:
            log_filter = st.text_input("Filter lines (contains)", "")
        with c2:
            tail_n = st.number_input("Tail (last N lines)", min_value=0, value=500, step=100,
                                     help="0 shows full log (may be slow for very large logs).")
        with c3:
            case_sensitive = st.checkbox("Case sensitive", value=False)

        lines = log_content.splitlines()
        if tail_n > 0:
            lines = lines[-tail_n:]
        if log_filter:
            cmp = (lambda s: log_filter in s) if case_sensitive else (lambda s: log_filter.lower() in s.lower())
            lines = [ln for ln in lines if cmp(ln)]

        display_text = "\n".join(lines) if lines else "(no matching lines)"
        st.text_area("Log", value=display_text, height=350, label_visibility="collapsed")

        st.download_button(
            "⬇️ Download full log",
            data=log_content.encode("utf-8"),
            file_name=f"Log_{selected_exp}.txt",
            mime="text/plain",
        )