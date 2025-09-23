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
import plotly.graph_objects as go
import plotly.colors as pc

from scripts.analysis import run_analysis
from scripts.utils import create_experiment_folders, get_experiments, metric_info, rank, sec_to_hms, safe_fmt, parse_params

st.set_page_config(layout="wide")

# Define directories
current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, 'data')
output_dir = os.path.join(current_dir, 'output')

# Load dataset
data_path = os.path.join(data_dir, 'Demo_Data.csv')
df = pd.read_csv(data_path)
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
    keyboard.press_and_release('ctrl+w')
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
            selected_data2 = st.selectbox("Data", [""] + data_files, label_visibility="collapsed")
            if selected_data2:
                file_path = os.path.join(data_dir, selected_data2)
                df_data2 = pd.read_csv(file_path)
                st.session_state.df = df_data2

                # Show a preview of the dataframe
                with st.expander("Preview Dataset", expanded=False):
                    st.dataframe(st.session_state.df.head(), use_container_width=True)

        with st.expander("2. FEATURES"):
            target = st.multiselect("Target", list(st.session_state.df.columns) + ['DASS_Sum'])
            target_transform = st.selectbox("Target Transform", ["none", "log", "yeo-johnson"])
            features_to_drop = st.multiselect("Features to Drop", st.session_state.df.columns)
            categorical_features = st.multiselect("Categorical Features", st.session_state.df.columns)

        with st.expander("3. FEATURE SELECTION"):
            nb_features = st.number_input("Number of Features", min_value=0, value=10)
            feature_scoring = st.selectbox("Feature Selection Scoring", feat_scores, placeholder="Choose a metric")

        with st.expander("4. REGRESSION"):
            nb_iterations = st.number_input("Number of Iterations", min_value=1, value=3)
            regressors = st.multiselect("Regressors", models, placeholder="Choose the models to optimize")
            metrics = st.multiselect("Metrics", metrics, placeholder="Choose a metric")
            kfold = st.number_input("K-Fold", min_value=2, value=5)
            n_trials = st.number_input("Number of Trials", min_value=10, value=50)
            n_jobs = st.number_input("Number of Jobs", min_value=1, value=4)
            save_models = st.checkbox("Save Models")

    with col2:

        if st.button("Show/Hide Current Configuration"):
            st.session_state.show_config = not st.session_state.show_config

        if st.session_state.show_config:
            config_message = {
                "selected_data":        selected_data2,
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
        
        if not selected_data2:
            st.error("Please select a dataset to start the analysis.")
        
        elif not target:
            st.error("Please select at least one target.")
        
        else:
            # Ensure output directory exists
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Get the number of existing experiments
            n_exps = len(os.listdir(output_dir))
            current_exp = n_exps + 1  # Since we are about to create a new one

            # Create the required folders for the experiment
            experiment_path, models_path, plots_path = create_experiment_folders(output_dir=output_dir, current_exp=current_exp)

            # Load the dataframe once
            file_path = os.path.join(data_dir, selected_data2)
            df_data2 = pd.read_csv(file_path)

            for tgt in target:
                st.subheader(f"🚀 Running analysis for target: **{tgt}**")

                # Prepare a fresh log buffer
                output_buffer = StringIO()
                sys.stdout = output_buffer
                log_display = st.empty()

                # Wrapper to call run_analysis for this single target
                def run_for_target():
                    run_analysis(
                        data                  = df_data2,
                        current_exp           = current_exp,
                        experiment_path       = experiment_path,
                        models_path           = models_path,
                        plots_path            = plots_path,
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
                analysis_thread = threading.Thread(target=run_for_target)
                analysis_thread.start()
                while analysis_thread.is_alive():
                    log_display.text(output_buffer.getvalue())
                    time.sleep(0.1)

                # Final flush of logs
                final_output = output_buffer.getvalue()
                log_display.text(final_output)
                sys.stdout = sys.__stdout__
                
                # Write out the per-experiment log inside the experiment folder
                log_name = f"Log_Exp{current_exp}.txt"
                log_path = os.path.join(output_dir, f"Exp{current_exp}", log_name)
                with open(log_path, "w", encoding="utf-8") as lf:
                    lf.write(final_output)

                st.success(f"✅ Saved log for experiment #{current_exp} at: `{log_path}`")


#####
# Dashboard Page
elif tab_selection == "Dashboard":

    st.header("Experiments Overview")

    # ---------- Helpers ----------
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
    t_summary, t_results, t_plots, t_models, t_logs = st.tabs(
        ["📊 Summary", "📋 Results Table", "🖼️ Plots", "🧠 Models", "📜 Logs"]
    )

    # ======================== SUMMARY ========================
    with t_summary:
        st.subheader(f"Summary — {selected_exp}")

        # Metric selector
        metric_choice = st.radio(
            "Primary metric",
            options=["R² (test_R2)", "MAE (test_MAE)", "MSE (test_MSE)"],
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
                # show mean metric
                try:
                    mean_val = np.nanmean(df_results[metric_col].astype(float))
                    st.metric(f"Avg {metric_label}", f"{mean_val:,.4f}")
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
                st.metric(f"Best {metric_label}", best_str)

            # Best model card
            with st.expander("🏆 Best model details", expanded=True):
                left, right = st.columns([1.2, 1])
                with left:
                    st.write(
                        f"**Model**: `{best_row.get('model_name', '—')}`  |  "
                        f"**ID**: `{best_row.get('model_id', '—')}`"
                    )
                    st.write(
                        f"**Target**: `{best_row.get('target', '—')}`  |  "
                        f"**Transform**: `{best_row.get('target_transform', '—')}`  |  "
                        f"**Trials**: `{best_row.get('n_trials', '—')}`"
                    )
                    st.write(
                        f"**Train**: R² {safe_fmt(best_row.get('train_R2'))}, "
                        f"MAE {safe_fmt(best_row.get('train_MAE'))}, "
                        f"MSE {safe_fmt(best_row.get('train_MSE'))}"
                    )
                    st.write(
                        f"**Test**:  R² {safe_fmt(best_row.get('test_R2'))}, "
                        f"MAE {safe_fmt(best_row.get('test_MAE'))}, "
                        f"MSE {safe_fmt(best_row.get('test_MSE'))}"
                    )
                with right:
                    # Show best_params nicely
                    params = parse_params(best_row.get("best_params", {}))
                    st.json(params)

            # Quick chart of top models by primary metric
            top_n = st.slider("Show top N in chart", min_value=3, max_value=min(30, len(ranked)), value=min(10, len(ranked)))
            to_plot = ranked.head(top_n)[["model_name", metric_col]].copy()
            to_plot.rename(columns={"model_name": "Model", metric_col: metric_label}, inplace=True)

            direction = "descending" if higher_is_better else "ascending"
            chart = (
                alt.Chart(to_plot)
                .mark_bar()
                .encode(
                    x=alt.X(f"{metric_label}:Q", sort=direction),
                    y=alt.Y("Model:N", sort="-x"),
                    tooltip=[alt.Tooltip("Model:N"), alt.Tooltip(f"{metric_label}:Q", format=".4f")],
                )
                .properties(height=28 * len(to_plot), width="container")
            )
            st.altair_chart(chart, use_container_width=True)

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

            st.dataframe(view_df, use_container_width=True, hide_index=True)

            # Download filtered CSV
            csv_bytes = view_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download filtered results (CSV)",
                data=csv_bytes,
                file_name=f"{selected_exp}_results_filtered.csv",
                mime="text/csv",
            )

    # ======================== PLOTS ========================
    with t_plots:
        st.subheader("Plots")
        if os.path.isdir(plots_dir):
            images = []
            for ext in ("*.png","*.jpg","*.jpeg","*.svg"):
                images.extend(glob.glob(os.path.join(plots_dir, ext)))
            if images:
                cols = st.columns(3)
                for i, img_path in enumerate(sorted(images)):
                    with cols[i % 3]:
                        st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
            else:
                st.info("No plot images found yet in the `plots/` folder.")
        else:
            st.info("`plots/` folder not found for this experiment.")

    # ======================== MODELS ========================
    with t_models:
        st.subheader("Models")
        if os.path.isdir(models_dir):
            files = sorted(glob.glob(os.path.join(models_dir, "*.joblib")))
            if not files:
                st.info("No .joblib models saved.")
            else:
                sizes = [os.path.getsize(f) for f in files]
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Saved models", len(files))
                with c2:
                    total_mb = sum(sizes) / (1024**2)
                    st.metric("Disk usage", f"{total_mb:.2f} MB")
                with st.expander("Show file list"):
                    st.write("\n".join(os.path.basename(f) for f in files))
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