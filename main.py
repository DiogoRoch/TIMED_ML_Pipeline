import os
import sys
import time
import pandas as pd
import streamlit as st
import psutil
import keyboard
from io import StringIO
import threading
import plotly.graph_objects as go
import plotly.colors as pc

from scripts.analysis import run_analysis

st.set_page_config(layout="wide")

# Define directories
current_dir = os.path.dirname(__file__)
results_dir = os.path.join(current_dir, 'results')
data_dir = os.path.join(current_dir, 'data')

# Load dataset
data_path = os.path.join(data_dir, 'WP1_Switzerland_processed.csv')
df = pd.read_csv(data_path)
if "df" not in st.session_state:
    st.session_state.df = df

# Load lists for multiselect options
models = ['RF', 'ET', 'XGB', 'LGB', 'NN', 'SVR', 'LR']
metrics = ['mse', 'mae', 'r2']
feat_scores = ['f_regression', 'mutual_info_regression']

# Load files
try:
    results_files = os.listdir(results_dir)
except FileNotFoundError:
    results_files = []
try:
    data_files = os.listdir(data_dir)
except FileNotFoundError:
    data_files = []

# Title of the Streamlit app
st.title("CSV Viewer & ML Analysis")

# Create radio buttons in the sidebar for tab selection
tab_selection = st.sidebar.radio("Select a page", ["Analysis", "CSV Viewer", "Results Plotting"])

# Button to close the Streamlit app
exit_app = st.sidebar.button("Shut Down")
if exit_app:
    time.sleep(1)
    keyboard.press_and_release('ctrl+w')
    pid = os.getpid()
    p = psutil.Process(pid)
    p.terminate()

# CSV Viewer Page
if tab_selection == "CSV Viewer":
    # Expander for the csv data files
    with st.expander('Data CSV Files'):
        selected_data1 = st.selectbox("Data1", [""] + data_files, label_visibility="collapsed")

        if selected_data1:
            file_path = os.path.join(data_dir, selected_data1)
            df_data1 = pd.read_csv(file_path)
            st.dataframe(df_data1)

    with st.expander('Results CSV Files'):
        selected_result = st.selectbox("Results", [""] + results_files, label_visibility="collapsed")

        if selected_result:
            file_path = os.path.join(results_dir, selected_result)
            df_result = pd.read_csv(file_path)
            st.dataframe(df_result)

# Analysis Page
elif tab_selection == "Analysis":
    st.header("Configure the Machine Learning Analysis")

    if "show_config" not in st.session_state:
        st.session_state.show_config = False

    col1, col2 = st.columns([1.5, 1])

    with col1:
        with st.expander("0. DATASET"):
            selected_data2 = st.selectbox("Data", [""] + data_files, label_visibility="collapsed")
            if selected_data2:
                file_path = os.path.join(data_dir, selected_data2)
                df_data2 = pd.read_csv(file_path)
                st.session_state.df = df_data2

        with st.expander("1. PATHS"):
            out_models_data_folder = st.text_input("Out Models Data Folder", "models")
            out_results_data_folder = st.text_input("Out Results Data Folder", "results")

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
                "selected_data": selected_data2,
                "out_models_data_folder": out_models_data_folder,
                "out_results_data_folder": out_results_data_folder,
                "target": target,
                "target_transform": target_transform,
                "features_to_drop": features_to_drop,
                "categorical_features": categorical_features,
                "nb_features": nb_features,
                "feature_scoring": feature_scoring,
                "nb_iterations": nb_iterations,
                "regressors": regressors,
                "metrics": metrics,
                "kfold": kfold,
                "n_trials": n_trials,
                "n_jobs": n_jobs,
                "save_models": save_models
            }
            st.write("### Current Configuration")
            st.json(config_message)

    if st.button("Start Analysis"):
        if not selected_data2:
            st.error("Please select a dataset to start the analysis.")
        elif not target:
            st.error("Please select at least one target.")
        else:
            # Ensure logs folder exists
            logs_dir = os.path.join(current_dir, 'logs')
            os.makedirs(logs_dir, exist_ok=True)

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
                        data=df_data2,
                        results_folder=out_results_data_folder,
                        models_folder=out_models_data_folder,
                        target=tgt,
                        target_transform=target_transform,
                        features_to_drop=features_to_drop,
                        categorical_features=categorical_features,
                        n_features=nb_features,
                        feature_scoring=feature_scoring,
                        models=regressors,
                        n_iter=nb_iterations,
                        k=kfold,
                        opti_scoring=metrics,
                        n_trials=n_trials,
                        n_jobs=n_jobs,
                        save_models=save_models
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

                # Get the number of existing results files
                results_list = os.listdir(results_dir)
                exp_counter = len(results_list) # Without +1 because the log file is created after the analysis and so the results list is already incremented
                # Write out the per-experiment log
                log_name = f"analysis_log_Exp{exp_counter}.txt"
                log_path = os.path.join(logs_dir, log_name)
                with open(log_path, "w", encoding="utf-8") as lf:
                    lf.write(final_output)

                st.success(f"✅ Saved log for experiment #{exp_counter} at: `{log_path}`")

# Model Evaluation Page (choose a model joblib file, choose a dataset -> plot residuals, plot shap values)
elif tab_selection == "Model Evaluation":
    st.header("Results Visualization")