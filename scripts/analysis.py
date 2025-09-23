from scripts.data_processing import feature_selection, data_processing
from scripts.regression_pipeline import regression
from scripts.utils import export_results, format_results, format_elapsed_time, create_experiment_folders

import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split


def run_analysis(data, current_exp, experiment_path, models_path, plots_path, target, target_transform, features_to_drop, categorical_features, n_features, feature_scoring, models, n_iter, k, opti_scoring, n_trials, n_jobs=4, save_models=False):
    
    # Start timer for the analysis
    start_time = time.time()

    # Log current date and time for the analysis
    print("\n" + "="*90)
    print(f'📆 Date: {datetime.now().strftime("%d/%m/%Y")}')
    print(f'⏰ Time: {datetime.now().strftime("%H:%M:%S")}')
    print("="*90)
    # Log function configuration
    print("\n" + "="*90)
    print("🔧 CONFIGURATION PARAMETERS")
    print("="*90)
    print(f"🧪 Experiment Number      : {current_exp}")
    print(f"📂 Experiment Path        : {experiment_path}")
    print(f"📂 Models Path            : {models_path}")
    print(f"📂 Plots Path             : {plots_path}")
    print(f"🎯 Target Variable        : {target}")
    print(f"🔄 Target Transform       : {target_transform}")
    print(f"🗑️ Features to Drop       : {features_to_drop}")
    print(f"🔢 Categorical Features   : {categorical_features}")
    print(f"🌟 Number of Features     : {n_features}")
    print(f"🔍 Feature Scoring Method : {feature_scoring}")
    print(f"🔄 Iterations per Model   : {n_iter}")
    print(f"🔢 K-Fold Value           : {k}")
    print(f"🎯 Optimization Scoring   : {opti_scoring}")
    print(f"🧪 Trials per Model       : {n_trials}")
    print(f"🔧 Number of Jobs         : {n_jobs}")
    print(f"💾 Save Models            : {save_models}")
    print("="*90)

    print("="*90)

    # Data loading and processing
    processed_data = data_processing(data, target=target, features_to_drop=features_to_drop, categorical_features=categorical_features)

    # Split data into features and target
    X = processed_data.drop(columns=[target])
    y = processed_data[target]

    # Feature selection if selected
    if n_features != 0:
        selected_features = feature_selection(X, y, n_features, score_func=feature_scoring)
        X = X[selected_features]
    
    # Regression
    regression_results = regression(
        models=models, X=X, y=y, target=target, target_transform=target_transform,
        current_exp=current_exp, models_exp_folder=models_path,
        n_iter=n_iter, k=k, scoring=opti_scoring, n_trials=n_trials, n_jobs=n_jobs, save_models=save_models
    )

    # Format results and convert to metrics
    formatted_results = format_results(regression_results)

    # Export results
    export_results(results=formatted_results, current_exp=current_exp, experiment_path=experiment_path)

    # End timer for the analysis
    end_time = time.time()
    elapsed_time = format_elapsed_time(start_time, end_time)

    # Log elapsed time
    print("\n" + "="*90)
    print(f"⏱️ Elapsed Time: {elapsed_time}")
    print("="*90)