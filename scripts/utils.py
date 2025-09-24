# File that contains utility functions for the regression task
    # Getshap function
    # Results formatting function
    # Results exporting to csv function

import os, json, ast, time
import shap
import pandas as pd
import numpy as np
from datetime import timedelta
from pycaret.regression import RegressionExperiment
from sklearn.pipeline import Pipeline


def performances_dict_to_df(performances):
    """
    Converts a dictionary of model performances into a pandas DataFrame.

    Parameters:
    -----------
    performances (dict): Dictionary of model performances returned by get_model_performances.

    Returns:
    --------
    pd.DataFrame: DataFrame with a single row for each model, containing metrics as columns.
    """
    
    rows = []
    for model_name, metrics in performances.items():
        row = {'model_name': model_name}
        for metric, stats in metrics.items():
            row[f'{metric}_train_mean'] = stats['train_mean']
            row[f'{metric}_train_std'] = stats['train_std']
            row[f'{metric}_val_mean'] = stats['val_mean']
            row[f'{metric}_val_std'] = stats['val_std']
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def get_model_performances(exp: RegressionExperiment , models: list):
    """
    Takes a regression experiment from PyCaret and a list of models and runs
    cross-validation over multiple metrics. Then returns a dictionary containing
    for each tested model: the mean and std of the cross-validation.

    Parameters
    ----------
    exp (RegressionExperiment): Experiment from PyCaret
    models (list): List of model estimators to train and evaluate

    Returns: Dictionary of performances for train/val sets for all models
        model_name (e.g., names of the estimators)
            metric (e.g, MAE, MSE, R2, RMSE, RMSLE, MAPE, ...)
                train_mean (mean of the cross-validation on the train sets)
                train_std (standard deviation of the cv on the train sets)
                val_mean (mean of the cv on the validation sets)
                val_std (standard deviation of the cv on the validation sets)
    """

    performances = {}
    for model in models:
        model_name = type(model).__name__ if not isinstance(model, str) else model
        performances[model_name] = {}
        current_model = exp.create_model(model, return_train_score=True, verbose=False)
        current_results = exp.pull()
        current_train = current_results.loc['CV-Train']
        current_val = current_results.loc['CV-Val']
        for metric in current_results.columns:
            performances[model_name][metric] = {}
            performances[model_name][metric]['train_mean'] = current_train.loc['Mean'][metric]
            performances[model_name][metric]['train_std'] = current_train.loc['Std'][metric]
            performances[model_name][metric]['val_mean'] = current_val.loc['Mean'][metric]
            performances[model_name][metric]['val_std'] = current_val.loc['Std'][metric]

    return performances


def generate_dynamic_hidden_layers(trial, max_layers=10, max_units_per_layer=1024):
    """
    Generate dynamic hidden layer configurations for MLP.
    Args:
        trial: Optuna trial object.
        max_layers: Maximum number of layers.
        max_units_per_layer: Maximum number of units in each layer.
    Returns:
        Tuple representing the layer sizes.
    """
    # Decide the number of layers
    n_layers = trial.suggest_int('NN__n_layers', 1, max_layers)
    
    # Generate layer sizes
    layers = []
    for i in range(n_layers):
        units = trial.suggest_int(f'NN__units_layer_{i+1}', 32, max_units_per_layer, step=32)
        layers.append(units)
    
    return tuple(layers)


def get_shap_values(model_name, pipe, X_test):

    # Check if pipe is a pipeline or a single model
    if isinstance(pipe, Pipeline):
        model = pipe[model_name]
    else:
        model = pipe

    if model_name in ["RF", "ET", "XGB", "LGB"]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        return shap_values
    
    elif model_name == "LR":
        explainer = shap.LinearExplainer(model, X_test)
        shap_values = explainer.shap_values(X_test)
        return shap_values
    
    else:
        raise ValueError(f"Model {model_name} not recognized.")
    

def plot_shap_values(shap_values, X_test, bar=False):

    if bar:
        shap.summary_plot(shap_values, X_test, plot_type="bar")
    else:
        shap.summary_plot(shap_values, X_test)


def results_to_series(results, exp_counter):
    
    results_copy = results.copy()

    # Create model_id (model_name + "_Exp" + str(exp_counter) + _score + _iteration)
    model_id = f"{results_copy['model_name']}_Exp{exp_counter}_{results_copy['score']}_{results_copy['iteration']}"
    # Add the model_id to the results_copy dictionary
    results_copy = {"model_id": model_id, **results_copy}

    # Pop the best_params column to re-insert it later as a JSON string
    best_params = results_copy.pop('best_params')

    series = pd.Series(results_copy)
    series['best_params'] = json.dumps(best_params)

    return series, model_id


def format_results(results):
    
    # Make a copy of the results dataframe
    results_copy = results.copy()

    # Convert the json strings back to dictionaries
    results_copy['best_params'] = results_copy['best_params'].apply(json.loads)

    return results_copy


def create_experiment_folders(output_dir, current_exp):
    """
    Creates the output folder if it does not yet exist and the experiment folder
    """

    # Create the output folder if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Create the experiment folder if it does not exist
    experiment_path = os.path.join(output_dir, f"Exp{current_exp}")
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
        print(f"Created directory: {experiment_path}")
    
    # Create the models folder inside current experiment folder if it does not exist
    models_path = os.path.join(experiment_path, "models")
    if not os.path.exists(models_path):
        os.makedirs(models_path)
        print(f"Created directory: {models_path}")

    # Create the plots folder inside current experiment folder if it does not exist
    plots_path = os.path.join(experiment_path, "plots")
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
        print(f"Created directory: {plots_path}")

    return experiment_path, models_path, plots_path


def export_results(results, current_exp, experiment_path):
    """Exports the results DataFrame to a CSV file in the specified results folder."""

    # Logs start of export
    print("\n" + "="*90)
    print("Exporting results to a CSV file...")

    # Start timer for export process
    start_time = time.time()

    # Create the file name and path
    file_name = f'results_Exp{current_exp}.csv'
    file_path = os.path.join(experiment_path, file_name)

    # Export the results to a CSV file
    results.to_csv(file_path, index=False)
    
    # End timer for export process
    end_time = time.time()
    elapsed_time = format_elapsed_time(start_time, end_time)

    # Log successful export details
    print("="*90)
    print(f"{'File Path':<20}: {file_path}")
    print(f"{'Export Status':<20}: Success 🎉")
    print(f"{'Export Time':<20}: {elapsed_time}")
    print("=" * 90)


def format_elapsed_time(start_time, end_time):
    """Formats the elapsed time between the start and end times in seconds."""

    elapsed_time = end_time - start_time
    
    # Calculate hours, minutes, and seconds
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60

    if hours > 0:
        return f"{hours}h {minutes}m {seconds:.2f}s"
    elif minutes > 0:
        return f"{minutes}m {seconds:.2f}s"
    else:
        return f"{seconds:.2f}s"
    

def get_experiments(output_dir):
    """Lists all experiment folders in the output directory."""

    if not os.path.exists(output_dir):
        return []

    exp_folders = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f)) and f.startswith("Exp")]
    exp_folders.sort(key=lambda x: int(x.replace("Exp", "")))  # Sort by experiment number

    return exp_folders


#### Dashboard helpers
def parse_params(cell):
    if pd.isna(cell):
        return {}
    # Try strict JSON first, then python-literal fallback
    try:
        return json.loads(cell)
    except Exception:
        try:
            return ast.literal_eval(str(cell))
        except Exception:
            return {"raw": str(cell)}

def sec_to_hms(seconds):
    try:
        s = float(seconds)
    except Exception:
        return "—"
    td = timedelta(seconds=s)
    # keep hours:minutes:seconds only
    total_seconds = int(td.total_seconds())
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:d}:{m:02d}:{s:02d}"

def rank(df, metric, higher_is_better):
    if metric not in df.columns:
        return None, df
    ascending = not higher_is_better
    ranked = df.sort_values(metric, ascending=ascending).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked.iloc[0].to_dict(), ranked

def metric_info(key):
    # returns (column_name, higher_is_better, label)
    mapping = {
        "R²": ("test_R2", True, "R²"),
        "MAE": ("test_MAE", False, "MAE"),
        "MSE": ("test_MSE", False, "MSE"),
    }
    return mapping[key]

# helper for safe formatting numbers in text
def safe_fmt(v):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{float(v):,.4f}"
    except Exception:
        return str(v) if v is not None else "—"


def store_metadata(exp_path, current_exp, data_path, target, target_transform, features_to_drop, categorical_features, n_features, kfold):
    """
    Store the metadata for the current experiment in a JSON file.
    The aim of the metadata is to keep track of most parameters and data used so that we can easily evaluate and interpret the models later on.

    Metadata to store:
    - Experiment ID.
    - Date of the experiment.
    - Location of the dataset used.
    - Target variable.
    - Target transformation.
    - Features removed.
    - Categorical features.
    - Feature selection method.
    - Number of features selected.
    - Kfold value.
    """
    
    metadata = {
        "experiment_id": f"experiment_{current_exp}",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_path": data_path,
        "target": target,
        "target_transform": target_transform,
        "features_to_drop": features_to_drop,
        "categorical_features": categorical_features,
        "n_features": n_features,
        "kfold": kfold
    }
    metadata_path = os.path.join(exp_path, f"metadata_Exp{current_exp}.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"✅ Saved metadata for experiment #{current_exp} at: `{metadata_path}`")