# File that contains the code for the main regression pipeline and the training of the models
import optuna
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import time
import numpy as np
import pandas as pd
import warnings

from scripts.utils import results_to_series, format_elapsed_time

# To ignore warnings
warnings.filterwarnings("ignore")
# To ignore optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)



def regression(models, X, y, target, n_iter=3, k=5, scoring=['mse'], n_trials=50, n_jobs=4):

    # Start Logs Printing
    # Dataset statistics
    print("\n" + "="*90)
    print("📊 DATASET STATISTICS")
    print("="*90)
    print(f" - Number of Samples/Subjects: {X.shape[0]}")
    print(f" - Target Variable to Predict : {target}")
    print("="*90 + "\n")
    # Models to be optimized
    print("="*90)
    print("🔍 MODELS TO BE OPTIMIZED & EVALUATED")
    print("="*90)
    for model_name in models:
        print(f" - {model_name}")
    print("="*90 + "\n")

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Initializing dataframes
    results_df = pd.DataFrame()

    # Model-wise loop
    for model_name in models:
        print(f"\n🚀 Starting optimization for model: {model_name}")
        print("="*90)

        for score in scoring:
            print(f"🎯 Scoring Metric: {score.upper()}")
            print("="*90)
            
            # Iteration-wise loop
            for i in range(n_iter):
                print(f"  ➡️  Iteration {i+1}/{n_iter} for model: {model_name}")
                start_time = time.time()

                # Optimize and evaluate model
                i_results = optimize_and_evaluate_model(model_name, i+1, target, X_train, y_train, X_test, y_test, k, score, n_trials, n_jobs)
                i_series = results_to_series(i_results)
                results_df = pd.concat([results_df, pd.DataFrame([i_series])], ignore_index=True)
                end_time = time.time()
                
                # Log iteration completion time
                elapsed_time = format_elapsed_time(start_time, end_time)
                current_time = time.strftime("%H:%M:%S", time.localtime())
                print(f"  ✅ Iteration {i+1}/{n_iter} for {model_name} completed in {elapsed_time}  |  {current_time}")
            
        print(f"🎉 Optimization for {model_name} completed.")
        print("="*90 + "\n")

    print("🏁 All model optimizations are complete!\n")
    return results_df


def optimize_and_evaluate_model(model_name, iteration, target, X_train, y_train, X_test, y_test, k, scoring, n_trials, n_jobs):
    
    objective, pipe = get_model(model_name, X_train, y_train, k, scoring)
    study = optuna.create_study(direction='minimize')
    
    # Start timer for optimization
    start_time = time.time()
    
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    best_params = study.best_params
    #best_score = study.best_value

    # Update pipeline with the best found parameters and fit it to the training data
    pipe.set_params(**best_params)

    # Do cross-validation with the best parameters on the train set
    val_mse, val_mae, val_r2 = cross_val_scores(pipe, X_train, y_train, k)

    # Fit the model to the entire training data
    pipe.fit(X_train, y_train)
    
    # Get predictions for both training and test sets
    train_preds = pipe.predict(X_train)

    # Get the scores for the best trial during optimization
    best_trial_scores = study.best_trial.user_attrs
    best_mse = best_trial_scores['MSE']
    best_mae = best_trial_scores['MAE']
    best_r2 = best_trial_scores['R2']

    # Evaluate the model on the train set
    train_mse = mean_squared_error(y_train, train_preds)
    train_mae = mean_absolute_error(y_train, train_preds)
    train_r2 = r2_score(y_train, train_preds)

    # Compute the differnce between train and test set metrics
    diff_mse = val_mse - train_mse
    diff_mae = val_mae - train_mae
    diff_r2 = val_r2 - train_r2

    # End timer for optimization and evaluation
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Store the results in a dictionary
    results = {
        'model_name': model_name,
        'iteration': iteration,
        'target': target,
        'n_trials': n_trials,
        'score': scoring,
        'best_params': best_params,
        'best_MSE': best_mse,
        'best_MAE': best_mae,
        'best_R2': best_r2,
        'train_MSE': train_mse,
        'train_MAE': train_mae,
        'train_R2': train_r2,
        'val_MSE': val_mse,
        'val_MAE': val_mae,
        'val_R2': val_r2,
        'tt_diff_MSE': diff_mse,
        'tt_diff_MAE': diff_mae,
        'tt_diff_R2': diff_r2,
        'optimization_time': elapsed_time
    }

    return results


def get_model(model_name, X_train, y_train, k, scoring='mse'):
    """Get the model pipeline and the objective function for Optuna."""

    # Determine if scaler is needed
    use_scaler = model_name in ['NN', 'SVR', 'LR']
    
    # Create a pipeline with placeholder model (no hyperparameters)
    if model_name == 'RF':
        model = RandomForestRegressor()
    elif model_name == 'XGB':
        model = XGBRegressor()
    elif model_name == 'NN':
        model = MLPRegressor()
    elif model_name == 'LR':
        model = LinearRegression()
    elif model_name == 'SVR':
        model = SVR()
    
    def objective(trial):
        # Fetch model-specific hyperparameters
        params = get_hyperparameters(trial, model_name)
        
        # Create and Set parameters to pipeline
        pipeline = create_pipeline(model_name, model, use_scaler)
        pipeline.set_params(**params)
        
        # Perform cross-validation and calculate the metrics
        mse, mae, r2 = cross_val_scores(pipeline, X_train, y_train, k)

        # Log all the metrics
        trial.set_user_attr('MSE', mse)
        trial.set_user_attr('MAE', mae)
        trial.set_user_attr('R2', r2)

        # Return the score to minimize (choice between mse, mae, and r2)
        if scoring == 'mse':
            return mse
        elif scoring == 'mae':
            return mae
        elif scoring == 'r2':
            return r2

    pipeline = create_pipeline(model_name, model, use_scaler)

    # Return both the objective function and the pipeline
    return objective, pipeline


def get_hyperparameters(trial, model_name):
    """Return model-specific hyperparameter suggestions for Optuna."""

    if model_name == 'RF':
        return {
            'RF__n_estimators': trial.suggest_int('RF__n_estimators', 100, 300),
            'RF__max_depth': trial.suggest_int('RF__max_depth', 5, 20),
            'RF__min_samples_split': trial.suggest_int('RF__min_samples_split', 2, 10),
            'RF__min_samples_leaf': trial.suggest_int('RF__min_samples_leaf', 1, 4),
            'RF__bootstrap': trial.suggest_categorical('RF__bootstrap', [True, False])
        }
    elif model_name == 'XGB':
        return {
            'XGB__n_estimators': trial.suggest_int('XGB__n_estimators', 100, 500),
            'XGB__max_depth': trial.suggest_int('XGB__max_depth', 3, 15),
            'XGB__gamma': trial.suggest_float('XGB__gamma', 0, 5),
            'XGB__reg_alpha': trial.suggest_int('XGB__reg_alpha', 0, 50),
            'XGB__reg_lambda': trial.suggest_float('XGB__reg_lambda', 0, 5),
            'XGB__colsample_bytree': trial.suggest_float('XGB__colsample_bytree', 0.5, 1),
            'XGB__min_child_weight': trial.suggest_int('XGB__min_child_weight', 0, 10)
        }
    elif model_name == 'NN':
        return {
            'NN__solver': trial.suggest_categorical('NN__solver', ['adam']),
            'NN__activation': trial.suggest_categorical('NN__activation', ['relu', 'tanh']),
            'NN__alpha': trial.suggest_float('NN__alpha', 1e-5, 1e-2, log=True),
            'NN__learning_rate_init': trial.suggest_float('NN__learning_rate_init', 1e-4, 1e-2, log=True),
            'NN__hidden_layer_sizes': trial.suggest_categorical('NN__hidden_layer_sizes', 
                [(32,), (64,), (128,), (256,), (512,), (64, 32), (128, 64), (256, 128)]),
            'NN__batch_size': trial.suggest_categorical('NN__batch_size', [16, 32, 64, 128]),
            'NN__learning_rate': trial.suggest_categorical('NN__learning_rate', ['constant', 'adaptive']),
            'NN__early_stopping': trial.suggest_categorical('NN__early_stopping', [True, False]),
            'NN__max_iter': trial.suggest_int('NN__max_iter', 1000, 4000)
        }
    elif model_name == 'SVR':
        params = {
            'SVR__C': trial.suggest_float('SVR__C', 1e-3, 1e4, log=True),
            'SVR__epsilon': trial.suggest_float('SVR__epsilon', 1e-4, 1.0, log=True),
            'SVR__kernel': trial.suggest_categorical('SVR__kernel', ['linear', 'rbf', 'poly']),
            'SVR__tol': trial.suggest_float('SVR__tol', 1e-5, 1e-1, log=True),
            'SVR__shrinking': trial.suggest_categorical('SVR__shrinking', [True, False]),
            'SVR__max_iter': trial.suggest_int('SVR__max_iter', 1000, 20000)
        }
        if params['SVR__kernel'] == 'poly':
            params['SVR__degree'] = trial.suggest_int('SVR__degree', 2, 6)
        if params['SVR__kernel'] in ['rbf', 'poly']:
            params['SVR__gamma'] = trial.suggest_float('SVR__gamma', 1e-6, 1e-1, log=True)
            params['SVR__coef0'] = trial.suggest_float('SVR__coef0', 0.0, 1.0)
        return params
    return {}  # Linear Regression has no hyperparameters


def create_pipeline(model_name, model, use_scaler=False):
    """Create a pipeline with an optional scaler."""

    steps = [('scaler', RobustScaler())] if use_scaler else []
    steps.append((model_name, model))
    return Pipeline(steps)


def cross_val_scores(pipeline, X_train, y_train, k):
    """Perform cross-validation and return the mse, mae and r2 scores."""

    cv_results = cross_validate(pipeline, X_train, y_train, cv=k, scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'], return_train_score=True)
    
    # Get the mean of MSE, MAE, and R2 from the cross-validation results
    mse = -cv_results['test_neg_mean_squared_error'].mean()
    mae = -cv_results['test_neg_mean_absolute_error'].mean()
    r2 = cv_results['test_r2'].mean()

    return mse, mae, r2