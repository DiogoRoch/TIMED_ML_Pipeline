# File that contains the code for the main regression pipeline and the training of the models
import os
import joblib
import optuna
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Model imports
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# General imports
import time
import numpy as np
import pandas as pd
import warnings

from scripts.utils import results_to_series, format_elapsed_time

# To ignore warnings
warnings.filterwarnings("ignore")
# To ignore optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)



def regression(models, X, y, random_state, target, current_exp, models_exp_folder, n_iter=3, k=5, scoring=['mse'], target_transform="none", n_trials=50, n_jobs=4, save_models=False):

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

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
                i_results, model = optimize_and_evaluate_model(
                    model_name=model_name,
                    iteration=i+1,
                    target=target,
                    X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test,
                    k=k,
                    scoring=score,
                    n_trials=n_trials,
                    n_jobs=n_jobs,
                    target_transform=target_transform
                )
                
                # Save the results to a dataframe
                i_series, model_id = results_to_series(i_results, current_exp)
                results_df = pd.concat([results_df, pd.DataFrame([i_series])], ignore_index=True)
                
                # Save the model
                if save_models:
                    save_model(model, model_id, models_exp_folder)

                end_time = time.time()
                
                # Log iteration completion time
                elapsed_time = format_elapsed_time(start_time, end_time)
                current_time = time.strftime("%H:%M:%S", time.localtime())
                print(f"    ✅ Iteration {i+1}/{n_iter} for {model_name} completed in {elapsed_time}  |  {current_time}")
            
        print(f"🎉 Optimization for {model_name} completed.")
        print("="*90 + "\n")

    print("🏁 All model optimizations are complete!\n")
    return results_df


def optimize_and_evaluate_model(model_name, iteration, target, X_train, y_train, X_test, y_test, k, scoring, n_trials, n_jobs, target_transform="none"):
    
    objective, pipeline = get_model(model_name, X_train, y_train, k, scoring, target_transform)
    
    if scoring == "r2":
        # For R2, we want to maximize the score, so we need to set the direction to 'maximize'
        study = optuna.create_study(direction='maximize')
    else:
        # For MSE and MAE, we want to minimize the score, so we set the direction to 'minimize'
        study = optuna.create_study(direction='minimize')
    
    # Start timer for optimization
    start_time = time.time()
    
    if model_name == "LR":
        # For Linear Regression, there is no need to optimize hyperparameters (because there are none)
        study.optimize(objective, n_trials=1, n_jobs=n_jobs)
    else:
        # For other models, optimize hyperparameters using Optuna
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    
    # End timer for optimization
    end_time = time.time()
    opt_time = end_time - start_time
    best_params = study.best_params

    # Update pipeline with the best found parameters and fit it to the training data
    pipeline = set_pipeline_params(pipeline, best_params)
    pipeline.fit(X_train, y_train)
    
    # Get predictions for both training and test sets
    train_preds = pipeline.predict(X_train)
    test_preds = pipeline.predict(X_test)

    # Get the scores for the best trial during optimization
    best_trial_scores = study.best_trial.user_attrs
    # MSE
    best_mse = best_trial_scores['MSE']
    best_mse_std = best_trial_scores['MSE_STD']
    best_mse_splits = best_trial_scores['MSE_SPLITS']
    # MAE
    best_mae = best_trial_scores['MAE']
    best_mae_std = best_trial_scores['MAE_STD']
    best_mae_splits = best_trial_scores['MAE_SPLITS']
    # R2
    best_r2 = best_trial_scores['R2']
    best_r2_std = best_trial_scores['R2_STD']
    best_r2_splits = best_trial_scores['R2_SPLITS']

    # Evaluate the model on the train set
    train_mse = mean_squared_error(y_train, train_preds)
    train_mae = mean_absolute_error(y_train, train_preds)
    train_r2 = r2_score(y_train, train_preds)

    # Evaluate the model on the test set
    test_mse = mean_squared_error(y_test, test_preds)
    test_mae = mean_absolute_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)


    # Store the results in a dictionary
    results = {
        # General information
        'model_name': model_name,
        'iteration': iteration,
        'target': target,
        'target_transform': target_transform,
        'n_trials': n_trials,
        'score': scoring,
        'best_params': best_params,
        
        # Scores from the best trial
        'best_MSE': best_mse,
        'best_MSE_STD': best_mse_std,
        'best_MSE_SPLITS': best_mse_splits,
        'best_MAE': best_mae,
        'best_MAE_STD': best_mae_std,
        'best_MAE_SPLITS': best_mae_splits,
        'best_R2': best_r2,
        'best_R2_STD': best_r2_std,
        'best_R2_SPLITS': best_r2_splits,
        
        # Train scores from the final model
        'train_MSE': train_mse,
        'train_MAE': train_mae,
        'train_R2': train_r2,
        
        # Test scores from the final model (not to use for hyperparameter tuning)
        'test_MSE': test_mse,
        'test_MAE': test_mae,
        'test_R2': test_r2,
        'optimization_time': opt_time
    }

    return results, pipeline


def save_model(model, model_id, models_exp_folder):
    """Save the trained model to a file."""
    
    # Create the filename
    filename = f"{model_id}.joblib"
    
    # Create the full path
    filepath = os.path.join(models_exp_folder, filename)
    
    # Save the model
    joblib.dump(model, filepath)

    print(f"    💾 Saved model: {filepath} | Size: {os.path.getsize(filepath) / 1e6:.2f} MB")


def get_model(model_name, X_train, y_train, k, scoring='mse', target_transform="none"):
    """Get the model pipeline and the objective function for Optuna."""

    # Determine if scaler is needed
    use_scaler = model_name in ['NN', 'SVR', 'LR']
    
    # Create a pipeline with placeholder model (no hyperparameters)
    if model_name == 'RF':
        model = RandomForestRegressor()
    elif model_name == 'ET':
        model = ExtraTreesRegressor()
    elif model_name == 'XGB':
        model = XGBRegressor()
    elif model_name == 'LGB':
        model = LGBMRegressor(verbose=-1)
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
        pipeline = create_pipeline(
            model_name=model_name,
            model=model,
            use_scaler=use_scaler,
            target_transform=target_transform
        )
        pipeline = set_pipeline_params(pipeline, params)
        
        # Perform cross-validation and calculate the metrics
        scores = cross_val_scores(pipeline, X_train, y_train, k)

        # Log all the metrics
        # MSE
        trial.set_user_attr('MSE', scores['mse_mean'])
        trial.set_user_attr('MSE_STD', scores['mse_std'])
        trial.set_user_attr('MSE_SPLITS', scores['mse_splits'])
        # MAE
        trial.set_user_attr('MAE', scores['mae_mean'])
        trial.set_user_attr('MAE_STD', scores['mae_std'])
        trial.set_user_attr('MAE_SPLITS', scores['mae_splits'])
        # R2
        trial.set_user_attr('R2', scores['r2_mean'])
        trial.set_user_attr('R2_STD', scores['r2_std'])
        trial.set_user_attr('R2_SPLITS', scores['r2_splits'])

        # Return the score to minimize (choice between mse, mae, and r2)
        if scoring == 'mse':
            return scores['mse_mean']
        elif scoring == 'mae':
            return scores['mae_mean']
        elif scoring == 'r2':
            return scores['r2_mean']

    pipeline = create_pipeline(
        model_name=model_name,
        model=model,
        use_scaler=use_scaler,
        target_transform=target_transform
    )

    # Return both the objective function and the pipeline
    return objective, pipeline


def get_hyperparameters(trial, model_name):
    """Return model-specific hyperparameter suggestions for Optuna."""

    if model_name == 'RF':
        return {
            'RF__n_estimators': trial.suggest_int('RF__n_estimators', 100, 2000, step=100),
            'RF__max_depth': trial.suggest_int('RF__max_depth', 5, 30),
            'RF__min_samples_split': trial.suggest_int('RF__min_samples_split', 2, 10),
            'RF__min_samples_leaf': trial.suggest_int('RF__min_samples_leaf', 1, 10),
            'RF__bootstrap': trial.suggest_categorical('RF__bootstrap', [True, False])
        }
    
    elif model_name == 'ET':
        return {
            'ET__n_estimators': trial.suggest_int('ET__n_estimators', 100, 2000, step=100),
            'ET__max_depth': trial.suggest_int('ET__max_depth', 5, 30),
            'ET__min_samples_split': trial.suggest_int('ET__min_samples_split', 2, 10),
            'ET__min_samples_leaf': trial.suggest_int('ET__min_samples_leaf', 1, 10),
            'ET__bootstrap': trial.suggest_categorical('ET__bootstrap', [True, False])
        }
    
    elif model_name == 'XGB':
        """
        Simplified the hyperparameter search space to avoid computation waste by focusing on the gbtree booster.
        Sources:
            - https://www.kaggle.com/code/cahyaalkahfi/xgboost-model-tuning-using-optuna
            - https://randomrealizations.com/posts/xgboost-parameter-tuning-with-optuna/
        """
        return {
            'XGB__objective': trial.suggest_categorical('XGB__objective', ['reg:squarederror']),
            'XGB__booster': trial.suggest_categorical('XGB__booster', ['gbtree']),
            'XGB__tree_method': trial.suggest_categorical('XGB__tree_method', ['approx', 'hist']),
            'XGB__n_estimators': trial.suggest_int('XGB__n_estimators', 100, 2000, step=100),
            'XGB__max_depth': trial.suggest_int('XGB__max_depth', 5, 30),
            'XGB__learning_rate': trial.suggest_float('XGB__learning_rate', 1e-3, 0.3, log=True),
            'XGB__min_child_weight': trial.suggest_int('XGB__min_child_weight', 1, 100),
            'XGB__subsample': trial.suggest_float('XGB__subsample', 0.5, 1.0, step=0.1),
            'XGB__colsample_bytree': trial.suggest_float('XGB__colsample_bytree', 0.5, 1.0, step=0.1),
            'XGB__gamma': trial.suggest_float('XGB__gamma', 0.0, 5.0, step=0.1),
            'XGB__lambda': trial.suggest_float('XGB__lambda', 1e-8, 10.0, log=True),
            'XGB__alpha': trial.suggest_float('XGB__alpha', 1e-8, 10.0, log=True),

        }
    
    elif model_name == 'LGB':
        """
        Increased the search space to allow for more flexibility in the model's complexity and regularization.
        Used same values when possible as XGBoost to allow for a fair comparison.
        Sources:
         - https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
        """
        return {
            'LGB__boosting_type': trial.suggest_categorical('LGB__boosting_type', ['gbdt', 'dart']),
            'LGB__n_estimators': trial.suggest_int('LGB__n_estimators', 100, 2000, step=100),
            'LGB__max_depth': trial.suggest_int('LGB__max_depth', 5, 30),
            'LGB__learning_rate': trial.suggest_float('LGB__learning_rate', 1e-3, 0.3, log=True),
            'LGB__num_leaves': trial.suggest_int('LGB__num_leaves', 31, 512),
            'LGB__min_child_samples': trial.suggest_int('LGB__min_child_samples', 5, 100),
            'LGB__subsample': trial.suggest_float('LGB__subsample', 0.5, 1.0, step=0.1),
            'LGB__colsample_bytree': trial.suggest_float('LGB__colsample_bytree', 0.5, 1.0, step=0.1),
            'LGB__reg_alpha': trial.suggest_float('LGB__reg_alpha', 1e-8, 10.0, log=True),
            'LGB__reg_lambda': trial.suggest_float('LGB__reg_lambda', 1e-8, 10.0, log=True),
            'LGB__min_split_gain': trial.suggest_float('LGB__min_split_gain', 0.0, 1.0),
            'LGB__bagging_freq': trial.suggest_int('LGB__bagging_freq', 1, 10),
            'LGB__max_bin': trial.suggest_int('LGB__max_bin', 128, 512)
        }
    
    elif model_name == 'NN':
        return {
            'NN__solver': trial.suggest_categorical('NN__solver', ['adam']),
            'NN__activation': trial.suggest_categorical('NN__activation', ['relu', 'tanh']),
            'NN__alpha': trial.suggest_float('NN__alpha', 1e-5, 1e-2, log=True),
            'NN__learning_rate_init': trial.suggest_float('NN__learning_rate_init', 1e-4, 1e-2, log=True),
            'NN__hidden_layer_sizes': trial.suggest_categorical('NN__hidden_layer_sizes', 
                [
                    # Single-layer configurations
                    (32,), (64,), (128,), (256,), (512,), (1024,),

                    # Two-layer configurations
                    (64, 32), (128, 64), (256, 128), (512, 256),
                    (1024, 512), (128, 32), (256, 64), (512, 128), 

                    # Three-layer configurations
                    (128, 64, 32), (256, 128, 64), (512, 256, 128),
                    (1024, 512, 256), (64, 32, 16), (256, 128, 32),
                    
                    # Four-layer configurations
                    (512, 256, 128, 64), (1024, 512, 256, 128), 
                    (128, 64, 32, 16), (256, 128, 64, 32), (512, 256, 128, 32),
                    
                    # Five-layer configurations
                    (1024, 512, 256, 128, 64), (512, 256, 128, 64, 32),
                    (256, 128, 64, 32, 16), (128, 64, 32, 16, 8),
                    
                    # Deep configurations with smaller layers
                    (64, 64, 64, 64), (128, 128, 128, 128), 
                    (256, 256, 256, 256), (512, 512, 512, 512),
                    
                    # Varied sizes for deeper networks
                    (1024, 512, 256, 128, 64, 32),
                    (512, 256, 128, 64, 32, 16),
                    (256, 128, 64, 32, 16, 8),
                    (128, 64, 32, 16, 8, 4),
                ]),
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


def create_pipeline(model_name, model, use_scaler=False, target_transform="none"):
    """Create a pipeline with an optional scaler."""

    steps = [("scaler", RobustScaler())] if use_scaler else []
    steps.append((model_name, model))
    pipeline = Pipeline(steps)

    # If transformation of the target is needed
    if target_transform == "none":
        return pipeline
    
    elif target_transform == "log":
        # Log 1p transformation to handle 0 values in case there are any.
        pipeline = TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log1p,
            inverse_func=np.expm1
        )
    
    elif target_transform == "yeo-johnson":
        # Yeo-Johnson transformation already handles 0 values so chosen instead of box-cox.
        pipeline = TransformedTargetRegressor(
            regressor=pipeline,
            transformer=PowerTransformer(method='yeo-johnson')
        )
    
    else:
        raise ValueError(f"Unknown target transformation: {target_transform}")
    
    return pipeline


def cross_val_scores(pipeline, X_train, y_train, k):
    """Perform cross-validation and return the negative mean squared error."""

    # Create a KFold object
    cv = KFold(n_splits=k, shuffle=False)   

    # Perform cross-validation
    cv_results = cross_validate(
        pipeline, X_train, y_train, cv=cv,
        scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
    )

    # Individual split scores
    mse_scores = -cv_results['test_neg_mean_squared_error']
    mae_scores = -cv_results['test_neg_mean_absolute_error']
    r2_scores = cv_results['test_r2']

    # Aggregate metrics (mean and std)
    # MSE
    mse_mean = np.mean(mse_scores)
    mse_std = np.std(mse_scores)
    # MAE
    mae_mean = np.mean(mae_scores)
    mae_std = np.std(mae_scores)
    # R2
    r2_mean = np.mean(r2_scores)
    r2_std = np.std(r2_scores)

    # Store in a dictionary
    scores_dict = {
        'mse_mean': mse_mean, 'mse_std': mse_std, 'mse_splits': mse_scores.tolist(),
        'mae_mean': mae_mean, 'mae_std': mae_std, 'mae_splits': mae_scores.tolist(),
        'r2_mean': r2_mean, 'r2_std': r2_std, 'r2_splits': r2_scores.tolist()
    }

    return scores_dict


def set_pipeline_params(pipeline, params):
    """Handles setting parameters even if the pipeline is nested."""
    
    if isinstance(pipeline, TransformedTargetRegressor):
        pipeline.regressor.set_params(**params)
    else:
        pipeline.set_params(**params)
    
    return pipeline