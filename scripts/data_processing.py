# File that contains the code for the processing of the dataset
# File that contains the functions for the selection of the features from the dataset
import os
import time
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

from scripts.utils import format_elapsed_time


# Dictionary that maps the target to the features to remove to avoid target leakage
task_leakage = {
    'CTPI_Mean': ['CTPI_Harried_Mean', 'CTPI_Cognitive_Mean'],
    'CTPI_Harried_Mean': ['CTPI_Mean', 'CTPI_Cognitive_Mean'],
    'CTPI_Cognitive_Mean': ['CTPI_Mean', 'CTPI_Harried_Mean'],
    'DASS_Sum': ['DASS_Depr_Sum', 'DASS_Stress_Sum', 'DASS_Anxiety_Sum'],
    'DASS_Depr_Sum': ['DASS_Sum', 'DASS_Stress_Sum', 'DASS_Anxiety_Sum'],
    'DASS_Stress_Sum': ['DASS_Sum', 'DASS_Depr_Sum', 'DASS_Anxiety_Sum'],
    'DASS_Anxiety_Sum': ['DASS_Sum', 'DASS_Depr_Sum', 'DASS_Stress_Sum']
}


def feature_selection(X, y, n_features, score_func='f_regression'):
    """
    Select the best features based on the provided scoring function.

    Parameters:
    - X (DataFrame or array-like): Feature matrix (processed data).
    - y (Series or array-like): Target variable.
    - n_features (int): Number of top features to select.
    - score_func (str): Scoring function for feature selection (f_regression, 'mir' for mutual_info_regression).

    Returns:
    - selected_features (Series): Names of the selected features.
    """
    
    # Start timing the feature selection process
    start_time = time.time()
    
    # Log the start of the process
    print('\n' + '='*90)
    print(f'🚀 Starting Feature Selection: Top {n_features} Features')
    print('='*90)
    print(f"🔧 Scoring Function: {score_func}")
    print(f'📊 Initial number of features: {X.shape[1]}')
    print(f'🎯 Target variable: {y.name}')
    print("="*90)
    
    # Initialize the SelectKBest object with the appropriate scoring function
    score_fn = f_regression if score_func == 'f_regression' else mutual_info_regression
    best_features = SelectKBest(score_func=score_fn, k=n_features)
    
    # Fit the model
    print('🔍 Fitting SelectKBest to identify the best features...')
    fit = best_features.fit(X, y)
    
    # Log feature scoring process completion
    print(f'✅ Feature selection completed. Now selecting the top {n_features} features.')
    
    # Create dataframes for scores, and feature names
    dfscores = pd.DataFrame(fit.scores_, columns=['Score'])
    dfcolumns = pd.DataFrame(X.columns, columns=['Feature'])
    
    # Concatenate the dataframes into a single dataframe
    feature_scores = pd.concat([dfcolumns, dfscores], axis=1)
    
    # Get the names of the top "n" features with the highest scores
    selected_features = feature_scores.nlargest(n_features, 'Score')['Feature']
    
    # Log the selected features and their scores
    print(f'\n🏆 Top {n_features} Features Selected:')
    print('='*90)
    print(feature_scores.nlargest(n_features, 'Score').to_string(index=False))
    print('='*90)
    
    # End timing the feature selection process
    end_time = time.time()
    elapsed_time = format_elapsed_time(start_time, end_time)
    print(f'\n⏱️ Feature selection completed in {elapsed_time}.')
    print('='*90 + "\n")

    return selected_features


import time
import numpy as np
import pandas as pd

def data_processing(
    data: pd.DataFrame,
    target: str,
    features_to_drop: list = None,
    categorical_features: list = None,
    single_frame: bool = True,
    verbose: bool = True,
):
    """
    Function that takes a dataset and a target and processes it:
     - Removes duplicate rows
     - Drops rows where the target is missing
     - Drops features that leak the target
     - Drops features that were asked to be dropped
     - Drops features with more than 20% missing values
     - Drops features with only constant values
     - Imputes values for features with less than 20% missing values (num -> median), (cat -> mode)
     - One-hot-encodes the categorical features

    Parameters:
    -----------
     - data (pd.DataFrame): Dataframe to process
     - target (str): Target feature, required for some preprocessing steps
        - Dropping rows where the target is missing
        - Dropping features that would leak the target
     - features_to_drop (list): Features that should be dropped from the dataset
     - categorical_features (list): Features to be considered as categorical
     - single_frame (bool): Whether to return a single dataframe with the features and target
        - If False, the function returns 2 dataframes: X, y
     - verbose (bool, default=True): If True, prints progress info; if False, prints nothing.

    Returns:
    --------
     - processed_data (pd.DataFrame): The processed dataframe, or (X, y) if single_frame=False
    """

    # helper "conditional print"
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    # Start timing the entire process
    vprint('\n' + '='*90)
    vprint('🔄 Starting data processing...')
    vprint('='*90)
    start_time = time.time()

    # Loading the dataset
    processed_data = data.copy(deep=True)
    original_shape = processed_data.shape
    vprint(f'✅ Data loaded. Original Shape: {original_shape}')
    vprint('='*90)

    # Replace infinite values with NaN
    vprint('- Checking for missing values:')
    vprint('  ➡️ Replacing infinite values with NaN.')
    processed_data = processed_data.replace([np.inf, -np.inf, '#NAME?'], np.nan)

    # Check percentage of missing values for each column
    vprint('  ➡️ Showing columns with missing values:')
    missing_percentages = round(processed_data.isnull().mean() * 100, 2)
    for feature, missing in missing_percentages.items():
        if missing > 0:
            vprint(f'\t-{feature}: {missing}%')
    vprint('='*90)

    # Drop rows where the target is missing
    vprint('- Dropping rows where the target is missing:')
    if target == 'DASS_Sum':
        # Get the DASS Columns
        dass_columns = task_leakage[target]  # assumes task_leakage is defined elsewhere
        # Drop rows with missing DASS values
        processed_data = data.dropna(subset=dass_columns)
        step1_shape = processed_data.shape
        if step1_shape == original_shape:
            vprint(f'  ❌ No missing target values to drop. Shape: {step1_shape}')
        else:
            vprint(f'  ✅ Dropped missing target values. New Shape: {step1_shape}')
        # If the target is DASS_Sum, we need to sum the DASS columns and add the target
        if 'DASS_Sum' not in processed_data.columns:
            vprint('- Adding the DASS_Sum target to the dataset:')
            processed_data[target] = processed_data[dass_columns].sum(axis=1)
            step1_shape = processed_data.shape
            vprint(f'  ✅ Added DASS_Sum. New Shape: {step1_shape}')
    else:
        processed_data = processed_data.dropna(subset=[target])
        step1_shape = processed_data.shape
        if step1_shape == original_shape:
            vprint(f'  ❌ No missing target values to drop. Shape: {step1_shape}')
        else:
            vprint(f'  ✅ Dropped missing target values. New Shape: {step1_shape}')

    ### Drop features not considered for the task
    vprint(f'- Dropping feature(s) not considered for the task:')
    if features_to_drop:
        processed_data = processed_data.drop(columns=features_to_drop)
        step3_shape = processed_data.shape
        vprint(f'  ✅ Dropped {len(features_to_drop)} features. New shape: {step3_shape}')
    else:
        vprint('  ❌ No features to drop.')

    # Dropping duplicated rows
    vprint('- Dropping duplicated rows:')
    n_duplicates = processed_data.duplicated().sum()
    if n_duplicates > 0:
        processed_data = processed_data.drop_duplicates()
        step4_shape = processed_data.shape
        vprint(f'  ✅ Dropped {n_duplicates} duplicated rows. New shape: {step4_shape}')
    else:
        vprint('  ❌ No duplicated rows to drop.')

    ### Handle missing values
    vprint('- Handling missing values:')

    # Drop features with more than 20% NaN values
    na_features = processed_data.columns[processed_data.isna().mean() > 0.2]
    if len(na_features) > 0:
        vprint(f'  ➡️ Dropping {len(na_features)} feature(s) with more than 20% NaN values.')
        processed_data = processed_data.drop(columns=na_features)
        step5_shape = processed_data.shape
        vprint(f'  ✅ Dropped high NaN features. New shape: {step5_shape}')
    else:
        vprint('  ➡️ No features have more than 20% NaN values.')

    ### Drop columns with constant values
    vprint('- Dropping columns with constant values:')
    initial_columns = processed_data.shape[1]
    processed_data = processed_data.loc[:, processed_data.apply(pd.Series.nunique) != 1]
    dropped_columns = initial_columns - processed_data.shape[1]
    vprint(f'  ➡️ Dropped {dropped_columns} constant column(s). New shape: {processed_data.shape}')

    # Replace NaN in numeric columns with the median (only if NaN exists)
    numeric_cols = processed_data.select_dtypes(include=['number']).columns
    numeric_cols_with_nan = processed_data[numeric_cols].columns[processed_data[numeric_cols].isna().any()]
    if len(numeric_cols_with_nan) > 0:
        vprint(f'  ➡️ Replacing NaN in {len(numeric_cols_with_nan)} numeric column(s) with the median.')
        processed_data[numeric_cols_with_nan] = processed_data[numeric_cols_with_nan].fillna(
            processed_data[numeric_cols_with_nan].median()
        )
        vprint('  ✅ Replaced NaN values in numeric columns.')
    else:
        vprint('  ➡️ No NaN values in numeric columns to replace.')

    # Replace NaN in categorical columns with the mode (only if NaN exists)
    if categorical_features:
        categorical_cols_with_nan = processed_data[categorical_features].columns[
            processed_data[categorical_features].isna().any()
        ]
        if len(categorical_cols_with_nan) > 0:
            vprint(f'  ➡️ Replacing NaN in {len(categorical_cols_with_nan)} categorical column(s) with the mode.')
            processed_data[categorical_cols_with_nan] = processed_data[categorical_cols_with_nan].apply(
                lambda col: col.fillna(col.mode()[0])
            )
            vprint('  ✅ Replaced NaN values in categorical columns.')
        else:
            vprint('  ➡️ No NaN values in categorical columns to replace.')

    ### One-hot encode categorical features
    vprint("- One-hot encoding categorical features:")
    if categorical_features:
        vprint(f'  ➡️ One-hot encoding {len(categorical_features)} specified categorical column(s).')
        processed_data = pd.get_dummies(processed_data, columns=categorical_features)
        step6_shape = processed_data.shape
        vprint(f'  ✅ One-hot encoding completed. New shape: {step6_shape}')
    else:
        vprint('  ➡️ No categorical column(s) found.')
        vprint('  ✅ No One-hot encoding.')

    ### End of data processing
    end_time = time.time()
    final_shape = processed_data.shape
    vprint(f'\n✅ Data processing completed. Final shape: {final_shape}')
    # assumes format_elapsed_time is defined elsewhere
    if 'format_elapsed_time' in globals():
        elapsed_time = format_elapsed_time(start_time, end_time)
        vprint(f'⏱️ Total processing time: {elapsed_time}.')
    else:
        vprint(f'⏱️ Total processing time (s): {end_time - start_time:.2f}')
    vprint("="*90 + "\n")

    if single_frame:
        return processed_data
    else:
        X = processed_data.drop(target, axis=1)
        y = processed_data[target]
        return X, y



def data_loading(wp, features_folder):

    wp = int(wp)
    ### Load data
    print('\n' + '='*90)
    print(f'📥 Loading WP{wp} data...')

    if wp == 1:
        data_folder = os.path.join(features_folder, 'Data_WP1')
        data_path = os.path.join(data_folder, 'WP1_Switzerland_processed.csv')
        df = pd.read_csv(data_path)
        return df

    elif wp == 2:
        data_folder = os.path.join(features_folder, 'Data_WP2')
        data_path = os.path.join(data_folder, 'WP2_Switzerland_processed.csv')
        df = pd.read_csv(data_path)
        return df

    else:
        return NameError('The chosen WP does not exist. Please choose between 1 and 2.')