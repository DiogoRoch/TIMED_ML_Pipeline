# Utility scripts for exploration of data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Reference for the thresholds: Evans JD. (1996) Straightforward Statistics for the Behavioral Sciences. Brooks/Cole Publishing; Pacific Grove, Calif.
#   correlations between 0.20-0.39 as weak
#   correlations 0.40-0.59 as moderate
#   correlations 0.60-0.79 as strong
#   correlations >0.80 as very strong
thresholds = {
    'very_strong':0.8,
    'strong':0.6,
    'moderate':0.4,
    'weak':0.2
}


def get_correlations_dict(data_df: pd.DataFrame, verbose: bool = True):
    """
    Get a dictionary with all the feature's sorted moderate and high correlations to other features.

    Parameters:
    -----------
    - data_df (pandas DataFrame): The dataframe to get correlations on.
    - verbose (bool): Choose whether to print results or not.

    Returns:
    --------
    - correlations_dict (dict): Dictionary containing the moderate and high correlations for each feature.
    """

    correlation_matrix = data_df.corr()
    features = sorted(correlation_matrix.columns)
    correlations_dict = {}

    for feature in features:

        correlations_dict[feature] = {}
        for threshold in thresholds:
            correlations_dict[feature][threshold] = []
        
        feature_correlations = correlation_matrix[feature]
        for index in features:
            index_correlation = round(feature_correlations.loc[index], 2)
            index_correlation_abs = np.abs(index_correlation)
            # Avoid storing self-correlations
            if feature != index:
                for threshold_name, threshold_value in thresholds.items():
                    if index_correlation_abs >= threshold_value:  # Problem here !!! -> Multiple assignments, fixed by adding a break statement after a threshold is fulfilled
                        correlations_dict[feature][threshold_name].append((index, index_correlation))
                        break

        # Sorting the correlations in descending order of their absolute values for each threshold
        for threshold_name, correlation_list in correlations_dict[feature].items():
            correlations_dict[feature][threshold_name] = sorted(correlation_list, key=lambda item: np.abs(item[1]), reverse=True)

    if verbose:
        for feature_name, feature_thresholds in correlations_dict.items():

            print(f'- {feature_name}')
            for threshold_name, correlations_values in feature_thresholds.items():
                print(f'\t- {threshold_name.title()}')
                if correlations_values:
                    for corr_feature, corr_value in correlations_values: print(f'\t\t- {corr_feature} -> {corr_value}')
                else:
                    print(f'\t\t- No {threshold_name.title()} correlations!')
            print()
    
    return correlations_dict


def plot_target_interactions(data_df: pd.DataFrame, categorical_features: list, target: str, thresholds: list):
    """
    Plots the scatterplots of all features on the dataframe with the target -> target against feature.
    """
    pass


def plot_coll_interactions(data_df: pd.DataFrame, categorical_features: list):
    """
    Plots the scatterplots of all features that have very high correlations.
    """
    pass


####
# Feature Summary Plots

# Plotting distribution function with histplot and kde
def plot_feature_distribution(data_df: pd.DataFrame, feature_to_plot: str, figsize: tuple=(10, 6), ax=None):
    """
    Plots the distribution of the feature from the provided dataframe.

    Parameters:
    -----------
    data_df (pd.DataFrame): DataFrame containing the dataset with the feature to plot.
    feature_to_plot (str): Name of the feature to plot.
    figsize (tuple): Size of the figure.
    ax (matplotlib.axes._axes.Axes): Axes to plot on, needed only if we need to specify the axes.

    Returns:
    --------
    None
    """

    features = list(data_df.columns)
    if feature_to_plot not in features:
        print(f'The chosen feature [{feature_to_plot}] is not inside the dataframe.')
        print('Choose one of:')
        print(features)
        return
    
    if not ax:
        plt.figure(figsize=figsize)
    sns.histplot(data_df[feature_to_plot], kde=True, ax=ax)
    if ax:
        ax.set_title(f'Distribution of {feature_to_plot}')
        ax.set_xlabel(feature_to_plot)
        ax.set_ylabel('Count')
    else:
        plt.title(f'Distribution of {feature_to_plot}')
        plt.xlabel(feature_to_plot)
        plt.ylabel('Count')
    if not ax:
        plt.show()


# Plotting boxplot function for specified feature
def plot_boxplot(data_df: pd.DataFrame, feature_to_plot: str, figsize: tuple=(10, 6), ax=None):
    """
    Plots the boxplot of the feature from the provided dataframe -> Distribution and Outliers.

    Parameters:
    -----------
    data_df (pd.DataFrame): DataFrame containing the dataset with the feature to plot.
    feature_to_plot (str): Name of the feature to plot.
    figsize (tuple): Size of the figure.
    ax (matplotlib.axes._axes.Axes): Axes to plot on, needed only if we need to specify the axes.

    Returns:
    --------
    None
    """

    features = list(data_df.columns)
    if feature_to_plot not in features:
        print(f'The chosen feature [{feature_to_plot}] is not inside the dataframe.')
        print('Choose one of:')
        print(features)
        return
    
    if not ax:
        plt.figure(figsize=figsize)
    sns.boxplot(data=data_df[feature_to_plot], ax=ax)
    if ax:
        ax.set_title(f'Boxplot of {feature_to_plot}')
        ax.set_xlabel(feature_to_plot)
        ax.set_ylabel('Values')
    else:
        plt.title(f'Boxplot of {feature_to_plot}')
        plt.xlabel(feature_to_plot)
        plt.ylabel('Values')
    if not ax:
        plt.show()


# Plotting violinplot for specified feature
def plot_violinplot(data_df: pd.DataFrame, feature_to_plot: str, figsize: tuple=(10, 6), ax=None):
    """
    Plots the violinplot of the feature from the provided dataframe.

    Parameters:
    -----------
    data_df (pd.DataFrame): DataFrame containing the dataset with the feature to plot.
    feature_to_plot (str): Name of the feature to plot.
    figsize (tuple): Size of the figure.
    ax (matplotlib.axes._axes.Axes): Axes to plot on, needed only if we need to specify the axes.

    Returns:
    --------
    None
    """

    features = list(data_df.columns)
    if feature_to_plot not in features:
        print(f'The chosen feature [{feature_to_plot}] is not inside the dataframe.')
        print('Choose one of:')
        print(features)
        return
    
    if not ax:
        plt.figure(figsize=figsize)
    sns.violinplot(data=data_df[feature_to_plot], ax=ax)
    if ax:
        ax.set_title(f'Violinplot of {feature_to_plot}')
        ax.set_xlabel(feature_to_plot)
        ax.set_ylabel('Values')
    else:
        plt.title(f'Violinplot of {feature_to_plot}')
        plt.xlabel(feature_to_plot)
        plt.ylabel('Values')
    if not ax:
        plt.show()


# Plotting all three types of plots for a feature
def plot_feature_summary(data_df: pd.DataFrame, feature_to_plot: str, figsize: tuple=(20, 6)):
    """
    Plots the 3 distribution/outlier plots in the same figure on 3 different columns:
        - Histplot with kde
        - Boxplot
        - Violinplot

    Parameters:
    -----------
    data_df (pd.DataFrame): DataFrame containing the dataset with the feature to plot.
    feature_to_plot (str): Name of the feature to plot.
    figsize (tuple): Size of the figure.

    Returns:
    --------
    None
    """


    features = list(data_df.columns)
    if feature_to_plot not in features:
        print(f'The chosen feature [{feature_to_plot}] is not inside the dataframe.')
        print('Choose one of:')
        print(features)
        return

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    plot_feature_distribution(data_df, feature_to_plot, ax=axes[0])
    plot_boxplot(data_df, feature_to_plot, ax=axes[1])
    plot_violinplot(data_df, feature_to_plot, ax=axes[2])
    plt.show()


# Plot feature interactions through a scatterplot
def plot_feature_interaction(data_df: pd.DataFrame, feature_to_plot: str, target_feature: str, figsize: tuple=(10, 6), title_corr :float=None):
    """
    Plots the regplot of a target feature against another feature (scatterplot + linear regression)

    Parameters:
    -----------
    data_df (pd.DataFrame): DataFrame containing the dataset with the feature to plot.
    feature_to_plot (str): Name of the feature to plot.
    target_feature (str): Name of the target feature.
    figsize (tuple): Size of the figure.
    title_corr (float): Correlation value between the target and the feature to plot.

    Returns:
    --------
    None
    """

    features = list(data_df.columns)
    if feature_to_plot not in features:
        print(f'The chosen feature [{feature_to_plot}] is not inside the dataframe.')
        print('Choose one of:')
        print(features)
        return
    if target_feature not in features:
        print(f'The chosen feature [{target_feature}] is not inside the dataframe.')
        print('Choose one of:')
        print(features)
        return
    
    plt.figure(figsize=figsize)
    sns.regplot(data_df, x=feature_to_plot, y=target_feature, line_kws=dict(color='r'))
    if title_corr:
        plt.title(f'Scatterplot of {target_feature} against {feature_to_plot} (r = {title_corr})')
    else:
        plt.title(f'Scatterplot of {target_feature} against {feature_to_plot}')
    plt.show()


def plot_target_interactions(data_df: pd.DataFrame, categorical_features: list, target: str, thresholds: list, save: bool=False):
    """
    Plots the scatterplots of all features on the dataframe with the target -> target against feature.

    Parameters:
    -----------
    data_df (pd.DataFrame): DataFrame containing the dataset with the feature to plot.
    categorical_features (list): List of categorical features to avoid doing regplots on them
    target (str): Name of the target feature
    thresholds (list): List of correlation strengths to take into account
        - very_strong
        - strong
        - moderate
        - weak
    save (bool): Whether to save the plots somewhere (True or False)

    Returns:
    --------
    None
    """
    print('Getting the correlations of the data...')
    correlations_dict = get_correlations_dict(data_df, verbose=False)
    target_correlations = correlations_dict[target]

    correlations = []
    for threshold in thresholds:
        correlations += target_correlations[threshold]

    for feature_name, correlation_value in correlations:
        print(feature_name, correlation_value)
        if feature_name not in categorical_features:
            plot_feature_interaction(data_df, feature_name, target, title_corr=correlation_value)


def plot_coll_interactions(data_df: pd.DataFrame, categorical_features: list, save: bool=False):
    """
    Plots the scatterplots of all features that have very high correlations.

    Parameters:
    -----------
    data_df (pd.DataFrame): DataFrame containing the dataset with the feature to plot.
    categorical_features (list): List of categorical features to avoid doing regplots on them
    save (bool): Whether to save the plots somewhere (True or False)

    Returns:
    --------
    None
    """
    print('Getting the correlations of the data...')
    print('='*70)
    correlations_dict = get_correlations_dict(data_df, verbose=False)

    for feature_name, feature_correlations in correlations_dict.items():
        if feature_name not in categorical_features and feature_correlations['very_strong']:
            print(feature_name)
            for correlated_feature, correlation_value in feature_correlations['very_strong']:
                if correlated_feature not in categorical_features and correlation_value >= 0.8:
                    print(correlated_feature, correlation_value)
                    plot_feature_interaction(data_df, correlated_feature, feature_name, title_corr=correlation_value)
            print('='*70)
            print('='*70)