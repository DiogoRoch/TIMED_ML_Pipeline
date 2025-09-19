# File that contains the functions for plotting the results of the regression task

# Plotting functions to use together with pycaret results
import plotly.graph_objects as go
import pandas as pd


def plot_all_scores(performances: dict, metrics: list):
    """
    Just plots the comparison of models across all the chosen metrics.

    Parameters
    ----------
    performances (dict): Dictionary containing the performances of the models
        model_name (e.g., names of the estimators)
            metric (e.g, MAE, MSE, R2, RMSE, RMSLE, MAPE, ...)
                train_mean (mean of the cross-validation on the train sets)
                train_std (standard deviation of the cv on the train sets)
                val_mean (mean of the cv on the validation sets)
                val_std (standard deviation of the cv on the validation sets)
    metrics (list): List of metrics to plot (e.g., R2, MSE, MAE, ...)
        They have to be present in the performances dict (can't plot things that don't exist)
    
    Returns: None
    """
    for metric in metrics:
        try:
            plot_model_comparison(performances, metric)
        except KeyError:
            print(f"Metric [{metric}] was not calculated beforehand.")


def plot_model_comparison_single(df: pd.DataFrame, metric: str, data_split: str):
    """
    Plots the comparison of all models' performances on the train or val sets
    (mean and std of the cross-validation) for a specified metric.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing model performances.
        Expected format:
        Columns = ['model_name', '{metric}_train_mean', '{metric}_train_std', '{metric}_val_mean', '{metric}_val_std']
    metric (str): Metric to plot (e.g., 'MAE', 'MSE', 'R2').
    data_split (str): Data split to plot ('train' or 'val').

    Returns:
    --------
    None
    """
    if data_split not in ['train', 'val']:
        raise ValueError('data_split must be either "train" or "val"')

    fig = go.Figure()

    for _, row in df.iterrows():
        model_name = row['model_name']

        if data_split == 'train':
            train_mean = row[f'{metric}_train_mean']
            train_std = row[f'{metric}_train_std']
            fig.add_trace(go.Bar(
                name=f"{model_name}",
                x=['Train'], 
                y=[train_mean],
                error_y=dict(type='data', array=[train_std])
            ))

        if data_split == 'val':
            val_mean = row[f'{metric}_val_mean']
            val_std = row[f'{metric}_val_std']
            fig.add_trace(go.Bar(
                name=f"{model_name}",
                x=['Val'], 
                y=[val_mean],
                error_y=dict(type='data', array=[val_std])
            ))

    fig.update_layout(
        barmode='group',
        title={
            'text': f"<b>Model Comparison for {metric} ({data_split})</b>",
            'x': 0.5,  # Center title
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24},
            'y': 0.95
        },
        xaxis_title="Data Split",
        yaxis_title=f"{metric} Value",
        legend=dict(
            title="Models",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        template='plotly_white',
        margin=dict(l=40, r=40, t=150, b=40),
        height=500,
    )

    # Add subtle gridlines for better readability
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='lightgrey', zerolinecolor='grey')

    fig.show()


def plot_model_comparison(performances: dict, metric: str):
    """
    Plots the comparison of all models' performances on the train and val sets
    (mean and std of the cross-validation) on a specified metric.

    Parameters
    ----------
    performances (dict): Dictionary containing the performances of the models
        model_name (e.g., names of the estimators)
            metric (e.g, MAE, MSE, R2, RMSE, RMSLE, MAPE, ...)
                train_mean (mean of the cross-validation on the train sets)
                train_std (standard deviation of the cv on the train sets)
                val_mean (mean of the cv on the validation sets)
                val_std (standard deviation of the cv on the validation sets)
    metric (str): Metric to plot

    Returns: None
    """
    models = list(performances.keys())
    fig = go.Figure()
    for model in models:
        fig.add_trace(go.Bar(
            name=model,
            x=['Train', 'Val'], y=[performances[model][metric]['train_mean'], performances[model][metric]['val_mean']],
            error_y=dict(type='data', array=[performances[model][metric]['train_std'], performances[model][metric]['val_std']])
        ))
    #fig.update_layout(barmode='group', title=f'Model Comparison for {metric}')
    fig.update_layout(
        barmode='group',
        title={
            'text': f"<b>Model Comparison for {metric}</b>",
            'x': 0.5,  # Center title
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        xaxis_title="Data Split",
        yaxis_title=f"{metric} Value",
        legend=dict(
            title="Models",
            orientation="h",  # Horizontal legend
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        template='plotly_white',  # Clean and professional theme
        margin=dict(l=40, r=40, t=150, b=40),  # Adjust margins for better spacing
        height=500,  # Adjust height for better readability
    )

    # Add subtle gridlines for better readability
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='lightgrey', zerolinecolor='grey')

    fig.show()