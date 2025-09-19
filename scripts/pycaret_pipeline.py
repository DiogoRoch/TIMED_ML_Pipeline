# Custom pipeline that takes advantage of the PyCaret library to compare a large number of models and then store/plot the results
from pycaret.regression import RegressionExperiment
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from scripts.data_processing import data_processing, task_leakage
from scripts.utils import get_model_performances, performances_dict_to_df


class RegressionAnalysis:

    def __init__(self, dataset: pd.DataFrame, target: str, session_ids: list, features_to_drop:list, categorical_features: list=None, n_experiments: int=1):
        
        # Check the parameters
        if target not in list(dataset.columns) and target != 'DASS_Sum':
            raise Exception(f'Target: [{target}] not in the dataset.')
        for feature in categorical_features:
            if feature not in list(dataset.columns):
                raise Exception(f'Categorical feature: [{categorical_features}] not in the dataset.')
        if len(session_ids) != n_experiments and type(session_ids) == list:
            raise Exception(f'The number of session ids [{len(session_ids)}] does not correspond to the number of experiments [{n_experiments}]!')
        
        self.experiments = {} # To store multiple RegressionExperiment objects and their results/models
        self.created_experiments = False # Checks whether experiments were created or not
        self.selected_models = False     # Checks whether models were selected or not
        self.tuned_models = False        # Checks whether models were tuned or not
        self.dataset = dataset
        self.target = target
        self.features_to_drop = features_to_drop
        self.categorical_features = categorical_features

        # To store the session_ids used for each RegressionExperiment
        self.n_experiments = n_experiments
        self.session_ids = self.select_session_ids(session_ids)

        # Process the Dataset and split it into a train and test set
        self.data_train, self.data_test = self.process_dataset()
        
    
    def select_session_ids(self, session_ids):
        """
        Method that selects the session ids to be used for the RegressionExperiment objects.
        """
        ids = []
        # Choose n_experiments random session_ids for the RegressionExperiments
        if session_ids == "random":
            for _ in range(self.n_experiments):
                present = True
                while present:
                    random_id = random.randint(0, 9999)
                    if random_id not in ids:
                        present = False
                        ids.append(random_id)
        # If the choice was not random, format the input into a list of ids
        else:
            ids = session_ids
        
        print(f'Session ids for the current session: {ids}')
        return ids


    def process_dataset(self):
        """
        Method that applies a data processing function to the dataset and returns the processed
        train data and test data.
        """
        # Adding the DASS_Sum column to the dataset from the start
        if 'DASS_Sum' not in self.dataset.columns:
            dass_scales = task_leakage['DASS_Sum']
            self.dataset['DASS_Sum'] = self.dataset[dass_scales].sum(axis=1)

        data_processed = data_processing(
            data=self.dataset,
            target=self.target,
            features_to_drop=self.features_to_drop,
            categorical_features=self.categorical_features,
            single_frame=True
        )
        data_train, data_test = train_test_split(data_processed, test_size=.2, random_state=42)

        return data_train, data_test


    def create_experiments(self):
        """
        Method that creates and setups the RegressionExperiments.
        """

        print(f'Creating {self.n_experiments} Regression Experiment(s):')
        for i, session_id in enumerate(self.session_ids):
            exp_number = i + 1
            exp = RegressionExperiment()
            exp.setup(
                data=self.data_train,
                test_data=self.data_test,
                target=self.target,
                fold=5,
                session_id=session_id,
                verbose=False
            )
            self.experiments[f'Experiment_{exp_number}'] = {}
            self.experiments[f'Experiment_{exp_number}']['exp'] = exp
            self.experiments[f'Experiment_{exp_number}']['models'] = []               # Will contain the selected non-tuned models
            self.experiments[f'Experiment_{exp_number}']['models_results'] = ''       # Will be a dataframe with the results of the selected non-tuned models
            self.experiments[f'Experiment_{exp_number}']['tuned_models'] = []         # Will contain the selected tuned models
            self.experiments[f'Experiment_{exp_number}']['tuned_models_results'] = '' # Will be a dataframe with the results of the selected tuned models
            print(f'  - Experiment_{exp_number} created with session id: {session_id}.')

        self.created_experiments = True


    def select_models(self):
        """
        Method that runs compare_models and selects the top 10 models to continue working on. Then stores the performances
        of the models that were selected in the experiments dictionary.
        """

        if self.created_experiments:
            print(f'Selecting top 10 models from each experiment and evaluating them:')
            for experiment in self.experiments.keys():
                exp = self.experiments[experiment]['exp']
                top_10_models = exp.compare_models(n_select=10, verbose=False)
                #model_names = [type(model).__name__ for model in top_10_models]
                #if 'LinearRegression' not in model_names:
                #    lr = exp.create_model('lr', verbose=False)
                #    top_10_models.append(lr)
                models_performances = get_model_performances(exp, top_10_models)
                results_df = performances_dict_to_df(models_performances)
                results_df = results_df.sort_values(by='R2_val_mean', ascending=False)
                self.experiments[experiment]['models'] = top_10_models
                self.experiments[experiment]['models_results'] = results_df
                print(f'  - {experiment} model selection completed.')
            
            self.selected_models = True
        else:
            raise Exception('Model selection not possible without first creating experiments with "create_experiments()".')



    def tune_models(self):
        """
        Goes through all selected models of each experiment, tunes them and then evaluates them. The results are then stored
        in the experiments dictionary. Can be quite computationally expensive to run!
        """
        
        if self.selected_models:
            print(f'Tuning the top 10 models from each experiment and evaluating them:')
            for experiment in self.experiments.keys():
                print(f'  - Tuning models of {experiment}:')
                exp = self.experiments[experiment]['exp']
                tuned_models = []
                for model in self.experiments[experiment]['models']:
                    model_name = type(model).__name__
                    tuned_model = exp.tune_model(model, fold=5, search_library='optuna', n_iter=200, choose_better=False, verbose=False)
                    tuned_models.append(tuned_model)
                    print(f'    - {model_name} tuning completed.')
                models_performances = get_model_performances(exp, tuned_models)
                results_df = performances_dict_to_df(models_performances)
                results_df = results_df.sort_values(by='R2_val_mean', ascending=False)
                self.experiments[experiment]['tuned_models'] = tuned_models
                self.experiments[experiment]['tuned_models_results'] = results_df
                print(f'  - Tuning models of {experiment} completed.')
                print('='*90)
            
            self.tuned_models = True
        else:
            raise Exception('Model tuning not possible without first selecting models with "select_models()".')


    def rank_models(self, metric: str='R2'):
        """
        Prints the ranking of the models from the current analysis according to the selected metric.
        """
        for experiment in self.experiments.keys():

            print(f'Models Ranking for {experiment}:')
            print(f'  - Non-Tuned Models:')
            if self.selected_models:
                models_results = self.experiments[experiment]['models_results']
                models_results = models_results.sort_values(by=f'{metric}_val_mean', ascending=False)
                for rank, model in enumerate(models_results['model_name']):
                    print(f'    {rank+1}) {model}')
            else:
                print('    - No selected models.')
            print(f'  - Tuned Models:')
            if self.tuned_models:
                tuned_models_results = self.experiments[experiment]['tuned_models_results']
                tuned_models_results = tuned_models_results.sort_values(by=f'{metric}_val_mean', ascending=False)
                for rank, tuned_model in enumerate(tuned_models_results['model_name']):
                    print(f'    {rank+1}) {tuned_model}')
            else:
                print('    - No tuned models.')
            print()