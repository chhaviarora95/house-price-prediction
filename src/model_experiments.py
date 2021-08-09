import pandas as pd
import utils as u
import evaluate as evl
import math

from typing import List, Dict
from sklearn.preprocessing import PolynomialFeatures
from termcolor import cprint

def run_model_experiments(
    model_obj,
    model_name: str,
    feat_train: pd.DataFrame,
    target_train: pd.Series,
    feature_sets_dict: Dict[str, List],
    metrics_list: List,
    is_poly: bool = False
) -> List[Dict]:
    
    """ Run model experiments for different feature sets.

    Args:
        model_obj: model object to run experiments on.
        model_name: name of the model.
        feat_train: training dataframe with features.
        target_train: training series with target variable.
        feature_sets_dict: dict of all feature sets to experiment on.
        metrics_List: list to append experiment results in.
        is_poly: bool value for whether there needs to be polynomial
                 transformation.

    Returns:
        List of experiment scoring results.
    """

    if is_poly == True:
        model_pipeline = u.create_model_pipeline(model_obj, is_poly=True)
        model_pipeline.steps.insert(2, ['poly_transform', PolynomialFeatures(2)])
    else:
        model_pipeline = u.create_model_pipeline(model_obj)

    for i, (feature_set_name, feature_set) in enumerate(feature_sets_dict.items()):

        metrics_dict = {}
        metrics_dict['model_name'] = model_name
        metrics_dict['feature_set'] = feature_set_name
        
        cprint(f'[TRAINING] {model_name} ------', color = 'red')

        select_feat_train = feat_train[feature_set]

        scores = evl.get_cross_val_scores(
                        select_feat_train,
                        target_train,
                        model_pipeline
                )
        
        metrics_dict['train_mse_mean'] = math.sqrt(-(scores['train_neg_mean_squared_error'].mean()))
        metrics_dict['test_mse_mean'] = math.sqrt(-(scores['test_neg_mean_squared_error'].mean()))
        metrics_dict['train_mse_std'] = math.sqrt((scores['train_neg_mean_squared_error'].std()))
        metrics_dict['test_mse_std'] = math.sqrt((scores['test_neg_mean_squared_error'].std()))
        
        metrics_dict['train_mae_mean'] = -(scores['train_neg_mean_absolute_error'].mean())
        metrics_dict['test_mae_mean'] = -(scores['test_neg_mean_absolute_error'].mean())
        metrics_dict['train_mae_std'] = (scores['train_neg_mean_absolute_error'].std())
        metrics_dict['test_mae_std'] = (scores['test_neg_mean_absolute_error'].std())

        metrics_dict['train_r2_mean'] = (scores['train_r2'].mean())
        metrics_dict['test_r2_mean'] = (scores['test_r2'].mean())
        metrics_dict['train_r2_std'] = (scores['train_r2'].std())
        metrics_dict['test_r2_std'] = (scores['test_r2'].std())

        metrics_list.append(metrics_dict)
        cprint(f'[EXP {i}] {feature_set_name} [DONE]', color = 'green')
    
    return metrics_list