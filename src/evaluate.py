import pandas as pd
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_validate

from typing import List, Dict, Union

SCORER = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']

def get_scoring_metrics(
    model_obj,
    feat: pd.DataFrame,
    target: pd.Series,
    pred: pd.DataFrame
) -> Dict[str, float]:
    
    """ Generate eval scoring metrics.

    Args:
        model_obj: trained model object.
        feat: dataframe with features.
        target: series with target variable.
        pred: array of predictions.

    Returns:
        dict of eval metric scores.
    """
    
    return {'mse': math.sqrt(mean_squared_error(target, pred)),
            'mae': mean_absolute_error(target, pred),
            'r2': model_obj.score(feat, target)
           }

def get_cross_val_scores(
    feat_train: pd.DataFrame,
    target_train: pd.Series,
    model_obj: str,
    scorer: List[str] = SCORER,
    cv: int = 5
) -> Dict:

    """ Generate cross validation scores.

    Args:
        feat_train: training dataframe with all features.
        target_train: training series with target variable.
        model_obj: model object.
        scorer: list of scoring metrics.
        cv: number of folds for cross validation.
    
    Returns:
        dict of cross val scores.
    """

    scores = cross_validate(
                    model_obj,
                    feat_train,
                    target_train, 
                    cv = cv,
                    scoring = scorer,
                    return_train_score=True
            )

    return scores
    