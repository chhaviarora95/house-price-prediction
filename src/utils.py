import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from termcolor import cprint
from typing import List, Dict, Any, Tuple

from preprocess import DataPreprocessor

""" Utility functions """

def get_train_test_split(
    feat_df: pd.DataFrame,
    target_df: pd.Series,
    test_size: float = 0.2,
    random_state: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    """ Create train, test data frames.

    Args:
        feat_df: dataframe with features.
        target_df: series with target variable.
        test_size: test size to split dataframes.
        random_state: random_state to split.

    Returns:
        feat_train, target_train, feat_test and target_test dfs
    """
    feat_train, feat_test, target_train, target_test = train_test_split(
                                                            feat_df,
                                                            target_df, 
                                                            test_size=test_size,
                                                            random_state=random_state
                                                        )
    return feat_train, feat_test, target_train, target_test

def get_fitted_grid_object(
    model_obj,
    params: List[Dict[str, List]],
    feat_train: pd.DataFrame,
    target_train: pd.Series,
    cv: int = 5
) -> Any:
    
    """ Generate trained GridSearchCV object.

    Args:
        model_obj: model object to run gridsearch on.
        params: list of parameters to run gridsearch. 
                For eg: [ {'alpha': [0.1, 1]
                          } 
                        ]
        feat_train: training dataframe with features.
        target_train: training series with target variable.
        random_state: random_state to split.
    
    Returns:
        grid: fitted GridSearchCV object.
    """
    cprint('[Running Gridsearch]', color = 'blue')
    grid_pipeline = create_model_pipeline(model_obj, is_grid=True)
    grid = GridSearchCV(
                    grid_pipeline,
                    params,
                    cv = cv
            )
    grid.fit(feat_train, target_train)

    return grid

def create_model_pipeline(
    model_obj,
    is_grid: bool = False,
    is_poly: bool = False
):
    """ Create model pipeline.

    Args:
        model_obj: model object to create pipeline.
        is_grid: bool value for whether this is GridSearchCV
                 pipeline. 
        is_poly: bool value for whether there needs to be polynomial
                 transformation.
            
    Returns:
        pipe: model pipeline.
    """

    input = [
                ('preprocess', DataPreprocessor(
                                datecols = ['date'],
                                cols_to_filter = ['id', 'sqft_above', 'sqft_basement']
                                )
                ),
                ('scale', StandardScaler()),
                ('model', model_obj)
            ]

    pipe = Pipeline(input)
    model = TransformedTargetRegressor(
                        regressor = pipe, 
                        func=np.log, 
                        inverse_func=np.exp 
                    )
    
    return pipe if (is_grid == True) | (is_poly == True) else model

