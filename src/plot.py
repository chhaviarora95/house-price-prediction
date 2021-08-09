import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns

"""
A set of helper functions to generate plots.
"""

def generate_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str
):

    """ Generate scatter plot. """
    
    fig = px.scatter(df, x=x_col, y=y_col)
    fig.show()

def generate_box_plot(
    df: pd.DataFrame,
    col: str
):
    
    """ Generate box plot. """
    
    fig = px.box(df, y=col)
    fig.show()

def generate_dist_plot(
    df: pd.DataFrame,
    col: str
):
    
    """ Generate distribution plot. """
    
    plt.figure(figsize = (12,8))
    sns.distplot(df[col])
    plt.xlabel(f'{col}')
    plt.show()

def generate_line_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str
):
    
    """ Generate line plot. """
    
    plt.figure(figsize=(16,8))
    plt.title(f'{title}')
    sns.lineplot(df[x_col], df[y_col])
    plt.ylabel(f'{y_col}')
    plt.xlabel(f'{x_col}')
    plt.show()

def generate_corr_matrix_plot(
    df: pd.DataFrame
):
    
    """ Generate correlation matrix plot. """
    
    df_corr = df.corr()
    plt.subplots(figsize=(20,10))
    sns.heatmap(df_corr, cmap = 'BuGn', linewidth =.005, annot = True)
    plt.show()

def generate_feature_importances_plot(
    model_obj,
    feat_df: pd.DataFrame
):
    
    """ Generate feature importances plot. """
    
    feat_importances = pd.Series(model_obj.feature_importances_,
                                 index=feat_df.columns
                                )

    feat_importances.nlargest(10).plot(kind='barh')
    plt.show()
