from numpy.lib.shape_base import column_stack
import pandas as pd
import numpy as np

from datetime import datetime
from geopy.distance import geodesic

"""
Preprocessor class

"""

class DataPreprocessor:
    
    def __init__(self, datecols, cols_to_filter=None):
        
        self.cols_to_filter = cols_to_filter
        self.datecols = datecols
    
    def fit(self, X, y=None):
        """learn any information from the training data we may need to 
           transform the test data"""
        
        # learn from the training data and return the class itself. 

        return self
    
    def transform(self, X, y=None):
        """transform the training or test data"""
        
        X_new = X.copy()

        # strip year and month from date col
        for x in self.datecols:
            X_new['date'] = pd.to_datetime(X_new['date'], format="%Y-%m-%d")
            X_new['sale_year'] = X_new['date'].dt.year
            X_new['sale_month'] = X_new['date'].dt.month
            X_new = X_new.drop('date', axis = 1)
        
        # create new feature - age of the house
        X_new['age'] = X_new['sale_year'] - (X_new['yr_built'])

        # partition the age into bins 
        bins = [-2,0,5,10,25,50,75,100,100000]
        labels = ['<1','1-5','6-10','11-25','26-50','51-75','76-100','>100']
        X_new['age_binned'] = pd.cut(X_new['age'], bins=bins, labels=labels)
        
        # one-hot encoding of age_binned
        X_new = pd.get_dummies(X_new, columns=['age_binned'])
        
        # create new features - is_renovated & has_basement
        X_new['is_renovated'] = np.where(X_new['yr_renovated']==0, 0, 1)
        X_new['has_basement'] = np.where(X_new['sqft_basement']==0, 0, 1)
        
        # create new features from zipcode
        counts_per_zip = X_new.groupby('zipcode').size().to_dict()
        X_new['num_houses_sold_in_zip_code'] = X_new['zipcode'].apply(lambda x: counts_per_zip[x])

        avg_sqft_per_zip = X_new.groupby('zipcode').agg({'sqft_living':'mean'}).to_dict()
        X_new['avg_house_size_in_zip_code'] = X_new['zipcode'].apply(lambda x: avg_sqft_per_zip['sqft_living'][x])
        
        # create new feature from lat and long
        downtown =( 47.6205, -122.3493)
        X_new['coordinates'] = list(zip(X_new.lat, X_new.long))
        X_new['km_dist_from_downtown'] = X_new['coordinates'].apply(lambda x: geodesic(downtown, x).km)
        
        # create ratio feature by neighborhood averages
        X_new['ratio_sqft_living_by_neighbours'] = X_new['sqft_living']/X_new['sqft_living15']
        X_new['ratio_sqft_lot_by_neighbours'] = X_new['sqft_lot']/X_new['sqft_lot15']

        # filter
        self.cols_to_filter.extend(['age','yr_built', 'yr_renovated',
                                    'zipcode', 'coordinates', 'sqft_living15',
                                    'sqft_lot15', 'lat', 'long'])
        X_new = X_new.drop(self.cols_to_filter, axis=1)

        return X_new
    
    def fit_transform(self, X, y=None):
        """fit and transform wrapper method, used for sklearn pipeline"""

        return self.fit(X).transform(X)