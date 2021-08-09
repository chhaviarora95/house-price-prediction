# Welcome to House Price Estimator!

![house-sale](https://images.app.goo.gl/eSiFzxdvUgxmMzTQA/to/.png)

**Attention Seattle folks! Do you wonder what could be the market worth of your humble abode?**

# Goal
The purpose of this project is to use data transformation and machine learning to create a model that will predict price for a house when given features like bedrooms, bathrooms, floors, view, condition, grade, sqft_living etc.

# Dataset Used 
[Kaggle Seattle Real Estate dataset](https://www.kaggle.com/harlfoxem/housesalesprediction)

# Methodology
- Data Wrangling - `Removed duplicates and irrelevant features.`
 - Exploratory Data Analysis - `Analyzed the data and visualized the features vs the target variable, price of the house.`
 - Data Visualization - `Used boxplot, scatter plot & correlation matrices to visualize the data and it's characteristics.`
 -  Feature Engineering - `Create multiple new features from manipulating existing features.`
 - Machine Learning Algorithms Used - `Linear Regression, Polynomial Transformation,Random Forest & Gradient Boosting.`
 - Evaluation Metrics Used - `Mean Absolute Error(MSE) and R-squared`

# Technologies/Libraries Used
``` javascript
 - Python 3
 - Jupyter
 - Pandas
 - Numpy
 - Seaborn
 - Matplotlib
 - Plotly
 - Scikit-Learn
 - Gradio
 ```

## Description of the features

-  **Bedrooms:** Number of bedrooms.
- **Bathrooms:** Number of bathrooms.
- **Sqft_living:** Square footage of the apartments interior living space.
- **Sqft_lot:** Square footage of the land space.
- **Floors:** Number of floors.
- **Waterfront:** A dummy variable for whether the apartment was overlooking the waterfront or not.
- **View:** An index from 0 to 4 of how good the view of the property was.
- **Condition:** An index from 1 to 5 on the condition of the apartment,  
- **Grade:** An index from 1 to 13, where 1-3 falls short of building construction and design, 7 has an average level of construction and design, and 11-13 have a high quality level of construction and design.  
- **Sqft_above:** The square footage of the interior housing space that is above ground level.
- **Sqft_basement:** The square footage of the interior housing space that is below ground level.
- **Yr_built:** The year the house was initially built.
- **Yr_renovated:** The year of the house’s last renovation.  
- **Zipcode:** What zipcode area the house is in (Seattle for this dataset).
- **Lat:**  Lattitude.  
- **Long:** Longitude. 
- **Sqft_living15:** The square footage of interior housing living space for the nearest 15 neighbors. 
- **Sqft_lot15:** The square footage of the land lots of the nearest 15 neighbors.

# Summary

*Scaling the features, generating new features with Gradient boosting regressor model with hyperparameter tuning led to the most accurate predictions with the least error. The result was a **mean absolute error of 92,000** with a **82% R^2 score**.<br>
This model can be used as a guide when determining house price estimates for Seattle since it leads to reasonable predictions.*

# To run locally

``` 
 - Clone this repo.
 - Change directory to the repo and run this cmd:
   python main.py
 - The predictions for the sample test file with details
   of two houses will be saved to the folder.
 ```

# Access live model

- **Go to this link-**  
- Enter all details of the house
- Please make sure to use **Seattle Zipcodes, lat and long**
   values. You could use the following example:
    - **Zipcode:** 98178
    - **Lat**:
    - **Long**:
 - Click Submit
 - **Scroll up to to the top to see the predcicted price!**

 
