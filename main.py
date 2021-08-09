import pandas as pd
import numpy as np
import sys

from joblib import load

sys.path.append('src/')

MODEL_LOCATION= 'trained_models/House_price_prediction_model.joblib'
TEST_DATA_LOCATION= 'testing_data/test.csv'

# load the model
model = load(MODEL_LOCATION)

# load sample testing dataset 
test_df = pd.read_csv(TEST_DATA_LOCATION)

def predict(
    df: pd.DataFrame
):
    pred = model.predict(df)
    return np.round(pred, 2)


if __name__ == '__main__':
    
    price_pred = predict(test_df)
    print('Predicted house price(s) are saved to folder successfully!')
    np.savetxt("predictions.csv", price_pred, delimiter=',')

