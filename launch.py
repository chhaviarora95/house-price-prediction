from random import sample
import pandas as pd
import numpy as np
import gradio as gr
import sys

from joblib import load

sys.path.append('src/')

FILENAME= 'trained_models/House_price_prediction_model.joblib'

# load the model
model = load(FILENAME)

# define the prediction function
def predict(
    id,
    date,
    bedrooms,
    bathrooms,
    sqft_living,
    sqft_lot,
    floors,
    is_waterfront,
    view,
    condition,
    grade, 
    sqft_above,
    sqft_basement,
    yr_built,
    yr_renovated,
    zipcode,
    lat,
    long,
    sqft_living15,
    sqft_lot15
):
    
  df = pd.DataFrame(
        {
            'id':[id] , 'date': [date], 'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],'sqft_living': [sqft_living],
            'sqft_lot': [sqft_lot], 'floors': [floors], 
            'waterfront': [is_waterfront], 'view': [view],
            'condition': [condition], 'grade': [grade],
            'sqft_above': [sqft_above], 'sqft_basement': [sqft_basement],
            'yr_built': [yr_built], 'yr_renovated': [yr_renovated],
            'zipcode': [zipcode],'lat': [lat], 'long': [long],
            'sqft_living15': [sqft_living15], 'sqft_lot15': [sqft_lot15]
                
        }
  )

  pred = model.predict(df)
  house_price = "Your estimated house price is: ${}".format(np.round(pred[0],2))
  return house_price

bedrooms = gr.inputs.Dropdown([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
bathrooms = gr.inputs.Dropdown([0, 1, 2, 3, 4, 5, 6, 7, 8])
floors = gr.inputs.Dropdown([0, 1, 2, 3, 3.5])
view = gr.inputs.Dropdown([0, 1, 2, 3, 4], label="How good is the view?")
condition = gr.inputs.Dropdown([1, 2, 3, 4, 5],
                              label="How good is the condition of the house?")
grade = gr.inputs.Dropdown([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                           label="How good is the quality of construction and design?")

sample_seattle= [

             [1310430130, '2014-05-02',3,2,1340,7912, 2, 0, 0, 3,
             3, 1340, 0, 1955, 2005, 98133, 47.7393, -122.3435, 1500, 7000]
]

iface = gr.Interface(
  fn=predict, 
  inputs=[
      "number", "text",bedrooms , bathrooms,
      gr.inputs.Slider(290, 13540), gr.inputs.Slider(520, 16000), floors,
      "checkbox", view, condition, 
      grade, gr.inputs.Slider(290, 9410), gr.inputs.Slider(0, 4820),
      "number", 'number', 'number', 'number', 'number', gr.inputs.Slider(399, 6210),
      gr.inputs.Slider(651, 16000)
      ],
  outputs=["text"],
  title="House price estimate",
  description='Attention Seattle folks! Do you wonder what could be the \
               market worth of your humble abode?',
  examples=sample_seattle
)

iface.launch(share=True)