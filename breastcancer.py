# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle

loaded_model = pickle.load(open('C:/cancer/trained_model.sav','rb'))

input_data = (20.57,17.77,132.9,1326,0.08474,0.07864,0.0869,0.07017,0.1812,0.05667,0.5435,0.7339,3.398,74.08,0.005225,0.01308,0.0186,0.0134,0.01389,0.003532,24.99,23.41,158.8,1956,0.1238,0.1866,0.2416,0.186,0.275,0.08902)
#change input data to numpy array
input_data_as_array = np.asarray(input_data)
#reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)
if(prediction == 0):
  print("Malignant cancer")
else:
  print("belignant cancer")