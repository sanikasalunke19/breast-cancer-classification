# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 11:59:33 2024

@author: Admin
"""

import numpy as np
import pickle
import streamlit

loaded_model = pickle.load(open('C:/cancer/trained_model.sav','rb'))

#creating a function

def breastcancer(input_data):
    #input_data = (20.57,17.77,132.9,1326,0.08474,0.07864,0.0869,0.07017,0.1812,0.05667,0.5435,0.7339,3.398,74.08,0.005225,0.01308,0.0186,0.0134,0.01389,0.003532,24.99,23.41,158.8,1956,0.1238,0.1866,0.2416,0.186,0.275,0.08902)
    #change input data to numpy array
    input_data_as_array = np.asarray(input_data)
    #reshape the numpy array as we are predicting for one datapoint
    input_data_reshaped = input_data_as_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    if(prediction == 0):
      return "Malignant cancer"
    else:
      return "belignant cancer"
  
def main():
    
    st.title('BREAST CANCER CLASSIFICATION WEBAPP')
    
    #"radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"
    radius_mean = st.text_input('radius')
    texture_mean = st.text_input('mean texture')
    perimeter_mean = st.text_input('mean perimeter')
    area_mean = st.text_input('mean area')
    smoothness_mean = st.text_input('mean smoothness')
    compactness_mean = st.text_input('mean compactness')
    concavity_mean = st.text_input('mean concativity')
    concave_points_mean = st.text_input('mean concave points')
    symmetry_mean = st.text_input('mean symmetry')
    fractal_dimension_mean = st.text_input('mean fractal dimension')
    radius_se = st.text_input('SE of radius')
    texture_se = st.text_input('SE of texture') 
    perimeter_se = st.text_input('SE of perimeter') 
    area_se = st.text_input('SE of area')
    smoothness_se = st.text_input('SE of smoothness')
    compactness_se = st.text_input('SE of compactness')
    concavity_se = st.text_input('SE of concativity')
    concavepoints_se = st.text_input('SE of concave points')
    symmetry_se = st.text_input('SE of symmetry')
    fractal_dimension_se = st.text_input('SE of fractal dimension')
    radius_worst = st.text_input('worst radius')
    texture_worst = st.text_input('worst texture')
    perimeter_worst = st.text_input('worst perimeter')
    area_worst = st.text_input('worst area')
    smoothness_worst = st.text_input('worst smoothness')
    compactness_worst = st.text_input('worst compactness')
    concavity_worst = st.text_input('worst concativity')
    concavepoints_worst = st.text_input('worst concave points')
    symmetry_worst = st.text_input('worst symmetry')
    fractal_dimension_worst = st.text_input('worst fractal dimension')

    #code for prediction
    diagnosis = ''

    #creating a button
    if st.button('cancer test result'):
        diagnosis = breastcancer([radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst])
        
    st.success(diagnosis)

if(__name__ == '__main__'):
   main()  
    

        
        
        
        