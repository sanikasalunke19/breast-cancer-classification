# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 11:59:33 2024
@author: Admin
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('C:/cancer/trained_model.sav', 'rb'))

def breastcancer(input_data):
    # Convert input_data to a NumPy array
    input_data_array = np.asarray(input_data, dtype=np.float64)
    
    # Reshape the array as we are predicting for one data point
    input_data_reshaped = input_data_array.reshape(1, -1)

    # Standardize the input data if needed
    # You can skip this step if your model was trained on non-standardized data
    # input_data_reshaped = standard_scaler.transform(input_data_reshaped)

    prediction = loaded_model.predict(input_data_reshaped)
    
    if prediction == 0:
        return "Malignant cancer"
    else:
        return "Benign cancer"

def main():
    st.title('BREAST CANCER CLASSIFICATION WEBAPP')

    radius_mean = st.text_input('radius')
    texture_mean = st.text_input('mean texture')
    perimeter_mean = st.text_input('mean perimeter')
    area_mean = st.text_input('mean area')
    smoothness_mean = st.text_input('mean smoothness')
    compactness_mean = st.text_input('mean compactness')
    concavity_mean = st.text_input('mean concavity')
    concave_points_mean = st.text_input('mean concave points')
    symmetry_mean = st.text_input('mean symmetry')
    fractal_dimension_mean = st.text_input('mean fractal dimension')
    radius_se = st.text_input('SE of radius')
    texture_se = st.text_input('SE of texture') 
    perimeter_se = st.text_input('SE of perimeter') 
    area_se = st.text_input('SE of area')
    smoothness_se = st.text_input('SE of smoothness')
    compactness_se = st.text_input('SE of compactness')
    concavity_se = st.text_input('SE of concavity')
    concavepoints_se = st.text_input('SE of concave points')
    symmetry_se = st.text_input('SE of symmetry')
    fractal_dimension_se = st.text_input('SE of fractal dimension')
    radius_worst = st.text_input('worst radius')
    texture_worst = st.text_input('worst texture')
    perimeter_worst = st.text_input('worst perimeter')
    area_worst = st.text_input('worst area')
    smoothness_worst = st.text_input('worst smoothness')
    compactness_worst = st.text_input('worst compactness')
    concavity_worst = st.text_input('worst concavity')
    concavepoints_worst = st.text_input('worst concave points')
    symmetry_worst = st.text_input('worst symmetry')
    fractal_dimension_worst = st.text_input('worst fractal dimension')

    diagnosis = ''

    if st.button('Cancer Test Result'):
        diagnosis = breastcancer([radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
                                  compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
                                  fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se,
                                  smoothness_se, compactness_se, concavity_se, concavepoints_se, symmetry_se,
                                  fractal_dimension_se, radius_worst, texture_worst, perimeter_worst,
                                  area_worst, smoothness_worst, compactness_worst, concavity_worst,
                                  concavepoints_worst, symmetry_worst, fractal_dimension_worst])

    st.success(diagnosis)

if __name__ == '__main__':
    main()

