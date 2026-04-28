import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("California House Price Predictor")
st.write("Enter house features to get a predicted median price.")

# load model and scaler
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# input fields
med_inc = st.slider("Median Income (in $10k)", 0.5, 15.0, 3.0)
house_age = st.slider("House Age", 1, 52, 20)
ave_rooms = st.slider("Average Rooms", 1.0, 10.0, 5.0)
ave_bedrms = st.slider("Average Bedrooms", 0.5, 5.0, 1.0)
population = st.number_input("Population in Block", 100, 40000, 1000)
ave_occup = st.slider("Average Occupants", 1.0, 10.0, 3.0)
latitude = st.slider("Latitude", 32.0, 42.0, 34.0)
longitude = st.slider("Longitude", -125.0, -114.0, -118.0)

# input dataframe with engineered features
input_data = pd.DataFrame({
    'MedInc': [med_inc],
    'HouseAge': [house_age],
    'AveRooms': [ave_rooms],
    'AveBedrms': [ave_bedrms],
    'Population': [population],
    'AveOccup': [ave_occup],
    'Latitude': [latitude],
    'Longitude': [longitude],
    'RoomsPerHouse': [ave_rooms / ave_occup],
    'BedroomsRatio': [ave_bedrms / ave_rooms],
    'PopulationPerHouse': [population / ave_occup],
    'IncomePerPerson': [med_inc / population] 
})

# scale input and predict
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]

st.success(f"**Predicted Median House Price: ${prediction*100000:,.0f}**")