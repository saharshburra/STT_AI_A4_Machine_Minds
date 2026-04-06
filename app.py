import streamlit as st
import pickle
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Load Model
model = pickle.load(open("models/best_rf_model.pkl", "rb"))

# Load Encoders
encoders = pickle.load(open("models/label_encoders.pkl", "rb"))

location_encoder = encoders["location"]
city_encoder = encoders["city"]
status_encoder = encoders["Status"]
type_encoder = encoders["property_type"]

# UI
st.title("🏠 UrbanNest Rent Prediction")

st.sidebar.header("About")
st.sidebar.write("Predict house rent using Machine Learning")

# Categorical Inputs
location = st.selectbox("Location", location_encoder.classes_)
city = st.selectbox("City", city_encoder.classes_)
status = st.selectbox("Status", status_encoder.classes_)
property_type = st.selectbox("Property Type", type_encoder.classes_)

# Numerical Inputs
latitude = st.number_input("Latitude", value=19.0)
longitude = st.number_input("Longitude", value=72.0)

bathrooms = st.number_input("Number of Bathrooms", min_value=1)
balconies = st.number_input("Number of Balconies", min_value=0)

isNegotiable = st.selectbox("Negotiable (0 = No, 1 = Yes)", [0, 1])

security = st.number_input("Security Deposit", min_value=0)

size = st.number_input("Size (sq ft)", min_value=100)
price_sqft = st.number_input("Price per sqft", min_value=1)

bhk = st.number_input("BHK", min_value=1)
rooms = st.number_input("Total Rooms", min_value=1)

verification_days = st.number_input("Verification Days", min_value=0)

# Prediction
if st.button("Predict Rent"):

    # Encode categorical features
    location_enc = location_encoder.transform([location])[0]
    city_enc = city_encoder.transform([city])[0]
    status_enc = status_encoder.transform([status])[0]
    type_enc = type_encoder.transform([property_type])[0]

    # Feature vector with proper column names to avoid sklearn warning
    features = pd.DataFrame([[location_enc, city_enc, latitude, longitude,
                              bathrooms, balconies, isNegotiable, security,
                              status_enc, size, price_sqft, bhk, rooms,
                              type_enc, verification_days]], 
                            columns=model.feature_names_in_)

    # Predict
    prediction = model.predict(features)

    # Output
    st.success(f"💰 Estimated Rent: ₹{prediction[0]:,.2f}")
