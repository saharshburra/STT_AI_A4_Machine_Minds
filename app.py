import streamlit as st
import pickle
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Load Model and encoders
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("models/best_rf_model.pkl", "rb"))
    encoders = pickle.load(open("models/label_encoders.pkl", "rb"))
    return model, encoders

model, encoders = load_artifacts()

# Extract encoders
location_encoder = encoders["location"]
city_encoder = encoders["city"]
status_encoder = encoders["Status"]
type_encoder = encoders["property_type"]

# Safe encoding function
def safe_encode(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        return -1  # unseen category fallback

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

# Input Validation
if size < 200:
    st.warning("⚠️ Size seems too small")

if bathrooms > rooms:
    st.error("❌ Bathrooms cannot exceed total rooms")

if price_sqft <= 0:
    st.error("❌ Price per sqft must be positive")
    
if latitude == 0 or longitude == 0:
    st.warning("⚠️ Location coordinates look unusual")

# Prediction
if st.button("Predict Rent"):
    # Stop if critical error
    if bathrooms > rooms or price_sqft <= 0:
        st.stop()

    # Encode categorical features
    location_enc = safe_encode(location_encoder, location)
    city_enc = safe_encode(city_encoder, city)
    status_enc = safe_encode(status_encoder, status)
    type_enc = safe_encode(type_encoder, property_type)

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
