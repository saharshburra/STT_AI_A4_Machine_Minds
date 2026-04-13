import streamlit as st
import pickle
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


@st.cache_data
def load_location_city_map():
    try:
        df = pd.read_csv("Dataset/train.csv")
        if "location" not in df.columns or "city" not in df.columns:
            return {}

        # Use the most frequent city seen for each location.
        location_city_map = (
            df.groupby("location")["city"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
            .to_dict()
        )
        return location_city_map
    except Exception:
        return {}


location_city_map = load_location_city_map()

# Match notebook preprocessing: unseen categories are encoded as -1.
def encode_with_unknown_as_minus_one(encoder, value):
    value_str = str(value)
    if value_str in encoder.classes_:
        return int(encoder.transform([value_str])[0])
    return -1


def get_default_city_for_location(location_value):
    mapped_city = location_city_map.get(location_value)
    if mapped_city in city_encoder.classes_:
        return mapped_city
    return city_encoder.classes_[0]


def update_city_from_location():
    selected_location = st.session_state.get("selected_location")
    st.session_state["selected_city"] = get_default_city_for_location(selected_location)


if "selected_location" not in st.session_state:
    st.session_state["selected_location"] = location_encoder.classes_[0]

if "selected_city" not in st.session_state:
    st.session_state["selected_city"] = get_default_city_for_location(st.session_state["selected_location"])

# UI
st.title("🏠 UrbanNest Rent Prediction")
st.caption("Fill property details below and get an instant rent estimate.")

st.sidebar.header("About")
st.sidebar.write("Predict house rent using Machine Learning")

st.subheader("Property Details")

# 3-column responsive grid for better input experience
col1, col2, col3 = st.columns(3)

with col1:
    location = st.selectbox(
        "Location",
        location_encoder.classes_,
        key="selected_location",
        on_change=update_city_from_location,
    )
    status = st.selectbox("Status", status_encoder.classes_)
    bathrooms = st.number_input("Number of Bathrooms", min_value=1, value=1, step=1)
    size = st.number_input("Size (sq ft)", min_value=100, value=600, step=10)
    verification_days = st.number_input("Verification Days", min_value=0, value=0, step=1)

with col2:
    city = st.selectbox("City", city_encoder.classes_, key="selected_city")
    property_type = st.selectbox("Property Type", type_encoder.classes_)
    balconies = st.number_input("Number of Balconies", min_value=0, value=1, step=1)
    bhk = st.number_input("BHK", min_value=1, value=2, step=1)
    rooms = st.number_input("Total Rooms", min_value=1, value=3, step=1)

with col3:
    latitude = st.number_input("Latitude", value=19.0, format="%.6f")
    longitude = st.number_input("Longitude", value=72.0, format="%.6f")
    isNegotiable = st.selectbox("Negotiable", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    security = st.number_input("Security Deposit", min_value=0, value=50000, step=1000)

# Input Validation
if size < 200:
    st.warning("⚠️ Size seems too small")

if bathrooms > rooms:
    st.error("❌ Bathrooms cannot exceed total rooms")

if latitude == 0 or longitude == 0:
    st.warning("⚠️ Location coordinates look unusual")

# Prediction
if st.button("Predict Rent"):
    # Stop if critical error
    if bathrooms > rooms:
        st.stop()

    # Encode categorical features
    location_enc = encode_with_unknown_as_minus_one(location_encoder, location)
    city_enc = encode_with_unknown_as_minus_one(city_encoder, city)
    status_enc = encode_with_unknown_as_minus_one(status_encoder, status)
    type_enc = encode_with_unknown_as_minus_one(type_encoder, property_type)

    # Feature vector with proper column names to avoid sklearn warning
    features = pd.DataFrame([[location_enc, city_enc, latitude, longitude,
                              bathrooms, balconies, isNegotiable, security,
                              status_enc, size, bhk, rooms,
                              type_enc, verification_days]], 
                            columns=model.feature_names_in_)

    # Predict
    prediction = model.predict(features)

    # Output
    st.success(f"💰 Estimated Rent: ₹{prediction[0]:,.2f}")
