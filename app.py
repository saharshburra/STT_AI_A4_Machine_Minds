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
def load_training_maps():
    try:
        df = pd.read_csv("Dataset/train.csv")
        if "location" not in df.columns or "city" not in df.columns:
            return {}, {}, {}

        # Use the most frequent city seen for each location.
        location_city_map = (
            df.groupby("location")["city"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
            .to_dict()
        )

        # Use mean coordinates per location as sensible defaults for the UI.
        if "latitude" in df.columns and "longitude" in df.columns:
            location_coords_map = (
                df.groupby("location")[["latitude", "longitude"]]
                .mean()
                .to_dict("index")
            )
            city_geo_bounds = (
                df.groupby("city")[["latitude", "longitude"]]
                .agg(["min", "max"])
                .to_dict()
            )
        else:
            location_coords_map = {}
            city_geo_bounds = {}

        return location_city_map, location_coords_map, city_geo_bounds
    except Exception:
        return {}, {}, {}


location_city_map, location_coords_map, city_geo_bounds = load_training_maps()

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


def get_default_coords_for_location(location_value):
    coords = location_coords_map.get(location_value, {})
    lat = float(coords.get("latitude", 19.0))
    lon = float(coords.get("longitude", 72.0))
    return lat, lon


def get_city_geo_bounds(city_value):
    # Build resilient lookup from the nested dict created by DataFrame.to_dict().
    lat_min = city_geo_bounds.get(("latitude", "min"), {}).get(city_value)
    lat_max = city_geo_bounds.get(("latitude", "max"), {}).get(city_value)
    lon_min = city_geo_bounds.get(("longitude", "min"), {}).get(city_value)
    lon_max = city_geo_bounds.get(("longitude", "max"), {}).get(city_value)

    if None in (lat_min, lat_max, lon_min, lon_max):
        return 6.0, 38.0, 68.0, 98.0

    # Add a small padding so users can make minor adjustments.
    return float(lat_min) - 0.2, float(lat_max) + 0.2, float(lon_min) - 0.2, float(lon_max) + 0.2


def update_city_from_location():
    selected_location = st.session_state.get("selected_location")
    st.session_state["selected_city"] = get_default_city_for_location(selected_location)
    default_lat, default_lon = get_default_coords_for_location(selected_location)
    st.session_state["lat_val"] = default_lat
    st.session_state["lon_val"] = default_lon


if "selected_location" not in st.session_state:
    st.session_state["selected_location"] = location_encoder.classes_[0]

if "selected_city" not in st.session_state:
    st.session_state["selected_city"] = get_default_city_for_location(st.session_state["selected_location"])

if "lat_val" not in st.session_state or "lon_val" not in st.session_state:
    default_lat, default_lon = get_default_coords_for_location(st.session_state["selected_location"])
    st.session_state["lat_val"] = default_lat
    st.session_state["lon_val"] = default_lon

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
    # Keep city consistent with selected location to avoid unrealistic combinations.
    city = st.text_input("City (Auto from Location)", value=st.session_state["selected_city"], disabled=True)
    property_type = st.selectbox("Property Type", type_encoder.classes_)
    balconies = st.number_input("Number of Balconies", min_value=0, value=1, step=1)
    bhk = st.selectbox("BHK (dataset uses binary encoding)", [0, 1], format_func=lambda x: "0" if x == 0 else "1")
    rooms = st.number_input("Total Rooms", min_value=1, value=3, step=1)

with col3:
    city_lat_min, city_lat_max, city_lon_min, city_lon_max = get_city_geo_bounds(st.session_state["selected_city"])
    latitude = st.number_input(
        "Latitude",
        min_value=float(city_lat_min),
        max_value=float(city_lat_max),
        format="%.6f",
        key="lat_val",
    )
    longitude = st.number_input(
        "Longitude",
        min_value=float(city_lon_min),
        max_value=float(city_lon_max),
        format="%.6f",
        key="lon_val",
    )
    st.caption(
        f"Expected range for {st.session_state['selected_city']}: "
        f"lat {city_lat_min:.2f} to {city_lat_max:.2f}, "
        f"lon {city_lon_min:.2f} to {city_lon_max:.2f}"
    )
    isNegotiable = st.selectbox("Negotiable", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    security = st.number_input("Security Deposit", min_value=0, value=50000, step=1000)

# Input Validation
if size < 200:
    st.warning("⚠️ Size seems too small")

if bathrooms > rooms:
    st.error("❌ Bathrooms cannot exceed total rooms")

if not (city_lat_min <= latitude <= city_lat_max) or not (city_lon_min <= longitude <= city_lon_max):
    st.warning("⚠️ Coordinates are outside the typical range for the selected city")

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
