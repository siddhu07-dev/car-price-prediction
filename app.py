
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

MODEL_FILE = Path("car_price_model.pkl")

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Used Car Price Prediction")
st.write("Estimate the resale price of a used car using Machine Learning.")
st.caption(
    "Educational estimate only. Actual prices depend on condition, "
    "location, service history and market demand."
)

if not MODEL_FILE.exists():
    st.error("Model not found. Run: python train_model.py")
    st.stop()

with open(MODEL_FILE, "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
features = saved["features"]
model_name = saved["model_name"]

st.info(f"Selected model: {model_name}")

with st.form("car_details"):
    brand_options = [
        "Maruti", "Hyundai", "Honda", "Toyota", "Tata",
        "Mahindra", "Ford", "Volkswagen", "Renault",
        "BMW", "Audi", "Other"
    ]

    brand = st.selectbox("Car brand", brand_options).lower()

    age = st.slider(
        "Age of car (years)",
        min_value=0,
        max_value=25,
        value=5
    )

    km_driven = st.number_input(
        "Distance driven (km)",
        min_value=0,
        max_value=1000000,
        value=45000,
        step=1000
    )

    fuel = st.selectbox(
        "Fuel type",
        ["Petrol", "Diesel", "CNG", "LPG"]
    ).lower()

    transmission = st.selectbox(
        "Transmission",
        ["Manual", "Automatic"]
    ).lower()

    owner = st.selectbox(
        "Previous ownership",
        [
            "First Owner",
            "Second Owner",
            "Third Owner",
            "Fourth & Above Owner",
            "Test Drive Car"
        ]
    ).lower()

    seller_type = st.selectbox(
        "Seller type",
        ["Individual", "Dealer", "Trustmark Dealer"]
    ).lower()

    submitted = st.form_submit_button("Predict Resale Price")

if submitted:
    row = {
        "brand": brand,
        "vehicle_age": age,
        "km_driven": km_driven,
        "fuel": fuel,
        "transmission": transmission,
        "owner": owner,
        "seller_type": seller_type
    }

    input_df = pd.DataFrame(
        [{feature: row[feature] for feature in features}]
    )

    try:
        predicted_price = float(model.predict(input_df)[0])
        predicted_price = max(0, predicted_price)

        st.subheader("Estimated Resale Price")
        st.success(f"₹{predicted_price:,.0f}")
        st.write(f"Prediction generated using **{model_name}**.")
        st.caption(
            "This is an estimate, not a guaranteed selling price."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
