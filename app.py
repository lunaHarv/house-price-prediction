import joblib
import pandas as pd
import streamlit as st

# Load trained pipeline
model = joblib.load("models/house_price_model.pkl")

st.set_page_config(page_title="House Price Predictor", layout="wide")
st.title("🏠 House Price Prediction Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
  st.subheader("Property Dimensions")
  num_rooms = st.number_input(
      "Number of Rooms", min_value=1, max_value=20, value=3
  )
  site_area = st.number_input(
      "Site Area (sqm)", min_value=10, max_value=2000, value=300
  )
  built_area = st.number_input(
      "Built Area (sqm)", min_value=10, max_value=1500, value=150
  )
  property_years = st.number_input(
      "Property Age (Years)", min_value=0, max_value=100, value=5
  )

with col2:
  st.subheader("Property Categorization")
  materials = st.selectbox(
      "Construction Materials", ["Mud&Wood", "Concrete"]
  )
  typology = st.selectbox(
      "Housing Typology", ["Detached", "Condominium", "Semi-detached"]
  )
  land_grading = st.selectbox(
      "Land Value Grading", ["Low", "Medium", "High"]
  )

with col3:
  st.subheader("Location & Access")
  cbd_km = st.number_input(
      "Proximity to CBD (km)", min_value=0.0, max_value=100.0, value=2.5
  )
  bus_km = st.number_input(
      "Proximity to Bus Station (km)",
      min_value=0.0,
      max_value=50.0,
      value=1.0,
  )
  schools_km = st.number_input(
      "Proximity to Schools (km)",
      min_value=0.0,
      max_value=50.0,
      value=1.5,
  )
  nearest_road = st.selectbox(
      "Type of Nearest Road", ["Asphalt", "Gravel"]
  )

if st.button("Predict Price", type="primary"):
  input_data = pd.DataFrame([{
      "Number_of_Rooms": num_rooms,
      "Site_Area_sqm": site_area,
      "Built_Area_sqm": built_area,
      "Property_Years": property_years,
      "Construction_Materials": materials,
      "Housing_Typology": typology,
      "Land_Value_Grading": land_grading,
      "Proximity_to_CBD_km": cbd_km,
      "Proximity_to_Bus_Station_km": bus_km,
      "Type_of_Nearest_Road": nearest_road,
      "Proximity_to_Schools_km": schools_km,
  }])

  prediction = model.predict(input_data)[0]
  st.success(f"### Estimated Price: {prediction:,.2f} ETB")