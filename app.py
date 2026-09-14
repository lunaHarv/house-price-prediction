import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import streamlit as st

# Page setup
st.set_page_config(
    page_title="House Price Predictor", page_icon="🏠", layout="wide"
)


@st.cache_resource
def load_or_train_model():
  """Loads the saved ML pipeline or trains a new one if missing/incompatible."""
  model_path = "models/house_price_model.pkl"
  data_path = (
      "houses_improved_data.csv"
      if os.path.exists("houses_improved_data.csv")
      else "data/houses_improved_data.csv"
  )

  # 1. Attempt loading saved model
  if os.path.exists(model_path):
    try:
      return joblib.load(model_path)
    except Exception:
      pass  # Fall back to training if pickle unpickling fails on cloud host

  # 2. Train model dynamically if model file is not available
  if not os.path.exists(data_path):
    st.error(
        f"Dataset file '{data_path}' not found! Please check repository file"
        " paths."
    )
    st.stop()

  df = pd.read_csv(data_path)
  X = df.drop("Price_ETB", axis=1)
  y = df["Price_ETB"]

  num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
  cat_features = X.select_dtypes(include=["object"]).columns.tolist()

  preprocessor = ColumnTransformer(
      transformers=[
          ("num", "passthrough", num_features),
          (
              "cat",
              OneHotEncoder(drop="first", handle_unknown="ignore"),
              cat_features,
          ),
      ]
  )

  pipeline = Pipeline(
      steps=[
          ("preprocessor", preprocessor),
          ("regressor", LinearRegression()),
      ]
  )

  pipeline.fit(X, y)

  os.makedirs("models", exist_ok=True)
  joblib.dump(pipeline, model_path)
  return pipeline


# Load model pipeline
model = load_or_train_model()

# App Header
st.title("🏠 Multivariate House Price Predictor")
st.write(
    "Enter property dimensions, construction details, and location metrics to"
    " calculate an estimated market price."
)
st.divider()

# Input Layout
col1, col2, col3 = st.columns(3)

with col1:
  st.subheader("📐 Physical Dimensions")
  num_rooms = st.number_input(
      "Number of Rooms", min_value=1, max_value=20, value=4, step=1
  )
  site_area = st.number_input(
      "Site Area (sqm)", min_value=10, max_value=5000, value=400, step=10
  )
  built_area = st.number_input(
      "Built Area (sqm)", min_value=10, max_value=3000, value=200, step=10
  )
  property_years = st.number_input(
      "Property Age (Years)", min_value=0, max_value=100, value=5, step=1
  )

with col2:
  st.subheader("🏗️ Property Specifications")
  materials = st.selectbox(
      "Construction Materials", ["Mud&Wood", "Concrete"]
  )
  typology = st.selectbox(
      "Housing Typology", ["Detached", "Condominium", "Semi-detached"]
  )
  land_grading = st.selectbox(
      "Land Value Grading", ["Medium", "High", "Low"]
  )

with col3:
  st.subheader("📍 Location & Accessibility")
  cbd_km = st.number_input(
      "Distance to CBD (km)",
      min_value=0.0,
      max_value=100.0,
      value=2.0,
      step=0.1,
  )
  bus_km = st.number_input(
      "Distance to Bus Station (km)",
      min_value=0.0,
      max_value=50.0,
      value=1.5,
      step=0.1,
  )
  schools_km = st.number_input(
      "Distance to Schools (km)",
      min_value=0.0,
      max_value=50.0,
      value=1.0,
      step=0.1,
  )
  nearest_road = st.selectbox(
      "Type of Nearest Road", ["Asphalt", "Gravel"]
  )

st.divider()

# Prediction Action
if st.button("Predict House Price", type="primary", use_container_width=True):
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

  try:
    predicted_price = model.predict(input_data)[0]
    st.success(f"### Estimated Price: **{predicted_price:,.2f} ETB**")
  except Exception as e:
    st.error(f"Error predicting price: {str(e)}")
