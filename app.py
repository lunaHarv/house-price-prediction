import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import streamlit as st


@st.cache_resource
def load_or_train_model():
  model_path = "models/house_price_model.pkl"
  data_path = "houses_improved_data.csv"

  # Fallback to check root directory if data folder isn't used
  if not os.path.exists(data_path) and os.path.exists(
      "data/houses_improved_data.csv"
  ):
    data_path = "data/houses_improved_data.csv"

  if os.path.exists(model_path):
    try:
      return joblib.load(model_path)
    except Exception:
      pass  # If pickling fails due to version mismatch, retrain below

  # Train model dynamically if model file is missing or corrupted
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


# Load model
model = load_or_train_model()
