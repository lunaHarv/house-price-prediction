import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Ensure directory paths exist
os.makedirs("models", exist_ok=True)

# 1. Load Dataset
df = pd.read_csv("data/houses_improved_data.csv")

# 2. Separate Features and Target
X = df.drop("Price_ETB", axis=1)
y = df["Price_ETB"]

# 3. Identify Numerical and Categorical Features
num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# 4. Define Preprocessor Pipeline
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

# 5. Build Pipeline
model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ]
)

# 6. Train-Test Split & Fitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model_pipeline.fit(X_train, y_train)

# 7. Evaluate
y_pred = model_pipeline.predict(X_test)
print("--- Model Performance ---")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(
    f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):,.2f} ETB"
)
print(f"MAE: {mean_absolute_error(y_test, y_pred):,.2f} ETB")

# 8. Save Pipeline
joblib.dump(model_pipeline, "models/house_price_model.pkl")
print("Model saved successfully to models/house_price_model.pkl")