# %%
import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import FunctionTransformer

# Dataset laden
df = pd.read_csv("sold_properties_mock_realistic_6000.csv")

# Features
X = df[[
    "surface_area",
    "bedrooms",
    "bathrooms",
    "build_year",
    "epc_score",
    "garden_area",
    "garage",
    "pool",
    "property_type",
    "city"
]]

# Target
y = df["sale_price"]

# Categorische features
categorical = ["property_type", "city"]

# Numerieke features
numeric = [
    "surface_area",
    "bedrooms",
    "bathrooms",
    "build_year",
    "epc_score",
    "garden_area",
    "garage",
    "pool"
]

# Preprocessing
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", FunctionTransformer(), numeric)
])

# Model pipeline
model = Pipeline([
    ("prep", preprocessor),
    ("reg", LinearRegression())
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model trainen
model.fit(X_train, y_train)

# Predictions op test set
pred = model.predict(X_test)

# RMSE berekenen
rmse = np.sqrt(mean_squared_error(y_test, pred))

print("Model RMSE:", round(rmse,2))

# Model + RMSE opslaan
joblib.dump({
    "model": model,
    "rmse": rmse
}, "valuation_model.pkl")

print("Model and RMSE saved!")
# %%
