# %%
import joblib
import pandas as pd

# Model laden
data = joblib.load("valuation_model.pkl")

model = data["model"]
rmse = data["rmse"]

# Nieuwe property input (zoals van dashboard)
property_data = pd.DataFrame([{
    "surface_area": 120,
    "bedrooms": 3,
    "bathrooms": 2,
    "build_year": 2005,
    "epc_score": 180,
    "garden_area": 80,
    "garage": True,
    "pool": False,
    "property_type": "house",
    "city": "Leuven"
}])

# Voorspelling
prediction = model.predict(property_data)[0]

# Confidence berekenen
confidence = max(0, 100 - (rmse / prediction) * 100)

# Waarde range
lower_bound = prediction - rmse
upper_bound = prediction + rmse

print("Estimated value:", round(prediction,2))
print("Confidence:", round(confidence,1), "%")
print("Value range:", round(lower_bound,2), "-", round(upper_bound,2))