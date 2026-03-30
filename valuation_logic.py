# %%
import joblib
import pandas as pd
import os

# Model laden
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "valuation_model.pkl")

data = joblib.load(MODEL_PATH)
model = data["model"]
rmse = data["rmse"]

def predict_price(property_data):
    # Exact jouw attributen en DataFrame opzet
    df = pd.DataFrame([{
        "surface_area": property_data["surface_area"],
        "bedrooms": property_data["bedrooms"],
        "bathrooms": property_data["bathrooms"],
        "build_year": property_data["build_year"],
        "epc_score": property_data["epc_score"],
        "garden_area": property_data["garden_area"],
        "garage": property_data["garage"],
        "pool": property_data["pool"],
        "property_type": property_data["property_type"],
        "city": property_data["city"]
    }])

    # Voorspelling
    prediction = model.predict(df)[0]

    # Confidence & Range berekenen (jouw originele formules)
    confidence = max(0, 100 - (rmse / prediction) * 100)
    lower_bound = prediction - rmse
    upper_bound = prediction + rmse

    return {
        "estimated_value": float(round(prediction, 2)),
        "confidence_score": float(round(confidence, 1)),
        "range_low": float(round(lower_bound, 2)),
        "range_high": float(round(upper_bound, 2))
    }
# %%
