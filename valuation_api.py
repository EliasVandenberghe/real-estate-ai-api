# %%
from fastapi import FastAPI
import joblib
import pandas as pd

from price_trend import generate_price_trend

app = FastAPI()

# model laden bij opstart (sneller dan per request)
data = joblib.load("valuation_model.pkl")

model = data["model"]
rmse = data["rmse"]


@app.post("/predict-price")
def predict_price(property_data: dict):

    # input omzetten naar DataFrame met dezelfde features als training
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

    # voorspelling
    prediction = model.predict(df)[0]

    # confidence berekenen
    confidence = max(0, 100 - (rmse / prediction) * 100)

    return {
        "estimated_value": float(prediction),
        "confidence_score": float(round(confidence, 1)),
        "range_low": float(prediction - rmse),
        "range_high": float(prediction + rmse)
    }


@app.post("/price-trend")
def price_trend(property_data: dict):

    trend = generate_price_trend(property_data)

    return trend
# %%
