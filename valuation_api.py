# %%
from fastapi import FastAPI
import joblib
import pandas as pd

from price_trend import generate_price_trend

app = FastAPI()

# model laden bij opstart
data = joblib.load("valuation_model.pkl")

model = data["model"]
rmse = data["rmse"]


@app.post("/predict-price")
def predict_price(property_data: dict):

    df = pd.DataFrame([property_data])

    prediction = model.predict(df)[0]

    confidence = max(0, 100 - (rmse / prediction) * 100)

    return {
        "estimated_value": float(prediction),
        "confidence_score": float(round(confidence,1)),
        "range_low": float(prediction - rmse),
        "range_high": float(prediction + rmse)
    }


@app.post("/price-trend")
def price_trend(property_data: dict):

    trend = generate_price_trend(property_data)

    return trend
# %%
