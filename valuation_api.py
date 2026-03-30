# %%
import os
import uvicorn
import joblib
import pandas as pd
from fastapi import FastAPI
from price_trend import generate_price_trend

app = FastAPI()

# --- 1. Modellen laden bij opstart ---
# Let op: dit kan op Render Free even duren en veel geheugen kosten
try:
    data = joblib.load("valuation_model.pkl")
    model = data["model"]
    rmse = data["rmse"]
    print("Succesvol valuation_model.pkl geladen.")
except Exception as e:
    print(f"Fout bij laden model: {e}")

# --- 2. Routes ---

@app.get("/")
def health_check():
    """Route voor Render om te zien of de API leeft."""
    return {
        "status": "online",
        "message": "Real Estate API is running",
        "environment": "production"
    }

@app.post("/predict-price")
def predict_price(property_data: dict):
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

    prediction = float(model.predict(df)[0])
    confidence = max(0, 100 - (rmse / prediction) * 100)

    return {
        "estimated_value": prediction,
        "confidence_score": float(round(confidence, 1)),
        "range_low": float(prediction - rmse),
        "range_high": float(prediction + rmse)
    }

@app.post("/price-trend")
def price_trend(property_data: dict):
    # Deze functie roept Chronos aan (via price_trend.py)
    trend = generate_price_trend(property_data)
    return trend

# --- 3. Start-up Logica ---
if __name__ == "__main__":
    # Render gebruikt de omgevingsvariabele $PORT
    # Als die niet bestaat (lokaal), gebruiken we 10000
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting app on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
# %%
