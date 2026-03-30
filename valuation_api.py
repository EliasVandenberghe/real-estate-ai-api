# %%
import os
import uvicorn
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from price_trend import generate_price_trend

app = FastAPI()

# Laad het lineaire model direct (is erg licht)
try:
    data = joblib.load("valuation_model.pkl")
    valuation_model = data["model"]
    rmse = data["rmse"]
    print("Succesvol valuation_model.pkl geladen.")
except Exception as e:
    print(f"Fout bij laden model: {e}")

@app.get("/")
def health_check():
    return {"status": "online", "message": "Real Estate API is running", "environment": "production"}

@app.post("/predict-all")
def predict_all(property_data: dict):
    """Combineert prijs en trend in één keer voor de frontend"""
    try:
        # 1. Prijs voorspellen
        df = pd.DataFrame([property_data])
        prediction = float(valuation_model.predict(df)[0])
        confidence = max(0, 100 - (rmse / prediction) * 100)

        # 2. Trend voorspellen (roept lazy loading van Chronos aan)
        trend_data = generate_price_trend(property_data)

        return {
            "valuation": {
                "estimated_value": int(prediction),
                "confidence_score": round(confidence, 1)
            },
            "trend": trend_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
# %%
