# %%
from fastapi import FastAPI, HTTPException
from valuation_logic import predict_price
from price_trend import generate_price_trend

app = FastAPI(title="Real Estate AI API")

@app.get("/")
def home():
    return {"status": "Online", "features": ["Valuation", "Trend Forecasting"]}

@app.post("/predict-all")
async def predict_all(property_input: dict):
    try:
        # 1. De waardeschatting (Jouw originele logica)
        valuation_res = predict_price(property_input)

        # 2. De prijstrend (Chronos logica)
        trend_res = generate_price_trend(property_input)

        # 3. Combineer resultaten
        return {
            "valuation": valuation_res,
            "trend": trend_res
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Fout bij berekening: {str(e)}")
# %%
