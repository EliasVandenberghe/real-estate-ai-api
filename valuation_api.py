# %%
from fastapi import FastAPI, HTTPException
from valuation_logic import predict_price, generate_price_trend

app = FastAPI(title="Real Estate AI API")

@app.get("/")
def home():
    return {"status": "Online", "features": ["Valuation", "Chronos Trend Forecasting"]}

@app.post("/predict-all")
async def predict_all(property_input: dict):
    try:
        # 1. Bereken de huidige waardeschatting
        valuation_res = predict_price(property_input)
        
        # 2. Genereer de historische en toekomstige trends
        trend_res = generate_price_trend(property_input)

        # 3. Voeg samen in de exact gewenste JSON structuur
        return {
            "valuation": valuation_res,
            "price_trend": trend_res
        }
    except Exception as e:
        print(f"❌ API Error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Fout bij verwerking: {str(e)}")
# %%
