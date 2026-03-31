# %%
from fastapi import FastAPI, HTTPException
from valuation_logic import predict_price, generate_price_trend
import uvicorn
import os

app = FastAPI(title="Real Estate AI API")


@app.get("/")
def home():
    return {
        "status": "Online",
        "features": [
            "Property Valuation",
            "Chronos Trend Forecasting"
        ]
    }


@app.post("/predict-all")
async def predict_all(property_input: dict):
    """
    Hoofd-endpoint:
    berekent waardeschatting + prijstrend.
    """

    try:
        valuation_res = predict_price(property_input)

        trend_res = generate_price_trend(property_input)

        return {
            "valuation": valuation_res,
            "price_trend": trend_res
        }

    except Exception as e:
        print(f"❌ API Error: {str(e)}")

        raise HTTPException(
            status_code=400,
            detail=f"Fout bij verwerking: {str(e)}"
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
# %%
