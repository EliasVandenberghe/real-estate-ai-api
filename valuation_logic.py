# %%
import joblib
import pandas as pd
import torch
import os
import gc
from chronos import ChronosPipeline

# --- Configuraties ---
BASE_DIR = os.path.dirname(__file__)
PKL_PATH = os.path.join(BASE_DIR, "valuation_model.pkl")
CSV_PATH = os.path.join(BASE_DIR, "sold_properties_mock_realistic_6000.csv")

# We gebruiken het tiny model direct van Amazon om lokale pad-fouten te voorkomen
CHRONOS_MODEL_ID = "amazon/chronos-t5-tiny"

# Globale placeholders
val_model = None
val_rmse = 0
df_global = None
chronos_pipeline = None

def load_resources():
    """Laadt de AI-modellen pas wanneer ze voor het eerst nodig zijn."""
    global val_model, val_rmse, df_global, chronos_pipeline
    
    if val_model is None:
        print("⏳ Eerste aanroep gedetecteerd: AI-resources laden...")
        try:
            # 1. Waardeschattingsmodel (.pkl)
            pkl_data = joblib.load(PKL_PATH)
            val_model = pkl_data["model"]
            val_rmse = pkl_data["rmse"]
            
            # 2. Dataset voor trends
            df_global = pd.read_csv(CSV_PATH)
            df_global["sale_date"] = pd.to_datetime(df_global["sale_date"])
            
            # 3. Chronos (Direct van HuggingFace om lokale corruptie te vermijden)
            chronos_pipeline = ChronosPipeline.from_pretrained(
                CHRONOS_MODEL_ID,
                device_map="cpu",
                dtype=torch.float32
            )
            gc.collect()
            print("✅ Alles succesvol geladen!")
        except Exception as e:
            print(f"❌ Fout bij laden resources: {e}")

def predict_price(property_data):
    load_resources()
    if val_model is None: return {"error": "Model niet beschikbaar"}
    
    df = pd.DataFrame([{
        "surface_area": property_data.get("surface_area"),
        "bedrooms": property_data.get("bedrooms"),
        "bathrooms": property_data.get("bathrooms"),
        "build_year": property_data.get("build_year"),
        "epc_score": property_data.get("epc_score"),
        "garden_area": property_data.get("garden_area"),
        "garage": property_data.get("garage"),
        "pool": property_data.get("pool"),
        "property_type": property_data.get("property_type"),
        "city": property_data.get("city")
    }])

    pred = val_model.predict(df)[0]
    conf = max(0, 100 - (val_rmse / pred) * 100) if pred > 0 else 0
    return {
        "estimated_value": float(round(pred, 2)),
        "confidence_score": float(round(conf, 1)),
        "range_low": float(round(pred - val_rmse, 2)),
        "range_high": float(round(pred + val_rmse, 2))
    }

def generate_price_trend(property_input):
    load_resources()
    if chronos_pipeline is None or df_global is None:
        return {"error": "Trend data niet beschikbaar"}

    surface = property_input.get("surface_area", 100)
    city = property_input.get("city")
    p_type = property_input.get("property_type")

    # Filteren op stad/type
    mask = (df_global["city"] == city) & (df_global["property_type"] == p_type)
    comparables = df_global[mask].copy()
    if len(comparables) < 10:
        comparables = df_global[df_global["property_type"] == p_type].copy()

    # Data voorbereiden voor Chronos
    comparables["price_per_m2"] = comparables["sale_price"] / comparables["surface_area"]
    comparables["year"] = comparables["sale_date"].dt.year
    yearly = comparables.groupby("year")["price_per_m2"].mean().sort_index()

    # Forecast
    context = torch.tensor(yearly.values, dtype=torch.float32).unsqueeze(0)
    forecast = chronos_pipeline.predict(context, prediction_length=3, num_samples=1)
    forecast_mean = forecast.mean(dim=1).flatten().cpu().numpy()

    hist_years = [int(y) for y in yearly.index.tolist()]
    hist_prices = [int(p * surface) for p in yearly.values]
    last_year = hist_years[-1]

    return {
        "historical_years": hist_years,
        "historical_prices": hist_prices,
        "forecast_years": [int(last_year + 1), int(last_year + 2), int(last_year + 3)],
        "forecast_prices": [int(p * surface) for p in forecast_mean]
    }
# %%
