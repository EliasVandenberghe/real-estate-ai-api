# %%
import joblib
import pandas as pd
import torch
import os
from chronos import ChronosPipeline

# --- Config ---
BASE_DIR = os.path.dirname(__file__)
PKL_PATH = os.path.join(BASE_DIR, "valuation_model.pkl")
CSV_PATH = os.path.join(BASE_DIR, "sold_properties_mock_realistic_6000.csv")
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "chronos_model_local")

# Globale variabelen om resources één keer te laden
val_model = None
val_rmse = 0
df_global = None
pipeline = None

def load_resources():
    global val_model, val_rmse, df_global, pipeline
    if val_model is None:
        try:
            # 1. Scikit-learn model laden
            if os.path.exists(PKL_PATH):
                pkl_data = joblib.load(PKL_PATH)
                val_model = pkl_data["model"]
                val_rmse = pkl_data["rmse"]
            
            # 2. Dataset laden voor historische trends
            if os.path.exists(CSV_PATH):
                df_global = pd.read_csv(CSV_PATH)
                df_global["sale_date"] = pd.to_datetime(df_global["sale_date"])
            
            # 3. Chronos laden (Check of lokaal bestaat, anders HF fallback)
            model_source = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else "amazon/chronos-t5-tiny"
            print(f"⏳ Chronos laden van: {model_source}...")
            
            pipeline = ChronosPipeline.from_pretrained(
                model_source,
                device_map="cpu",
                torch_dtype=torch.float32,
            )
            print("✅ Alle resources (inclusief Chronos) succesvol geladen!")
        except Exception as e:
            print(f"❌ Kritieke fout bij laden: {e}")

def predict_price(property_data):
    """Berekent de huidige waarde van een woning."""
    load_resources()
    if val_model is None:
        return {"error": "Valuation model not loaded"}

    # DataFrame maken van input
    df = pd.DataFrame([property_data])
    
    # Zorg dat de kolommen exact matchen met de training (image_f658c1)
    cols = ["surface_area", "bedrooms", "bathrooms", "build_year", "epc_score", 
            "garden_area", "garage", "pool", "property_type", "city"]
    
    # Filter en vul missende waarden indien nodig
    df = df.reindex(columns=cols, fill_value=0)
    
    # Voorspelling
    pred = val_model.predict(df)[0]
    
    # Betrouwbaarheidsscore op basis van RMSE
    conf = max(0, 100 - (val_rmse / pred) * 100) if pred > 0 else 0
    
    return {
        "estimated_value": float(round(pred, 2)),
        "confidence_score": float(round(conf, 1)),
        "range_low": float(round(pred - val_rmse, 2)),
        "range_high": float(round(pred + val_rmse, 2))
    }

def generate_price_trend(property_input):
    """Genereert historische data en een 3-jarige prognose met fallback logica."""
    load_resources()
    if pipeline is None or df_global is None:
        return {"error": "Resources voor trend niet loaded"}
    
    surface = property_input.get("surface_area", 100)
    city = property_input.get("city")
    p_type = property_input.get("property_type")
    
    # 1. Probeer te filteren op Stad EN Type
    comparables = df_global[(df_global["city"] == city) & (df_global["property_type"] == p_type)].copy()
    
    # 2. Fallback 1: Alleen op Type (als stad geen matches heeft)
    if len(comparables) < 3:
        print(f"⚠️ Te weinig data voor {city}, we kijken naar de algemene trend voor {p_type}")
        comparables = df_global[df_global["property_type"] == p_type].copy()
    
    # 3. Fallback 2: Alles (als noodgreep)
    if len(comparables) < 3:
        comparables = df_global.copy()

    # Bereken prijs per m2
    comparables["price_per_m2"] = comparables["sale_price"] / comparables["surface_area"]
    yearly = comparables.groupby(df_global["sale_date"].dt.year)["price_per_m2"].mean().sort_index()

    # Check of we nu wel data hebben om IndexError te voorkomen
    hist_years = [int(y) for y in yearly.index.tolist()]
    if not hist_years:
        # Ultieme fallback als zelfs de CSV leeg zou zijn
        hist_years = [2022, 2023, 2024]
        yearly_values = [3000, 3100, 3200] 
    else:
        yearly_values = yearly.values.tolist()

    # Chronos voorspelling
    context = torch.tensor(yearly_values, dtype=torch.float32)
    forecast = pipeline.predict(context, prediction_length=3)
    forecast_median = forecast[0].median(dim=0).values.numpy()

    hist_prices = [int(p * surface) for p in yearly_values]
    
    return {
        "historical_years": hist_years,
        "historical_prices": hist_prices,
        "forecast_years": [hist_years[-1] + i for i in range(1, 4)],
        "forecast_prices": [int(p * surface) for p in forecast_median.tolist()]
    }
# %%
