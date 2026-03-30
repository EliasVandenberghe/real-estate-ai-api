# %%
import joblib
import pandas as pd
import torch
import os
import gc
from chronos import ChronosPipeline

# --- Configuratie ---
BASE_DIR = os.path.dirname(__file__)
PKL_PATH = os.path.join(BASE_DIR, "valuation_model.pkl")
CSV_PATH = os.path.join(BASE_DIR, "sold_properties_mock_realistic_6000.csv")
MODEL_PATH = os.path.join(BASE_DIR, "chronos_model_local")

# We maken de variabelen leeg aan het begin
val_model = None
val_rmse = 0
df_global = None
chronos_pipeline = None

def load_resources():
    """Laadt de resources alleen als ze nog niet in het geheugen zitten."""
    global val_model, val_rmse, df_global, chronos_pipeline
    
    if val_model is None:
        print("⏳ Resources laden...")
        try:
            pkl_data = joblib.load(PKL_PATH)
            val_model = pkl_data["model"]
            val_rmse = pkl_data["rmse"]
            
            df_global = pd.read_csv(CSV_PATH)
            df_global["sale_date"] = pd.to_datetime(df_global["sale_date"])
            
            chronos_pipeline = ChronosPipeline.from_pretrained(
                MODEL_PATH,
                device_map="cpu",
                dtype=torch.float32
            )
            gc.collect()
            print("✅ Alles succesvol geladen!")
        except Exception as e:
            print(f"❌ Fout bij laden: {e}")

def predict_price(property_data):
    load_resources() # Zorg dat model geladen is
    if val_model is None: return {"error": "Model niet beschikbaar"}
    
    df = pd.DataFrame([property_data]) # Versimpelde DF creatie
    # Zorg dat de kolommen exact matchen (zoals in je eerdere versie)
    # ... (gebruik hier de uitgebreide DF uit je vorige code voor veiligheid) ...
    
    pred = val_model.predict(df)[0]
    conf = max(0, 100 - (val_rmse / pred) * 100) if pred > 0 else 0
    return {
        "estimated_value": float(round(pred, 2)),
        "confidence_score": float(round(conf, 1)),
        "range_low": float(round(pred - val_rmse, 2)),
        "range_high": float(round(pred + val_rmse, 2))
    }

def generate_price_trend(property_input):
    """Trend output exact volgens de gevraagde JSON structuur."""
    if chronos_pipeline is None or df_global is None:
        return {"error": "Data of model niet beschikbaar"}

    surface = property_input.get("surface_area", 100)
    city = property_input.get("city")
    p_type = property_input.get("property_type")

    # 1. Filteren
    comparables = df_global[
        (df_global["city"] == city) & 
        (df_global["property_type"] == p_type)
    ].copy()

    if len(comparables) < 10:
        comparables = df_global[df_global["property_type"] == p_type].copy()

    # 2. Historische data groeperen
    comparables["price_per_m2"] = comparables["sale_price"] / comparables["surface_area"]
    comparables["year"] = comparables["sale_date"].dt.year
    yearly = comparables.groupby("year")["price_per_m2"].mean().sort_index()

    # 3. Chronos Voorspelling (num_samples=1 voor Render RAM)
    context = torch.tensor(yearly.values, dtype=torch.float32).unsqueeze(0)
    forecast = chronos_pipeline.predict(context, prediction_length=3, num_samples=1)
    forecast_mean = forecast.mean(dim=1).flatten().cpu().numpy()

    # 4. Resultaten opbouwen in exact het juiste formaat
    hist_years = [int(y) for y in yearly.index.tolist()]
    hist_prices = [int(p * surface) for p in yearly.values]

    last_year = hist_years[-1]
    fc_years = [int(last_year + 1), int(last_year + 2), int(last_year + 3)]
    fc_prices = [int(p * surface) for p in forecast_mean]

    return {
        "historical_years": hist_years,
        "historical_prices": hist_prices,
        "forecast_years": fc_years,
        "forecast_prices": fc_prices
    }
# %%
