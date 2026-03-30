#%%
import pandas as pd
import numpy as np
import torch
import os
from chronos import ChronosPipeline

# --- 1. Global Setup (Geheugen-efficiënt) ---
BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, "sold_properties_mock_realistic_6000.csv")
MODEL_PATH = os.path.join(BASE_DIR, "chronos_model_local")

# Laad de dataset één keer bij het opstarten
df_global = pd.read_csv(CSV_PATH)

# Global pipeline variabele
pipeline = None

def generate_price_trend(property_input):
    global pipeline
    
    # --- 2. Load Chronos model LOKAAL ---
    if pipeline is None:
        print(f"Chronos model wordt nu LOKAAL geladen vanuit: {MODEL_PATH}")
        try:
            pipeline = ChronosPipeline.from_pretrained(
                MODEL_PATH,  # <--- GEBRUIK HET LOKALE PAD!
                device_map="cpu",
                torch_dtype=torch.float32 # Gebruik 'torch_dtype' i.p.v. 'dtype'
            )
            print("✅ Chronos succesvol geladen!")
        except Exception as e:
            print(f"❌ Fout bij laden model: {e}")
            return {"error": f"Model laden mislukt: {str(e)}"}

    # 3. Gebruik de globale dataset
    df = df_global.copy()

    # Gebruik .get() om crashes bij missende data te voorkomen
    city = property_input.get("city")
    property_type = property_input.get("property_type")
    surface = property_input.get("surface_area", 100)

    # 4. Filter comparable properties
    comparables = df[
        (df["city"] == city) &
        (df["property_type"] == property_type) &
        (df["surface_area"].between(surface * 0.7, surface * 1.3))
    ].copy()

    if len(comparables) < 30:
        comparables = df[
            (df["city"] == city) &
            (df["property_type"] == property_type)
        ].copy()

    if len(comparables) == 0:
        comparables = df.copy()

    # 5. Compute price per m² & Remove outliers
    comparables["price_per_m2"] = comparables["sale_price"] / comparables["surface_area"]
    q1 = comparables["price_per_m2"].quantile(0.25)
    q3 = comparables["price_per_m2"].quantile(0.75)
    iqr = q3 - q1
    comparables = comparables[
        (comparables["price_per_m2"] > q1 - 1.5 * iqr) &
        (comparables["price_per_m2"] < q3 + 1.5 * iqr)
    ]

    # 6. Extract year and aggregate
    comparables["year"] = pd.to_datetime(comparables["sale_date"]).dt.year
    yearly = comparables.groupby("year")["price_per_m2"].mean().sort_index()

    if len(yearly) < 3:
        # Fallback naar algemene trend als er te weinig data is
        yearly = (df.groupby(pd.to_datetime(df["sale_date"]).dt.year)["sale_price"].mean() / df["surface_area"].mean()).sort_index()

    # 7. Chronos forecast
    series = torch.tensor(yearly.values, dtype=torch.float32)
    context = series.unsqueeze(0) 
    
    # Voorspel 3 jaar vooruit
    forecast = pipeline.predict(context, prediction_length=3, num_samples=20)
    forecast_mean = forecast.mean(dim=1).flatten().cpu().numpy()

    # 8. Formatteren resultaten
    historical_prices = yearly.values * surface
    forecast_prices = forecast_mean * surface
    historical_years = yearly.index.tolist()
    last_year = historical_years[-1]
    forecast_years = [int(last_year + 1), int(last_year + 2), int(last_year + 3)]

    return {
        "historical_years": [int(y) for y in historical_years],
        "historical_prices": [int(p) for p in historical_prices],
        "forecast_years": forecast_years,
        "forecast_prices": [int(p) for p in forecast_prices]
    }
# %%
