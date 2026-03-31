# %%
import joblib
import pandas as pd
import torch
import os
import numpy as np
from chronos import ChronosPipeline

# --- Config ---
BASE_DIR = os.path.dirname(__file__)
PKL_PATH = os.path.join(BASE_DIR, "valuation_model.pkl")
CSV_PATH = os.path.join(BASE_DIR, "sold_properties_mock_realistic_6000.csv")

# Globale variabelen
val_model = None
val_rmse = 0
df_global = None
pipeline = None


def load_resources():
    """
    Lazy loading van alle zware resources.
    Wordt slechts één keer uitgevoerd.
    """
    global val_model, val_rmse, df_global, pipeline

    # Als alles al geladen is → stop
    if val_model is not None and pipeline is not None and df_global is not None:
        return

    try:
        # --- 1. Scikit-learn model ---
        if val_model is None and os.path.exists(PKL_PATH):
            pkl_data = joblib.load(PKL_PATH)
            val_model = pkl_data["model"]
            val_rmse = pkl_data.get("rmse", 25000)

        # --- 2. Dataset laden ---
        if df_global is None and os.path.exists(CSV_PATH):
            df_global = pd.read_csv(CSV_PATH)
            df_global["sale_date"] = pd.to_datetime(df_global["sale_date"])

        # --- 3. Chronos model laden ---
        if pipeline is None:
            model_source = "amazon/chronos-t5-tiny"
            print(f"⏳ Chronos laden van: {model_source}")

            pipeline = ChronosPipeline.from_pretrained(
                model_source,
                device_map="cpu",
                dtype=torch.float32,
            )

            print("✅ Chronos model geladen!")

    except Exception as e:
        print(f"❌ Kritieke fout bij laden resources: {e}")


def predict_price(property_data):
    """
    Berekent de huidige waarde van een woning.
    """
    load_resources()

    if val_model is None:
        return {"error": "Valuation model not loaded"}

    # Verwachte features
    cols = [
        "surface_area",
        "bedrooms",
        "bathrooms",
        "build_year",
        "epc_score",
        "garden_area",
        "garage",
        "pool",
        "property_type",
        "city",
    ]

    df = pd.DataFrame([property_data])
    df = df.reindex(columns=cols, fill_value=0)

    pred = float(val_model.predict(df)[0])

    conf = max(0, 100 - (val_rmse / pred) * 100) if pred > 0 else 0

    return {
        "estimated_value": float(round(pred, 2)),
        "confidence_score": float(round(conf, 1)),
        "range_low": float(round(pred - val_rmse, 2)),
        "range_high": float(round(pred + val_rmse, 2)),
    }


def generate_price_trend(property_input):
    """
    Genereert historische prijzen + 3 jaar forecast.
    """
    load_resources()

    if pipeline is None or df_global is None:
        return {"error": "Resources voor trend niet geladen"}

    surface = property_input.get("surface_area", 100)
    city = property_input.get("city")
    p_type = property_input.get("property_type")

    # --- Comparable filtering ---
    comparables = df_global[
        (df_global["city"] == city) & (df_global["property_type"] == p_type)
    ].copy()

    if len(comparables) < 3:
        comparables = df_global[df_global["property_type"] == p_type].copy()

    if len(comparables) < 3:
        comparables = df_global.copy()

    # --- Price per m² ---
    comparables["price_per_m2"] = comparables["sale_price"] / comparables["surface_area"]

    # FIX: correcte groupby
    yearly = (
        comparables.groupby(comparables["sale_date"].dt.year)["price_per_m2"]
        .mean()
        .sort_index()
    )

    hist_years = [int(y) for y in yearly.index.tolist()]

    # fallback indien weinig data
    if not hist_years:
        hist_years = [2022, 2023, 2024]
        yearly_values = [3000, 3100, 3200]
    else:
        yearly_values = yearly.values.tolist()

    # --- Chronos input ---
    context = torch.tensor(yearly_values, dtype=torch.float32)

    forecast = pipeline.predict(context, prediction_length=3)

    # Median over samples
    forecast_median = (
        forecast[0]
        .median(dim=0)
        .values
        .cpu()
        .numpy()
    )

    hist_prices = [int(p * surface) for p in yearly_values]

    return {
        "historical_years": hist_years,
        "historical_prices": hist_prices,
        "forecast_years": [hist_years[-1] + i for i in range(1, 4)],
        "forecast_prices": [int(p * surface) for p in forecast_median.tolist()],
    }
# %%
