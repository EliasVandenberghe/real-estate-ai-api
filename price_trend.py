#%%
import pandas as pd
import numpy as np
import torch
from chronos import ChronosPipeline

# We maken de pipeline variabele globaal, maar laten hem leeg (None)
pipeline = None

def generate_price_trend(property_input):
    global pipeline
    
    # --------------------------------------------------
    # Load Chronos model ONLY when needed (Lazy Loading)
    # --------------------------------------------------
    if pipeline is None:
        print("Chronos model (tiny) wordt nu geladen in het geheugen...")
        pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-tiny",  # VEEL lichter voor Render Free
            device_map="cpu",
            dtype=torch.float32
        )
        print("Chronos succesvol geladen!")

    # 1. Load dataset
    df = pd.read_csv("sold_properties_mock_realistic_6000.csv")

    city = property_input["city"]
    property_type = property_input["property_type"]
    surface = property_input["surface_area"]

    # 2. Filter comparable properties
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

    # 3. Compute price per m²
    comparables["price_per_m2"] = comparables["sale_price"] / comparables["surface_area"]

    # 4. Remove outliers
    q1 = comparables["price_per_m2"].quantile(0.25)
    q3 = comparables["price_per_m2"].quantile(0.75)
    iqr = q3 - q1
    comparables = comparables[
        (comparables["price_per_m2"] > q1 - 1.5 * iqr) &
        (comparables["price_per_m2"] < q3 + 1.5 * iqr)
    ]

    # 5. Extract year and aggregate
    comparables["year"] = pd.to_datetime(comparables["sale_date"]).dt.year
    yearly = comparables.groupby("year")["price_per_m2"].mean().sort_index()

    if len(yearly) < 3:
        yearly = (df.groupby(pd.to_datetime(df["sale_date"]).dt.year)["sale_price"].mean() / df["surface_area"].mean()).sort_index()

    # 6. Convert to tensor
    series = torch.tensor(yearly.values, dtype=torch.float32)

    # 7. Chronos forecast
    # We voegen een extra dimensie toe voor de pipeline input
    context = series.unsqueeze(0) 
    forecast = pipeline.predict(context, prediction_length=3, num_samples=20)
    
    forecast_mean = forecast.mean(dim=1).flatten().cpu().numpy()

    # 8. Convert €/m² → property price
    historical_prices = yearly.values * surface
    forecast_prices = forecast_mean * surface
    historical_years = yearly.index.tolist()
    last_year = historical_years[-1]
    forecast_years = [last_year + 1, last_year + 2, last_year + 3]

    return {
        "historical_years": [int(y) for y in historical_years],
        "historical_prices": [int(p) for p in historical_prices],
        "forecast_years": [int(y) for y in forecast_years],
        "forecast_prices": [int(p) for p in forecast_prices]
    }
# %%
