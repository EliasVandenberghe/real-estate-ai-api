#%%
import pandas as pd
import numpy as np
import torch
from chronos import ChronosPipeline


# --------------------------------------------------
# Load Chronos model once (important for performance)
# --------------------------------------------------

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",
    dtype=torch.float32
)


# --------------------------------------------------
# Main forecasting function
# --------------------------------------------------

def generate_price_trend(property_input):

    # ----------------------------------------
    # 1 Load dataset
    # ----------------------------------------

    df = pd.read_csv("sold_properties_mock_realistic_6000.csv")

    city = property_input["city"]
    property_type = property_input["property_type"]
    surface = property_input["surface_area"]

    # ----------------------------------------
    # 2 Filter comparable properties
    # ----------------------------------------

    comparables = df[
        (df["city"] == city) &
        (df["property_type"] == property_type) &
        (df["surface_area"].between(surface * 0.7, surface * 1.3))
    ].copy()

    # fallback if too few comparables
    if len(comparables) < 30:

        comparables = df[
            (df["city"] == city) &
            (df["property_type"] == property_type)
        ].copy()

    # fallback if still empty
    if len(comparables) == 0:
        comparables = df.copy()

    # ----------------------------------------
    # 3 Compute price per m²
    # ----------------------------------------

    comparables["price_per_m2"] = (
        comparables["sale_price"] /
        comparables["surface_area"]
    )

    # ----------------------------------------
    # 4 Remove outliers (IQR method)
    # ----------------------------------------

    q1 = comparables["price_per_m2"].quantile(0.25)
    q3 = comparables["price_per_m2"].quantile(0.75)
    iqr = q3 - q1

    comparables = comparables[
        (comparables["price_per_m2"] > q1 - 1.5 * iqr) &
        (comparables["price_per_m2"] < q3 + 1.5 * iqr)
    ]

    # ----------------------------------------
    # 5 Extract year and aggregate yearly
    # ----------------------------------------

    comparables["year"] = pd.to_datetime(
        comparables["sale_date"]
    ).dt.year

    yearly = (
        comparables
        .groupby("year")["price_per_m2"]
        .mean()
        .sort_index()
    )

    # Safety fallback if too few years
    if len(yearly) < 3:
        yearly = (
            df.groupby(
                pd.to_datetime(df["sale_date"]).dt.year
            )["sale_price"].mean() / df["surface_area"].mean()
        ).sort_index()

    # ----------------------------------------
    # 6 Convert to tensor for Chronos
    # ----------------------------------------

    series = torch.tensor(
        yearly.values,
        dtype=torch.float32
    )

    # ----------------------------------------
    # 7 Chronos forecast
    # ----------------------------------------

    forecast = pipeline.predict(
    series,
    prediction_length=3,
    num_samples=20
    )

    forecast = forecast.detach().cpu().numpy()

    # average prediction across samples
    forecast_mean = np.mean(forecast, axis=0)
    forecast_mean = np.squeeze(forecast_mean)

    # force correct 1D array
    forecast_mean = np.array(forecast_mean).reshape(-1)
    forecast_mean = forecast_mean[:3]

    # ----------------------------------------
    # 8 Convert €/m² → property price
    # ----------------------------------------

    historical_prices = yearly.values * surface
    forecast_prices = forecast_mean * surface

    historical_years = yearly.index.tolist()

    last_year = historical_years[-1]

    forecast_years = [
        last_year + 1,
        last_year + 2,
        last_year + 3
    ]

    # ----------------------------------------
    # 9 Return data for frontend chart
    # ----------------------------------------

    return {

        "historical_years": historical_years,

        "historical_prices": [
            int(p) for p in historical_prices
        ],

        "forecast_years": forecast_years,

        "forecast_prices": [
            int(float(p)) for p in forecast_prices
        ]

    }


# --------------------------------------------------
# Local test (for development)
# --------------------------------------------------

if __name__ == "__main__":

    test_property = {
        "city": "Antwerp",
        "property_type": "house",
        "surface_area": 140
    }

    result = generate_price_trend(test_property)

    print("\nPrice Trend Forecast Test\n")
    print(result)
# %%
