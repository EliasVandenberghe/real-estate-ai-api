# %%
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# dataset laden
df = pd.read_csv("sold_properties_mock_realistic_6000.csv")

def generate_price_trend(property_input):

    city = property_input["city"]
    property_type = property_input["property_type"]
    surface = property_input["surface_area"]

    # vergelijkbare woningen selecteren
    comparables = df[
        (df["city"] == city) &
        (df["property_type"] == property_type) &
        (df["surface_area"].between(surface * 0.7, surface * 1.3))
    ].copy()

    # prijs per m² berekenen
    comparables["price_per_m2"] = (
        comparables["sale_price"] / comparables["surface_area"]
    )

    # jaar kolom
    comparables["year"] = pd.to_datetime(
        comparables["sale_date"]
    ).dt.year

    # gemiddelde €/m² per jaar
    yearly = comparables.groupby("year")["price_per_m2"].mean()

    # FIX: maak een echte tijdindex
    yearly.index = pd.PeriodIndex(yearly.index, freq="Y")

    # exponential smoothing model
    model = ExponentialSmoothing(
        yearly,
        trend="add",
        seasonal=None
    ).fit()

    # forecast 3 jaar
    forecast_years = 3
    forecast = model.forecast(forecast_years)

    # combine historical + forecast
    historical_years = yearly.index.year.tolist()
    historical_values = yearly.values.tolist()

    forecast_year_list = list(
        range(historical_years[-1] + 1,
              historical_years[-1] + 1 + forecast_years)
    )

    forecast_values = forecast.values.tolist()

    # prijs voor specifieke woning
    property_price_history = [
        v * surface for v in historical_values
    ]

    property_price_forecast = [
        v * surface for v in forecast_values
    ]

    return {
        "historical_years": historical_years,
        "historical_prices": property_price_history,
        "forecast_years": forecast_year_list,
        "forecast_prices": property_price_forecast
    }


#SCRIPT TESTEN
property_input = {
    "surface_area": 120,
    "city": "Leuven",
    "property_type": "house"
}

result = generate_price_trend(property_input)

print(result)
# %%
