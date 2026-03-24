# %%
import pandas as pd
import numpy as np
import timesfm

# 1. Dataset laden
try:
    df = pd.read_csv("sold_properties_mock_realistic_6000.csv")
except FileNotFoundError:
    print("Fout: Het CSV-bestand is niet gevonden in de huidige map.")
    exit()

# 2. TimesFM model initialisatie (Hparams + Checkpoint structuur)
# Dit matcht de (hparams, checkpoint) signature van jouw versie.
hparams = timesfm.TimesFmHparams(
    context_len=128,
    horizon_len=3,
    backend="cpu"
)

# Laad de model-gewichten
checkpoint = timesfm.TimesFmCheckpoint(hparams=hparams)

# Initialiseer het model
model = timesfm.TimesFm(
    hparams=hparams,
    checkpoint=checkpoint
)

def generate_price_trend(property_input):
    city = property_input["city"]
    property_type = property_input["property_type"]
    surface = property_input["surface_area"]

    # Filteren op vergelijkbare panden
    comparables = df[
        (df["city"] == city) &
        (df["property_type"] == property_type) &
        (df["surface_area"].between(surface * 0.7, surface * 1.3))
    ].copy()

    # Fallback indien te weinig data
    if len(comparables) < 20:
        comparables = df[
            (df["city"] == city) &
            (df["property_type"] == property_type)
        ].copy()

    # Prijs per m2 berekenen en groeperen per jaar
    comparables["price_per_m2"] = comparables["sale_price"] / comparables["surface_area"]
    comparables["year"] = pd.to_datetime(comparables["sale_date"]).dt.year
    yearly = comparables.groupby("year")["price_per_m2"].mean().sort_index()

    if len(yearly) < 2:
        raise ValueError("Niet genoeg historische data voor deze stad/type combinatie.")

    historical_years = [int(y) for y in yearly.index.tolist()]
    historical_values = yearly.values.astype(np.float32)

    # Input voor model (Shape: 1 serie, T tijdstappen)
    inputs = historical_values.reshape(1, -1)

    # Voorspelling uitvoeren (geeft een tuple terug: forecast, full_output)
    forecast, _ = model.forecast(
        inputs=inputs,
        freq=[0]  # 0 voor onregelmatige/jaarlijkse data
    )

    # Resultaten extraheren (eerste 3 stappen uit de voorspelling)
    forecast_values = forecast[0][:3]
    forecast_years = [int(historical_years[-1] + i) for i in range(1, 4)]

    # Resultaat object opbouwen
    return {
        "historical_years": historical_years,
        "historical_prices": [round(float(v * surface), 2) for v in historical_values],
        "forecast_years": forecast_years,
        "forecast_prices": [round(float(v * surface), 2) for v in forecast_values]
    }

# SCRIPT TESTEN
if __name__ == "__main__":
    property_input = {
        "surface_area": 120, 
        "city": "Leuven", 
        "property_type": "house"
    }
    
    try:
        res = generate_price_trend(property_input)
        print("\n" + "="*30)
        print(" SUCCESS: TREND GEGENEREERD ")
        print("="*30)
        print(f"Historische jaren: {res['historical_years']}")
        print(f"Laatste prijs: €{res['historical_prices'][-1]:,.2f}")
        print("-" * 30)
        print(f"Voorspelling jaren: {res['forecast_years']}")
        print(f"Voorspelde prijs (jaar 1): €{res['forecast_prices'][0]:,.2f}")
        print("="*30)
    except Exception as e:
        print(f"\nFout tijdens uitvoering: {e}")
# %%
