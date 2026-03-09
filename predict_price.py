# %%
import joblib
import pandas as pd

# Model laden
data = joblib.load("valuation_model.pkl")

model = data["model"]
rmse = data["rmse"]


# Functie om prijs te voorspellen op basis van property input
def predict_price(property_data):

    # Nieuwe property input (zoals van dashboard)
    # Convert de input dictionary naar een pandas DataFrame
    df = pd.DataFrame([{
        "surface_area": property_data["surface_area"],
        "bedrooms": property_data["bedrooms"],
        "bathrooms": property_data["bathrooms"],
        "build_year": property_data["build_year"],
        "epc_score": property_data["epc_score"],
        "garden_area": property_data["garden_area"],
        "garage": property_data["garage"],
        "pool": property_data["pool"],
        "property_type": property_data["property_type"],
        "city": property_data["city"]
    }])

    # Voorspelling
    prediction = model.predict(df)[0]

    # Confidence berekenen
    confidence = max(0, 100 - (rmse / prediction) * 100)

    # Waarde range
    lower_bound = prediction - rmse
    upper_bound = prediction + rmse

    # Resultaat teruggeven (voor API response)
    return {
        "estimated_value": round(prediction, 2),
        "confidence_score": round(confidence, 1),
        "range_low": round(lower_bound, 2),
        "range_high": round(upper_bound, 2)
    }
# %%
