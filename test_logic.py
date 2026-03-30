#%%
from valuation_logic import predict_price, generate_price_trend
import json

# Test data (Huis in Leuven)
test_property = {
    "surface_area": 95,
    "bedrooms": 2,
    "bathrooms": 1,
    "build_year": 1998,
    "epc_score": 220,
    "garden_area": 40,
    "garage": False,
    "pool": False,
    "property_type": "house",
    "city": "Leuven"
}

def run_full_test():
    print("--- START INTEGRALE TEST ---")
    
    try:
        # Haal de data op zoals de API dat zou doen
        val_res = predict_price(test_property)
        trend_res = generate_price_trend(test_property)

        # Bouw de finale JSON
        final_json = {
            "valuation": val_res,
            "price_trend": trend_res
        }

        # Print de JSON op een mooie manier (indent=2)
        print("\nGEGENEREERDE JSON OUTPUT:")
        print(json.dumps(final_json, indent=2))
        
        # Kleine validatie-check
        if "historical_years" in trend_res:
            print(f"\n✅ Test geslaagd: {len(trend_res['historical_years'])} jaren aan data gevonden.")
        else:
            print("\n⚠️ Waarschuwing: Trend data lijkt onvolledig.")

    except Exception as e:
        print(f"\n❌ KRITIEKE FOUT TIJDENS TEST: {e}")

    print("\n--- TEST AFGEROND ---")

if __name__ == "__main__":
    run_full_test()