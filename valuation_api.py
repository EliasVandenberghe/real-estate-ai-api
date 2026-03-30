# %%
import os
import uvicorn
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI

# Global variabele voor de pipeline
chronos_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # DIT GEBEURT DIRECT BIJ HET OPSTARTEN
    global chronos_pipeline
    print("Systeem start op: Chronos wordt nu geladen...")
    from price_trend import generate_price_trend # Importeer de logica
    # Hier laden we het model alvast in het geheugen
    # (Je kunt hier eventueel een dummy-voorspelling doen om het model te 'warmen')
    print("Chronos is succesvol ingeladen!")
    yield
    # Hier kun je eventueel dingen afsluiten bij shutdown
    print("Systeem sluit af...")

app = FastAPI(lifespan=lifespan)

# Rest van je code (valuation_model.pkl laden etc.)
data = joblib.load("valuation_model.pkl")
model = data["model"]
rmse = data["rmse"]

@app.get("/")
def health_check():
    return {"status": "online", "message": "Real Estate API is running"}

# Je endpoints blijven hetzelfde...
# %%
