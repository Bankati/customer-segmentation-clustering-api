"""API de segmentation client.

Charge le préprocesseur et le modèle K-Means entraînés (voir train.py),
et expose un endpoint /predict qui prend une fiche client brute et
retourne le cluster prédit ainsi que le persona associé.
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.preprocessing import engineer_features

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

ml_artifacts: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_artifacts["preprocessor"] = joblib.load(MODELS_DIR / "preprocessor.joblib")
    ml_artifacts["kmeans"] = joblib.load(MODELS_DIR / "kmeans_model.joblib")
    ml_artifacts["metadata"] = json.loads((MODELS_DIR / "metadata.json").read_text(encoding="utf-8"))
    yield
    ml_artifacts.clear()


app = FastAPI(title="Customer Segmentation API", lifespan=lifespan)


class CustomerData(BaseModel):
    Year_Birth: int
    Education: str
    Marital_Status: str
    Income: Optional[float] = None
    Kidhome: int
    Teenhome: int
    Dt_Customer: str = Field(description="Format JJ-MM-AAAA, ex: 04-09-2012")
    Recency: int
    MntWines: int
    MntFruits: int
    MntMeatProducts: int
    MntFishProducts: int
    MntSweetProducts: int
    MntGoldProds: int
    NumDealsPurchases: int
    NumWebPurchases: int
    NumCatalogPurchases: int
    NumStorePurchases: int
    NumWebVisitsMonth: int
    AcceptedCmp1: int
    AcceptedCmp2: int
    AcceptedCmp3: int
    AcceptedCmp4: int
    AcceptedCmp5: int
    Response: int


class PredictionResponse(BaseModel):
    cluster: int
    persona: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    raw_df = pd.DataFrame([customer.model_dump()])
    features = engineer_features(raw_df)

    preprocessor = ml_artifacts["preprocessor"]
    kmeans = ml_artifacts["kmeans"]
    metadata = ml_artifacts["metadata"]

    X_scaled = preprocessor.transform(features)
    cluster = int(kmeans.predict(X_scaled)[0])
    persona = metadata["persona_map"].get(str(cluster), "Persona non défini")

    return PredictionResponse(cluster=cluster, persona=persona)
