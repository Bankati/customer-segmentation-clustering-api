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
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

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
    # L'exemple ci-dessous remplace les valeurs par défaut que Swagger
    # générerait sinon ("string", 0...) — notamment Dt_Customer, qui ferait
    # planter le parsing de date si on clique "Try it out" sans le modifier.
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Year_Birth": 1985,
                "Education": "Graduation",
                "Marital_Status": "Married",
                "Income": 58000,
                "Kidhome": 0,
                "Teenhome": 1,
                "Dt_Customer": "04-09-2012",
                "Recency": 12,
                "MntWines": 450,
                "MntFruits": 30,
                "MntMeatProducts": 320,
                "MntFishProducts": 40,
                "MntSweetProducts": 25,
                "MntGoldProds": 60,
                "NumDealsPurchases": 3,
                "NumWebPurchases": 8,
                "NumCatalogPurchases": 4,
                "NumStorePurchases": 7,
                "NumWebVisitsMonth": 5,
                "AcceptedCmp1": 0,
                "AcceptedCmp2": 0,
                "AcceptedCmp3": 0,
                "AcceptedCmp4": 1,
                "AcceptedCmp5": 0,
                "Response": 1,
            }
        }
    )

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

    try:
        features = engineer_features(raw_df)
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Données invalides : {exc}. Vérifiez notamment Dt_Customer (format attendu JJ-MM-AAAA).",
        ) from exc

    preprocessor = ml_artifacts["preprocessor"]
    kmeans = ml_artifacts["kmeans"]
    metadata = ml_artifacts["metadata"]

    try:
        X_scaled = preprocessor.transform(features)
        cluster = int(kmeans.predict(X_scaled)[0])
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Impossible de traiter ces données : {exc}") from exc

    persona = metadata["persona_map"].get(str(cluster), "Persona non défini")

    return PredictionResponse(cluster=cluster, persona=persona)
