"""Feature engineering pour le clustering client.

Reproduit exactement les transformations du notebook d'exploration :
suppression des colonnes inutiles, création des variables Age, Number_kids,
Monetary, Frequence, Categorie_Marketing et Customer_since.
"""

import numpy as np
import pandas as pd

# Colonnes numériques effectivement utilisées par le modèle de clustering
NUM_VAR = ["Income", "Recency", "Age", "Number_kids", "Monetary", "Frequence"]

_MNT_COLS = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
]

_PURCHASE_COLS = [
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
]

_CAMPAIGN_COLS = [
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
]


def engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Transforme les données brutes en features prêtes pour le preprocessor sklearn."""
    df = raw_df.copy()

    df.drop(columns=["ID", "Complain", "Z_CostContact", "Z_Revenue"], inplace=True, errors="ignore")

    df["Age"] = pd.Timestamp.now().year - df["Year_Birth"]
    df.drop(columns=["Year_Birth"], inplace=True)

    df["Number_kids"] = df["Kidhome"] + df["Teenhome"]
    df.drop(columns=["Kidhome", "Teenhome"], inplace=True)

    df["Monetary"] = df[_MNT_COLS].sum(axis=1)
    df.drop(columns=_MNT_COLS, inplace=True)

    df["Frequence"] = df[_PURCHASE_COLS].sum(axis=1)
    df.drop(columns=_PURCHASE_COLS, inplace=True)

    marketing = df[_CAMPAIGN_COLS].sum(axis=1) + df["Response"]
    df["Categorie_Marketing"] = np.select(
        [marketing == 0, marketing.between(1, 2), marketing >= 3],
        ["1", "2", "3"],
        default="0",
    )
    df.drop(columns=_CAMPAIGN_COLS + ["Response"], inplace=True)

    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="mixed")
    df["Customer_since"] = (pd.Timestamp.now() - df["Dt_Customer"]).dt.days
    df.drop(columns=["Dt_Customer"], inplace=True)

    return df

