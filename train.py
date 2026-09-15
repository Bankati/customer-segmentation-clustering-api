"""Entraînement du modèle de segmentation client.

Charge les données brutes, applique le feature engineering, ajuste le
préprocesseur (imputation + scaling) et le K-Means, puis sauvegarde les
artefacts dans models/ avec joblib pour être réutilisés par l'API.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.preprocessing import NUM_VAR, engineer_features

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "customer_segmentation.csv"
MODELS_DIR = BASE_DIR / "models"
K_RANGE = range(2, 11)
RANDOM_STATE = 42

PERSONA_NAMES_BY_MONETARY_RANK = ["Clientèle Économe", "Clientèle Premium VIP"]


def find_optimal_k(X_scaled: np.ndarray) -> tuple[int, list[float]]:
    silhouette_scores = []
    for k in K_RANGE:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
    optimal_k = list(K_RANGE)[int(np.argmax(silhouette_scores))]
    return optimal_k, silhouette_scores


def build_persona_map(df: pd.DataFrame, optimal_k: int) -> dict:
    """Associe chaque cluster à un persona en se basant sur le Monetary moyen,
    plutôt que sur l'identifiant de cluster (arbitraire selon le run)."""
    if optimal_k != len(PERSONA_NAMES_BY_MONETARY_RANK):
        return {}

    clusters_by_monetary = df.groupby("cluster")["Monetary"].mean().sort_values().index.tolist()
    return {
        int(cluster_id): persona
        for cluster_id, persona in zip(clusters_by_monetary, PERSONA_NAMES_BY_MONETARY_RANK)
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)

    numerical_pipeline = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler())
    preprocessor = make_column_transformer((numerical_pipeline, NUM_VAR))
    X_scaled = preprocessor.fit_transform(df)

    optimal_k, silhouette_scores = find_optimal_k(X_scaled)
    print(f"Nombre optimal de clusters : {optimal_k} (silhouette={max(silhouette_scores):.3f})")

    final_model = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
    df["cluster"] = final_model.fit_predict(X_scaled)

    persona_map = build_persona_map(df, optimal_k)
    if not persona_map:
        print(f"Avertissement : pas de mapping persona défini pour k={optimal_k} clusters.")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    joblib.dump(final_model, MODELS_DIR / "kmeans_model.joblib")

    metadata = {
        "num_var": NUM_VAR,
        "optimal_k": optimal_k,
        "silhouette_score": float(max(silhouette_scores)),
        "persona_map": persona_map,
    }
    with open(MODELS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Artefacts sauvegardés dans {MODELS_DIR.resolve()}")


if __name__ == "__main__":
    main()
