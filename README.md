# Customer Segmentation — Clustering API

Segmentation client par K-Means, industrialisée depuis une exploration notebook jusqu'à une API FastAPI.

## Structure

```
.
├── app/
│   ├── main.py           # API FastAPI (endpoint /predict)
│   └── preprocessing.py  # Feature engineering partagé entraînement/API
├── data/
│   └── customer_segmentation.csv  # Non versionné (voir .gitignore)
├── models/                # Artefacts entraînés (joblib + metadata.json)
├── notebooks/
│   └── exploration.ipynb  # Notebook d'analyse exploratoire d'origine
├── train.py                # Script d'entraînement
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Entraîner le modèle

Placez `customer_segmentation.csv` dans `data/`, puis :

```bash
python train.py
```

Génère `models/preprocessor.joblib`, `models/kmeans_model.joblib` et `models/metadata.json`.

## Lancer l'API

```bash
uvicorn app.main:app --reload
```

- `GET /health` — vérification de disponibilité
- `POST /predict` — prend une fiche client brute, retourne le cluster prédit et le persona associé

Documentation interactive : http://127.0.0.1:8000/docs
