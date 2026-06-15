"""Constantes centralisées du projet."""

import os

# --- Données ---
RAW_DATA_PATH = os.path.join("fraud_data", "transactions.csv")
ARTIFACT_DIR = "artifacts"
TARGET_COLUMN = "is_fraud"

FEATURE_COLUMNS = [
    "amount",
    "hour",
    "distance_from_home_km",
    "is_new_device",
    "is_foreign_country",
    "transactions_last_24h",
    "avg_amount_last_30d",
    "amount_to_avg_ratio",
]

TEST_SIZE = 0.25
RANDOM_STATE = 42

# --- MLflow ---
# Serveur de tracking : local par défaut, surchargé par la variable d'env MLFLOW_TRACKING_URI.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "fraud_detection"
REGISTERED_MODEL_NAME = "FraudDetector"
# Alias moderne (remplace les "stages" Staging/Production dépréciés depuis MLflow 2.9)
CHAMPION_ALIAS = "champion"

# Seuil de décision pour qualifier une transaction de frauduleuse
DECISION_THRESHOLD = 0.5
