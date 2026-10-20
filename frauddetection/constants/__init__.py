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
# Nouvelle version candidate, comparée au champion avant promotion
CHALLENGER_ALIAS = "challenger"

# Référence du drift versionnée avec chaque modèle (voir utils/reference.py)
REFERENCE_ARTIFACT_DIR = "reference"
REFERENCE_FILE = "test.csv"
REFERENCE_TAG = "reference_sha256"

# Seuil de décision pour qualifier une transaction de frauduleuse
DECISION_THRESHOLD = 0.5

# --- Monitoring du drift (Evidently) ---
DRIFT_EXPERIMENT_NAME = "fraud_drift_monitoring"
COMPARISON_EXPERIMENT_NAME = "fraud_model_comparison"
REPORTS_DIR = "reports"
# Features binaires : traitées comme catégorielles par Evidently (test adapté)
CATEGORICAL_COLUMNS = ["is_new_device", "is_foreign_country"]
# Part de features en drift au-delà de laquelle on déclare le jeu "en drift"
DRIFT_SHARE = 0.5
# Tranche de performance suivie à l'arrivée des labels (zone peu apprise)
SLICE_COLUMN = "is_foreign_country"
# En dessous de ce nombre de fraudes, les métriques d'une tranche sont fragiles
FRAGILE_MIN_FRAUDS = 30
