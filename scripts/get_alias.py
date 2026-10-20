"""Affiche le numéro de version porté par un alias de FraudDetector (ou rien).

Utilisé par scripts/run_demo_local.ps1 pour enchaîner les étapes sans coder
les numéros de version en dur.

Usage :
    python scripts/get_alias.py champion
    python scripts/get_alias.py challenger
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
from mlflow.tracking import MlflowClient

from frauddetection.constants import MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage : python scripts/get_alias.py <alias>")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        print(MlflowClient().get_model_version_by_alias(REGISTERED_MODEL_NAME, sys.argv[1]).version)
    except Exception:
        sys.exit(1)
