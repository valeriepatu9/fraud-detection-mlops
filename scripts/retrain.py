"""Réentraînement après drift : nouveau RandomForest sur le lot récent.

Le lot de production (désormais labellisé) sert de jeu d'entraînement, et un
RandomForest est enregistré comme NOUVELLE version de FraudDetector.

Pourquoi PAS l'historique ? En cas de concept drift, les anciens labels sont
périmés : les transactions "card testing" de l'historique sont étiquetées
légitimes (ce n'était pas de la fraude à l'époque), ce qui contredit les
nouveaux labels. Avec --with-history, le recall sur le mois suivant plafonne
vers 0.5 ; sur la fenêtre récente seule, il remonte vers 0.9.

L'alias 'champion' n'est PAS déplacé : on compare d'abord les métriques dans
MLflow, puis on promeut en live :

    python scripts/promote_model.py <version>

Usage :
    python scripts/retrain.py prod_concept_drift                 # fenêtre récente
    python scripts/retrain.py prod_concept_drift --with-history  # le piège
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from frauddetection.components.data_ingestion import DataIngestion
from frauddetection.components.model_trainer import ModelTrainer
from frauddetection.constants import ARTIFACT_DIR, MLFLOW_TRACKING_URI, RAW_DATA_PATH
from frauddetection.entity.config_entity import DataIngestionConfig, ModelTrainerConfig


def retrain(batch_name: str, with_history: bool = False):
    batch_path = os.path.join("fraud_data", f"{batch_name}.csv")
    if not os.path.exists(batch_path):
        raise SystemExit(f"Lot introuvable : {batch_path}")

    # Dossier dédié : ne pas écraser artifacts/test.csv, qui sert de référence au drift
    work_dir = os.path.join(ARTIFACT_DIR, f"retrain_{batch_name}")
    os.makedirs(work_dir, exist_ok=True)
    frames = [pd.read_csv(RAW_DATA_PATH)] if with_history else []
    frames.append(pd.read_csv(batch_path))
    training_path = os.path.join(work_dir, "training_data.csv")
    pd.concat(frames).to_csv(training_path, index=False)

    ingestion = DataIngestion(DataIngestionConfig(raw_data_path=training_path, artifact_dir=work_dir)).initiate()
    config = ModelTrainerConfig()
    trainer = ModelTrainer(config, ingestion)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.experiment_name)
    Xtr, ytr = trainer._load_xy(ingestion.train_path)
    Xte, yte = trainer._load_xy(ingestion.test_path)
    suffix = "_with_history" if with_history else ""
    result = trainer._train_and_log(
        f"v_retrain_{batch_name}{suffix}",
        "RandomForestClassifier",
        RandomForestClassifier(random_state=config.random_state, **config.random_forest_params),
        Xtr, ytr, Xte, yte,
    )

    print(f"\nNouvelle version : {config.registered_model_name} v{result.registered_version}"
          f" (recall test = {result.metrics.recall:.3f}, f1 = {result.metrics.f1_score:.3f})")
    print(f"Pour la mettre en production : python scripts/promote_model.py {result.registered_version}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--with-history"]
    if len(args) != 1:
        raise SystemExit("Usage : python scripts/retrain.py <nom_du_lot> [--with-history]")
    retrain(args[0], with_history="--with-history" in sys.argv)
