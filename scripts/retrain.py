"""Réentraînement après drift : nouvelle version de FraudDetector sous l'alias @challenger.

Le lot de production, désormais labellisé, sert à entraîner un RandomForest
(mêmes hyperparamètres que la v2). La nouvelle version reçoit l'alias
'challenger' ; l'alias 'champion' n'est PAS touché. Sa référence de drift est
son propre jeu de test (découpage 75/25 des données d'entraînement).

Deux stratégies, selon la nature du drift :

  - Data drift (prod_foreign_card) : --with-history. La distribution a changé
    mais les règles de fraude non : les anciens labels restent valides, et
    l'historique apporte du volume sur la zone déjà connue.
  - Concept drift (prod_card_testing) : lot récent seul. Dans l'historique, les
    rafales de petits paiements sont étiquetées légitimes (ce n'était pas de la
    fraude à l'époque) : ces labels périmés contredisent les nouveaux.
    --with-history reste disponible pour montrer le piège.

Ensuite :
    python scripts/compare_models.py <lot_du_mois_suivant>
    python scripts/promote_model.py <version>

Usage :
    python scripts/retrain.py prod_foreign_card --with-history
    python scripts/retrain.py prod_card_testing
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier

from frauddetection.components.data_ingestion import DataIngestion
from frauddetection.components.model_trainer import ModelTrainer
from frauddetection.constants import ARTIFACT_DIR, CHALLENGER_ALIAS, MLFLOW_TRACKING_URI, RAW_DATA_PATH
from frauddetection.entity.config_entity import DataIngestionConfig, ModelTrainerConfig


def retrain(batch_name: str, with_history: bool = False):
    batch_path = os.path.join("fraud_data", f"{batch_name}.csv")
    if not os.path.exists(batch_path):
        raise SystemExit(f"Lot introuvable : {batch_path}")

    # Dossier dédié : ne pas écraser artifacts/test.csv, référence de la v1 et de la v2
    suffix = "_with_history" if with_history else ""
    work_dir = os.path.join(ARTIFACT_DIR, f"retrain_{batch_name}{suffix}")
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
    result = trainer._train_and_log(
        f"v_retrain_{batch_name}{suffix}",
        "RandomForestClassifier",
        RandomForestClassifier(random_state=config.random_state, **config.random_forest_params),
        Xtr, ytr, Xte, yte,
    )

    client = MlflowClient()
    client.set_registered_model_alias(config.registered_model_name, CHALLENGER_ALIAS, result.registered_version)
    client.set_model_version_tag(config.registered_model_name, result.registered_version,
                                 "training_data", f"{batch_name}{' + historique' if with_history else ''}")

    print(f"\nNouvelle version : {config.registered_model_name} v{result.registered_version} "
          f"-> alias @{CHALLENGER_ALIAS} (champion inchangé)")
    print(f"Données : {batch_name}{' + historique' if with_history else ' seul'}, "
          f"{len(Xtr) + len(Xte)} transactions ; recall sur son propre test = {result.metrics.recall:.3f}")
    print(f"Comparer : python scripts/compare_models.py {batch_name}_next")
    print(f"Promouvoir : python scripts/promote_model.py {result.registered_version}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--with-history"]
    if len(args) != 1:
        raise SystemExit("Usage : python scripts/retrain.py <nom_du_lot> [--with-history]")
    retrain(args[0], with_history="--with-history" in sys.argv)
