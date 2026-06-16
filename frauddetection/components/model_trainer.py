"""Étape 3 — Entraînement et tracking MLflow.

C'est ici que vit toute la logique MLflow de la démo :

  1. On entraîne DEUX modèles dans des "runs" MLflow séparés :
       - v1 : DecisionTree(max_depth=2) sans rééquilibrage  -> le modèle "en prod"
       - v2 : RandomForest(class_weight="balanced")          -> le bon modèle
     Chaque run logge ses hyperparamètres, ses métriques (accuracy/recall/F1)
     et le modèle lui-même (Pipeline scaler + classifieur).

  2. On enregistre les deux modèles dans le Model Registry sous le même nom
     ("FraudDetector"), ce qui crée les versions 1 et 2.

  3. On pose l'alias 'champion' sur la VERSION 1 (le mauvais modèle), pour
     reproduire la situation de départ : c'est ce que l'app sert en prod.

  La promotion du bon modèle (champion -> v2) se fait ensuite EN LIVE, soit via
  scripts/promote_model.py, soit directement dans l'UI MLflow.
"""

import sys

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from frauddetection.constants import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    MLFLOW_TRACKING_URI,
    CHAMPION_ALIAS,
)
from frauddetection.components.data_transformation import DataTransformation
from frauddetection.entity.config_entity import ModelTrainerConfig
from frauddetection.entity.artifact_entity import DataIngestionArtifact, ModelRunArtifact
from frauddetection.exception.exception import FraudDetectionException
from frauddetection.logging.logger import logger
from frauddetection.utils.ml_utils import compute_classification_metrics


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, ingestion: DataIngestionArtifact):
        self.config = config
        self.ingestion = ingestion

    def _load_xy(self, path):
        df = pd.read_csv(path)
        return df[FEATURE_COLUMNS], df[TARGET_COLUMN]

    def _train_and_log(self, model_label, algo_name, estimator, Xtr, ytr, Xte, yte):
        """Entraîne un modèle, le logge dans un run MLflow et l'enregistre."""
        with mlflow.start_run(run_name=model_label) as run:
            pipe = Pipeline(
                steps=[
                    ("scaler", DataTransformation.build_preprocessor()),
                    ("classifier", estimator),
                ]
            )
            pipe.fit(Xtr, ytr)
            preds = pipe.predict(Xte)
            metrics = compute_classification_metrics(yte, preds)

            mlflow.log_param("algorithm", algo_name)
            mlflow.log_params({f"clf__{k}": v for k, v in estimator.get_params().items()
                               if k in ("max_depth", "n_estimators", "class_weight")})
            mlflow.log_metric("accuracy", metrics.accuracy)
            mlflow.log_metric("precision", metrics.precision)
            mlflow.log_metric("recall", metrics.recall)
            mlflow.log_metric("f1_score", metrics.f1_score)
            mlflow.set_tag("model_label", model_label)

            # Logge ET enregistre le modèle dans le Registry (crée une nouvelle version)
            result = mlflow.sklearn.log_model(
                sk_model=pipe,
                artifact_path="model",
                registered_model_name=self.config.registered_model_name,
                input_example=Xte.head(2),
            )
            logger.info(
                "[%s] recall=%.3f f1=%.3f accuracy=%.4f | run_id=%s",
                model_label, metrics.recall, metrics.f1_score, metrics.accuracy, run.info.run_id,
            )

            client = MlflowClient()
            version = self._latest_version(client)
            client.set_model_version_tag(
                self.config.registered_model_name, version, "algo", algo_name
            )

            return ModelRunArtifact(
                run_id=run.info.run_id,
                model_label=model_label,
                algo=algo_name,
                registered_version=version,
                metrics=metrics,
            )

    def _latest_version(self, client: MlflowClient) -> str:
        versions = client.search_model_versions(
            f"name='{self.config.registered_model_name}'"
        )
        return str(max(int(v.version) for v in versions))

    def initiate(self):
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(self.config.experiment_name)
            logger.info("Tracking MLflow : %s | experiment=%s",
                        MLFLOW_TRACKING_URI, self.config.experiment_name)

            Xtr, ytr = self._load_xy(self.ingestion.train_path)
            Xte, yte = self._load_xy(self.ingestion.test_path)

            # --- Run A : le mauvais modèle (celui "en prod") ---
            v1 = self._train_and_log(
                "v1_decision_tree",
                "DecisionTreeClassifier",
                DecisionTreeClassifier(random_state=self.config.random_state,
                                       **self.config.decision_tree_params),
                Xtr, ytr, Xte, yte,
            )

            # --- Run B : le bon modèle ---
            v2 = self._train_and_log(
                "v2_random_forest",
                "RandomForestClassifier",
                RandomForestClassifier(random_state=self.config.random_state,
                                       **self.config.random_forest_params),
                Xtr, ytr, Xte, yte,
            )

            # --- Situation de départ : le champion en prod = le MAUVAIS modèle (v1) ---
            client = MlflowClient()
            client.set_registered_model_alias(
                self.config.registered_model_name, CHAMPION_ALIAS, v1.registered_version
            )
            logger.info(
                "Alias '%s' posé sur la version %s (%s) — c'est le modèle servi en prod.",
                CHAMPION_ALIAS, v1.registered_version, v1.algo,
            )
            logger.info(
                "Pour promouvoir le bon modèle en live : "
                "python scripts/promote_model.py %s", v2.registered_version,
            )
            return v1, v2
        except Exception as e:
            raise FraudDetectionException(e, sys) from e
