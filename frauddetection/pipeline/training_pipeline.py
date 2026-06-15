"""Orchestration du pipeline d'entraînement de bout en bout."""

import sys

from frauddetection.components.data_ingestion import DataIngestion
from frauddetection.components.model_trainer import ModelTrainer
from frauddetection.entity.config_entity import DataIngestionConfig, ModelTrainerConfig
from frauddetection.exception.exception import FraudDetectionException
from frauddetection.logging.logger import logger


class TrainingPipeline:
    def run(self):
        try:
            logger.info("=== Démarrage du pipeline d'entraînement ===")
            ingestion_artifact = DataIngestion(DataIngestionConfig()).initiate()
            v1, v2 = ModelTrainer(ModelTrainerConfig(), ingestion_artifact).initiate()
            logger.info("=== Pipeline terminé ===")
            logger.info("Récapitulatif :")
            for v in (v1, v2):
                logger.info(
                    "  %s (v%s, %s) : recall=%.3f f1=%.3f accuracy=%.4f",
                    v.model_label, v.registered_version, v.algo,
                    v.metrics.recall, v.metrics.f1_score, v.metrics.accuracy,
                )
            return v1, v2
        except Exception as e:
            raise FraudDetectionException(e, sys) from e
