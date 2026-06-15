"""Étape 1 — Ingestion des données : lecture du CSV et découpage train/test."""

import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from frauddetection.constants import TARGET_COLUMN
from frauddetection.entity.config_entity import DataIngestionConfig
from frauddetection.entity.artifact_entity import DataIngestionArtifact
from frauddetection.exception.exception import FraudDetectionException
from frauddetection.logging.logger import logger


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate(self) -> DataIngestionArtifact:
        try:
            logger.info("Ingestion des données : lecture de %s", self.config.raw_data_path)
            df = pd.read_csv(self.config.raw_data_path)
            logger.info("  %d lignes, taux de fraude = %.3f%%",
                        len(df), 100 * df[TARGET_COLUMN].mean())

            train_df, test_df = train_test_split(
                df,
                test_size=self.config.test_size,
                stratify=df[TARGET_COLUMN],
                random_state=self.config.random_state,
            )

            os.makedirs(self.config.artifact_dir, exist_ok=True)
            train_df.to_csv(self.config.train_path, index=False)
            test_df.to_csv(self.config.test_path, index=False)
            logger.info("  train=%d / test=%d écrits dans %s",
                        len(train_df), len(test_df), self.config.artifact_dir)

            return DataIngestionArtifact(
                train_path=self.config.train_path,
                test_path=self.config.test_path,
            )
        except Exception as e:
            raise FraudDetectionException(e, sys) from e
