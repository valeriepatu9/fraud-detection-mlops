"""Objets de configuration du pipeline (les 'entrées' de chaque étape)."""

from dataclasses import dataclass, field
import os

from frauddetection.constants import (
    RAW_DATA_PATH,
    ARTIFACT_DIR,
    TEST_SIZE,
    RANDOM_STATE,
    EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
)


@dataclass
class DataIngestionConfig:
    raw_data_path: str = RAW_DATA_PATH
    artifact_dir: str = ARTIFACT_DIR
    test_size: float = TEST_SIZE
    random_state: int = RANDOM_STATE

    @property
    def train_path(self) -> str:
        return os.path.join(self.artifact_dir, "train.csv")

    @property
    def test_path(self) -> str:
        return os.path.join(self.artifact_dir, "test.csv")


@dataclass
class ModelTrainerConfig:
    experiment_name: str = EXPERIMENT_NAME
    registered_model_name: str = REGISTERED_MODEL_NAME
    random_state: int = RANDOM_STATE
    # Hyperparamètres des deux modèles comparés pendant la démo
    decision_tree_params: dict = field(default_factory=lambda: {"max_depth": 2})
    random_forest_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 300,
            "class_weight": "balanced",
            "n_jobs": -1,
        }
    )
