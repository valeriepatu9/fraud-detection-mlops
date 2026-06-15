"""Objets 'artefact' : ce que produit chaque étape du pipeline."""

from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    train_path: str
    test_path: str


@dataclass
class ClassificationMetric:
    accuracy: float
    precision: float
    recall: float
    f1_score: float


@dataclass
class ModelRunArtifact:
    run_id: str
    model_label: str          # "v1_decision_tree" / "v2_random_forest"
    algo: str                 # nom de l'algorithme
    registered_version: str   # version dans le Model Registry
    metrics: ClassificationMetric
