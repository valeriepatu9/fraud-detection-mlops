"""Fonctions utilitaires ML : calcul des métriques de classification."""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from frauddetection.entity.artifact_entity import ClassificationMetric


def compute_classification_metrics(y_true, y_pred) -> ClassificationMetric:
    """Calcule accuracy, precision, recall, F1 pour la classe positive (fraude).

    Sur un jeu très déséquilibré, l'accuracy est trompeuse : un modèle qui
    prédit toujours "légitime" obtient ~99% d'accuracy tout en ratant 100%
    des fraudes. C'est pourquoi on suit surtout le RECALL (part des fraudes
    réellement détectées) et le F1.
    """
    return ClassificationMetric(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1_score=f1_score(y_true, y_pred, zero_division=0),
    )
