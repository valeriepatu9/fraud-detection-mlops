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


def slice_metrics(y_true, y_pred, slice_values, slice_name: str, min_frauds: int) -> dict:
    """Performance en global et par tranche (valeurs 0 / 1 de slice_values).

    Chaque entrée donne l'effectif, le nombre de fraudes, le taux de fraude,
    recall / précision / F1, et un drapeau 'fragile' quand la tranche compte
    moins de min_frauds fraudes (métriques alors peu fiables).
    """
    import pandas as pd

    y_true, y_pred, slice_values = (pd.Series(x).reset_index(drop=True)
                                    for x in (y_true, y_pred, slice_values))
    parts = [("global", slice_values == slice_values)] + [
        (f"{slice_name}={v}", slice_values == v) for v in (0, 1)
    ]
    out = {}
    for name, mask in parts:
        yt, yp = y_true[mask], y_pred[mask]
        m = compute_classification_metrics(yt, yp)
        n_frauds = int(yt.sum())
        out[name] = {
            "n": int(mask.sum()),
            "n_frauds": n_frauds,
            "fraud_rate": float(yt.mean()) if len(yt) else 0.0,
            "recall": m.recall,
            "precision": m.precision,
            "f1": m.f1_score,
            "fragile": n_frauds < min_frauds,
        }
    return out


def format_slice(s: dict) -> str:
    """Ligne de tableau : n, fraudes, taux, recall, précision, F1 (+ FRAGILE)."""
    flag = "  FRAGILE" if s["fragile"] else ""
    return (f"{s['n']:>7}{s['n_frauds']:>8}{s['fraud_rate']:>8.2%}"
            f"{s['recall']:>8.3f}{s['precision']:>8.3f}{s['f1']:>7.3f}{flag}")
