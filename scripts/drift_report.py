"""Rapport de drift Evidently : compare un lot de production à la référence.

  - Référence : le jeu de test hold-out (artifacts/test.csv), que le modèle n'a
    pas vu à l'entraînement -> ses performances y sont représentatives.
  - Lot courant : un CSV de production (voir scripts/generate_drift_data.py).

Le rapport couvre deux volets :
  1. Data drift   : la distribution de chaque feature a-t-elle bougé ?
  2. Performance  : le champion détecte-t-il toujours la fraude ? (suppose que
                    les labels du lot sont connus, ce qui en vrai arrive avec
                    retard : rétrofacturations, enquêtes...)

Tout est loggé dans MLflow (expérience 'fraud_drift_monitoring') : scores de
drift et métriques en metrics, rapport HTML en artifact.

Usage :
    python scripts/drift_report.py prod_ok
    python scripts/drift_report.py prod_data_drift
    python scripts/drift_report.py fraud_data/mon_lot.csv
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
import mlflow.sklearn
import pandas as pd
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.presets import ClassificationPreset, DataDriftPreset
from mlflow.tracking import MlflowClient

from frauddetection.constants import (
    CATEGORICAL_COLUMNS,
    CHAMPION_ALIAS,
    DRIFT_EXPERIMENT_NAME,
    DRIFT_SHARE,
    FEATURE_COLUMNS,
    MLFLOW_TRACKING_URI,
    RAW_DATA_PATH,
    REGISTERED_MODEL_NAME,
    REPORTS_DIR,
    TARGET_COLUMN,
)
from frauddetection.entity.config_entity import DataIngestionConfig
from frauddetection.utils.ml_utils import compute_classification_metrics

PREDICTION_COLUMN = "prediction"


def resolve_batch(arg: str) -> tuple[str, str]:
    """Accepte un nom de lot ('prod_ok') ou un chemin vers un CSV."""
    path = arg if arg.endswith(".csv") else os.path.join("fraud_data", f"{arg}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"Lot introuvable : {path} (lancer scripts/generate_drift_data.py ?)")
    return os.path.splitext(os.path.basename(path))[0], path


def reference_path() -> str:
    test_path = DataIngestionConfig().test_path
    if os.path.exists(test_path):
        return test_path
    print(f"[!] {test_path} absent (python main.py pas lancé ?) : référence = {RAW_DATA_PATH}")
    return RAW_DATA_PATH


def load_champion():
    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}")
    mv = MlflowClient().get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    return model, mv.version, mv.tags.get("algo", "?")


def to_dataset(df: pd.DataFrame) -> Dataset:
    definition = DataDefinition(
        numerical_columns=[c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS],
        categorical_columns=CATEGORICAL_COLUMNS,
        classification=[BinaryClassification(
            target=TARGET_COLUMN, prediction_labels=PREDICTION_COLUMN, pos_label=1,
        )],
    )
    return Dataset.from_pandas(df, data_definition=definition)


def extract_drift(snapshot) -> tuple[int, float, dict]:
    """Récupère dans le snapshot Evidently le nb/la part de features en drift
    et le score de drift par feature."""
    count, share, scores = 0, 0.0, {}
    for m in snapshot.dict()["metrics"]:
        name = m["metric_name"]
        if name.startswith("DriftedColumnsCount"):
            count, share = int(m["value"]["count"]), float(m["value"]["share"])
        elif name.startswith("ValueDrift"):
            col = m["config"]["column"]
            scores[col] = {
                "score": float(m["value"]),
                "method": m["config"].get("method"),
                "threshold": m["config"].get("threshold"),
            }
    return count, share, scores


def is_drifted(info: dict) -> bool:
    # Tests à p-value (KS, chi²) : drift si p < seuil ; distances (Wasserstein,
    # Jensen-Shannon...) : drift si distance >= seuil.
    method = (info["method"] or "").lower()
    if "p_value" in method or "test" in method:
        return info["score"] < info["threshold"]
    return info["score"] >= info["threshold"]


def run(batch_arg: str):
    batch_name, batch_path = resolve_batch(batch_arg)
    ref_path = reference_path()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model, version, algo = load_champion()

    ref = pd.read_csv(ref_path)
    cur = pd.read_csv(batch_path)
    for df in (ref, cur):
        df[PREDICTION_COLUMN] = model.predict(df[FEATURE_COLUMNS])

    report = Report([DataDriftPreset(columns=FEATURE_COLUMNS, drift_share=DRIFT_SHARE),
                     ClassificationPreset()])
    snapshot = report.run(current_data=to_dataset(cur), reference_data=to_dataset(ref))

    os.makedirs(REPORTS_DIR, exist_ok=True)
    html_path = os.path.join(REPORTS_DIR, f"drift_{batch_name}.html")
    snapshot.save_html(html_path)

    count, share, scores = extract_drift(snapshot)
    dataset_drift = share >= DRIFT_SHARE
    ref_metrics = compute_classification_metrics(ref[TARGET_COLUMN], ref[PREDICTION_COLUMN])
    cur_metrics = compute_classification_metrics(cur[TARGET_COLUMN], cur[PREDICTION_COLUMN])

    mlflow.set_experiment(DRIFT_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"{batch_name}__v{version}"):
        mlflow.log_params({
            "batch": batch_name,
            "reference": ref_path,
            "model_version": version,
            "model_algo": algo,
            "n_rows": len(cur),
        })
        mlflow.log_metric("drifted_columns_count", count)
        mlflow.log_metric("drifted_columns_share", share)
        for col, info in scores.items():
            mlflow.log_metric(f"drift_score__{col}", info["score"])
        mlflow.log_metric("fraud_rate", cur[TARGET_COLUMN].mean())
        mlflow.log_metric("recall", cur_metrics.recall)
        mlflow.log_metric("precision", cur_metrics.precision)
        mlflow.log_metric("f1_score", cur_metrics.f1_score)
        mlflow.log_metric("recall_reference", ref_metrics.recall)
        mlflow.log_metric("recall_delta", cur_metrics.recall - ref_metrics.recall)
        mlflow.set_tag("dataset_drift", str(dataset_drift))
        mlflow.log_artifact(html_path, artifact_path="evidently")

    print(f"\n=== Drift : {batch_name} vs référence — champion v{version} ({algo}) ===\n")
    print(f"{'feature':<24}{'méthode':<32}{'score':>9}{'seuil':>8}  drift")
    for col, info in scores.items():
        flag = "OUI" if is_drifted(info) else "-"
        print(f"{col:<24}{info['method'] or '':<32}{info['score']:>9.3f}{info['threshold']:>8.2f}  {flag}")
    print(f"\nFeatures en drift : {count}/{len(scores)} ({share:.0%})"
          f" -> drift global : {'OUI' if dataset_drift else 'non'}")
    print(f"Taux de fraude    : {ref[TARGET_COLUMN].mean():.2%} (réf) -> {cur[TARGET_COLUMN].mean():.2%}")
    print(f"Recall champion   : {ref_metrics.recall:.3f} (réf) -> {cur_metrics.recall:.3f}")
    print(f"Precision         : {ref_metrics.precision:.3f} (réf) -> {cur_metrics.precision:.3f}")
    print(f"\nRapport HTML : {os.path.abspath(html_path)} (aussi dans MLflow, expérience '{DRIFT_EXPERIMENT_NAME}')")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage : python scripts/drift_report.py <prod_ok|prod_data_drift|prod_concept_drift|chemin.csv>")
    run(sys.argv[1])
