"""Compare @champion et @challenger sur un lot qu'aucun des deux n'a vu.

Le lot est en principe celui du mois suivant (prod_foreign_card_next,
prod_card_testing_next) : le challenger a été entraîné sur le mois précédent,
l'évaluer sur ses propres données serait trop flatteur.

La comparaison porte sur la performance (labels connus), en global et par
tranche is_foreign_country, et elle est loggée dans MLflow (expérience
'fraud_model_comparison', un run par comparaison).

Usage :
    python scripts/compare_models.py prod_foreign_card_next
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from frauddetection.constants import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    COMPARISON_EXPERIMENT_NAME,
    FEATURE_COLUMNS,
    FRAGILE_MIN_FRAUDS,
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
    SLICE_COLUMN,
    TARGET_COLUMN,
)
from frauddetection.utils.ml_utils import format_slice, slice_metrics


def resolve_batch(arg: str) -> tuple[str, str]:
    path = arg if arg.endswith(".csv") else os.path.join("fraud_data", f"{arg}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"Lot introuvable : {path} (lancer scripts/generate_drift_data.py ?)")
    return os.path.splitext(os.path.basename(path))[0], path


def compare(batch_arg: str):
    batch_name, batch_path = resolve_batch(batch_arg)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    df = pd.read_csv(batch_path)

    results = {}
    for alias in (CHAMPION_ALIAS, CHALLENGER_ALIAS):
        try:
            mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias)
        except Exception:
            raise SystemExit(f"Aucune version sous l'alias @{alias} (lancer scripts/retrain.py ?)")
        model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/{mv.version}")
        pred = model.predict(df[FEATURE_COLUMNS])
        results[alias] = {
            "version": mv.version,
            "training_data": mv.tags.get("training_data", "historique"),
            "perf": slice_metrics(df[TARGET_COLUMN], pred, df[SLICE_COLUMN], SLICE_COLUMN, FRAGILE_MIN_FRAUDS),
        }

    champ, chall = results[CHAMPION_ALIAS], results[CHALLENGER_ALIAS]
    mlflow.set_experiment(COMPARISON_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"{batch_name}__v{champ['version']}_vs_v{chall['version']}"):
        mlflow.log_params({
            "batch": batch_name,
            "champion_version": champ["version"],
            "challenger_version": chall["version"],
            "challenger_training_data": chall["training_data"],
        })
        for alias, r in results.items():
            for name, s in r["perf"].items():
                key = "" if name == "global" else "__" + name.replace("=", "_")
                for metric in ("recall", "precision", "f1"):
                    mlflow.log_metric(f"{alias}__{metric}{key}", s[metric])
        delta = chall["perf"]["global"]["recall"] - champ["perf"]["global"]["recall"]
        mlflow.log_metric("recall_delta", delta)

    print(f"\n=== {batch_name} : champion v{champ['version']} contre challenger v{chall['version']} "
          f"(entraîné sur {chall['training_data']}) ===\n")
    print(f"{'tranche':<24}{'':<13}{'n':>7}{'fraudes':>8}{'taux':>8}{'recall':>8}{'précis.':>8}{'F1':>7}")
    for name in champ["perf"]:
        print(f"{name:<24}{'champion':<13}{format_slice(champ['perf'][name])}")
        print(f"{'':<24}{'challenger':<13}{format_slice(chall['perf'][name])}")
    print(f"\nRecall global : {champ['perf']['global']['recall']:.3f} -> {chall['perf']['global']['recall']:.3f} "
          f"({delta:+.3f})")
    if delta > 0:
        print(f"Promouvoir le challenger : python scripts/promote_model.py {chall['version']}")
    else:
        print("Le challenger ne fait pas mieux que le champion : ne pas le promouvoir.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage : python scripts/compare_models.py <lot_du_mois_suivant>")
    compare(sys.argv[1])
