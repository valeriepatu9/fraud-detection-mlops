"""Rapport de drift Evidently : compare un lot de production à la référence du champion.

  - Référence : le jeu de test hold-out du champion en cours, versionné avec
    le modèle (artifact reference/test.csv, tag reference_sha256). Après un
    réentraînement, la référence du nouveau champion est son propre jeu de test.
  - Lot courant : un CSV de production (voir scripts/generate_drift_data.py).

Le rapport sépare ce que l'on sait à deux moments différents :

  A. Disponible tout de suite (sans labels) : drift variable par variable,
     verdict global, taux d'alertes du champion.
  B. Disponible à l'arrivée des labels, un à trois mois plus tard en fraude
     (contestations, enquêtes) : taux de fraude, recall, précision, F1, en
     global et par tranche is_foreign_country.

Tout est loggé dans MLflow (expérience 'fraud_drift_monitoring', un run par
lot) : métriques "now__*" pour le bloc A, "labels__*" pour le bloc B, rapport
HTML Evidently en artifact.

Usage :
    python scripts/drift_report.py prod_normal
    python scripts/drift_report.py prod_foreign_card
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
    FRAGILE_MIN_FRAUDS,
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
    REPORTS_DIR,
    SLICE_COLUMN,
    TARGET_COLUMN,
)
from frauddetection.utils.ml_utils import compute_classification_metrics
from frauddetection.utils.reference import load_reference

PREDICTION_COLUMN = "prediction"


def resolve_batch(arg: str) -> tuple[str, str]:
    """Accepte un nom de lot ('prod_normal') ou un chemin vers un CSV."""
    path = arg if arg.endswith(".csv") else os.path.join("fraud_data", f"{arg}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"Lot introuvable : {path} (lancer scripts/generate_drift_data.py ?)")
    return os.path.splitext(os.path.basename(path))[0], path


def load_champion():
    mv = MlflowClient().get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/{mv.version}")
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


def slice_metrics(df: pd.DataFrame) -> dict:
    """Performance en global et par tranche de SLICE_COLUMN (0 / 1)."""
    out = {}
    parts = [("global", df)] + [(f"{SLICE_COLUMN}={v}", df[df[SLICE_COLUMN] == v]) for v in (0, 1)]
    for name, part in parts:
        m = compute_classification_metrics(part[TARGET_COLUMN], part[PREDICTION_COLUMN])
        n_frauds = int(part[TARGET_COLUMN].sum())
        out[name] = {
            "n": len(part),
            "n_frauds": n_frauds,
            "fraud_rate": part[TARGET_COLUMN].mean() if len(part) else 0.0,
            "recall": m.recall,
            "precision": m.precision,
            "f1": m.f1_score,
            "fragile": n_frauds < FRAGILE_MIN_FRAUDS,
        }
    return out


def fmt_slice(s: dict) -> str:
    flag = "  FRAGILE" if s["fragile"] else ""
    return (f"{s['n']:>7}{s['n_frauds']:>8}{s['fraud_rate']:>8.2%}"
            f"{s['recall']:>8.3f}{s['precision']:>8.3f}{s['f1']:>7.3f}{flag}")


def run(batch_arg: str):
    batch_name, batch_path = resolve_batch(batch_arg)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model, version, algo = load_champion()
    ref, ref_sha = load_reference(REGISTERED_MODEL_NAME, version)

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
    drifted = [col for col, info in scores.items() if is_drifted(info)]
    alert_ref = float(ref[PREDICTION_COLUMN].mean())
    alert_cur = float(cur[PREDICTION_COLUMN].mean())
    perf_ref, perf_cur = slice_metrics(ref), slice_metrics(cur)
    fragile = [k for k, s in perf_cur.items() if s["fragile"]]

    mlflow.set_experiment(DRIFT_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"{batch_name}__v{version}"):
        mlflow.log_params({
            "batch": batch_name,
            "model_version": version,
            "model_algo": algo,
            "reference_sha256": ref_sha,
            "n_rows": len(cur),
        })
        # A. Disponible tout de suite
        mlflow.log_metric("now__drifted_columns_count", count)
        mlflow.log_metric("now__drifted_columns_share", share)
        for col, info in scores.items():
            mlflow.log_metric(f"now__drift_score__{col}", info["score"])
        mlflow.log_metric("now__alert_rate", alert_cur)
        mlflow.log_metric("now__alert_rate_reference", alert_ref)
        mlflow.set_tag("dataset_drift", str(dataset_drift))
        mlflow.set_tag("drifted_columns", ",".join(drifted) or "-")
        # B. Disponible à l'arrivée des labels
        for name, s in perf_cur.items():
            key = "" if name == "global" else "__" + name.replace("=", "_")
            for metric in ("fraud_rate", "recall", "precision", "f1", "n", "n_frauds"):
                mlflow.log_metric(f"labels__{metric}{key}", s[metric])
        mlflow.log_metric("labels__recall_reference", perf_ref["global"]["recall"])
        mlflow.log_metric("labels__recall_delta", perf_cur["global"]["recall"] - perf_ref["global"]["recall"])
        mlflow.set_tag("fragile_slices", ",".join(fragile) or "-")
        mlflow.log_artifact(html_path, artifact_path="evidently")

    print(f"\n=== {batch_name} vs référence du champion v{version} ({algo}), "
          f"référence sha256 {ref_sha[:12]} ===")

    print("\n[A] DISPONIBLE TOUT DE SUITE (sans labels)\n")
    print(f"{'variable':<24}{'méthode':<32}{'score':>9}{'seuil':>8}  drift")
    for col, info in scores.items():
        flag = "OUI" if is_drifted(info) else "-"
        print(f"{col:<24}{info['method'] or '':<32}{info['score']:>9.3f}{info['threshold']:>8.2f}  {flag}")
    print(f"\nVariables en drift : {count}/{len(scores)} ({share:.0%})"
          f" -> verdict global (seuil {DRIFT_SHARE:.0%} des variables) : {'OUI' if dataset_drift else 'non'}")
    if drifted and not dataset_drift:
        print(f"ATTENTION : verdict global « non », mais {len(drifted)} variable(s) en drift "
              f"({', '.join(drifted)}). Le verdict global ne suffit pas : analyser chaque variable.")
    print(f"Taux d'alertes du champion : {alert_ref:.2%} (réf) -> {alert_cur:.2%}")

    print("\n[B] DISPONIBLE À L'ARRIVÉE DES LABELS (un à trois mois plus tard)\n")
    print(f"{'tranche':<24}{'':<6}{'n':>7}{'fraudes':>8}{'taux':>8}{'recall':>8}{'précis.':>8}{'F1':>7}")
    for name in perf_cur:
        print(f"{name:<24}{'réf':<6}{fmt_slice(perf_ref[name])}")
        print(f"{'':<24}{'lot':<6}{fmt_slice(perf_cur[name])}")
    print(f"\nFRAGILE : moins de {FRAGILE_MIN_FRAUDS} fraudes dans la tranche, métriques peu fiables.")
    print(f"\nRapport HTML : {os.path.abspath(html_path)} (aussi dans MLflow, expérience '{DRIFT_EXPERIMENT_NAME}')")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage : python scripts/drift_report.py <prod_normal|prod_premium|prod_foreign_card|"
                         "prod_card_testing|prod_black_friday|chemin.csv>")
    run(sys.argv[1])
