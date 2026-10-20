"""Référence du drift versionnée avec le modèle.

Chaque version du modèle embarque SA référence : le jeu de test hold-out sur
lequel elle a été évaluée. Elle est loggée dans le run d'entraînement (dataset
MLflow + artifact CSV) et son empreinte SHA-256 est posée en tag sur la version
du modèle. Le rapport de drift compare donc toujours un lot à la référence du
champion en cours, et non à un fichier local implicite.
"""

import hashlib
import os
import shutil
import tempfile

import mlflow
import mlflow.data
import pandas as pd
from mlflow.tracking import MlflowClient

from frauddetection.constants import (
    REFERENCE_ARTIFACT_DIR,
    REFERENCE_FILE,
    REFERENCE_TAG,
    TARGET_COLUMN,
)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log_reference(test_path: str) -> str:
    """À appeler dans le run actif : logge la référence, renvoie son SHA-256."""
    digest = sha256_file(test_path)
    df = pd.read_csv(test_path)
    dataset = mlflow.data.from_pandas(df, source=os.path.abspath(test_path),
                                      name="reference_test", targets=TARGET_COLUMN)
    mlflow.log_input(dataset, context="reference")
    with tempfile.TemporaryDirectory() as tmp:
        # Nom de fichier fixe dans l'artifact, quel que soit le chemin local
        tmp_path = os.path.join(tmp, REFERENCE_FILE)
        shutil.copyfile(test_path, tmp_path)
        mlflow.log_artifact(tmp_path, artifact_path=REFERENCE_ARTIFACT_DIR)
    mlflow.set_tag(REFERENCE_TAG, digest)
    return digest


def load_reference(model_name: str, version: str) -> tuple[pd.DataFrame, str]:
    """Télécharge la référence d'une version du modèle et vérifie son empreinte."""
    client = MlflowClient()
    mv = client.get_model_version(model_name, version)
    expected = mv.tags.get(REFERENCE_TAG)
    if not expected:
        raise RuntimeError(
            f"La version {version} de {model_name} n'a pas de tag {REFERENCE_TAG} : "
            "elle a été entraînée avant le versionnement de la référence (relancer main.py)."
        )
    with tempfile.TemporaryDirectory() as tmp:
        path = mlflow.artifacts.download_artifacts(
            run_id=mv.run_id, artifact_path=f"{REFERENCE_ARTIFACT_DIR}/{REFERENCE_FILE}", dst_path=tmp
        )
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Référence corrompue : SHA-256 {actual} au lieu de {expected}.")
        return pd.read_csv(path), expected
