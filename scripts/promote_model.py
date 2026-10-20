"""Promotion d'un modèle : déplace l'alias 'champion' vers une version donnée.

C'est l'opération que tu exécutes EN LIVE pendant la démo pour faire passer le
bon modèle (RandomForest, v2) en production. L'application qui charge
`models:/FraudDetector@champion` servira alors automatiquement cette version
au prochain rafraîchissement.

Rollback : la promotion ne supprime rien. Revenir en arrière consiste à
redéplacer l'alias vers l'ancienne version, que le script affiche à chaque
promotion :
    python scripts/promote_model.py <ancienne version>

Usage :
    python scripts/promote_model.py 2        # promeut la version 2
    python scripts/promote_model.py          # promeut la dernière version
"""

import sys
import os

# Permet d'exécuter le script directement depuis la racine du projet
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
from mlflow.tracking import MlflowClient

from frauddetection.constants import (
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
    CHAMPION_ALIAS,
)


def promote(version: str | None = None):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        raise SystemExit(f"Aucune version trouvée pour le modèle '{REGISTERED_MODEL_NAME}'.")

    if version is None:
        version = str(max(int(v.version) for v in versions))

    target = next((v for v in versions if str(v.version) == str(version)), None)
    if target is None:
        raise SystemExit(f"Version {version} introuvable.")

    try:
        previous = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS).version
    except Exception:
        previous = None

    algo = target.tags.get("algo", "?")
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version)
    print(f"OK — alias '{CHAMPION_ALIAS}' déplacé vers {REGISTERED_MODEL_NAME} v{version} ({algo}).")
    print("Rafraîchis l'application : elle sert désormais ce modèle.")
    if previous and previous != str(version):
        print(f"Rollback si besoin : python scripts/promote_model.py {previous}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    promote(arg)
