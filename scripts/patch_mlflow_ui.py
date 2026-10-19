"""Corrige le bug de l'UI MLflow 3.x lors de la modification d'un alias.

Symptôme : dans la page d'un modèle enregistré, enregistrer un alias affiche
« Cannot read properties of undefined (reading 'loadPage') ». L'alias est bien
sauvegardé côté serveur, mais la page ne se rafraîchit pas ; en réessayant on
envoie des ajouts / suppressions en cascade et l'alias peut disparaître.

Cause : la page passe `this.loadModelVersions` en callback sans le lier à
l'instance, donc `this` est undefined au moment de l'appel. Le patch remplace
ce callback par une fonction fléchée.

Le patch modifie le bundle JS installé avec le package mlflow : il est à
rejouer après chaque (ré)installation de mlflow. Il est idempotent.

Usage :
    python scripts/patch_mlflow_ui.py
"""

import pathlib
import sys

import mlflow

BUGGY = "onMetadataUpdated:this.loadModelVersions,"
FIXED = "onMetadataUpdated:()=>this.loadModelVersions(),"


def main() -> int:
    js_dir = pathlib.Path(mlflow.__file__).parent / "server" / "js" / "build" / "static" / "js"
    patched, already = [], []
    for path in js_dir.glob("*.js"):
        content = path.read_text(encoding="utf-8")
        if BUGGY in content:
            path.write_text(content.replace(BUGGY, FIXED), encoding="utf-8")
            patched.append(path.name)
        elif FIXED in content:
            already.append(path.name)

    if patched:
        print(f"OK — UI MLflow {mlflow.__version__} patchée : {', '.join(patched)}")
    elif already:
        print(f"OK — UI MLflow {mlflow.__version__} déjà patchée : {', '.join(already)}")
    else:
        # Le code de l'UI a changé (nouvelle version de mlflow) : vérifier si le
        # bug est corrigé en amont avant de supprimer ce script.
        print(f"ERREUR — motif introuvable dans l'UI MLflow {mlflow.__version__}.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
