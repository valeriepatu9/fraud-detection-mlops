"""
Génère des lots de transactions "de production" pour la démo de data drift.

Trois lots, comparés ensuite au jeu d'entraînement par scripts/drift_report.py :

  - prod_ok            : même générateur, autre seed -> aucun drift attendu.
  - prod_data_drift    : la clientèle change (plus d'international, de nouveaux
                         appareils, de high rollers, de vélocité). Les règles de
                         fraude sont les mêmes -> Evidently détecte le drift.
  - prod_concept_drift : distributions IDENTIQUES, mais les fraudeurs adoptent un
                         nouveau mode opératoire (card testing). Evidently ne voit
                         rien sur les features... et pourtant le recall s'effondre.
  - prod_concept_drift_next : le mois suivant, même mode opératoire. Sert à
                         évaluer honnêtement le modèle réentraîné.

Usage :
    python scripts/generate_drift_data.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_data import TARGET_COLUMN, generate

N_BATCH = 15_000

BATCHES = {
    "prod_ok": dict(random_state=7),
    "prod_data_drift": dict(
        random_state=8,
        p_high_roller=0.25,
        p_international=0.35,
        p_new_device=0.25,
        p_foreign=0.25,
        tx_rate=4,
    ),
    "prod_concept_drift": dict(random_state=9, card_testing=True),
    # Le mois suivant, même mode opératoire : sert à évaluer le modèle réentraîné
    # sur des données qu'il n'a jamais vues
    "prod_concept_drift_next": dict(random_state=10, card_testing=True),
}


if __name__ == "__main__":
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fraud_data"))
    for name, params in BATCHES.items():
        df = generate(n_samples=N_BATCH, **params)
        path = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"{name:<20} {len(df)} lignes, taux de fraude = {df[TARGET_COLUMN].mean():.3%}  -> {path}")
