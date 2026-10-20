"""
Génère les lots de transactions "de production" pour la démo de surveillance du drift.

Chaque lot (15 000 transactions, seed fixe) est comparé à la référence du
champion par scripts/drift_report.py. Les règles de fraude ne changent que dans
les lots card testing.

  - prod_normal           : paramètres de la référence, autre seed
                            -> ni drift ni dégradation.
  - prod_premium          : clientèle plus aisée (30 % de "high rollers" au lieu de 10 %)
                            -> drift sans dégradation.
  - prod_foreign_card     : carte sans frais à l'étranger (15 % de paiements à
                            l'étranger au lieu de 0,5 %), zone peu apprise
                            -> drift AVEC dégradation.
  - prod_card_testing     : nouveau mode opératoire des fraudeurs (concept drift)
                            -> dégradation SANS drift.
  - prod_black_friday     : montants des achats légitimes x 1,6
                            -> drift, recall stable, précision en forte baisse.
  - *_next                : le mois suivant, mêmes conditions. Sert à évaluer
                            honnêtement un modèle réentraîné sur le lot précédent.

Usage :
    python scripts/generate_drift_data.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_data import TARGET_COLUMN, generate

N_BATCH = 15_000
BLACK_FRIDAY_FACTOR = 1.6

BATCHES = {
    "prod_normal": dict(random_state=7),
    "prod_premium": dict(random_state=8, p_high_roller=0.30),
    "prod_foreign_card": dict(random_state=11, p_foreign=0.15),
    "prod_foreign_card_next": dict(random_state=13, p_foreign=0.15),
    "prod_card_testing": dict(random_state=9, card_testing=True),
    "prod_card_testing_next": dict(random_state=10, card_testing=True),
    "prod_black_friday": dict(random_state=12),
}


def black_friday(df):
    """Gonfle les montants des seuls achats légitimes, APRÈS le calcul des labels.

    Le générateur déduit la fraude des variables (un montant > 3 x la moyenne
    fait partie de plusieurs règles). Gonfler les montants avant le calcul des
    labels transformerait des clients dépensiers en fraudeurs par construction ;
    or un achat de Black Friday reste légitime.
    """
    df = df.copy()
    legit = df[TARGET_COLUMN] == 0
    df.loc[legit, "amount"] = (df.loc[legit, "amount"] * BLACK_FRIDAY_FACTOR).round(2)
    df["amount_to_avg_ratio"] = (df["amount"] / df["avg_amount_last_30d"]).round(3)
    return df


if __name__ == "__main__":
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fraud_data"))
    print(f"{'lot':<24}{'lignes':>8}{'fraude':>9}{'nb fraudes':>12}{'étranger':>10}")
    for name, params in BATCHES.items():
        df = generate(n_samples=N_BATCH, **params)
        if name == "prod_black_friday":
            df = black_friday(df)
        path = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"{name:<24}{len(df):>8}{df[TARGET_COLUMN].mean():>9.2%}"
              f"{int(df[TARGET_COLUMN].sum()):>12}{df['is_foreign_country'].mean():>10.1%}")
    print(f"\nLots écrits dans {out_dir}")
