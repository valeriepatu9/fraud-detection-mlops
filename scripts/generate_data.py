"""
Génère un jeu de données synthétique de transactions bancaires pour la démo
de détection de fraude.

Principe pédagogique
--------------------
La fraude N'EST PAS séparable par un seuil simple. Des transactions légitimes
peuvent être grosses, nocturnes ou internationales (le dirigeant qui règle son
hôtel à Tokyo). Le label "fraude" est porté par des CONJONCTIONS de 3 indices
(ex. nouvel appareil ET étranger ET montant inhabituel), plus un peu de bruit.

Conséquences :
  - DecisionTree(max_depth=2) sans rééquilibrage : ne peut faire que ~2 coupures,
    ne peut pas isoler une conjonction de 3 conditions, et sur une fraude rare
    (~1%) il s'effondre en prédisant "légitime" partout -> recall ~ 0, et il
    classe MÊME la fraude évidente comme légitime.
  - RandomForest(class_weight="balanced") : capte les conjonctions, rééquilibre
    la classe rare -> bon recall, et flague la fraude évidente.

Les distributions ont des queues réalistes (distance bimodale domestique /
international, soldes "high-roller") pour que le scénario "4200€ depuis
Varsovie" soit rare mais PRÉSENT dans les données — sinon le modèle
extrapolerait dans le vide.
"""

import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_SAMPLES = 60_000
LABEL_NOISE = 0.04  # fraction de fraudes "ratées" (fraud -> legit) pour un recall < 1

FEATURE_COLUMNS = [
    "amount",
    "hour",
    "distance_from_home_km",
    "is_new_device",
    "is_foreign_country",
    "transactions_last_24h",
    "avg_amount_last_30d",
    "amount_to_avg_ratio",
]
TARGET_COLUMN = "is_fraud"


def generate(n_samples=N_SAMPLES, label_noise=LABEL_NOISE, random_state=RANDOM_STATE,
             p_high_roller=0.10, p_international=0.18, p_new_device=0.12,
             p_foreign=0.005, tx_rate=3, card_testing=False):
    """Les paramètres p_* / tx_rate pilotent les distributions (=> data drift) ;
    card_testing ajoute une règle de fraude inédite (=> concept drift).
    Les valeurs par défaut reproduisent exactement le jeu d'entraînement.

    p_foreign=0.005 : à l'entraînement, les paiements à l'étranger sont rares.
    C'est la "zone peu apprise" que le lancement d'une carte sans frais à
    l'étranger (lot prod_foreign_card, 15 %) viendra solliciter."""
    rng = np.random.default_rng(random_state)

    # Solde moyen : majorité de clients standards + une minorité de "high rollers"
    is_high_roller = rng.random(n_samples) < p_high_roller
    avg_amount = np.where(
        is_high_roller,
        np.abs(rng.normal(200, 80, n_samples)) + 50,
        np.abs(rng.normal(70, 30, n_samples)) + 10,
    )

    # Montant : ratio lognormal autour de la moyenne du client
    ratio = np.exp(rng.normal(0.0, 0.8, n_samples))
    amount = np.clip(avg_amount * ratio, 1, None)

    hour = rng.integers(0, 24, n_samples)
    is_night = (hour < 6) | (hour >= 23)

    # Distance bimodale : domestique (proche) vs international (loin)
    is_international = rng.random(n_samples) < p_international
    distance = np.where(
        is_international,
        np.abs(rng.normal(700, 500, n_samples)) + 100,
        np.abs(rng.normal(15, 35, n_samples)),
    )

    is_new_device = rng.random(n_samples) < p_new_device
    is_foreign = rng.random(n_samples) < p_foreign
    tx_24h = rng.poisson(tx_rate, n_samples)

    big = amount > 3 * avg_amount
    far = distance > 150
    veloc = tx_24h >= 6

    # Signal = union de conjonctions à 3 conditions (DNF) : hors de portée d'un arbre profondeur 2
    fraud = (
        (is_new_device & is_foreign & big)
        | (is_night & big & far)
        | (is_foreign & far & veloc)
        | (is_new_device & big & is_night)
        | (is_new_device & is_foreign & far)
        | (is_night & is_foreign & veloc)
    )
    if card_testing:
        # Nouveau mode opératoire : rafales de petits paiements pour tester une
        # carte volée. Jamais vu à l'entraînement : le modèle ne peut pas le capter.
        small = amount < 0.5 * avg_amount
        fraud = fraud | (veloc & small)
    y = np.asarray(fraud).astype(int)

    # Bruit de label : on "rate" une fraction des fraudes pour éviter un signal parfait
    fraud_idx = np.where(y == 1)[0]
    missed = rng.choice(fraud_idx, size=int(label_noise * len(fraud_idx)), replace=False)
    y[missed] = 0

    df = pd.DataFrame({
        "amount": amount,
        "hour": hour,
        "distance_from_home_km": distance,
        "is_new_device": is_new_device.astype(int),
        "is_foreign_country": is_foreign.astype(int),
        "transactions_last_24h": tx_24h,
        "avg_amount_last_30d": avg_amount,
        "amount_to_avg_ratio": amount / avg_amount,
        TARGET_COLUMN: y,
    })

    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    df = df.round({"amount": 2, "distance_from_home_km": 1,
                   "avg_amount_last_30d": 2, "amount_to_avg_ratio": 3})
    return df


if __name__ == "__main__":
    import os
    out = os.path.join(os.path.dirname(__file__), "..", "fraud_data", "transactions.csv")
    out = os.path.abspath(out)
    df = generate()
    df.to_csv(out, index=False)
    print(f"Dataset généré : {out}")
    print(f"  {len(df)} lignes, taux de fraude = {df['is_fraud'].mean():.3%}")
