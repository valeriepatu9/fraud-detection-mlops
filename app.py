"""Application de scoring de fraude — Crédit Agricole (démo).

L'app ne pointe PAS vers un fichier de modèle : elle charge la version aliasée
'champion' depuis le Model Registry MLflow :

    models:/FraudDetector@champion

Conséquence : changer le modèle en production = déplacer cet alias dans le
Registry (voir scripts/promote_model.py). Au rafraîchissement de la page,
l'app sert automatiquement le nouveau modèle. C'est tout l'intérêt de la démo.

Lancement :
    streamlit run app.py
"""

import os

import mlflow
import mlflow.sklearn
import pandas as pd
import streamlit as st
from mlflow.tracking import MlflowClient

from frauddetection.constants import (
    FEATURE_COLUMNS,
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
    CHAMPION_ALIAS,
    DECISION_THRESHOLD,
)

st.set_page_config(page_title="Détection de fraude — CA", page_icon="🛡️", layout="centered")

# --- Scénarios pré-remplis (valeurs validées pour la démo) ---
# Ordre des features : amount, hour, distance_from_home_km, is_new_device,
# is_foreign_country, transactions_last_24h, avg_amount_last_30d, amount_to_avg_ratio
SCENARIOS = {
    "normal": {
        "label": "Transaction normale",
        "desc": "Achat en ligne · 50€ · Paris · 14h · appareil connu",
        "features": [50.0, 14, 8.0, 0, 0, 3, 60.0, 0.83],
        "human": {
            "Montant": "50,00 €",
            "Localisation": "Paris, France",
            "Heure": "14h32",
            "Appareil": "iPhone 13 (connu)",
            "Pays habituel ?": "Oui",
        },
    },
    "fraude": {
        "label": "Fraude évidente",
        "desc": "Virement · 4 200€ · Varsovie · 2h du matin · appareil inconnu",
        "features": [4200.0, 3, 1400.0, 1, 1, 7, 150.0, 28.0],
        "human": {
            "Montant": "4 200,00 €",
            "Localisation": "Varsovie, Pologne",
            "Heure": "02h17",
            "Appareil": "Appareil inconnu",
            "Pays habituel ?": "Non — habituel : France",
        },
    },
}


def load_champion():
    """Charge le modèle 'champion' depuis le Registry + ses métadonnées (version, algo)."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}")
    client = MlflowClient()
    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    algo = mv.tags.get("algo", "?")
    return model, mv.version, algo


st.title("🛡️ Détection de fraude - Cédric Agricole")
st.caption("Analyse une transaction en temps réel · démo MLflow")

# Chargement du modèle (rechargé à chaque exécution -> reflète la promotion en live)
try:
    model, version, algo = load_champion()
    st.info(f"Modèle servi en production : **v{version} · {algo}** "
            f"(alias `@{CHAMPION_ALIAS}`)")
except Exception as e:
    st.error(
        "Impossible de charger le modèle depuis le Registry MLflow.\n\n"
        f"Vérifie que le serveur tourne sur `{MLFLOW_TRACKING_URI}` et que "
        "`python main.py` a bien été exécuté.\n\n"
        f"Détail : {e}"
    )
    st.stop()

st.subheader("Choisir un scénario")
col1, col2 = st.columns(2)
choice = None
with col1:
    if st.button(f"🟢 {SCENARIOS['normal']['label']}", use_container_width=True):
        choice = "normal"
    st.caption(SCENARIOS["normal"]["desc"])
with col2:
    if st.button(f"🔴 {SCENARIOS['fraude']['label']}", use_container_width=True):
        choice = "fraude"
    st.caption(SCENARIOS["fraude"]["desc"])

if "choice" not in st.session_state:
    st.session_state.choice = None
if choice is not None:
    st.session_state.choice = choice

if st.session_state.choice:
    sc = SCENARIOS[st.session_state.choice]
    st.divider()
    st.markdown("**Détail de la transaction**")
    st.table(pd.DataFrame(sc["human"].items(), columns=["Champ", "Valeur"]).set_index("Champ"))

    X = pd.DataFrame([sc["features"]], columns=FEATURE_COLUMNS)
    proba = float(model.predict_proba(X)[0][1])
    is_fraud = proba >= DECISION_THRESHOLD

    if is_fraud:
        st.error(f"🚨 **Alerte fraude** — score de fraude : {proba:.2f} "
                 f"(seuil {DECISION_THRESHOLD:.1f}) → transaction bloquée pour vérification")
    else:
        st.success(f"✅ **Transaction légitime** — score de fraude : {proba:.2f} "
                   f"(seuil {DECISION_THRESHOLD:.1f})")
        if st.session_state.choice == "fraude":
            st.warning("Quelque chose vous semble normal ? 🤔")

with st.sidebar:
    st.header("À propos")
    st.write(
        "Cette application charge le modèle aliasé `champion` depuis le "
        "**MLflow Model Registry**. Promouvoir un nouveau modèle ne touche "
        "pas au code de l'app : il suffit de déplacer l'alias."
    )
    st.code("models:/FraudDetector@champion", language="text")
    if st.button("🔄 Recharger le modèle"):
        st.rerun()
