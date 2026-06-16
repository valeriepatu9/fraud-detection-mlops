"""Application de scoring de fraude — Crédit Agricole (démo).

L'app charge la version aliasée 'champion' depuis le Model Registry MLflow :
    models:/FraudDetector@champion

Promouvoir un modèle = déplacer cet alias. L'app reflète le changement
au prochain rafraîchissement, sans modifier une ligne de code.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import streamlit as st
from mlflow.tracking import MlflowClient

from frauddetection.constants import (
    CHAMPION_ALIAS,
    DECISION_THRESHOLD,
    FEATURE_COLUMNS,
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
)

st.set_page_config(
    page_title="CA Fraud Intelligence",
    page_icon="assets/favicon.png" if False else None,
    layout="centered",
)

# --- CSS minimaliste : couleurs CA, typographie sobre ---
st.markdown("""
<style>
    /* Header personnalisé */
    .ca-header {
        background-color: #006A4E;
        padding: 18px 24px;
        border-radius: 4px;
        margin-bottom: 24px;
    }
    .ca-header h1 {
        color: white;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: 0.03em;
    }
    .ca-header p {
        color: rgba(255,255,255,0.75);
        font-size: 0.78rem;
        margin: 4px 0 0 0;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    /* Badge modèle */
    .model-badge {
        display: inline-block;
        background: #f0f2f6;
        border: 1px solid #d0d3da;
        border-left: 3px solid #006A4E;
        padding: 6px 14px;
        border-radius: 2px;
        font-size: 0.82rem;
        color: #333;
        margin-bottom: 20px;
        font-family: monospace;
    }
    /* Verdict */
    .verdict-fraud {
        background: #FFF0F0;
        border: 1px solid #E53935;
        border-left: 4px solid #E53935;
        padding: 14px 18px;
        border-radius: 2px;
        color: #B71C1C;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .verdict-legit {
        background: #F0FFF4;
        border: 1px solid #2E7D32;
        border-left: 4px solid #2E7D32;
        padding: 14px 18px;
        border-radius: 2px;
        color: #1B5E20;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .verdict-warning {
        background: #FFFDE7;
        border: 1px solid #F9A825;
        border-left: 4px solid #F9A825;
        padding: 10px 18px;
        border-radius: 2px;
        color: #7A5000;
        font-size: 0.85rem;
        margin-top: 10px;
    }
    /* Score bar */
    .score-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-top: 12px;
    }
    .score-label { font-size: 0.78rem; color: #666; min-width: 90px; }
    .score-bar-bg {
        flex: 1; height: 6px; background: #e0e0e0; border-radius: 3px;
    }
    .score-bar-fill {
        height: 6px; border-radius: 3px;
    }
    .score-value { font-size: 0.82rem; font-weight: 600; min-width: 36px; text-align: right; }
    /* Table */
    table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    th { background: #f5f5f5; font-weight: 600; padding: 7px 12px; text-align: left;
         border-bottom: 2px solid #ddd; color: #444; }
    td { padding: 7px 12px; border-bottom: 1px solid #eee; color: #333; }
    /* Supprimer padding Streamlit autour du markdown */
    .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# --- Scénarios ---
SCENARIOS = {
    "normal": {
        "label": "Transaction standard",
        "features": [50.0, 14, 8.0, 0, 0, 3, 60.0, 0.83],
        "human": {
            "Référence": "TXN-20240614-00412",
            "Montant": "50,00 €",
            "Localisation": "Paris, France",
            "Heure": "14:32 CET",
            "Terminal": "iPhone 13 — enregistré",
            "Pays habituel": "Oui",
            "Transactions / 24h": "3",
        },
    },
    "fraude": {
        "label": "Transaction suspecte",
        "features": [4200.0, 3, 1400.0, 1, 1, 7, 150.0, 28.0],
        "human": {
            "Référence": "TXN-20240614-00413",
            "Montant": "4 200,00 €",
            "Localisation": "Varsovie, Pologne",
            "Heure": "02:17 CET",
            "Terminal": "Appareil non enregistré",
            "Pays habituel": "Non (habituel : France)",
            "Transactions / 24h": "7",
        },
    },
}


def load_champion():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}")
    client = MlflowClient()
    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    algo = mv.tags.get("algo", "?")
    return model, mv.version, algo


# --- Header ---
st.markdown("""
<div class="ca-header">
    <h1>CA Fraud Intelligence</h1>
    <p>Système de scoring de transactions en temps réel</p>
</div>
""", unsafe_allow_html=True)

# --- Chargement modèle ---
try:
    model, version, algo = load_champion()
    st.markdown(
        f'<div class="model-badge">Modèle actif &nbsp;|&nbsp; '
        f'<strong>{algo}</strong> &nbsp;·&nbsp; version {version} &nbsp;·&nbsp; '
        f'alias <code>@{CHAMPION_ALIAS}</code></div>',
        unsafe_allow_html=True,
    )
except Exception as e:
    st.error(f"Impossible de charger le modèle depuis le Registry MLflow.\n\nDétail : {e}")
    st.stop()

# --- Sélection scénario ---
st.markdown("**Sélectionner une transaction**")
col1, col2 = st.columns(2)
choice = None
with col1:
    if st.button("Transaction standard", use_container_width=True):
        choice = "normal"
with col2:
    if st.button("Transaction suspecte", use_container_width=True):
        choice = "fraude"

if "choice" not in st.session_state:
    st.session_state.choice = None
if choice is not None:
    st.session_state.choice = choice

# --- Résultat ---
if st.session_state.choice:
    sc = SCENARIOS[st.session_state.choice]
    st.divider()

    st.markdown("**Détail de la transaction**")
    rows = "".join(f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>"
                   for k, v in sc["human"].items())
    st.markdown(f"<table><thead><tr><th>Champ</th><th>Valeur</th></tr></thead>"
                f"<tbody>{rows}</tbody></table>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    X = pd.DataFrame([sc["features"]], columns=FEATURE_COLUMNS)
    proba = float(model.predict_proba(X)[0][1])
    is_fraud = proba >= DECISION_THRESHOLD
    pct = int(proba * 100)
    color = "#E53935" if is_fraud else "#2E7D32"

    # Score bar
    st.markdown(f"""
    <div class="score-row">
        <span class="score-label">Score de fraude</span>
        <div class="score-bar-bg">
            <div class="score-bar-fill" style="width:{pct}%; background:{color};"></div>
        </div>
        <span class="score-value" style="color:{color};">{proba:.2f}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if is_fraud:
        st.markdown(
            f'<div class="verdict-fraud">ALERTE — Transaction bloquée pour vérification '
            f'(score {proba:.2f} &gt; seuil {DECISION_THRESHOLD})</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="verdict-legit">Transaction autorisée '
            f'(score {proba:.2f} &lt; seuil {DECISION_THRESHOLD})</div>',
            unsafe_allow_html=True,
        )
        if st.session_state.choice == "fraude":
            st.markdown(
                '<div class="verdict-warning">Ce résultat est incorrect. '
                'Le modèle actuellement en production ne détecte pas cette transaction '
                'comme frauduleuse.</div>',
                unsafe_allow_html=True,
            )

# --- Sidebar ---
with st.sidebar:
    st.markdown("**Environnement**")
    st.markdown(f"- Registry : `{MLFLOW_TRACKING_URI}`")
    st.markdown(f"- Modèle : `{REGISTERED_MODEL_NAME}`")
    st.markdown(f"- Alias actif : `@{CHAMPION_ALIAS}`")
    st.markdown(f"- Seuil de décision : `{DECISION_THRESHOLD}`")
    st.divider()
    st.markdown("**Référence modèle**")
    st.code(f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}", language="text")
    st.divider()
    if st.button("Recharger le modèle"):
        st.rerun()