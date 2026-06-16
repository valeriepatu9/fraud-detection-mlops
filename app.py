import base64, os
import mlflow, mlflow.sklearn
import pandas as pd
import streamlit as st
from mlflow.tracking import MlflowClient
from frauddetection.constants import (
    CHAMPION_ALIAS, DECISION_THRESHOLD, FEATURE_COLUMNS,
    MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME,
)

st.set_page_config(page_title="Fraud Intelligence", layout="centered")

st.markdown("""
<style>
body, .stApp { background-color: #0d1117; color: #e6edf3; }
[data-testid="stSidebar"] { background-color: #090d13; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

SCENARIOS = {
    "normal": {
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


@st.cache_resource(show_spinner="Chargement du modèle depuis MLflow...")
def get_model():
    """Charge le modèle champion. Mis en CACHE : ne se recharge pas à chaque clic.
    Le bouton 'Recharger le modèle' vide ce cache pour refléter une promotion."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}")
    client = MlflowClient()
    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    return model, mv.version, mv.tags.get("algo", "?")


# --- Header ---
logo_path = "assets/nda_logo.png"
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f'<img src="data:image/png;base64,{b64}" '
        f'style="height:48px;filter:brightness(0) invert(1);margin-bottom:8px;">',
        unsafe_allow_html=True,
    )

st.markdown("## Fraud Intelligence")
st.caption("Crédit Agricole · Scoring de transactions en temps réel")
st.divider()

# --- Modèle ---
try:
    model, version, algo = get_model()
    st.info(f"Modèle actif : **{algo}** · version {version} · alias `@{CHAMPION_ALIAS}`")
except Exception as e:
    st.error(f"Impossible de charger le modèle : {e}")
    st.stop()

# --- Scénarios ---
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
    st.table(pd.DataFrame(sc["human"].items(), columns=["", "Valeur"]).set_index(""))

    X = pd.DataFrame([sc["features"]], columns=FEATURE_COLUMNS)
    proba = float(model.predict_proba(X)[0][1])
    is_fraud = proba >= DECISION_THRESHOLD

    st.markdown(f"**Score de risque : `{proba:.3f}`** (seuil `{DECISION_THRESHOLD}`)")
    st.progress(proba)

    if is_fraud:
        st.error(f"TRANSACTION BLOQUÉE — score {proba:.3f} supérieur au seuil")
    else:
        st.success(f"TRANSACTION AUTORISÉE — score {proba:.3f} inférieur au seuil")
        if st.session_state.choice == "fraude":
            st.warning("Le modèle en production ne détecte pas cette fraude.")

# --- Sidebar ---
with st.sidebar:
    st.markdown("**Environnement**")
    st.code(f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}", language="text")
    st.divider()
    if st.button("Recharger le modèle"):
        get_model.clear()   # vide le cache -> recharge le champion (après promotion)
        st.rerun()
