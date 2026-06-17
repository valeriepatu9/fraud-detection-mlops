"""Application de scoring de fraude — NDA Partners (démo MLflow).

L'UI est rendue en HTML/CSS custom (via components.html) pour un contrôle total
du design. Mais les SCORES viennent du VRAI modèle chargé depuis le registry :
Python calcule predict_proba pour chaque scénario, puis injecte les valeurs
dans la page. Aucune valeur n'est codée en dur.

Le bouton 'Recharger le modèle' (natif Streamlit) vide le cache et recharge le
champion depuis MLflow — c'est le geste à faire après une promotion.
"""

import base64
import json
import os

import mlflow
import mlflow.sklearn
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from mlflow.tracking import MlflowClient

from frauddetection.constants import (
    CHAMPION_ALIAS,
    DECISION_THRESHOLD,
    FEATURE_COLUMNS,
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
)

st.set_page_config(page_title="Fraud Intelligence", layout="wide")

# Masque le chrome Streamlit et le padding ; fond sombre pour l'enrobage
st.markdown("""
<style>
  #MainMenu, footer, header {visibility: hidden;}
  .block-container {padding: 0 !important; max-width: 100% !important;}
  .stApp {background: #0d1117;}
  div[data-testid="stDecoration"] {display: none;}
  /* Bouton recharger */
  div.stButton > button {
    background: #161b22; color: #c9d1d9; border: 1px solid #30363d;
    border-radius: 6px; font-size: 13px; padding: 8px 16px;
  }
  div.stButton > button:hover { background: #1c2128; border-color: #8b949e; color: #f0f6fc; }
</style>
""", unsafe_allow_html=True)

SCENARIOS = {
    "normal": {
        "features": [50.0, 14, 8.0, 0, 0, 3, 60.0, 0.83],
        "data": {
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
        "data": {
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
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}")
    client = MlflowClient()
    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    return model, mv.version, mv.tags.get("algo", "?")


def load_logo():
    for path in ("assets/nda_logo.png", "assets/nda_logo.jpg"):
        if os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    return None


try:
    model, version, algo = get_model()
except Exception as e:
    st.error(f"Impossible de charger le modèle depuis le registry MLflow : {e}")
    st.stop()

# --- Calcul des VRAIS scores du modèle pour chaque scénario ---
payload = {}
for key, sc in SCENARIOS.items():
    X = pd.DataFrame([sc["features"]], columns=FEATURE_COLUMNS)
    score = float(model.predict_proba(X)[0][1])
    payload[key] = {"score": score, "data": sc["data"]}

logo_b64 = load_logo()
if logo_b64:
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="height:28px;filter:brightness(0) invert(1);">'
else:
    logo_html = '<span style="font-size:15px;font-weight:700;color:#0d1117;letter-spacing:0.08em;">NDA</span>'

is_rf = "RandomForest" in algo
badge_color = "#3fb950" if is_rf else "#388bfd"

TEMPLATE = r"""
<!DOCTYPE html><html lang="fr"><head><meta charset="UTF-8">
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background:#0d1117; color:#e6edf3; }
  .layout { display:flex; min-height:760px; }
  .sidebar { width:250px; background:#010409; border-right:1px solid #21262d; padding:24px 16px; display:flex; flex-direction:column; gap:22px; flex-shrink:0; }
  .sidebar-logo { display:flex; align-items:center; gap:10px; padding-bottom:18px; border-bottom:1px solid #21262d; }
  .logo-mark { width:32px; height:32px; background:#fff; border-radius:6px; display:flex; align-items:center; justify-content:center; }
  .logo-text { font-size:13px; font-weight:600; color:#f0f6fc; letter-spacing:0.04em; }
  .logo-sub { font-size:11px; color:#8b949e; margin-top:1px; }
  .s-label { font-size:11px; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px; }
  .s-item { margin-bottom:10px; }
  .s-key { font-size:11px; color:#8b949e; display:block; margin-bottom:3px; }
  .s-val { font-size:12px; color:#79c0ff; font-family:'SF Mono',monospace; background:#161b22; border:1px solid #30363d; border-radius:4px; padding:4px 8px; display:block; word-break:break-all; }
  .main { flex:1; padding:32px 40px; }
  .title { font-size:22px; font-weight:600; color:#f0f6fc; }
  .sub { font-size:13px; color:#8b949e; margin-top:4px; margin-bottom:26px; }
  .badge { display:inline-flex; align-items:center; background:#161b22; border:1px solid #30363d; border-left:3px solid __BADGE__; border-radius:0 6px 6px 0; padding:8px 14px; font-size:12px; font-family:'SF Mono',monospace; color:#8b949e; margin-bottom:26px; }
  .badge strong { color:#f0f6fc; } .badge .alias { color:#79c0ff; }
  .sc-label { font-size:11px; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:12px; }
  .sc-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:26px; }
  .sc-btn { padding:12px 16px; background:#161b22; border:1px solid #30363d; border-radius:6px; cursor:pointer; display:flex; align-items:center; gap:10px; transition:all .15s; text-align:left; }
  .sc-btn:hover { background:#1c2128; border-color:#8b949e; }
  .sc-btn.on-n { border-color:#3fb950; } .sc-btn.on-f { border-color:#f85149; }
  .dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
  .dot-g { background:#3fb950; } .dot-r { background:#f85149; }
  .sc-t { font-size:13px; font-weight:500; color:#c9d1d9; }
  .sc-d { font-size:11px; color:#8b949e; margin-top:2px; }
  .card { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px 20px; margin-bottom:16px; }
  .card-label { font-size:11px; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:14px; }
  table { width:100%; border-collapse:collapse; }
  tr { border-bottom:1px solid #21262d; } tr:last-child { border-bottom:none; }
  td { padding:8px 0; font-size:13px; }
  td:first-child { color:#8b949e; width:48%; }
  td:last-child { color:#f0f6fc; text-align:right; font-weight:500; }
  .sc-head { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:10px; }
  .sc-num { font-size:28px; font-weight:600; }
  .num-d { color:#f85149; } .num-s { color:#3fb950; }
  .bar-bg { height:6px; background:#21262d; border-radius:3px; overflow:hidden; margin-bottom:6px; }
  .bar-fill { height:100%; border-radius:3px; }
  .fill-d { background:#f85149; } .fill-s { background:#3fb950; }
  .ticks { display:flex; justify-content:space-between; font-size:11px; color:#8b949e; }
  .verdict { padding:14px 18px; font-size:13px; font-weight:600; border-radius:0 6px 6px 0; margin-bottom:10px; }
  .v-d { background:rgba(248,81,73,.1); border:1px solid rgba(248,81,73,.3); border-left:3px solid #f85149; color:#ffa198; }
  .v-s { background:rgba(63,185,80,.1); border:1px solid rgba(63,185,80,.3); border-left:3px solid #3fb950; color:#56d364; }
  .warn { background:rgba(210,153,34,.1); border:1px solid rgba(210,153,34,.25); border-radius:6px; padding:10px 14px; font-size:12px; color:#d29922; }
  .hidden { display:none; }
</style></head><body>
<div class="layout">
  <aside class="sidebar">
    <div class="sidebar-logo">
      <div class="logo-mark">__LOGO__</div>
      <div><div class="logo-text">NDA Partners</div><div class="logo-sub">Fraud Intelligence</div></div>
    </div>
    <div>
      <div class="s-label">Environnement</div>
      <div class="s-item"><span class="s-key">Registry</span><span class="s-val">MLflow Model Registry</span></div>
      <div class="s-item"><span class="s-key">Modèle</span><span class="s-val">__MODEL__</span></div>
      <div class="s-item"><span class="s-key">Alias actif</span><span class="s-val">@__ALIAS__</span></div>
      <div class="s-item"><span class="s-key">Seuil de décision</span><span class="s-val">__THRESH__</span></div>
    </div>
  </aside>
  <main class="main">
    <div class="title">Fraud Intelligence</div>
    <div class="sub">Crédit Agricole · Scoring de transactions en temps réel</div>
    <div class="badge">Modèle actif &nbsp;/&nbsp; <strong>__ALGO__</strong> &nbsp;·&nbsp; v__VERSION__ &nbsp;·&nbsp; <span class="alias">@__ALIAS__</span></div>
    <div class="sc-label">Sélectionner une transaction</div>
    <div class="sc-grid">
      <div class="sc-btn" id="b-n" onclick="pick('normal')"><span class="dot dot-g"></span><div><div class="sc-t">Transaction standard</div><div class="sc-d">50€ · Paris · 14h · appareil connu</div></div></div>
      <div class="sc-btn" id="b-f" onclick="pick('fraude')"><span class="dot dot-r"></span><div><div class="sc-t">Transaction suspecte</div><div class="sc-d">4 200€ · Varsovie · 02h · inconnu</div></div></div>
    </div>
    <div id="result" class="hidden">
      <div class="card"><div class="card-label">Détail de la transaction</div><table id="tx"></table></div>
      <div class="card">
        <div class="card-label">Score de risque</div>
        <div class="sc-head"><div></div><span class="sc-num" id="num">0.000</span></div>
        <div class="bar-bg"><div class="bar-fill" id="bar" style="width:0%"></div></div>
        <div class="ticks"><span>0.0</span><span>seuil __THRESH__</span><span>1.0</span></div>
      </div>
      <div class="verdict v-s hidden" id="v-legit">TRANSACTION AUTORISÉE</div>
      <div class="verdict v-d hidden" id="v-fraud">TRANSACTION BLOQUÉE — vérification requise</div>
      <div class="warn hidden" id="v-warn">Le modèle en production ne détecte pas cette transaction comme frauduleuse.</div>
    </div>
  </main>
</div>
<script>
const DATA = __DATA__;
const THRESH = __THRESH__;
function pick(k){
  document.getElementById('b-n').classList.remove('on-n');
  document.getElementById('b-f').classList.remove('on-f');
  document.getElementById(k==='normal'?'b-n':'b-f').classList.add(k==='normal'?'on-n':'on-f');
  const s = DATA[k]; const score = s.score; const fraud = score >= THRESH;
  document.getElementById('tx').innerHTML = Object.entries(s.data).map(([a,b])=>`<tr><td>${a}</td><td>${b}</td></tr>`).join('');
  const num = document.getElementById('num');
  num.textContent = score.toFixed(3); num.className = 'sc-num ' + (fraud?'num-d':'num-s');
  const bar = document.getElementById('bar');
  bar.style.width = (score*100).toFixed(1)+'%'; bar.className = 'bar-fill ' + (fraud?'fill-d':'fill-s');
  document.getElementById('v-legit').className = 'verdict v-s' + (fraud?' hidden':'');
  document.getElementById('v-fraud').className = 'verdict v-d' + (fraud?'':' hidden');
  document.getElementById('v-warn').className = 'warn' + ((k==='fraude'&&!fraud)?'':' hidden');
  document.getElementById('result').classList.remove('hidden');
}
</script></body></html>
"""

html = (TEMPLATE
        .replace("__LOGO__", logo_html)
        .replace("__BADGE__", badge_color)
        .replace("__ALGO__", algo)
        .replace("__VERSION__", str(version))
        .replace("__ALIAS__", CHAMPION_ALIAS)
        .replace("__MODEL__", REGISTERED_MODEL_NAME)
        .replace("__THRESH__", str(DECISION_THRESHOLD))
        .replace("__DATA__", json.dumps(payload)))

components.html(html, height=800, scrolling=True)

# --- Contrôle natif : recharge le champion depuis MLflow (après une promotion) ---
c1, c2, c3 = st.columns([2, 1, 2])
with c2:
    if st.button("↻ Recharger le modèle", use_container_width=True):
        get_model.clear()
        st.rerun()