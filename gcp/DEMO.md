# Démo du vendredi : marche à suivre (GCP)

Toutes les commandes se lancent dans **PowerShell, à la racine du projet, venv activé** (`.\venv\Scripts\Activate.ps1`).

| | URL |
|---|---|
| App Streamlit | https://app-773988469584.europe-west9.run.app |
| MLflow (utilisateur `admin`) | https://mlflow-773988469584.europe-west9.run.app |

Mot de passe MLflow :
```powershell
gcloud secrets versions access latest --secret mlflow-admin-password --project nda-ss-tech-radar-mlops-drift
```

## 1. Une heure avant

1. Ouvrir l'accès et allumer les instances (environ 5 min) :
   ```powershell
   .\gcp\demo-mode.ps1 on
   ```
2. Dans le terminal qui servira pendant la démo, pointer les scripts vers le MLflow GCP (le point au début compte) :
   ```powershell
   . .\gcp\demo-env.ps1
   ```
   La ligne `MLflow cible : https://mlflow-...run.app` doit s'afficher. Sans cette étape, les scripts écrivent dans le MLflow local.
3. Ouvrir trois onglets : l'app, MLflow (se connecter en `admin`), et le dossier `reports\`.
4. Vérifier le point de départ : le badge de l'app affiche **Modèle actif / DecisionTreeClassifier · v1 · @champion**. Sinon :
   ```powershell
   python scripts/promote_model.py 1
   ```
   puis cliquer sur « ↻ Recharger le modèle » dans l'app.

## 2. Déroulé

### Acte 1 : promouvoir le bon modèle
1. App : la v1 laisse passer la fraude évidente.
2. MLflow : Models → FraudDetector → Version 2 → Aliases → ajouter `champion` → Save.
   Plan B : `python scripts/promote_model.py 2`.
3. App : « ↻ Recharger le modèle » → badge **RandomForestClassifier · v2**, la fraude est détectée.

### Acte 2 : data drift (la clientèle change)
```powershell
python scripts/drift_report.py prod_ok           # 0/8 variables en drift, recall stable
python scripts/drift_report.py prod_data_drift   # 6/8 en drift -> drift global OUI, recall stable
```
Montrer `reports\drift_prod_data_drift.html`, puis dans MLflow l'expérience `fraud_drift_monitoring`.

### Acte 3 : concept drift (les fraudeurs changent de méthode)
```powershell
python scripts/drift_report.py prod_concept_drift   # 0/8 en drift... recall 0,92 -> 0,44
```
Message : le monitoring des variables ne voit rien, seul le suivi de la performance alerte.

### Acte 4 : réentraîner
```powershell
python scripts/retrain.py prod_concept_drift        # affiche "Nouvelle version : FraudDetector vN"
python scripts/promote_model.py N                   # N = le numéro affiché
python scripts/drift_report.py prod_concept_drift_next   # recall remonte vers 0,89
```
Le réentraînement prend une à deux minutes : le lancer, puis expliquer pendant qu'il tourne.

Facultatif, le piège : `python scripts/retrain.py prod_concept_drift --with-history`, promouvoir cette version, relancer le rapport → recall vers 0,50. Puis remettre la bonne version en champion.

Si on demande pourquoi la colonne « réf » de la version réentraînée affiche un recall et une précision bas : la référence est l'ancien jeu de test, où les rafales de petits paiements étaient étiquetées légitimes. C'est le mois suivant qui fait foi.

## 3. Après la démo

```powershell
.\gcp\demo-mode.ps1 off
```
Accès coupé, plus aucune instance allumée. Le mode `on` coûte environ 1 €/jour.

Pour rejouer la démo : `python scripts/promote_model.py 1`. Les versions créées par les réentraînements restent dans le registre, ce qui n'empêche rien.

## En cas de problème

| Symptôme | Cause / remède |
|---|---|
| URL en 404 | Accès coupé → `.\gcp\demo-mode.ps1 on` |
| `CERTIFICATE_VERIFY_FAILED` dans un script | `. .\gcp\demo-env.ps1` oublié dans ce terminal |
| L'app affiche toujours l'ancien modèle | Bouton « ↻ Recharger le modèle » (le modèle est mis en cache) |
| Erreur en modifiant un alias dans l'UI | Recharger la page (Ctrl+F5), sinon passer par `promote_model.py` |
| Premier affichage très lent | Instances éteintes → `demo-mode.ps1 on` une heure avant |
