# Démo du vendredi : marche à suivre

Toutes les commandes se lancent dans **PowerShell, à la racine du projet, venv activé** (`.\venv\Scripts\Activate.ps1`).

| | URL |
|---|---|
| App Streamlit | https://app-773988469584.europe-west9.run.app |
| MLflow (utilisateur `admin`) | https://mlflow-773988469584.europe-west9.run.app |

Mot de passe MLflow :
```powershell
gcloud secrets versions access latest --secret mlflow-admin-password --project nda-ss-tech-radar-mlops-drift
```

Les chiffres ci-dessous viennent de `scripts/run_demo_local.ps1`, rejoué sur un registre vierge (référence entraînée avec 0,5 % de paiements à l'étranger). Les numéros de version supposent un registre vierge : v1 et v2 par `main.py`, puis une version par réentraînement. Toujours promouvoir le numéro affiché par `retrain.py`.

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
4. Vérifier le point de départ : le badge de l'app affiche **DecisionTreeClassifier · v1 · @champion**. Sinon `python scripts/promote_model.py 1`, puis « ↻ Recharger le modèle » dans l'app.

## 2. Déroulé

### Acte 1 : promouvoir le bon modèle (épisode 1)
1. App, « Transaction suspecte » (4 200 €, Varsovie) : score 0,257 avec la v1, transaction autorisée.
2. MLflow : Models → FraudDetector → Version 2 → Aliases → `champion` (plan B : `python scripts/promote_model.py 2`).
3. App : « ↻ Recharger le modèle » → badge **RandomForestClassifier · v2**, score 0,687, transaction bloquée.

### Acte 2 : un drift qui ne dégrade rien
```powershell
python scripts/drift_report.py prod_normal    # témoin : 0 variable en drift, recall 0,944
python scripts/drift_report.py prod_premium   # 2 variables en drift (amount, avg_amount_last_30d), recall 0,935
```
Message : la clientèle s'enrichit, le drift sonne, le modèle tient. Le drift est une alerte à analyser, pas une panne.

### Acte 3 : un drift qui annonce une dégradation (le cas qui justifie la surveillance)
```powershell
python scripts/drift_report.py prod_foreign_card
```
- Bloc A, **disponible tout de suite** : une seule variable dérive, `is_foreign_country` (Jensen-Shannon 0,208). Le verdict global dit « non » (1 variable sur 8), mais le script affiche l'avertissement : c'est le cas le plus grave. Taux d'alertes 0,61 % → 0,78 %.
- Bloc B, **à l'arrivée des labels** : recall 0,908 → **0,507**, et **0,227 sur les paiements à l'étranger** (132 fraudes).
- App, « Rafale à l'étranger » (Lisbonne, 7 transactions en 24 h) : score 0,273 avec la v2, **transaction autorisée**. « Achat à l'étranger » (2 transactions) : 0,020, autorisé à raison.

### Acte 4 : réentraîner, comparer, promouvoir, revenir en arrière
```powershell
python scripts/retrain.py prod_foreign_card --with-history   # -> vN sous @challenger
python scripts/compare_models.py prod_foreign_card_next      # recall 0,487 (champion) -> 0,915 (challenger), étranger 0,233 -> 0,891
python scripts/promote_model.py N
```
App : « ↻ Recharger le modèle », puis « Rafale à l'étranger » : score **0,840**, transaction bloquée.

Rollback, si la question vient : `python scripts/promote_model.py 2` (le script affiche toujours la commande de retour à la version précédente).

Pourquoi l'historique : la carte change la clientèle, pas les règles de la fraude, donc les anciens labels restent valides. Sans historique, le recall global reste proche (0,905), mais le recall sur les paiements domestiques tombe de 0,957 à 0,857 : le modèle oublie la zone qu'il connaissait.

### Acte 5 : une dégradation que le drift ne voit pas (concept drift)
```powershell
python scripts/drift_report.py prod_card_testing    # 0 variable en drift, recall 0,312
python scripts/retrain.py prod_card_testing         # lot récent seul -> vN sous @challenger
python scripts/compare_models.py prod_card_testing_next   # recall 0,263 -> 0,867
```
Facultatif, le piège : `python scripts/retrain.py prod_card_testing --with-history` puis `compare_models.py prod_card_testing_next` → recall 0,316. Les anciens labels disent « légitime » pour les rafales de petits paiements : ils sont périmés.

### Complément : Black Friday, un drift qui dégrade la précision
```powershell
python scripts/drift_report.py prod_black_friday
```
2 variables en drift (`amount`, `amount_to_avg_ratio`), recall stable (0,934), mais **précision 0,978 → 0,366** : le modèle bloque des achats légitimes. Le signal immédiat est le **taux d'alertes, 0,61 % → 1,55 %**, visible sans attendre les labels.

## 3. Après la démo

```powershell
.\gcp\demo-mode.ps1 off
```
Accès coupé, plus aucune instance allumée. Le mode `on` coûte environ 1 €/jour.

Pour rejouer depuis le début : `python scripts/promote_model.py 1`. Les versions créées par les réentraînements restent dans le registre, ce qui n'empêche rien.

## 4. Rejouer toute la démo d'un coup

```powershell
.\scripts\run_demo_local.ps1            # contre le MLflow local (serveur démarré au préalable)
. .\gcp\demo-env.ps1; .\scripts\run_demo_local.ps1   # contre le MLflow GCP
```
Le script enchaîne génération des données, entraînement de v1 et v2, les cinq rapports, les quatre réentraînements avec leurs comparaisons, une promotion et un rollback. Durée : environ 11 minutes en local.

## En cas de problème

| Symptôme | Cause / remède |
|---|---|
| URL en 404 | Accès coupé → `.\gcp\demo-mode.ps1 on` |
| `CERTIFICATE_VERIFY_FAILED` dans un script | `. .\gcp\demo-env.ps1` oublié dans ce terminal |
| L'app affiche toujours l'ancien modèle | Bouton « ↻ Recharger le modèle » (le modèle est mis en cache) |
| Erreur en modifiant un alias dans l'UI | Recharger la page (Ctrl+F5), sinon passer par `promote_model.py` |
| `n'a pas de tag reference_sha256` | Version entraînée avant le versionnement de la référence : relancer `main.py` |
| Premier affichage très lent | Instances éteintes → `demo-mode.ps1 on` une heure avant |
