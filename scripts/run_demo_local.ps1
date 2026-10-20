<#
Rejoue toute la démo de surveillance du drift, de la génération des données
jusqu'aux réentraînements.

Cible : le serveur MLflow désigné par MLFLOW_TRACKING_URI (par défaut le
serveur local http://127.0.0.1:5000, à démarrer avant). Pour viser le MLflow
de GCP, lancer d'abord `. .\gcp\demo-env.ps1` dans le même terminal.

Le registre doit être vierge pour retrouver les numéros de version de
gcp/DEMO.md (v1 et v2 créées par main.py, puis v3 à v6). Sur un registre déjà
rempli, le script fonctionne aussi : il lit les versions via les alias.

Usage (racine du projet, venv activé ou non) :
    .\scripts\run_demo_local.ps1
    .\scripts\run_demo_local.ps1 -SkipData   # garde les CSV déjà générés
#>
param([switch]$SkipData)

# "Continue" : Python écrit ses logs sur stderr ; les échecs sont détectés par code retour.
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore"
$env:MLFLOW_DISABLE_AGENT_HINT = "1"
Set-Location (Join-Path $PSScriptRoot "..")
$python = ".\venv\Scripts\python.exe"
$uri = if ($env:MLFLOW_TRACKING_URI) { $env:MLFLOW_TRACKING_URI } else { "http://127.0.0.1:5000" }

function Step([string]$title) { Write-Host "`n##### $title" -ForegroundColor Cyan }

function Py {
    & $python @args 2>&1 | ForEach-Object { "$_" } | Where-Object { $_ -notmatch "^\[ .* \] INFO|Downloading artifacts|it/s\]|RemoteException|is deprecated" }
    if ($LASTEXITCODE -ne 0) { throw "Échec : python $args" }
}

function AliasVersion([string]$alias) {
    $v = & $python scripts\get_alias.py $alias 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $v) { throw "Aucune version sous l'alias @$alias" }
    return "$v".Trim()
}

try { Invoke-WebRequest -UseBasicParsing -Uri "$uri/health" -TimeoutSec 30 | Out-Null }
catch { throw "MLflow injoignable sur $uri : démarrer le serveur (voir README) ou lancer . .\gcp\demo-env.ps1" }
Write-Host "MLflow cible : $uri" -ForegroundColor Green

if (-not $SkipData) {
    Step "Données : référence (0,5 % à l'étranger) et lots de production"
    Py scripts\generate_data.py
    Py scripts\generate_drift_data.py
}

Step "Épisode 1 : entraînement de v1 et v2, champion sur v1"
Py main.py
$v1 = AliasVersion "champion"
$v2 = [string]([int]$v1 + 1)

Step "Acte 1 : promotion de la v$v2 (forêt aléatoire)"
Py scripts\promote_model.py $v2

foreach ($batch in "prod_normal", "prod_premium", "prod_foreign_card", "prod_card_testing", "prod_black_friday") {
    Step "Rapport de drift : $batch (champion v$v2)"
    Py scripts\drift_report.py $batch
}

Step "Carte à l'étranger : réentraînement avec historique (data drift)"
Py scripts\retrain.py prod_foreign_card --with-history
$foreignHistory = AliasVersion "challenger"
Py scripts\compare_models.py prod_foreign_card_next

Step "Carte à l'étranger : variante sans historique"
Py scripts\retrain.py prod_foreign_card
Py scripts\compare_models.py prod_foreign_card_next

Step "Promotion du challenger avec historique (v$foreignHistory), puis rapport sur le mois suivant"
Py scripts\promote_model.py $foreignHistory
Py scripts\drift_report.py prod_foreign_card_next

Step "Card testing : réentraînement sur le lot récent seul (concept drift)"
Py scripts\retrain.py prod_card_testing
$cardTesting = AliasVersion "challenger"
Py scripts\compare_models.py prod_card_testing_next

Step "Card testing : le piège de l'historique"
Py scripts\retrain.py prod_card_testing --with-history
Py scripts\compare_models.py prod_card_testing_next

Step "Promotion du challenger card testing (v$cardTesting), puis rollback vers v$foreignHistory"
Py scripts\promote_model.py $cardTesting
Py scripts\promote_model.py $foreignHistory

Write-Host "`nDémo rejouée. Champion final : v$(AliasVersion 'champion'). Rapports HTML dans reports\." -ForegroundColor Green
