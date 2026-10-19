<#
Déploie la démo sur GCP : serveur MLflow + app Streamlit sur Cloud Run.

Architecture
  - mlflow : Cloud Run (1 instance max), authentification basique (admin),
             mlflow.db + artefacts sur un bucket GCS monté en /mlflow.
  - app    : Cloud Run (1 instance max), charge models:/FraudDetector@champion.
  - images : build local puis push vers Artifact Registry.
  - secrets: mot de passe admin MLflow + clé CSRF dans Secret Manager.

Prérequis (déjà faits une fois) : dépôt Artifact Registry "fraud", bucket
<projet>-mlflow, compte de service app-sa avec accès au bucket.

Usage (depuis la racine du projet) :
    .\gcp\deploy.ps1                 # tout (build, push, mlflow, app)
    .\gcp\deploy.ps1 -Target mlflow  # uniquement le serveur MLflow
    .\gcp\deploy.ps1 -Target app     # uniquement l'app Streamlit
#>
param(
    [string]$Project = "nda-ss-tech-radar-mlops-drift",
    [string]$Region = "europe-west9",
    [ValidateSet("all", "mlflow", "app")]
    [string]$Target = "all"
)

# "Continue" : gcloud écrit sa progression sur stderr, ce que PowerShell 5.1 prendrait
# pour une erreur fatale avec "Stop". Les échecs sont détectés par Invoke-Checked.
$ErrorActionPreference = "Continue"
$env:Path = "$env:LOCALAPPDATA\Google\Cloud SDK\google-cloud-sdk\bin;$env:Path"

$Registry = "$Region-docker.pkg.dev/$Project/fraud"
$Bucket = "$Project-mlflow"
$ServiceAccount = "app-sa@$Project.iam.gserviceaccount.com"

# Les commandes natives ne lèvent pas d'exception en PowerShell 5.1 : on vérifie le code retour.
function Invoke-Checked {
    param([scriptblock]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "Échec (code $LASTEXITCODE) : $Command" }
}

function Ensure-Secret {
    param([string]$Name)
    & gcloud secrets describe $Name --project $Project --format="value(name)" 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        $bytes = New-Object byte[] 24
        [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
        $value = [Convert]::ToBase64String($bytes) -replace "[+/=]", ""
        # Fichier temporaire sans retour à la ligne : un pipe ajouterait un "\n" au secret
        $tmp = [IO.Path]::GetTempFileName()
        [IO.File]::WriteAllText($tmp, $value)
        try {
            Invoke-Checked { gcloud secrets create $Name --project $Project --replication-policy=user-managed --locations=$Region --data-file=$tmp }
        } finally { Remove-Item $tmp }
    }
    Invoke-Checked { gcloud secrets add-iam-policy-binding $Name --project $Project --member="serviceAccount:$ServiceAccount" --role=roles/secretmanager.secretAccessor --format="value(etag)" | Out-Null }
}

Invoke-Checked { gcloud auth configure-docker "$Region-docker.pkg.dev" --quiet 2>$null }

if ($Target -in "all", "mlflow") {
    Write-Host "== Secrets" -ForegroundColor Cyan
    Ensure-Secret "mlflow-admin-password"
    Ensure-Secret "mlflow-flask-secret"

    Write-Host "== Image MLflow" -ForegroundColor Cyan
    Invoke-Checked { docker build -f Dockerfile.mlflow -t "$Registry/mlflow:latest" . }
    Invoke-Checked { docker push "$Registry/mlflow:latest" }

    Write-Host "== Cloud Run : mlflow" -ForegroundColor Cyan
    # Migre le schéma d'une base existante avant de démarrer (no-op si à jour)
    $serverCmd = "if [ -f /mlflow/mlflow.db ]; then mlflow db upgrade sqlite:////mlflow/mlflow.db; fi && exec mlflow server --app-name basic-auth --backend-store-uri sqlite:////mlflow/mlflow.db --artifacts-destination /mlflow/mlartifacts --serve-artifacts --host 0.0.0.0 --port 8080 --workers 1"
    Invoke-Checked {
        gcloud run deploy mlflow --project $Project --region $Region `
            --image "$Registry/mlflow:latest" `
            --service-account $ServiceAccount `
            --execution-environment gen2 --cpu 1 --memory 2Gi --cpu-boost `
            --min-instances 0 --max-instances 1 --timeout 300 --port 8080 `
            --allow-unauthenticated `
            --add-volume "name=mlflow,type=cloud-storage,bucket=$Bucket" `
            --add-volume-mount "volume=mlflow,mount-path=/mlflow" `
            --set-env-vars "MLFLOW_SERVER_ALLOWED_HOSTS=*.run.app,MLFLOW_AUTH_CONFIG_PATH=/etc/mlflow/basic_auth.ini,MALLOC_ARENA_MAX=2" `
            --set-secrets "MLFLOW_AUTH_ADMIN_PASSWORD=mlflow-admin-password:latest,MLFLOW_FLASK_SERVER_SECRET_KEY=mlflow-flask-secret:latest" `
            --command sh --args "-c,$serverCmd"
    }
}

$MlflowUrl = (& gcloud run services describe mlflow --project $Project --region $Region --format="value(status.url)")
if (-not $MlflowUrl) { throw "Service mlflow introuvable : déployer d'abord avec -Target mlflow." }

if ($Target -in "all", "app") {
    Write-Host "== Image app" -ForegroundColor Cyan
    Invoke-Checked { docker build -f Dockerfile -t "$Registry/app:latest" . }
    Invoke-Checked { docker push "$Registry/app:latest" }

    Write-Host "== Cloud Run : app" -ForegroundColor Cyan
    Invoke-Checked {
        gcloud run deploy app --project $Project --region $Region `
            --image "$Registry/app:latest" `
            --service-account $ServiceAccount `
            --cpu 1 --memory 1Gi --cpu-boost `
            --min-instances 0 --max-instances 1 --session-affinity --timeout 3600 --port 8080 `
            --allow-unauthenticated `
            --set-env-vars "MLFLOW_TRACKING_URI=$MlflowUrl,MLFLOW_TRACKING_USERNAME=admin" `
            --set-secrets "MLFLOW_TRACKING_PASSWORD=mlflow-admin-password:latest" `
            --command streamlit --args "run,app.py,--server.address=0.0.0.0,--server.port=8080,--server.headless=true"
    }
}

Write-Host ""
Write-Host "MLflow : $MlflowUrl  (utilisateur admin)" -ForegroundColor Green
Write-Host "Mot de passe : gcloud secrets versions access latest --secret mlflow-admin-password --project $Project"
if ($Target -in "all", "app") {
    $AppUrl = (& gcloud run services describe app --project $Project --region $Region --format="value(status.url)")
    Write-Host "App    : $AppUrl" -ForegroundColor Green
}
