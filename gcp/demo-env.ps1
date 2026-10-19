<#
Configure le terminal courant pour que les scripts Python (promote_model,
drift_report, retrain...) parlent au MLflow de Cloud Run au lieu du local.

À "dot-sourcer" (le point au début est indispensable) :
    . .\gcp\demo-env.ps1

Ce que ça fait :
  - MLFLOW_TRACKING_URI / USERNAME / PASSWORD (mot de passe lu dans Secret Manager) ;
  - REQUESTS_CA_BUNDLE : le réseau NDA inspecte le HTTPS avec un certificat que
    Python ne connaît pas -> on construit un bundle certifi + magasin Windows.

Pour revenir au MLflow local : fermer le terminal (ou Remove-Item Env:MLFLOW_*).
#>
param(
    [string]$Project = "nda-ss-tech-radar-mlops-drift",
    [string]$Region = "europe-west9"
)

$env:Path = "$env:LOCALAPPDATA\Google\Cloud SDK\google-cloud-sdk\bin;$env:Path"
$python = Join-Path $PSScriptRoot "..\venv\Scripts\python.exe"

$bundle = Join-Path $env:LOCALAPPDATA "nda-ca-bundle.pem"
if (-not (Test-Path $bundle)) {
    $certifi = & $python -c "import certifi; print(certifi.where())"
    $sb = New-Object Text.StringBuilder
    [void]$sb.Append([IO.File]::ReadAllText($certifi))
    foreach ($store in "Cert:\LocalMachine\Root", "Cert:\CurrentUser\Root", "Cert:\LocalMachine\CA") {
        foreach ($cert in Get-ChildItem $store) {
            [void]$sb.Append("`n-----BEGIN CERTIFICATE-----`n" +
                [Convert]::ToBase64String($cert.RawData, "InsertLineBreaks") +
                "`n-----END CERTIFICATE-----`n")
        }
    }
    [IO.File]::WriteAllText($bundle, $sb.ToString().Replace("`r`n", "`n"))
}

$env:REQUESTS_CA_BUNDLE = $bundle
$env:MLFLOW_TRACKING_URI = & gcloud run services describe mlflow --project $Project --region $Region --format="value(status.url)"
$env:MLFLOW_TRACKING_USERNAME = "admin"
$env:MLFLOW_TRACKING_PASSWORD = & gcloud secrets versions access latest --secret mlflow-admin-password --project $Project 2>$null
$env:MLFLOW_DISABLE_AGENT_HINT = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "MLflow cible : $env:MLFLOW_TRACKING_URI (admin)" -ForegroundColor Green
