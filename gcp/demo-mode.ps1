<#
Ouvre ou ferme la démo sur Cloud Run (app + mlflow).

    .\gcp\demo-mode.ps1 on    # accès public + 1 instance chaude (pas de démarrage à froid)
    .\gcp\demo-mode.ps1 off   # accès coupé (404) + 0 instance : plus rien ne tourne

"on" coûte environ 1 €/jour (2 instances allumées en permanence) : repasser
en "off" après la démo.
#>
param(
    [Parameter(Mandatory)][ValidateSet("on", "off")][string]$Mode,
    [string]$Project = "nda-ss-tech-radar-mlops-drift",
    [string]$Region = "europe-west9"
)

$env:Path = "$env:LOCALAPPDATA\Google\Cloud SDK\google-cloud-sdk\bin;$env:Path"
$ingress, $min = if ($Mode -eq "on") { "all", 1 } else { "internal", 0 }

foreach ($service in "mlflow", "app") {
    Write-Host "== $service : ingress=$ingress, min-instances=$min" -ForegroundColor Cyan
    & gcloud run services update $service --project $Project --region $Region `
        --ingress $ingress --min-instances $min --quiet 2>&1 | Select-String "Done|ERROR"
    if ($LASTEXITCODE -ne 0) { throw "Échec de la mise à jour de $service" }
}

if ($Mode -eq "on") {
    foreach ($service in "mlflow", "app") {
        $url = & gcloud run services describe $service --project $Project --region $Region --format="value(status.url)"
        Write-Host "$service : $url" -ForegroundColor Green
    }
}
