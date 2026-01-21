# Script PowerShell pour filtrer uniquement les logs Zammad

Write-Host "=== LOGS ZAMMAD (filtre) ===" -ForegroundColor Cyan
Write-Host ""

# Vérifier si Docker est utilisé
$dockerRunning = docker ps --filter "name=rag_backend" --format "{{.Names}}" 2>$null

if ($dockerRunning -eq "rag_backend") {
    Write-Host "Docker détecté - Filtrage des logs Zammad..." -ForegroundColor Green
    Write-Host ""
    docker logs rag_backend 2>&1 | Select-String -Pattern "Zammad|KB|knowledge|ticket" -CaseSensitive:$false
    Write-Host ""
    Write-Host "=== SUIVI EN TEMPS RÉEL (Ctrl+C pour arrêter) ===" -ForegroundColor Yellow
    docker logs -f rag_backend 2>&1 | Select-String -Pattern "Zammad|KB|knowledge|ticket" -CaseSensitive:$false
} else {
    $logFile = "logs\app.log"
    
    if (Test-Path $logFile) {
        Write-Host "Fichier de log trouvé: $logFile" -ForegroundColor Green
        Write-Host ""
        Write-Host "=== DERNIÈRES LIGNES ZAMMAD ===" -ForegroundColor Yellow
        Get-Content $logFile | Select-String -Pattern "Zammad|KB|knowledge|ticket" -CaseSensitive:$false | Select-Object -Last 50
        Write-Host ""
        Write-Host "=== SUIVI EN TEMPS RÉEL (Ctrl+C pour arrêter) ===" -ForegroundColor Yellow
        Get-Content $logFile -Wait | Select-String -Pattern "Zammad|KB|knowledge|ticket" -CaseSensitive:$false
    } else {
        Write-Host "Fichier de log non trouvé: $logFile" -ForegroundColor Red
    }
}
