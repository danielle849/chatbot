# Script PowerShell pour voir les logs du backend

Write-Host "=== LOGS DU BACKEND ===" -ForegroundColor Cyan
Write-Host ""

# Vérifier si Docker est utilisé
$dockerRunning = docker ps --filter "name=rag_backend" --format "{{.Names}}" 2>$null

if ($dockerRunning -eq "rag_backend") {
    Write-Host "Docker détecté - Affichage des logs du conteneur..." -ForegroundColor Green
    Write-Host ""
    docker logs -f rag_backend
} else {
    # Vérifier si le fichier de log existe
    $logFile = "logs\app.log"
    
    if (Test-Path $logFile) {
        Write-Host "Fichier de log trouvé: $logFile" -ForegroundColor Green
        Write-Host ""
        Write-Host "=== DERNIÈRES 50 LIGNES ===" -ForegroundColor Yellow
        Get-Content $logFile -Tail 50
        Write-Host ""
        Write-Host "=== SUIVI EN TEMPS RÉEL (Ctrl+C pour arrêter) ===" -ForegroundColor Yellow
        Get-Content $logFile -Wait -Tail 20
    } else {
        Write-Host "Fichier de log non trouvé: $logFile" -ForegroundColor Red
        Write-Host "Le serveur backend doit être démarré pour générer des logs." -ForegroundColor Yellow
    }
}
