# Script pour relancer la synchronisation Zammad et voir les logs en temps reel

$headers = @{
    "X-API-Key" = "Q43Yz9Jio2ZBdNOPUR614drRhOmV3IICsaMu0O8eYGwUNusUbpBfbmz9jWyfNNwW"
    "Content-Type" = "application/json"
}

Write-Host "=== SYNCHRONISATION ZAMMAD ===" -ForegroundColor Cyan
Write-Host ""

# Verifier si le serveur est en cours d'execution
Write-Host "Verification du serveur backend..." -ForegroundColor Yellow
try {
    Invoke-RestMethod -Uri "http://localhost:8000/docs" -Method GET -TimeoutSec 5 -ErrorAction Stop | Out-Null
    #$healthCheck = Invoke-RestMethod -Uri "http://localhost:8000/docs" -Method GET -TimeoutSec 5 -ErrorAction Stop | Out-Null
    Write-Host "[OK] Serveur backend actif" -ForegroundColor Green
} catch {
    Write-Host "[ERREUR] Le serveur backend n'est pas accessible" -ForegroundColor Red
    Write-Host "Demarrez le serveur backend d'abord!" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "=== DEMARRAGE DE LA SYNCHRONISATION ===" -ForegroundColor Cyan
Write-Host "Les logs vont s'afficher ci-dessous..." -ForegroundColor Yellow
Write-Host ""

# Verifier si Docker est utilise
$dockerRunning = docker ps --filter "name=rag_backend" --format "{{.Names}}" 2>$null

if ($dockerRunning -eq "rag_backend") {
    Write-Host "Mode Docker detecte" -ForegroundColor Green
    Write-Host ""
    
    # Lancer la synchronisation en arriere-plan et afficher les logs
    Write-Host "Lancement de la synchronisation..." -ForegroundColor Yellow
    $syncJob = Start-Job -ScriptBlock {
        $headers = @{
            "X-API-Key" = "Q43Yz9Jio2ZBdNOPUR614drRhOmV3IICsaMu0O8eYGwUNusUbpBfbmz9jWyfNNwW"
            "Content-Type" = "application/json"
        }
        try {
            Invoke-RestMethod -Uri "http://localhost:8000/api/documents/sync/zammad" -Method POST -Headers $headers -TimeoutSec 300
        } catch {
            Write-Output "ERROR: $($_.Exception.Message)"
        }
    }
    
    # Afficher les logs Docker en temps reel
    Write-Host "=== LOGS EN TEMPS REEL ===" -ForegroundColor Cyan
    docker logs -f rag_backend 2>&1 | Select-String -Pattern "Zammad|KB|knowledge|ticket|Loading|Loaded|ERROR|Fetched|Retrieved|answer_id|Successfully processed" -CaseSensitive:$false
    
    # Attendre que le job se termine
    Wait-Job $syncJob | Out-Null
    $result = Receive-Job $syncJob
    Remove-Job $syncJob
    
    Write-Host ""
    Write-Host "=== RESULTAT DE LA SYNCHRONISATION ===" -ForegroundColor Cyan
    $result | ConvertTo-Json -Depth 10
    
} else {
    Write-Host "Mode local detecte" -ForegroundColor Green
    Write-Host ""
    
    # Verifier si le fichier de log existe
    $logFile = "logs\app.log"
    if (-not (Test-Path $logFile)) {
        Write-Host "[ATTENTION] Fichier de log non trouve: $logFile" -ForegroundColor Yellow
        Write-Host "Le serveur doit etre demarre pour generer des logs." -ForegroundColor Yellow
    }
    
    # Lancer la synchronisation
    Write-Host "Lancement de la synchronisation..." -ForegroundColor Yellow
    Write-Host ""
    
    try {
        # Lancer la synchronisation dans un job
        $syncJob = Start-Job -ScriptBlock {
            $headers = @{
                "X-API-Key" = "Q43Yz9Jio2ZBdNOPUR614drRhOmV3IICsaMu0O8eYGwUNusUbpBfbmz9jWyfNNwW"
                "Content-Type" = "application/json"
            }
            try {
                Invoke-RestMethod -Uri "http://localhost:8000/api/documents/sync/zammad" -Method POST -Headers $headers -TimeoutSec 300
            } catch {
                Write-Output "ERROR: $($_.Exception.Message)"
            }
        }
        
        # Afficher les logs en temps reel pendant la synchronisation
        if (Test-Path $logFile) {
            Write-Host "=== LOGS EN TEMPS REEL (filtre Zammad) ===" -ForegroundColor Cyan
            Write-Host "Appuyez sur Ctrl+C pour arreter l'affichage des logs" -ForegroundColor Yellow
            Write-Host ""
            
            # Lire les dernieres lignes et suivre en temps reel
            Get-Content $logFile -Wait -Tail 50 | Select-String -Pattern "Zammad|KB|knowledge|ticket|Loading|Loaded|ERROR|Fetched|Retrieved|answer_id|Successfully processed|Found.*answer IDs" -CaseSensitive:$false
        } else {
            # Si pas de fichier de log, attendre et afficher le resultat
            Write-Host "Attente de la synchronisation..." -ForegroundColor Yellow
            Wait-Job $syncJob | Out-Null
            $result = Receive-Job $syncJob
            Remove-Job $syncJob
            
            Write-Host ""
            Write-Host "=== RESULTAT ===" -ForegroundColor Cyan
            $result | ConvertTo-Json -Depth 10
        }
        
    } catch {
        Write-Host "[ERREUR] Erreur lors de la synchronisation" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "=== FIN ===" -ForegroundColor Cyan