# Script pour tester la connexion Zammad et voir les diagnostics

$headers = @{
    "X-API-Key" = "Q43Yz9Jio2ZBdNOPUR614drRhOmV3IICsaMu0O8eYGwUNusUbpBfbmz9jWyfNNwW"
    "Content-Type" = "application/json"
}

Write-Host "=== TEST DE CONNEXION ZAMMAD ===" -ForegroundColor Cyan
Write-Host ""

try {
    Write-Host "Test de l'endpoint de diagnostic..." -ForegroundColor Yellow
    $response = Invoke-RestMethod `
        -Uri "http://localhost:8000/api/documents/zammad/test" `
        -Method GET `
        -Headers $headers `
        -ErrorAction Stop
    
    Write-Host "`n=== CONFIGURATION ===" -ForegroundColor Green
    $response.config | ConvertTo-Json -Depth 2
    
    Write-Host "`n=== TEST KNOWLEDGE BASES ===" -ForegroundColor Green
    $response.knowledge_bases_test | ConvertTo-Json -Depth 2
    
    Write-Host "`n=== TEST KB ENTRIES ===" -ForegroundColor Green
    $response.kb_entries_test | ConvertTo-Json -Depth 2
    
    if ($response.status -eq "success") {
        Write-Host "`n✓ Connexion Zammad OK" -ForegroundColor Green
        if ($response.kb_entries_test.entries_count -gt 0) {
            Write-Host "✓ Entrées KB trouvées: $($response.kb_entries_test.entries_count)" -ForegroundColor Green
        } else {
            Write-Host "⚠ Aucune entrée KB trouvée" -ForegroundColor Yellow
        }
    } else {
        Write-Host "`n✗ Erreur de connexion" -ForegroundColor Red
        Write-Host $response.message -ForegroundColor Red
    }
    
} catch {
    Write-Host "`n✗ ERREUR lors du test" -ForegroundColor Red
    Write-Host "Message: $($_.Exception.Message)" -ForegroundColor Red
    
    if ($_.Exception.Response) {
        $statusCode = $_.Exception.Response.StatusCode.value__
        Write-Host "Code HTTP: $statusCode" -ForegroundColor Yellow
        
        try {
            $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
            $responseBody = $reader.ReadToEnd()
            Write-Host "Réponse: $responseBody" -ForegroundColor Yellow
        } catch {
            Write-Host "Impossible de lire la réponse" -ForegroundColor Yellow
        }
    }
    
    Write-Host "`nVérifiez que:" -ForegroundColor Cyan
    Write-Host "1. Le serveur backend est démarré (http://localhost:8000)" -ForegroundColor White
    Write-Host "2. La configuration Zammad est correcte dans .env.backend" -ForegroundColor White
    Write-Host "3. Les logs du serveur pour plus de détails" -ForegroundColor White
}
