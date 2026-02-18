# Test retrieval – prüfen ob "Webhook lokal testen" und "ngrok" gefunden werden
# Nach Zammad-Sync ausführen: .\Test_search.ps1

$apiKey = "Q43Yz9Jio2ZBdNOPUR614drRhOmV3IICsaMu0O8eYGwUNusUbpBfbmz9jWyfNNwW"
$headers = @{
    "Content-Type" = "application/json"
    "X-API-Key"   = $apiKey
}

$queries = @(
    "Webhook lokal testen",
    "ngrok",
    "Webhook lokal testen ngrok"
)

foreach ($query in $queries) {
    Write-Host ""
    Write-Host "=== Query: `"$query`" ===" -ForegroundColor Cyan
    Write-Host ""

    $body = @{ query = $query; top_k = 8 } | ConvertTo-Json

    try {
        $response = Invoke-RestMethod `
            -Method Post `
            -Uri "http://localhost:8000/api/documents/search" `
            -Headers $headers `
            -Body $body `
            -TimeoutSec 30

        Write-Host "Treffer: $($response.total_results)" -ForegroundColor Yellow
        Write-Host ""

        foreach ($r in $response.results) {
            $score = [math]::Round($r.score, 3)
            $scoreColor = if ($score -ge 0.4) { "Green" } elseif ($score -ge 0.3) { "Yellow" } else { "Red" }
            Write-Host "  #$($r.rank) Score: $score" -ForegroundColor $scoreColor
            $titleInfo = if ($r.title) { " | KB: $($r.title)" } else { "" }
            Write-Host "     Doc: $($r.filename) | Chunk: $($r.chunk_index)$titleInfo"
            Write-Host "     Text: $($r.text)"
            Write-Host ""
        }
    }
    catch {
        Write-Host "FEHLER: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Backend läuft? (http://localhost:8000)" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "=== Ende ===" -ForegroundColor Cyan
Write-Host "Erwartung: 'Webhook lokal testen' sollte Chunks mit ngrok-Anleitung finden."
Write-Host "Wenn Score < 0.35: Chunk wird vom RAG gefiltert."
Write-Host "Wenn falsche Docs: Zammad-Sync erneut ausführen (.\sync-and-watch-logs.ps1)"
