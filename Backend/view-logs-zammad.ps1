# PowerShell script to filter only Zammad logs

Write-Host "=== ZAMMAD LOGS (filtered) ===" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is used
$dockerRunning = docker ps --filter "name=rag_backend" --format "{{.Names}}" 2>$null

if ($dockerRunning -eq "rag_backend") {
    Write-Host "Docker detected - Filtering Zammad logs..." -ForegroundColor Green
    Write-Host ""
    docker logs rag_backend 2>&1 | Select-String -Pattern "Zammad|KB|knowledge|ticket" -CaseSensitive:$false
    Write-Host ""
    Write-Host "=== REAL-TIME FOLLOWING (Ctrl+C to stop) ===" -ForegroundColor Yellow
    docker logs -f rag_backend 2>&1 | Select-String -Pattern "Zammad|KB|knowledge|ticket" -CaseSensitive:$false
} else {
    $logFile = "logs\app.log"
    
    if (Test-Path $logFile) {
        Write-Host "Log file found: $logFile" -ForegroundColor Green
        Write-Host ""
        Write-Host "=== LAST ZAMMAD LINES ===" -ForegroundColor Yellow
        Get-Content $logFile | Select-String -Pattern "Zammad|KB|knowledge|ticket" -CaseSensitive:$false | Select-Object -Last 50
        Write-Host ""
        Write-Host "=== REAL-TIME FOLLOWING (Ctrl+C to stop) ===" -ForegroundColor Yellow
        Get-Content $logFile -Wait | Select-String -Pattern "Zammad|KB|knowledge|ticket" -CaseSensitive:$false
    } else {
        Write-Host "Log file not found: $logFile" -ForegroundColor Red
    }
}
