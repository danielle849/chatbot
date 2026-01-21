# PowerShell script to view backend logs

Write-Host "=== BACKEND LOGS ===" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is used
$dockerRunning = docker ps --filter "name=rag_backend" --format "{{.Names}}" 2>$null

if ($dockerRunning -eq "rag_backend") {
    Write-Host "Docker detected - Displaying container logs..." -ForegroundColor Green
    Write-Host ""
    docker logs -f rag_backend
} else {
    # Check if log file exists
    $logFile = "logs\app.log"
    
    if (Test-Path $logFile) {
        Write-Host "Log file found: $logFile" -ForegroundColor Green
        Write-Host ""
        Write-Host "=== LAST 50 LINES ===" -ForegroundColor Yellow
        Get-Content $logFile -Tail 50
        Write-Host ""
        Write-Host "=== REAL-TIME FOLLOWING (Ctrl+C to stop) ===" -ForegroundColor Yellow
        Get-Content $logFile -Wait -Tail 20
    } else {
        Write-Host "Log file not found: $logFile" -ForegroundColor Red
        Write-Host "The backend server must be started to generate logs." -ForegroundColor Yellow
    }
}
