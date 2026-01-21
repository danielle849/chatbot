# Script to restart Zammad synchronization and view logs in real time

$headers = @{
    "X-API-Key" = "Q43Yz9Jio2ZBdNOPUR614drRhOmV3IICsaMu0O8eYGwUNusUbpBfbmz9jWyfNNwW"
    "Content-Type" = "application/json"
}

Write-Host "=== ZAMMAD SYNCHRONIZATION ===" -ForegroundColor Cyan
Write-Host ""

# Check if server is running
Write-Host "Checking backend server..." -ForegroundColor Yellow
try {
    Invoke-RestMethod -Uri "http://localhost:8000/docs" -Method GET -TimeoutSec 5 -ErrorAction Stop | Out-Null
    #$healthCheck = Invoke-RestMethod -Uri "http://localhost:8000/docs" -Method GET -TimeoutSec 5 -ErrorAction Stop | Out-Null
    Write-Host "[OK] Backend server active" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Backend server is not accessible" -ForegroundColor Red
    Write-Host "Start the backend server first!" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "=== STARTING SYNCHRONIZATION ===" -ForegroundColor Cyan
Write-Host "Logs will be displayed below..." -ForegroundColor Yellow
Write-Host ""

# Check if Docker is used
$dockerRunning = docker ps --filter "name=rag_backend" --format "{{.Names}}" 2>$null

if ($dockerRunning -eq "rag_backend") {
    Write-Host "Docker mode detected" -ForegroundColor Green
    Write-Host ""
    
    # Launch synchronization in background and display logs
    Write-Host "Starting synchronization..." -ForegroundColor Yellow
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
    
    # Display Docker logs in real time
    Write-Host "=== REAL-TIME LOGS ===" -ForegroundColor Cyan
    docker logs -f rag_backend 2>&1 | Select-String -Pattern "Zammad|KB|knowledge|ticket|Loading|Loaded|ERROR|Fetched|Retrieved|answer_id|Successfully processed" -CaseSensitive:$false
    
    # Wait for job to complete
    Wait-Job $syncJob | Out-Null
    $result = Receive-Job $syncJob
    Remove-Job $syncJob
    
    Write-Host ""
    Write-Host "=== SYNCHRONIZATION RESULT ===" -ForegroundColor Cyan
    $result | ConvertTo-Json -Depth 10
    
} else {
    Write-Host "Local mode detected" -ForegroundColor Green
    Write-Host ""
    
    # Check if log file exists
    $logFile = "logs\app.log"
    if (-not (Test-Path $logFile)) {
        Write-Host "[WARNING] Log file not found: $logFile" -ForegroundColor Yellow
        Write-Host "Server must be started to generate logs." -ForegroundColor Yellow
    }
    
    # Launch synchronization
    Write-Host "Starting synchronization..." -ForegroundColor Yellow
    Write-Host ""
    
    try {
        # Launch synchronization in a job
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
        
        # Display logs in real time during synchronization
        if (Test-Path $logFile) {
            Write-Host "=== REAL-TIME LOGS (Zammad filter) ===" -ForegroundColor Cyan
            Write-Host "Press Ctrl+C to stop log display" -ForegroundColor Yellow
            Write-Host ""
            
            # Read last lines and follow in real time
            Get-Content $logFile -Wait -Tail 50 | Select-String -Pattern "Zammad|KB|knowledge|ticket|Loading|Loaded|ERROR|Fetched|Retrieved|answer_id|Successfully processed|Found.*answer IDs" -CaseSensitive:$false
        } else {
            # If no log file, wait and display result
            Write-Host "Waiting for synchronization..." -ForegroundColor Yellow
            Wait-Job $syncJob | Out-Null
            $result = Receive-Job $syncJob
            Remove-Job $syncJob
            
            Write-Host ""
            Write-Host "=== RESULT ===" -ForegroundColor Cyan
            $result | ConvertTo-Json -Depth 10
        }
        
    } catch {
        Write-Host "[ERROR] Error during synchronization" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "=== END ===" -ForegroundColor Cyan