# Script to test Zammad connection and view diagnostics

$headers = @{
    "X-API-Key" = "Q43Yz9Jio2ZBdNOPUR614drRhOmV3IICsaMu0O8eYGwUNusUbpBfbmz9jWyfNNwW"
    "Content-Type" = "application/json"
}

Write-Host "=== ZAMMAD CONNECTION TEST ===" -ForegroundColor Cyan
Write-Host ""

try {
    Write-Host "Testing diagnostic endpoint..." -ForegroundColor Yellow
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
        Write-Host "`n✓ Zammad connection OK" -ForegroundColor Green
        if ($response.kb_entries_test.entries_count -gt 0) {
            Write-Host "✓ KB entries found: $($response.kb_entries_test.entries_count)" -ForegroundColor Green
        } else {
            Write-Host "⚠ No KB entries found" -ForegroundColor Yellow
        }
    } else {
        Write-Host "`n✗ Connection error" -ForegroundColor Red
        Write-Host $response.message -ForegroundColor Red
    }
    
} catch {
    Write-Host "`n✗ ERROR during test" -ForegroundColor Red
    Write-Host "Message: $($_.Exception.Message)" -ForegroundColor Red
    
    if ($_.Exception.Response) {
        $statusCode = $_.Exception.Response.StatusCode.value__
        Write-Host "HTTP Code: $statusCode" -ForegroundColor Yellow
        
        try {
            $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
            $responseBody = $reader.ReadToEnd()
            Write-Host "Response: $responseBody" -ForegroundColor Yellow
        } catch {
            Write-Host "Unable to read response" -ForegroundColor Yellow
        }
    }
    
    Write-Host "`nCheck that:" -ForegroundColor Cyan
    Write-Host "1. The backend server is running (http://localhost:8000)" -ForegroundColor White
    Write-Host "2. Zammad configuration is correct in .env.backend" -ForegroundColor White
    Write-Host "3. Server logs for more details" -ForegroundColor White
}
