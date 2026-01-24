$headers = @{
    "Content-Type" = "application/json"
    "X-API-Key" = "your-api-key-here"

  }
  
  $body = @{ message = "Morgen, wie geht es dir?" } | ConvertTo-Json
  
  Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/chat" -Headers $headers -Body $body

  