$headers = @{
    "Content-Type" = "application/json"
    "X-API-Key" = "Q43Yz9Jio2ZBdNOPUR614drRhOmV3IICsaMu0O8eYGwUNusUbpBfbmz9jWyfNNwW"

  }
  
  $body = @{ message = "Morgen, wie geht es dir?" } | ConvertTo-Json
  
  Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/chat" -Headers $headers -Body $body

  