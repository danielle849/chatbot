"""Authentication middleware for API key validation."""
from fastapi import HTTPException, Security, Request
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN
from app.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    request: Request,
    api_key: str = Security(api_key_header)
):
    """Verify API key from request header."""
    # 1) X-API-Key header
    if api_key and api_key == settings.api_key:
        return api_key

    # 2) Authorization: Bearer <key>
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
        if token == settings.api_key:
            return token
    
    if not api_key:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="API key is missing"
        )
    if api_key != settings.api_key:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
    return api_key
