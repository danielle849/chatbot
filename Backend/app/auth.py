"""Authentication middleware for API key validation."""
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from starlette.status import HTTP_403_FORBIDDEN
from app.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)

async def verify_api_key(
    api_key: str = Security(api_key_header),
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
):
    # 1) X-API-Key
    if api_key and api_key == settings.api_key:
        return api_key

    # 2) Authorization: Bearer <key>
    if credentials and credentials.scheme and credentials.scheme.lower() == "bearer":
        token = (credentials.credentials or "").strip()
        if token == settings.api_key:
            return token

    # 3) Otherwise -> refuse
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN,
        detail="Invalid or missing API key",
    )
