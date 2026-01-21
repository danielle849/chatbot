"""Health check API endpoints."""
from fastapi import APIRouter, HTTPException
from datetime import datetime
from app.models import HealthResponse
from app.config import settings
from qdrant_client import QdrantClient
from app.utils.logger import logger

router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health_check():
    """Basic health check."""
    return HealthResponse(
        status="healthy",
        service="backend",
        timestamp=datetime.now()
    )


@router.get("/qdrant", response_model=HealthResponse)
async def qdrant_health():
    """Check Qdrant connection health."""
    try:
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        collections = client.get_collections()
        
        return HealthResponse(
            status="healthy",
            service="qdrant",
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Qdrant connection unavailable"
        )
