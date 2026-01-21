"""Pydantic models for API requests and responses."""
from pydantic import BaseModel,Field
from typing import List, Optional
from datetime import datetime


class ChatMessage(BaseModel):
    """Chat message model."""
    message: str = Field(..., min_length=1, max_length=10000, description="User message (max 10000 characters)")
    conversation_id: Optional[str] = Field(None, max_length=100, description="Optional conversation ID")


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str
    conversation_id: str
    sources: Optional[List[str]] = None


class DocumentInfo(BaseModel):
    """Document information model."""
    id: str
    filename: str
    file_type: str
    chunks_count: int
    ingested_at: datetime


class DocumentListResponse(BaseModel):
    """Document list response model."""
    documents: List[DocumentInfo]
    total: int


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    service: str
    timestamp: datetime

class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., min_length=1, max_length=10000, description="Search query")
    top_k: Optional[int] = Field(None, ge=1, le=50, description="Number of results to return")
