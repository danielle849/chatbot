"""Configuration management for the RAG chatbot backend."""
from pydantic_settings import BaseSettings
from typing import Optional, List


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Qdrant Configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "company_documents"
    
    # HuggingFace Configuration
    hf_api_token: Optional[str] = None
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    llm_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # API Configuration
    api_key: str = "your-api-key-here"
    api_title: str = "Company RAG Chatbot API"
    api_version: str = "1.0.0"

    # CORS Configuration
    cors_origins: str = "*"  # Comma-separated list of allowed origins, or "*" for all
    
    # Document Processing
    documents_folder: str = "../data"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Zammad Configuration
    zammad_base_url: Optional[str] = None
    zammad_api_token: Optional[str] = None
    zammad_sync_tickets: bool = True
    zammad_sync_kb: bool = True
    zammad_sync_attachments: bool = True
    zammad_ticket_limit: int = 100
    zammad_kb_id: Optional[int] = 1  # Knowledge Base ID to sync (default 1)
    
    # LLM Parameters
    temperature: float = 0.7
    max_tokens: int = 512
    top_k: int = 4  # Number of documents to retrieve

    # Retrieval Optimization
    retrieval_score_threshold: float = 0.5  # Minimum similarity score (0.0-1.0)
    retrieval_fetch_k: int = 20  # Fetch more candidates before filtering
    use_mmr: bool = False  # Maximal Marginal Relevance for diversity
    mmr_diversity: float = 0.5  # MMR diversity parameter (0.0-1.0)
    
    # Memory Management
    max_memory_length: int = 10  # Max conversation turns to keep in memory
    memory_ttl_hours: int = 24  # Memory expiration time in hours
    enable_per_conversation_memory: bool = True  # Separate memory per conversation_id
    
    # Performance Optimization
    enable_embedding_cache: bool = True
    batch_embedding_size: int = 32
    max_retries: int = 3  # Retry attempts for failed operations
    retry_delay: float = 1.0  # Delay between retries in seconds
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    class Config:
        env_file = ".env.backend"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
