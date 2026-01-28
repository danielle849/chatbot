"""Chat API endpoints - Optimized version."""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from app.models import ChatMessage, ChatResponse
from app.auth import verify_api_key
from app.rag.chain import RAGChain
from app.rag.vector_store import VectorStore
from app.rag.embeddings import EmbeddingGenerator
from app.config import settings
from app.utils.logger import logger
import json
import asyncio

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Initialize RAG components (singleton pattern)
_rag_chain: RAGChain = None
_vector_store: VectorStore = None
_embedding_generator: EmbeddingGenerator = None


def get_rag_chain() -> RAGChain:
    """Get or initialize RAG chain."""
    global _rag_chain, _vector_store, _embedding_generator
    
    if _rag_chain is None:
        try:
            _embedding_generator = EmbeddingGenerator()
            _vector_store = VectorStore()
            _vector_store.create_collection(_embedding_generator.dimension)
            _rag_chain = RAGChain(_vector_store)
        except Exception as e:
            logger.error(f"Error initializing RAG chain: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize RAG chain: {str(e)}")
    
    return _rag_chain


@router.post("", response_model=ChatResponse)
async def chat(
    message: ChatMessage,
    api_key: str = Depends(verify_api_key)
):
    """Send a chat message and get a RAG response."""
    try:
        rag_chain = get_rag_chain()
        
        # Execute request in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
           # logger.info(f"type(message.message)={type(message.message)} value={message.message!r}")
           # logger.info(f"type(message.conversation_id)={type(message.conversation_id)} value={message.conversation_id!r}")
            rag_chain.query,
            message.message,
            message.conversation_id
        )
        
        return ChatResponse(
            response=result["answer"],
            conversation_id=result["conversation_id"],
            sources=result.get("sources", [])
        )
    except Exception as e:
        logger.Exception(f"Error processing chat message")
        raise HTTPException(status_code=500, detail=f"Error processing message: {type(e).__name__}: {e!r}")


@router.post("/stream")
async def chat_stream(
    message: ChatMessage,
    api_key: str = Depends(verify_api_key)
):
    """Stream the chat response."""
    async def generate():
        try:
            rag_chain = get_rag_chain()
            
            # Execute request in thread pool
            loop = asyncio.get_event_loop()
            yield f"data: {json.dumps({'content':'(Suche in Datenbank…) '})}\n\n"
            result = await loop.run_in_executor(
                None,
                rag_chain.query,
                message.message,
                message.conversation_id
            )
            
            # Stream response word by word
            words = result["answer"].split()
            for word in words:
                chunk = {
                    "content": word + " ",
                    "conversation_id": result["conversation_id"]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            
            # Send final chunk with sources
            final_chunk = {
                "done": True,
                "sources": result.get("sources", [])
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            
        except Exception as e:
            logger.exception("Error processing chat stream")  # <-- stacktrace complet
            error_chunk = {"error": f"{type(e).__name__}: {e!r}"}
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@router.delete("/memory/{conversation_id}")
async def clear_conversation_memory(
    conversation_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Clear memory of a specific conversation."""
    try:
        rag_chain = get_rag_chain()
        rag_chain.clear_memory(conversation_id)
        return {"message": f"Memory cleared for conversation: {conversation_id}"}
    except Exception as e:
        logger.error(f"Error clearing memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))
