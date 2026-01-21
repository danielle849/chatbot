"""Document management API endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional, Dict, Tuple
from datetime import datetime
from app.models import DocumentInfo, DocumentListResponse, SearchRequest
from app.auth import verify_api_key
from app.rag.document_loader import DocumentLoader
from app.rag.zammad_loader import ZammadLoader
from app.rag.document_processor import DocumentProcessor
from app.rag.embeddings import EmbeddingGenerator
from app.rag.vector_store import VectorStore
from app.config import settings
from app.utils.logger import logger
from app.utils.file_helpers import temp_document_file

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Initialize components
_vector_store: VectorStore = None
_embedding_generator: EmbeddingGenerator = None


def get_components():
    """Get or initialize vector store and embedding generator."""
    global _vector_store, _embedding_generator
    
    if _vector_store is None:
        _embedding_generator = EmbeddingGenerator()
        _vector_store = VectorStore()
        _vector_store.create_collection(_embedding_generator.dimension)
    
    return _vector_store, _embedding_generator


def _infer_file_type(filename: str) -> str:
    """Infer file type from filename."""
    if filename.endswith('.pdf'):
        return 'pdf'
    elif filename.endswith('.docx'):
        return 'docx'
    elif filename.endswith('.txt'):
        return 'txt'
    elif filename.endswith('.md'):
        return 'markdown'
    elif filename.endswith(('.html', '.htm')):
        return 'html'
    return 'unknown'


def _process_and_store_document(
    doc_data: Dict,
    processor: DocumentProcessor,
    embedding_generator: EmbeddingGenerator,
    vector_store: VectorStore,
    text_content: str
) -> Tuple[int, Dict]:
    """Process a document and store it in the vector database.
    
    Returns:
        Tuple of (chunks_count, doc_info_dict)
    """
    with temp_document_file(text_content, suffix='.txt') as tmp_path:
        chunks = processor.process_document(tmp_path)
        
        if not chunks:
            return 0, None
        
        # Update metadata
        for chunk in chunks:
            chunk['metadata'].update(doc_data.get('metadata', {}))
        
        # Generate embeddings and store
        texts = [chunk["text"] for chunk in chunks]
        embeddings = embedding_generator.generate_embeddings(texts)
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        vector_store.add_documents(texts, embeddings, metadatas)
        
        doc_info = {
            "title": doc_data.get('title', doc_data.get('filename', 'Unknown')),
            "source": doc_data.get('source', 'local'),
            "chunks": len(chunks)
        }
        
        return len(chunks), doc_info


def _process_attachment(
    attachment_info: Dict,
    zammad_loader: ZammadLoader,
    processor: DocumentProcessor
) -> Optional[str]:
    """Process an attachment and return its text content."""
    attachment = zammad_loader.load_attachment(
        attachment_info['ticket_id'],
        attachment_info['article_id'],
        attachment_info['attachment_id'],
        attachment_info['filename']
    )
    
    if not attachment:
        return None
    
    import os
    suffix = os.path.splitext(attachment['filename'])[1]
    
    with temp_document_file(attachment['content'], suffix=suffix, mode='wb') as tmp_path:
        chunks = processor.process_document(tmp_path)
        if chunks:
            attachment_text = "\n\n".join([chunk['text'] for chunk in chunks])
            return f"\n\nAttachment: {attachment['filename']}\n{attachment_text}"
    
    return None


async def _ingest_zammad(
    vector_store: VectorStore,
    embedding_generator: EmbeddingGenerator,
    processor: DocumentProcessor
) -> Dict:
    """Ingest documents from Zammad.""" #verifie la configuration de Zammad
    if not settings.zammad_base_url or not settings.zammad_api_token:
        raise HTTPException(
            status_code=400,
            detail="Zammad configuration missing. Set ZAMMAD_BASE_URL and ZAMMAD_API_TOKEN"
        )
    # Charge les documents de Zammad/ creer le loader Zammad
    zammad_loader = ZammadLoader()
    zammad_documents = []
    
    # Diagnostic information
    tickets_count = 0
    kb_entries_count = 0
    processing_errors = []
    
    # Load tickets /charge les tickets (si activé)
    if settings.zammad_sync_tickets:
        logger.info("Loading tickets from Zammad...")
        tickets = zammad_loader.load_tickets(limit=settings.zammad_ticket_limit)
        tickets_count = len(tickets)
        logger.info(f"Retrieved {tickets_count} tickets from Zammad API")
        zammad_documents.extend(tickets)
    else:
        logger.info("Ticket sync is disabled (zammad_sync_tickets=False)")
    
    # Load knowledge base entries
    if settings.zammad_sync_kb:
        logger.info("Loading knowledge base entries from Zammad...")
        # Use configured KB ID if available
        kb_id = getattr(settings, 'zammad_kb_id', None)
        if kb_id:
            logger.info(f"Using KB ID: {kb_id}")
        kb_entries = zammad_loader.load_knowledge_base_entries(kb_id=kb_id)
        kb_entries_count = len(kb_entries)
        logger.info(f"Retrieved {kb_entries_count} KB entries from Zammad API")
        zammad_documents.extend(kb_entries)
    else:
        logger.info("KB sync is disabled (zammad_sync_kb=False)")
    
    logger.info(f"Total documents retrieved from Zammad: {len(zammad_documents)}")
    
    total_chunks = 0
    ingested_docs = []
    
    # Process Zammad documents
    for doc in zammad_documents:
        try:
            text_content = doc.get('text', '')
            
            if not text_content or not text_content.strip():
                logger.warning(f"Document '{doc.get('title', 'unknown')}' has empty text content, skipping")
                processing_errors.append(f"Empty content: {doc.get('title', 'unknown')}")
                continue
            
            # Process attachments if enabled
            if settings.zammad_sync_attachments and doc.get('attachments'):
                for attachment_info in doc['attachments']:
                    attachment_text = _process_attachment(attachment_info, zammad_loader, processor)
                    if attachment_text:
                        text_content += attachment_text
            
            # Process and store document
            chunks_count, doc_info = _process_and_store_document(
                doc_data={
                    'title': doc['title'],
                    'source': doc['source'],
                    'metadata': {
                        'source': doc['source'],
                        'source_id': doc['source_id'],
                        'title': doc['title'],
                        **doc.get('metadata', {})
                    }
                },
                processor=processor,
                embedding_generator=embedding_generator,
                vector_store=vector_store,
                text_content=text_content
            )
            
            if doc_info:
                total_chunks += chunks_count
                ingested_docs.append(doc_info)
                logger.info(f"Ingested {doc['title']}: {chunks_count} chunks")
            else:
                logger.warning(f"Document '{doc.get('title', 'unknown')}' produced 0 chunks, not stored")
                processing_errors.append(f"No chunks: {doc.get('title', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error ingesting {doc.get('title', 'unknown')}: {e}")
            processing_errors.append(f"Error processing {doc.get('title', 'unknown')}: {str(e)}")
            continue
    
    # Build response with diagnostic information
    response = {
        "message": f"Successfully ingested {len(ingested_docs)} documents from Zammad",
        "ingested": len(ingested_docs),
        "total_chunks": total_chunks,
        "documents": ingested_docs,
        "source": "zammad",
        "diagnostics": {
            "tickets_retrieved": tickets_count,
            "kb_entries_retrieved": kb_entries_count,
            "total_documents_retrieved": len(zammad_documents),
            "sync_tickets_enabled": settings.zammad_sync_tickets,
            "sync_kb_enabled": settings.zammad_sync_kb,
            "sync_attachments_enabled": settings.zammad_sync_attachments,
            "kb_id_used": getattr(settings, 'zammad_kb_id', None),
            "processing_errors": processing_errors if processing_errors else None
        }
    }
    
    return response


async def _ingest_local(
    vector_store: VectorStore,
    embedding_generator: EmbeddingGenerator,
    processor: DocumentProcessor
) -> Dict:
    """Ingest documents from local folder."""
    loader = DocumentLoader()
    documents = loader.scan_documents()
    
    if not documents:
        return {"message": "No documents found to ingest", "ingested": 0, "source": "local"}
    
    total_chunks = 0
    ingested_docs = []
    
    for doc_info in documents:
        try:
            chunks = processor.process_document(doc_info["path"])
            
            if not chunks:
                logger.warning(f"No chunks extracted from {doc_info['filename']}")
                continue
            
            # Generate embeddings and store
            texts = [chunk["text"] for chunk in chunks]
            embeddings = embedding_generator.generate_embeddings(texts)
            metadatas = [chunk["metadata"] for chunk in chunks]
            
            vector_store.add_documents(texts, embeddings, metadatas)
            
            total_chunks += len(chunks)
            ingested_docs.append({
                "filename": doc_info["filename"],
                "chunks": len(chunks)
            })
            
            logger.info(f"Ingested {doc_info['filename']}: {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error ingesting {doc_info['filename']}: {e}")
            continue
    
    return {
        "message": f"Successfully ingested {len(ingested_docs)} documents",
        "ingested": len(ingested_docs),
        "total_chunks": total_chunks,
        "documents": ingested_docs,
        "source": "local"
    }


@router.post("/ingest")
async def ingest_documents(
    source: Optional[str] = "local",
    api_key: str = Depends(verify_api_key)
):
    """Ingest documents from local folder or Zammad.
    
    Query parameters:
        source: Source to ingest from - "local" (default) or "zammad"
    """
    try:
        vector_store, embedding_generator = get_components()
        processor = DocumentProcessor()
        
        if source == "zammad":
            return await _ingest_zammad(vector_store, embedding_generator, processor)
        else:
            return await _ingest_local(vector_store, embedding_generator, processor)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error ingesting documents: {str(e)}")


@router.get("", response_model=DocumentListResponse)
async def list_documents(api_key: str = Depends(verify_api_key)):
    """List all ingested documents."""
    try:
        vector_store, _ = get_components()
        
        doc_ids = vector_store.list_documents()
        
        documents = []
        seen_filenames = set()
        
        for doc_id in doc_ids:
            try:
                # Get one chunk from this document to extract metadata
                chunks_count = vector_store.get_document_count(doc_id)
                metadata = vector_store.get_document_metadata(doc_id)
                
                if metadata:
                    filename = metadata.get("filename", "Unknown")
                    
                    # Avoid duplicates
                    if filename not in seen_filenames:
                        seen_filenames.add(filename)
                        file_type = metadata.get("file_type", "unknown")
                        if not file_type or file_type == "unknown":
                            file_type = _infer_file_type(filename)
                        
                        documents.append(DocumentInfo(
                            id=doc_id,
                            filename=filename,
                            file_type=file_type,
                            chunks_count=chunks_count,
                            ingested_at=datetime.now()  # We don't store this, using current time
                        ))
            except Exception as e:
                logger.warning(f"Error getting info for doc {doc_id}: {e}")
                continue
        
        return DocumentListResponse(
            documents=documents,
            total=len(documents)
        )
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


@router.post("/sync/zammad")
async def sync_zammad(api_key: str = Depends(verify_api_key)):
    """Sync documents from Zammad (tickets and knowledge base)."""
    # Call the main ingest function with zammad source
    return await ingest_documents(source="zammad", api_key=api_key)


@router.get("/zammad/test")
async def test_zammad_connection(api_key: str = Depends(verify_api_key)):
    """Test Zammad connection and configuration."""
    try:
        # Check configuration
        config_status = {
            "zammad_base_url": settings.zammad_base_url if settings.zammad_base_url else "NOT SET",
            "zammad_api_token": "SET" if settings.zammad_api_token else "NOT SET",
            "zammad_sync_tickets": settings.zammad_sync_tickets,
            "zammad_sync_kb": settings.zammad_sync_kb,
            "zammad_kb_id": getattr(settings, 'zammad_kb_id', None),
            "zammad_ticket_limit": settings.zammad_ticket_limit
        }
        
        if not settings.zammad_base_url or not settings.zammad_api_token:
            return {
                "status": "error",
                "message": "Zammad configuration missing",
                "config": config_status
            }
        
        # Try to connect
        from app.rag.zammad_client import ZammadClient
        client = ZammadClient(settings.zammad_base_url, settings.zammad_api_token)
        
        # Test knowledge bases endpoint
        kb_result = client._make_request('GET', 'knowledge_bases')
        kb_test = {
            "endpoint": f"{settings.zammad_base_url}/api/v1/knowledge_bases",
            "status": "success" if kb_result is not None else "failed",
            "result_type": type(kb_result).__name__ if kb_result else None,
            "result_keys": list(kb_result.keys()) if isinstance(kb_result, dict) else None,
            "result_preview": str(kb_result)[:200] if kb_result else None
        }
        
        # Test specific KB endpoint
        kb_id = getattr(settings, 'zammad_kb_id', 1)
        kb_entries_result = client._make_request('GET', f'knowledge_bases/{kb_id}/answers')
        kb_entries_test = {
            "endpoint": f"{settings.zammad_base_url}/api/v1/knowledge_bases/{kb_id}/answers",
            "status": "success" if kb_entries_result is not None else "failed",
            "result_type": type(kb_entries_result).__name__ if kb_entries_result else None,
            "result_keys": list(kb_entries_result.keys()) if isinstance(kb_entries_result, dict) else None,
            "entries_count": len(kb_entries_result) if isinstance(kb_entries_result, list) else None,
            "result_preview": str(kb_entries_result)[:500] if kb_entries_result else None
        }
        
        return {
            "status": "success",
            "config": config_status,
            "knowledge_bases_test": kb_test,
            "kb_entries_test": kb_entries_test
        }
        
    except Exception as e:
        logger.error(f"Error testing Zammad connection: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "config": config_status if 'config_status' in locals() else {}
        }

@router.post("/search")
async def search_documents(
    request: SearchRequest,
    api_key: str = Depends(verify_api_key)
):
    """Test retrieval/embedding by searching for similar documents."""
    try:
        vector_store, embedding_generator = get_components()
        
        # Generate embedding for the query
        query_embedding = embedding_generator.generate_embedding(request.query)
        
        # Search for similar documents
        top_k = request.top_k or settings.top_k
        results = vector_store.search(query_embedding, top_k=top_k)
        
        # Format results
        formatted_results = []
        for i, result in enumerate(results, 1):
            metadata = result.get("metadata", {})
            formatted_results.append({
                "rank": i,
                "score": result.get("score", 0.0),
                "text": result.get("text", "")[:200] + "..." if len(result.get("text", "")) > 200 else result.get("text", ""),
                "filename": metadata.get("filename", "Unknown"),
                "doc_id": metadata.get("doc_id", "Unknown"),
                "chunk_index": metadata.get("chunk_index", None)
            })
        
        return {
            "query": request.query,
            "total_results": len(formatted_results),
            "results": formatted_results,
            "embedding_dimension": len(query_embedding)
        }
        
    except Exception as e:
        logger.error(f"Error during search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during search: {str(e)}")

# Delete a document and all its chunks
@router.delete("/{doc_id}")
async def delete_document(doc_id: str, api_key: str = Depends(verify_api_key)):
    """Delete a document and all its chunks."""
    try:
        vector_store, _ = get_components()
        vector_store.delete_document(doc_id)
        
        return {"message": f"Document {doc_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")
