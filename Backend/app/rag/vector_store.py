"""Vector store integration with Qdrant."""
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from app.config import settings
from app.utils.logger import logger


class VectorStore:
    """Manages vector storage and retrieval using Qdrant."""
    
    def __init__(self, collection_name: Optional[str] = None):
        """Initialize Qdrant client and collection."""
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        logger.info(f"Connected to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}")
    
    def create_collection(self, vector_size: int):
        """Create collection if it doesn't exist."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection already exists: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def add_documents(self, texts: List[str], embeddings: List[List[float]], metadatas: List[Dict]):
        """Add documents with embeddings to the vector store."""
        try:
            points = []
            for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
                point_id = self._generate_point_id(metadata.get("doc_id", ""), metadata.get("chunk_index", i))
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": text,
                        **metadata
                    }
                )
                points.append(point)
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Added {len(points)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def search(self, query_embedding: List[float], top_k: int = None, filter_dict: Optional[Dict] = None, score_threshold: Optional[float] = None) -> List[Dict]:
        """Search for similar documents with optional score threshold."""
        try:
            top_k = top_k or settings.top_k
            
            # Build filter if provided
            qdrant_filter = None
            if filter_dict:
                conditions = []
                for key, value in filter_dict.items():
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
                if conditions:
                    qdrant_filter = Filter(must=conditions)
            
            # Search with higher top_k if we want to filter by score
            search_limit = top_k
            if score_threshold is not None:
                search_limit = max(top_k, settings.retrieval_fetch_k)
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=search_limit,
                query_filter=qdrant_filter,
                score_threshold=score_threshold  # Qdrant score threshold
            )
            
            documents = []
            for result in results:
                documents.append({
                    "text": result.payload.get("text", ""),
                    "score": result.score,
                    "metadata": {k: v for k, v in result.payload.items() if k != "text"}
                })
            
            # Limit to final top_k if we retrieved more
            return documents[:top_k]
        except Exception as e:
            logger.error(f"Error searching in vector store: {e}")
            raise
    
    def delete_document(self, doc_id: str):
        """Delete all chunks of a document."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                    ]
                )
            )
            logger.info(f"Deleted document: {doc_id}")
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            raise
    
    def get_document_count(self, doc_id: Optional[str] = None) -> int:
        """Get total number of documents or chunks for a specific document."""
        try:
            if doc_id:
                # Count chunks for specific document
                results = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                        ]
                    ),
                    limit=10000
                )
                return len(results[0])
            else:
                # Count all documents
                collection_info = self.client.get_collection(self.collection_name)
                return collection_info.points_count
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0
    
    def list_documents(self) -> List[str]:
        """List all unique document IDs."""
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True
            )
            doc_ids = set()
            for point in results[0]:
                doc_id = point.payload.get("doc_id")
                if doc_id:
                    doc_ids.add(doc_id)
            return list(doc_ids)
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []
    
    def get_document_metadata(self, doc_id: str) -> Optional[Dict]:
        """Get metadata for a specific document by retrieving one chunk."""
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                    ]
                ),
                limit=1,
                with_payload=True
            )
            if results[0]:
                return results[0][0].payload
            return None
        except Exception as e:
            logger.error(f"Error getting document metadata: {e}")
            return None
    
    def _generate_point_id(self, doc_id: str, chunk_index: int) -> int:
        """Generate a unique point ID from doc_id and chunk_index."""
        import hashlib
        combined = f"{doc_id}_{chunk_index}"
        return int(hashlib.md5(combined.encode()).hexdigest()[:8], 16)
