"""LangChain RAG chain implementation with optimizations."""
from typing import List, Dict, Optional
from collections import defaultdict
from datetime import datetime, timedelta
from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
try:
    from pydantic import ConfigDict, model_validator
    PYDANTIC_V2 = True
except ImportError:
    try:
        from pydantic import model_validator
        PYDANTIC_V2 = False
    except ImportError:
        PYDANTIC_V2 = False
        model_validator = None
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from functools import wraps
import time
from app.config import settings
from app.rag.vector_store import VectorStore
from app.utils.logger import logger


def retry_on_failure(max_retries=3, delay=1.0):
    """Decorator to retry on failure."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed after {max_retries} attempts: {e}")
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying in {delay}s...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


class RAGChain:
    """Optimized RAG chain using LangChain, Qdrant, and Mistral."""
    
    def __init__(self, vector_store: VectorStore):
        """Initialize the RAG chain with the vector store."""
        self.vector_store = vector_store
        self.llm = None
        self.embeddings = None
        self.retriever = None
        self.base_chain = None
        
        # Per-conversation memory management
        self.memories = defaultdict(lambda: ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            k=settings.max_memory_length  # Keep the last N turns
        ))
        self.memory_timestamps = {}  # For TTL
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize LLM and embeddings models."""
        try:
            # Initialize embeddings
            logger.info("Initializing HuggingFace embeddings...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
            )
            
            # Initialize LLM
            logger.info(f"Loading LLM model: {settings.llm_model}")
            tokenizer = AutoTokenizer.from_pretrained(
                settings.llm_model,
                token=settings.hf_api_token
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                settings.llm_model,
                token=settings.hf_api_token,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=settings.max_tokens,
                temperature=settings.temperature,
                return_full_text=False
            )
            
            self.llm = HuggingFacePipeline(pipeline=pipe)
            
            # Create retriever from vector store
            from langchain_community.vectorstores import Qdrant
            from langchain.schema import Document
            
            # Create a wrapper retriever
            self.retriever = QdrantRetrieverWrapper(self.vector_store, self.embeddings)
            
            logger.info("RAG chain initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG chain: {e}")
            raise
    
    def _get_or_create_chain(self, conversation_id: str):
        """Get or create a chain with memory for a specific conversation."""
        # Clean up expired memories
        self._cleanup_expired_memories()
        
        # Get or create memory for this conversation
        memory = self.memories[conversation_id]
        self.memory_timestamps[conversation_id] = datetime.now()
        
        # Improved prompt template in English
        prompt_template = """You are a helpful AI assistant for a company.

TASK: Answer questions precisely and helpfully based on the provided context.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- If the question concerns an error or problem:
  1. Analyze the error systematically
  2. Explain the likely cause
  3. Provide concrete solution steps in numbered form

- If the question concerns processes or policies:
  1. Explain the process step by step
  2. Refer to relevant documentation
  3. Provide practical examples when possible

- If the answer is not contained in the context:
  Honestly state that the information is not available and provide general guidance if possible.

- Structure your answer clearly and use bullet points or numbered lists when appropriate.

ANSWER:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=True,
            verbose=False  # Reduce verbosity for production
        )
        
        return chain
    
    def _cleanup_expired_memories(self):
        """Clean up expired memories according to TTL."""
        if not settings.enable_per_conversation_memory:
            return
            
        now = datetime.now()
        expired_conversations = [
            conv_id for conv_id, timestamp in self.memory_timestamps.items()
            if now - timestamp > timedelta(hours=settings.memory_ttl_hours)
        ]
        
        for conv_id in expired_conversations:
            logger.info(f"Cleaning up expired memory for conversation: {conv_id}")
            if conv_id in self.memories:
                self.memories[conv_id].clear()
                del self.memories[conv_id]
            if conv_id in self.memory_timestamps:
                del self.memory_timestamps[conv_id]
    
    @retry_on_failure(max_retries=settings.max_retries, delay=settings.retry_delay)
    def query(self, question: str, conversation_id: Optional[str] = None) -> Dict:
        """Query the RAG chain with a question."""
        try:
            # Use "default" if no conversation_id
            conv_id = conversation_id or "default"
            
            # Get or create chain for this conversation
            if settings.enable_per_conversation_memory:
                chain = self._get_or_create_chain(conv_id)
            else:
                # Fallback: use a single chain (original behavior)
                if self.base_chain is None:
                    memory = ConversationBufferWindowMemory(
                        memory_key="chat_history",
                        return_messages=True,
                        output_key="answer",
                        k=settings.max_memory_length
                    )
                    prompt_template = """You are a helpful AI assistant for a company.

CONTEXT: {context}
QUESTION: {question}
ANSWER:"""
                    PROMPT = PromptTemplate(
                        template=prompt_template,
                        input_variables=["context", "question"]
                    )
                    self.base_chain = ConversationalRetrievalChain.from_llm(
                        llm=self.llm,
                        retriever=self.retriever,
                        memory=memory,
                        combine_docs_chain_kwargs={"prompt": PROMPT},
                        return_source_documents=True,
                        verbose=False
                    )
                chain = self.base_chain
            
            result = chain.invoke({"question": question})
            
            # Extract sources with scores
            sources = []
            source_scores = {}
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    if hasattr(doc, 'metadata'):
                        filename = doc.metadata.get("filename", "Unknown")
                        sources.append(filename)
                        # Store score if available
                        if hasattr(doc, 'metadata') and 'score' in doc.metadata:
                            source_scores[filename] = doc.metadata['score']
            
            return {
                "answer": result.get("answer", ""),
                "sources": list(set(sources)),
                "source_scores": source_scores,  # Similarity scores
                "conversation_id": conv_id
            }
        except Exception as e:
            logger.error(f"Error querying the RAG chain: {e}")
            raise
    
    def clear_memory(self, conversation_id: Optional[str] = None):
        """Clear memory of a specific conversation or all."""
        if conversation_id:
            if conversation_id in self.memories:
                self.memories[conversation_id].clear()
                logger.info(f"Memory cleared for conversation: {conversation_id}")
        else:
            # Clear all memories
            for memory in self.memories.values():
                memory.clear()
            self.memories.clear()
            self.memory_timestamps.clear()
            logger.info("All memories have been cleared")


class QdrantRetrieverWrapper(BaseRetriever):  # Hérite de BaseRetriever
    """Wrapper to make Qdrant vector store compatible with LangChain retriever interface."""
    
    # Configuration Pydantic pour permettre les types arbitraires
    # Pour Pydantic v2
    try:
        model_config = ConfigDict(arbitrary_types_allowed=True)
    except:
        pass
    
    # Pour Pydantic v1
    class Config:
        arbitrary_types_allowed = True
    
    if model_validator:
        @model_validator(mode='before')
        @classmethod
        def validate_model(cls, values):
            """Ignore validation for non-serializable attributes."""
            return values
    
    def __init__(self, vector_store: VectorStore, embeddings):
        """Initialize retriever wrapper."""
        super().__init__()  # Appel au constructeur parent
        self.vector_store = vector_store
        self.embeddings = embeddings
    
    def _get_relevant_documents(self, query: str) -> List:
        """Retrieve relevant documents for a query with optimization."""
        try:
            from langchain_core.documents import Document
        except ImportError:
            # Fallback for older LangChain versions
            try:
                from langchain.schema import Document
            except ImportError:
                from langchain.documents import Document
        
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Search with fetch_k to get more candidates
        results = self.vector_store.search(
            query_embedding,
            top_k=settings.retrieval_fetch_k  # Retrieve more candidates
        )
        
        # Filter by score threshold
        filtered_results = [
            result for result in results
            if result.get("score", 0) >= settings.retrieval_score_threshold
        ]
        
        # Limit to final top_k
        filtered_results = filtered_results[:settings.top_k]
        
        # Convert to LangChain documents with enriched metadata
        documents = []
        for result in filtered_results:
            metadata = result.get("metadata", {})
            metadata["score"] = result.get("score", 0)  # Add score
            doc = Document(
                page_content=result["text"],
                metadata=metadata
            )
            documents.append(doc)
        
        logger.debug(f"Retrieved {len(documents)} documents (threshold: {settings.retrieval_score_threshold})")
        return documents
    
    def get_relevant_documents(self, query: str) -> List:
        """Public method for LangChain compatibility."""
        return self._get_relevant_documents(query)
    
    def invoke(self, query: str, **kwargs) -> List:
        """Invoke method for LangChain compatibility."""
        return self._get_relevant_documents(query)
    
    async def aget_relevant_documents(self, query: str) -> List:
        """Async version for LangChain compatibility."""
        return self._get_relevant_documents(query)