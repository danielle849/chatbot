"""LangChain RAG chain implementation with optimizations."""
from typing import List, Dict, Optional
from collections import defaultdict
from datetime import datetime, timedelta
from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from functools import wraps
import time
from app.config import settings
from app.rag.vector_store import VectorStore
from app.utils.logger import logger


def retry_on_failure(max_retries=3, delay=1.0):
    """Decorator pour réessayer en cas d'échec."""
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
    """RAG chain optimisée utilisant LangChain, Qdrant, et Mistral."""
    
    def __init__(self, vector_store: VectorStore):
        """Initialise la chaîne RAG avec le vector store."""
        self.vector_store = vector_store
        self.llm = None
        self.embeddings = None
        self.retriever = None
        self.base_chain = None
        
        # Gestion de la mémoire par conversation
        self.memories = defaultdict(lambda: ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            k=settings.max_memory_length  # Garde les N derniers tours
        ))
        self.memory_timestamps = {}  # Pour TTL
        
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
            
            model = AutoModelForCausalLM.from_pretrained(
                settings.llm_model,
                **model_kwargs
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
            
            logger.info("Chaîne RAG initialisée avec succès")
            
        except Exception as e:
            logger.error(f"Error initializing RAG chain: {e}")
            raise
    
    def _get_or_create_chain(self, conversation_id: str):
        """Obtient ou crée une chaîne avec mémoire pour une conversation spécifique."""
        # Nettoyer les mémoires expirées
        self._cleanup_expired_memories()
        
        # Obtenir ou créer la mémoire pour cette conversation
        memory = self.memories[conversation_id]
        self.memory_timestamps[conversation_id] = datetime.now()
        
        # Template de prompt amélioré en allemand
        prompt_template = """Du bist ein hilfreicher KI-Assistent für ein Unternehmen.

AUFGABE: Beantworte Fragen präzise und hilfreich basierend auf dem bereitgestellten Kontext.

KONTEXT:
{context}

FRAGE: {question}

ANWEISUNGEN:
- Wenn die Frage einen Fehler oder ein Problem betrifft:
  1. Analysiere den Fehler systematisch
  2. Erkläre die wahrscheinliche Ursache
  3. Gib konkrete Lösungsschritte in nummerierter Form

- Wenn die Frage Prozesse oder Richtlinien betrifft:
  1. Erkläre den Prozess Schritt für Schritt
  2. Verweise auf relevante Dokumentation
  3. Gib praktische Beispiele wenn möglich

- Wenn die Antwort nicht im Kontext enthalten ist:
  Sage ehrlich, dass die Information nicht verfügbar ist und gib wenn möglich allgemeine Hinweise.

- Strukturiere deine Antwort klar und verwende Aufzählungen oder nummerierte Listen wenn angebracht.

ANTWORT:"""
        
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
            verbose=False  # Réduire le verbosity pour la production
        )
        
        return chain
    
    def _cleanup_expired_memories(self):
        """Nettoie les mémoires expirées selon TTL."""
        if not settings.enable_per_conversation_memory:
            return
            
        now = datetime.now()
        expired_conversations = [
            conv_id for conv_id, timestamp in self.memory_timestamps.items()
            if now - timestamp > timedelta(hours=settings.memory_ttl_hours)
        ]
        
        for conv_id in expired_conversations:
            logger.info(f"Nettoyage de la mémoire expirée pour conversation: {conv_id}")
            if conv_id in self.memories:
                self.memories[conv_id].clear()
                del self.memories[conv_id]
            if conv_id in self.memory_timestamps:
                del self.memory_timestamps[conv_id]
    
    @retry_on_failure(max_retries=settings.max_retries, delay=settings.retry_delay)
    def query(self, question: str, conversation_id: Optional[str] = None) -> Dict:
        """Interroge la chaîne RAG avec une question."""
        try:
            # Utiliser "default" si pas de conversation_id
            conv_id = conversation_id or "default"
            
            # Obtenir ou créer la chaîne pour cette conversation
            if settings.enable_per_conversation_memory:
                chain = self._get_or_create_chain(conv_id)
            else:
                # Fallback: utiliser une chaîne unique (comportement original)
                if self.base_chain is None:
                    memory = ConversationBufferWindowMemory(
                        memory_key="chat_history",
                        return_messages=True,
                        output_key="answer",
                        k=settings.max_memory_length
                    )
                    prompt_template = """Du bist ein hilfreicher KI-Assistent für ein Unternehmen.

KONTEXT: {context}
FRAGE: {question}
ANTWORT:"""
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
            
            # Extraire les sources avec scores
            sources = []
            source_scores = {}
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    if hasattr(doc, 'metadata'):
                        filename = doc.metadata.get("filename", "Unknown")
                        sources.append(filename)
                        # Stocker le score si disponible
                        if hasattr(doc, 'metadata') and 'score' in doc.metadata:
                            source_scores[filename] = doc.metadata['score']
            
            return {
                "answer": result.get("answer", ""),
                "sources": list(set(sources)),
                "source_scores": source_scores,  # Scores de similarité
                "conversation_id": conv_id
            }
        except Exception as e:
            logger.error(f"Erreur lors de l'interrogation de la chaîne RAG: {e}")
            raise
    
    def clear_memory(self, conversation_id: Optional[str] = None):
        """Efface la mémoire d'une conversation spécifique ou toutes."""
        if conversation_id:
            if conversation_id in self.memories:
                self.memories[conversation_id].clear()
                logger.info(f"Mémoire effacée pour conversation: {conversation_id}")
        else:
            # Effacer toutes les mémoires
            for memory in self.memories.values():
                memory.clear()
            self.memories.clear()
            self.memory_timestamps.clear()
            logger.info("Toutes les mémoires ont été effacées")


class QdrantRetrieverWrapper:
    """Wrapper to make Qdrant vector store compatible with LangChain retriever interface."""
    
    def __init__(self, vector_store: VectorStore, embeddings):
        """Initialize retriever wrapper."""
        self.vector_store = vector_store
        self.embeddings = embeddings
    
    def get_relevant_documents(self, query: str) -> List:
        """Récupère les documents pertinents pour une requête avec optimisation."""
        try:
            from langchain_core.documents import Document
        except ImportError:
            # Fallback for older LangChain versions
            try:
                from langchain.schema import Document
            except ImportError:
                from langchain.documents import Document
        
        # Générer l'embedding de la requête
        query_embedding = self.embeddings.embed_query(query)
        
        # Rechercher avec fetch_k pour avoir plus de candidats
        results = self.vector_store.search(
            query_embedding,
            top_k=settings.retrieval_fetch_k  # Récupérer plus de candidats
        )
        
        # Filtrer par seuil de score
        filtered_results = [
            result for result in results
            if result.get("score", 0) >= settings.retrieval_score_threshold
        ]
        
        # Limiter à top_k final
        filtered_results = filtered_results[:settings.top_k]
        
        # Convertir en documents LangChain avec métadonnées enrichies
        documents = []
        for result in filtered_results:
            metadata = result.get("metadata", {})
            metadata["score"] = result.get("score", 0)  # Ajouter le score
            doc = Document(
                page_content=result["text"],
                metadata=metadata
            )
            documents.append(doc)
        
        logger.debug(f"Récupéré {len(documents)} documents (seuil: {settings.retrieval_score_threshold})")
        return documents
    
    def invoke(self, query: str, **kwargs) -> List:
        """Invoke method for LangChain compatibility."""
        return self.get_relevant_documents(query)
    
    def get_relevant_documents_async(self, query: str) -> List:
        """Async version for LangChain compatibility."""
        return self.get_relevant_documents(query)
