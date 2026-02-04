"""LangChain RAG chain implementation with optimizations."""
from typing import List, Dict, Optional, Any
from collections import defaultdict
from datetime import datetime, timedelta
from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
#from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from pydantic import ConfigDict
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from functools import wraps
from app.utils.logger import logger
import time
from app.config import settings
from app.rag.vector_store import VectorStore


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

        # Improved prompt template in German
        prompt_template = """Du bist ein interner KI-Assistent für ein Unternehmen.

KONTEXT:
{context}

FRAGE:
{question}

REGELN (SEHR WICHTIG):
- Beantworte die Frage AUSSCHLIESSLICH mit Informationen aus dem KONTEXT.
- Verwende KEIN externes Wissen, KEINE Annahmen, KEINE Vermutungen, KEINE zusätzlichen FAQs..
- Wenn die Antwort im Kontext enthalten ist:
  - Gib eine kurze, präzise Antwort.
  - ZITIERE anschließend die exakte Textstelle aus dem Kontext, die die Antwort belegt.
- Wenn die Antwort NICHT im Kontext enthalten ist:
  - Antworte exakt mit: "Nicht im Kontext gefunden."
- Antworte NUR auf Deutsch.
- Erfinde keine Informationen.

ANTWORTFORMAT:
Antwort: <kurze Antwort>
Beleg: "<exaktes Zitat aus dem Kontext>"

ANTWORT:(auf Deutsch)
"""


        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            #memory=memory,
            chain_type_kwargs={"prompt": PROMPT,
                               "document_variable_name": "context",
            },
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
        conv_id = conversation_id or "default"

        if self.base_chain is None:

            prompt_template =  """Du bist ein hilfreicher KI-Assistent für ein Unternehmen.


WICHTIGE REGELN:
1. PRIORITÄT: Verwende zuerst die Informationen aus dem KONTEXT (Wissensdatenbank)
2. Wenn der KONTEXT relevante Informationen enthält, beantworte die Frage basierend darauf
3. Wenn der KONTEXT leer ist oder keine relevanten Informationen enthält, nutze dein Allgemeinwissen
4. Antworte IMMER auf Deutsch, auch wenn der KONTEXT auf Englisch ist

VERBOTEN:
- Kopiere NIEMALS die Formatierung "FRAGE:" oder "ANTWORT:" aus dem Kontext
- Kopiere NIEMALS Q&A-Paare aus dem Kontext
- Verwende KEINE Formatierung mit "FRAGE:" und "ANTWORT:" in deiner Antwort
- Antworte NICHT mit mehreren Fragen und Antworten

ERLAUBT:
- Extrahiere NUR die relevanten Informationen aus dem Kontext
- Formuliere die Antwort in deinen eigenen Worten
- Antworte direkt und natürlich auf die gestellte Frage
- Verwende einen fließenden, natürlichen Text


KONTEXT: {context}
FRAGE: {question}
ANTWORT(auf Deutsch):"""

            PROMPT = PromptTemplate(
                 template=prompt_template,
                 input_variables=["context", "question"]
            )
            self.base_chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT,
                                   "document_variable_name": "context",
                },
                verbose=False,
            )

        result = self.base_chain.invoke({"question": question})

        sources = []
        source_scores = {}
        min_source_score = 0.4

        for doc in result.get("source_documents", []) or []:
            meta = doc.metadata or {}
            title = meta.get("title")
            filename = meta.get("filename", "Unknown")
            #chunk_index = meta.get("chunk_index")
            score = meta.get("score", 0.0)

            # Only include sources with good similarity scores
            if score is None or score < min_source_score:
                continue

            parts = []
            if title:
                parts.append(title)
            parts.append(filename)
            #if chunk_index is not None:
             #   parts.append(f"Chunk {chunk_index}")

            label = " | ".join(parts)

            sources.append(label)

            if score is not None:
                try:
                    source_scores[label] = round(float(score), 4)

                except Exception:
                    pass
        try:
            memory = self.memories[conv_id]
            self.memory_timestamps[conv_id] = datetime.now()
            memory.save_context({"question": question}, {"answer": result.get("answer", "")})
        except Exception as e:
            logger.warning(f"Memory save failed for conversation {conv_id}: {e}")

        return {
            "answer": result.get("answer", ""),
            "sources": list(dict.fromkeys(sources)),
            "source_scores": source_scores,
            "conversation_id": conv_id
        }
    #except Exception as e:
    #logger.error(f"Error querying the RAG chain: {e}")
    #raise

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


class QdrantRetrieverWrapper(BaseRetriever):
    """Wrapper to make Qdrant vector store compatible with LangChain retriever interface."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    vector_store: Any
    embeddings: Any

    def __init__(self, vector_store: VectorStore, embeddings):
        """Initialize retriever wrapper."""
        super().__init__(vector_store = vector_store, embeddings = embeddings)


    def _get_relevant_documents(self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None) -> List[Document]:
        query_embedding = self.embeddings.embed_query(query)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=settings.retrieval_fetch_k,
            score_threshold=settings.retrieval_score_threshold ,
        )
        results = results[:settings.top_k]
        logger.info("Top scores: %s", [round(r["score"], 3) for r in results[:5]])

        docs: List[Document] = []
        for r in results:
            metadata = r.get("metadata", {})
            metadata["score"] = r.get("score", 0)
            text = (r.get("text") or "")[:2000]
            docs.append(Document(page_content=text, metadata=metadata))

        return docs

    async def _aget_relevant_documents(self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None) -> List[Document]:
        # simple fallback async
         return self._get_relevant_documents(query)

