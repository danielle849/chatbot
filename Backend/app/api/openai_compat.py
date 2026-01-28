# app/api/openai_compat.py
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import time, uuid, json, asyncio
from app.auth import verify_api_key
from app.utils.logger import logger
from app.rag.chain import RAGChain
from app.rag.vector_store import VectorStore
from app.rag.embeddings import EmbeddingGenerator

from app.openai_models import (
    OpenAIModelList,
    OpenAIModel,
    OpenAIChatCompletionsRequest,
    OpenAIChatCompletionsResponse,
    OpenAIChatMessage,
    OpenAIChatChoice,
    OpenAIChatCompletionsChunk,
    OpenAIChatChunkChoice,
    OpenAIChatDelta,
)

router = APIRouter(prefix="/v1", tags=["openai-compat"])

_rag_chain: RAGChain = None
_vector_store: VectorStore = None
_embedding_generator: EmbeddingGenerator = None


def get_rag_chain() -> RAGChain:
    global _rag_chain, _vector_store, _embedding_generator

    if _rag_chain is None:
        try:
            _embedding_generator = EmbeddingGenerator()
            _vector_store = VectorStore()
            _vector_store.create_collection(_embedding_generator.dimension)
            _rag_chain = RAGChain(_vector_store)
        except Exception as e:
            logger.error(f"Error initializing RAG chain: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize RAG chain: {str(e)}",
            )

    return _rag_chain


def extract_last_user_message(messages: list[OpenAIChatMessage]) -> str:
    # Take the last non-empty user message
    for m in reversed(messages):
        if m.role == "user" and m.content and m.content.strip():
            return m.content.strip()
    return ""


def get_conversation_id(
    req: OpenAIChatCompletionsRequest,
    request: Request,
) -> str | None:
    """
    OpenAI does not have a standard conversation_id.

    Options:
    1) Custom header: X-Conversation-Id
    2) req.user (string)
    3) None => rag_chain creates a new conversation_id
    """
    hdr = request.headers.get("x-conversation-id")
    if hdr:
        return hdr

    if req.user:
        return req.user

    return None


@router.get("/models", response_model=OpenAIModelList)
def list_models(api_key: str = Depends(verify_api_key)):
    now = int(time.time())
    return OpenAIModelList(
        data=[
            OpenAIModel(
                id="rag-fastapi",
                created=now,
                owned_by="local",)]
    )


@router.post("/chat/completions")
async def chat_completions(
    payload: OpenAIChatCompletionsRequest,
    request: Request,
    api_key: str = Depends(verify_api_key),
):
    created = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"

    rag_chain = get_rag_chain()
    user_text = extract_last_user_message(payload.messages)

    if not user_text:
        raise HTTPException(
            status_code=400,
            detail="No user message found in 'messages'.",
        )

    conversation_id = get_conversation_id(payload, request)
    loop = asyncio.get_event_loop()

    # STREAMING MODE
    if payload.stream:

        async def sse_gen():
            try:
                # Initial empty chunk
                first = OpenAIChatCompletionsChunk(
                    id=completion_id,
                    created=created,
                    model=payload.model,
                    choices=[
                        OpenAIChatChunkChoice(
                            index=0,
                            delta=OpenAIChatDelta(),
                        )],
                )
                yield f"data: {first.model_dump_json()}\n\n"

                result = await loop.run_in_executor(
                    None,
                    rag_chain.query,
                    user_text,
                    conversation_id,
                )
                answer = result["answer"]

                # Stream word by word
                for word in answer.split():
                    chunk = OpenAIChatCompletionsChunk(
                        id=completion_id,
                        created=created,
                        model=payload.model,
                        choices=[
                            OpenAIChatChunkChoice(
                                index=0,
                                delta=OpenAIChatDelta(content=word + " "),
                                finish_reason=None,
                            )],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

                # Final chunk
                last = OpenAIChatCompletionsChunk(
                    id=completion_id,
                    created=created,
                    model=payload.model,
                    choices=[
                        OpenAIChatChunkChoice(
                            index=0,
                            delta=OpenAIChatDelta(),
                            finish_reason="stop",
                        )
                    ],
                )
                yield f"data: {last.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                err = {"error": str(e)}
                yield f"data: {json.dumps(err)}\n\n"

        return StreamingResponse(
            sse_gen(),
            media_type="text/event-stream",
        )

    # NON-STREAM MODE
    try:
        result = await loop.run_in_executor(
            None,
            rag_chain.query, user_text, conversation_id,
        )

        sources = result.get("sources", [])
        answer = result["answer"]

        if sources:
            answer += "\n\nSources:\n- " + "\n- ".join(sources)

        response = OpenAIChatCompletionsResponse(
            id=completion_id,
            created=created,
            model=payload.model or "rag-fastapi",
            choices=[
                OpenAIChatChoice(
                    index=0,
                    message=OpenAIChatMessage(
                        role="assistant",
                        content=answer,
                    ),
                    finish_reason="stop",
                )
            ],
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        )

        return JSONResponse(response.model_dump())

    except Exception as e:
        logger.exception(f"Error processing chat completions: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
