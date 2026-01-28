# app/openai_models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Any, Dict

# ---- /v1/models ----
class OpenAIModel(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local"

class OpenAIModelList(BaseModel):
    object: str = "list"
    data: List[OpenAIModel]

# ---- /v1/chat/completions ----
class OpenAIChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    name: Optional[str] = None

class OpenAIChatCompletionsRequest(BaseModel):
    model: str
    messages: List[OpenAIChatMessage]
    stream: Optional[bool] = False

    # optional
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    user: Optional[str] = None

class OpenAIChatChoice(BaseModel):
    index: int
    message: OpenAIChatMessage
    finish_reason: Optional[str] = "stop"

class OpenAIChatCompletionsResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[OpenAIChatChoice]
    usage: Optional[Dict[str, int]] = None

# ---- Streaming chunks ----
class OpenAIChatDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class OpenAIChatChunkChoice(BaseModel):
    index: int
    delta: OpenAIChatDelta
    finish_reason: Optional[str] = None

class OpenAIChatCompletionsChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[OpenAIChatChunkChoice]
