"""HTTP API for the Barrister RAG application."""

from contextlib import asynccontextmanager
from functools import lru_cache
import logging
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .config import settings

logger = logging.getLogger(__name__)


# Loading retrieval models and the index is expensive; cache one pipeline per process.
_pipeline_lock = Lock()


@lru_cache(maxsize=1)
def _get_pipeline():
    """Return the process-wide pipeline, importing it only once.

    Importing the pipeline initializes its retriever and model dependencies, so
    caching avoids repeating that expensive setup for later chat requests.
    """
    with _pipeline_lock:
        from rag.src.pipeline import rag_pipeline

        return rag_pipeline


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize the pipeline before serving requests.

    Eager loading keeps the first chat request from paying the initialization
    cost and surfaces missing configuration or RAG data during startup.
    """
    logger.info("Loading Barrister RAG pipeline...")
    await run_in_threadpool(_get_pipeline)
    logger.info("Barrister RAG pipeline loaded.")
    yield


app = FastAPI(
    title="Barrister API",
    description="Question answering over selected Nigerian legal sources.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    # These local Streamlit origins are defined explicitly in config.py.
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ChatMessage(BaseModel):
    """One earlier user or assistant message in the conversation."""

    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    """Payload accepted by the chat endpoint."""

    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=30)


class ChatResponse(BaseModel):
    """Answer returned to the frontend."""

    answer: str


@app.get("/")
def root():
    """Return basic API links for a quick check that the service is running."""
    return {"name": "Barrister API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    """Report that the API process is responsive."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Generate a legal information answer for the frontend chat UI."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be blank.")

    history = [
        {"role": message.role, "content": message.content.strip()}
        for message in request.history
        if message.content.strip()
    ]

    try:
        # RAG initialization and invocation are synchronous, so run them off the event loop.
        pipeline = await run_in_threadpool(_get_pipeline)
        answer = await run_in_threadpool(
            pipeline.invoke,
            {"question": question, "history": history},
        )
    except Exception as exc:
        logger.exception("Barrister failed to answer a chat request")
        raise HTTPException(
            status_code=503,
            detail="Barrister could not process the question. Check the API key and RAG data, then try again.",
        ) from exc

    return ChatResponse(answer=str(answer).strip())
