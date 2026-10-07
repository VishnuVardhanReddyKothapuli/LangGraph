"""Single-user local application. Run one worker when using embedded Qdrant."""

import hashlib
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import RLock
from typing import Literal
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from folio.rag import (MAX_FILE_BYTES, build_prompt, clip, generate, open_store,
                       provider_names, read_document, split_documents)

load_dotenv()
log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
# ponytail: one shared local library; add authentication and per-user collections before multi-user hosting.
# ponytail: serialize embedded storage access; use a Qdrant server for concurrent workers.
lock = RLock()


@asynccontextmanager
async def lifespan(app):
    url = os.getenv("QDRANT_URL")
    app.state.client = QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY") or None, timeout=30) if url else QdrantClient(
        path=os.getenv("QDRANT_PATH", "./data/qdrant"))
    app.state.collection = os.getenv("QDRANT_COLLECTION", "folio_documents")
    app.state.store = None  # First upload downloads models; the interface can open immediately.
    yield
    app.state.client.close()


app = FastAPI(title="Folio", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def get_store():
    if app.state.store is None:
        app.state.store = open_store(app.state.client, app.state.collection)
    return app.state.store


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    with lock:
        try:
            client, collection = app.state.client, app.state.collection
            count = client.count(collection, exact=True).count if client.collection_exists(collection) else 0
            return {"chunks": count, "providers": provider_names(), "context_window": 5000}
        except Exception as exc:
            log.warning("Status failed (%s)", type(exc).__name__)
            raise HTTPException(503, "Could not connect to Qdrant. Check server configuration.") from exc


@app.post("/api/documents")
def upload(file: UploadFile):
    try:
        content = file.file.read(MAX_FILE_BYTES + 1)
        documents = read_document(file.filename or "", content)
        chunks = split_documents(documents)
        digest = hashlib.sha256(content).hexdigest()
        # Deterministic IDs make a retry or re-upload of the same file idempotent.
        ids = [str(uuid5(NAMESPACE_URL, f"{documents[0].metadata['source']}:{digest}:{i}")) for i in range(len(chunks))]
        with lock:
            get_store().add_documents(chunks, ids=ids, batch_size=32)
        return {"name": documents[0].metadata["source"], "chunks": len(chunks)}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        log.warning("Indexing failed (%s)", type(exc).__name__)
        raise HTTPException(503, "Indexing failed. Check Qdrant and model-download connectivity. Retrying is safe.") from exc
    finally:
        file.file.close()


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=16000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    history: list[Turn] = Field(default_factory=list, max_length=20)


@app.post("/api/chat")
def chat(body: ChatRequest):
    try:
        history = [HumanMessage(t.content) if t.role == "user" else AIMessage(t.content) for t in body.history]
        build_prompt(body.question, [], history)  # Validate before retrieval or an API call.
        previous = next((t.content for t in reversed(body.history) if t.role == "user"), "")
        query = clip(previous, 150) + "\n" + body.question
        with lock:
            if not app.state.client.collection_exists(app.state.collection) or not app.state.client.count(app.state.collection).count:
                raise HTTPException(409, "Upload a document before asking a question.")
            hits = get_store().similarity_search(query, k=8)
        messages, sources, used = build_prompt(body.question, hits, history)
        if not sources:
            return {"answer": "I could not find evidence in the uploaded documents.", "sources": [],
                    "provider": "none", "input_tokens": used}
        answer, provider = generate(messages)
        return {"answer": answer, "sources": sources, "provider": provider, "input_tokens": used}
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        log.warning("Chat failed (%s)", type(exc).__name__)
        raise HTTPException(503, "Could not complete the request. Check Qdrant and your connection, then retry.") from exc
