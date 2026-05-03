"""
src/api/routes.py  –  FastAPI route handlers.

POST /query          – non-streaming RAG query
POST /query/stream   – SSE streaming (recommended — user sees tokens immediately)
POST /ingest         – index a new document
DELETE /cache        – invalidate semantic cache
GET  /health         – health check
GET  /collection     – Qdrant collection info
"""
from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from src.api.schemas import HealthResponse, IngestRequest, IngestResponse, QueryRequest, QueryResponse
from src.auth.rate_limiter import check_rate_limit
from src.cache.semantic_cache import cache_stats, get_cached_answer, invalidate_cache
from src.config import get_settings
from src.graph.workflow import get_workflow
from src.retrieval.qdrant_client import collection_info

router = APIRouter()
_settings = get_settings()

_SYSTEM_PROMPT = """\
You are a helpful enterprise customer support assistant.
Answer using ONLY the provided context. Cite sources as [doc_id].
If context lacks the answer, say: "I don't have enough information. Please contact support."
Be concise and professional."""


# ─── Non-streaming ────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest, user: str = Depends(check_rate_limit)) -> QueryResponse:
    graph = get_workflow()
    initial_state = {
        "query": req.query, "retry_count": 0, "cached": False, "expansion_done": False,
        "metadata_filters": {"product_category": req.product_category or None, "doc_type": None},
    }
    try:
        state = await graph.ainvoke(initial_state) or {}
        return QueryResponse(
            answer=state.get("answer") or "",
            citations=state.get("citations") or [],
            intent=state.get("intent"),
            cached=state.get("cached", False),
            query_variations=state.get("query_variations") or [],
            top_rerank_score=state.get("top_rerank_score", 0.0),
            expansion_triggered=state.get("expansion_done", False),
        )
    except Exception as exc:
        logger.error(f"Query error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ─── SSE Streaming ────────────────────────────────────────────────────────────

async def _stream_tokens(query: str, context_str: str, citations: list) -> AsyncIterator[str]:
    llm = ChatOpenAI(
        model=_settings.openai_model,
        temperature=_settings.openai_temperature,
        api_key=_settings.openai_api_key,
        max_tokens=_settings.openai_max_tokens,
        streaming=True,
    )
    user_msg = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer (cite [doc_id]):"
    try:
        async for chunk in llm.astream([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]):
            if chunk.content:
                yield f"data: {json.dumps({'event': 'token', 'data': chunk.content})}\n\n"

        for cit in citations:
            yield f"data: {json.dumps({'event': 'citation', 'data': cit})}\n\n"

        yield f"data: {json.dumps({'event': 'done', 'data': ''})}\n\n"

    except Exception as exc:
        logger.error(f"Streaming error: {exc}")
        yield f"data: {json.dumps({'event': 'error', 'data': str(exc)})}\n\n"


@router.post("/query/stream")
async def query_stream_endpoint(req: QueryRequest, user: str = Depends(check_rate_limit)) -> StreamingResponse:
    """
    SSE streaming endpoint.
    User sees first token in ~300ms even for slow queries.
    Events: token | citation | done | error
    """
    # Cache check first
    cached = await get_cached_answer(req.query)
    if cached:
        async def _stream_cached():
            chunk_size = 20
            for i in range(0, len(cached), chunk_size):
                yield f"data: {json.dumps({'event': 'token', 'data': cached[i:i+chunk_size]})}\n\n"
                await asyncio.sleep(0)
            yield f"data: {json.dumps({'event': 'done', 'data': '[cached]'})}\n\n"
        return StreamingResponse(_stream_cached(), media_type="text/event-stream",
                                  headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # Run pipeline through context builder
    graph = get_workflow()
    initial_state = {
        "query": req.query, "retry_count": 0, "cached": False, "expansion_done": False,
        "metadata_filters": {"product_category": req.product_category or None, "doc_type": None},
    }

    try:
        state = await graph.ainvoke(initial_state)
        if state is None:
            raise ValueError("Graph returned None state")
    except Exception as exc:
        _msg = str(exc)
        async def _err():
            yield f"data: {json.dumps({'event': 'error', 'data': _msg})}\n\n"
        return StreamingResponse(_err(), media_type="text/event-stream")

    context_str = (state or {}).get("context_str") or ""
    citations   = (state or {}).get("citations") or []

    # Chitchat / fallback — no streaming needed
    if not context_str and (state or {}).get("answer"):
        answer = state["answer"]
        async def _direct():
            yield f"data: {json.dumps({'event': 'token', 'data': answer})}\n\n"
            yield f"data: {json.dumps({'event': 'done', 'data': ''})}\n\n"
        return StreamingResponse(_direct(), media_type="text/event-stream")

    return StreamingResponse(
        _stream_tokens(req.query, context_str, citations),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── Ingest ───────────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(req: IngestRequest) -> IngestResponse:
    try:
        from src.ingestion.pipeline import IngestPipeline
        pipeline = IngestPipeline()
        loop = asyncio.get_event_loop()
        count = await loop.run_in_executor(
            None, lambda: pipeline.ingest_text(req.text, req.source, req.doc_type, req.product_category)
        )
        return IngestResponse(chunks_indexed=count, source=req.source)
    except Exception as exc:
        logger.error(f"Ingest error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ─── Cache ────────────────────────────────────────────────────────────────────

@router.delete("/cache")
async def invalidate_cache_endpoint() -> dict:
    deleted = await invalidate_cache()
    return {"deleted": deleted, "status": "ok"}


# ─── Health ───────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health_endpoint() -> HealthResponse:
    try:
        q_info = collection_info()
    except Exception as exc:
        q_info = {"error": str(exc)}
    try:
        c_stats = await cache_stats()
    except Exception as exc:
        c_stats = {"error": str(exc)}
    return HealthResponse(status="ok", qdrant=q_info, cache=c_stats)


@router.get("/collection")
def collection_endpoint() -> dict:
    try:
        return collection_info()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))