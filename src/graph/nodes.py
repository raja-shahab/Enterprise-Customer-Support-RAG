"""
src/graph/nodes.py  –  All LangGraph node functions.

Key latency optimisations in this file:
  1. cache_lookup_node     – short-circuits entire pipeline on cache hit (<100ms)
  2. chitchat_node         – bypasses all retrieval for greetings
  3. retrieval_node        – uses original query only (no LLM variation call)
  4. confidence_check_node – checks reranker top score AFTER first retrieval
                             Only if score < threshold → expand queries (smart expansion)
  5. expand_and_retrieve_node – only called when confidence is low
                                generates variations + does a second retrieval pass
  6. rerank_node           – TinyBERT on 8 candidates (~10ms vs old 10s)
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from src.config import get_settings
from src.graph.router import classify_intent
from src.graph.state import ASAState, RetrievedDocument
from src.ingestion.embedder import embed_text_dense, embed_text_sparse
from src.retrieval.query_transform import extract_metadata_filters, generate_query_variations
from src.retrieval.qdrant_client import build_filter, hybrid_search
from src.retrieval.reranker import rerank, get_top_score

_settings = get_settings()


def _get_llm(streaming: bool = False) -> ChatOpenAI:
    return ChatOpenAI(
        model=_settings.openai_model,
        temperature=_settings.openai_temperature,
        api_key=_settings.openai_api_key,
        max_tokens=_settings.openai_max_tokens,
        streaming=streaming,
    )


_SYSTEM_PROMPT = """\
You are a helpful enterprise customer support assistant.
Answer using ONLY the provided context. Cite sources as [doc_id].
If context lacks the answer, say: "I don't have enough information. Please contact support."
Be concise and professional. Max 3 paragraphs."""

_CHITCHAT_RESPONSES = {
    "hi": "Hello! How can I help you today?",
    "hello": "Hi there! What can I assist you with?",
    "hey": "Hey! What do you need help with?",
    "thanks": "You're welcome! Anything else I can help with?",
    "thank you": "My pleasure! Feel free to ask if you need anything else.",
    "bye": "Goodbye! Have a great day!",
    "goodbye": "Take care! Come back if you need help.",
}


# ─── Node 1: Cache Lookup ─────────────────────────────────────────────────────

async def cache_lookup_node(state: ASAState) -> ASAState:
    """Check semantic cache. On hit, skip entire RAG pipeline → <100ms."""
    try:
        from src.cache.semantic_cache import get_cached_answer
        cached = await get_cached_answer(state["query"])
        if cached:
            logger.info("Cache HIT")
            return {**state, "answer": cached, "cached": True}
    except Exception as exc:
        logger.warning(f"Cache lookup error: {exc}")
    return {**state, "cached": False}


# ─── Node 2: Intent Router ────────────────────────────────────────────────────

async def intent_router_node(state: ASAState) -> ASAState:
    intent = await classify_intent(state["query"])
    return {
        **state,
        "intent": intent,
        "retry_count": state.get("retry_count", 0),
        "expansion_done": False,
        "needs_expansion": False,
        "top_rerank_score": 0.0,
    }


# ─── Node 3: Query Transform (filter extraction only, no variations) ──────────

async def query_transform_node(state: ASAState) -> ASAState:
    """
    Extract metadata filters from query.
    Does NOT generate query variations — that only happens if confidence is low.
    This saves 1 LLM call for the majority of queries.
    """
    query = state["query"]
    filters = await extract_metadata_filters(query)
    # Start with original query only
    return {
        **state,
        "query_variations": [query],
        "metadata_filters": filters,
    }


# ─── Node 4: Retrieval ────────────────────────────────────────────────────────

async def _retrieve_one(query: str, filters: Optional[Any], top_k: int) -> List[RetrievedDocument]:
    dense_vec = embed_text_dense(query)
    sparse_vec = embed_text_sparse(query)
    results = await hybrid_search(
        dense_vector=dense_vec,
        sparse_vector=sparse_vec,
        top_k=top_k,
        filters=filters,
        dense_weight=_settings.hybrid_dense_weight,
        sparse_weight=_settings.hybrid_sparse_weight,
    )
    docs = []
    for point in results:
        payload = point.payload or {}
        docs.append(RetrievedDocument(
            doc_id=str(point.id),
            text=payload.get("text", ""),
            parent_text=payload.get("parent_text", ""),
            source=payload.get("source", "unknown"),
            section=payload.get("section"),
            doc_type=payload.get("doc_type"),
            product_category=payload.get("product_category"),
            score=point.score,
        ))
    return docs


async def retrieval_node(state: ASAState) -> ASAState:
    """
    Retrieve candidates using current query_variations.
    First pass: single original query (fast).
    Second pass (if called again after expansion): multiple variations.
    """
    variations = state.get("query_variations") or [state["query"]]
    mf = state.get("metadata_filters", {})
    qdrant_filter = build_filter(
        product_category=mf.get("product_category"),
        doc_type=mf.get("doc_type"),
    )

    tasks = [_retrieve_one(q, qdrant_filter, _settings.retrieval_top_k) for q in variations]
    results_per_variation = await asyncio.gather(*tasks, return_exceptions=True)

    seen: Dict[str, RetrievedDocument] = {}
    for result in results_per_variation:
        if isinstance(result, Exception):
            logger.error(f"Retrieval error: {result}")
            continue
        for doc in result:
            if doc.doc_id not in seen or doc.score > seen[doc.doc_id].score:
                seen[doc.doc_id] = doc

    merged = sorted(seen.values(), key=lambda d: d.score, reverse=True)
    logger.info(f"Retrieved {len(merged)} candidates from {len(variations)} variation(s)")

    if not merged:
        return {**state, "retrieved_docs": [], "error": "no_results"}
    return {**state, "retrieved_docs": merged, "error": None}


# ─── Node 5: Confidence Check (Smart Expansion Decision) ─────────────────────

async def confidence_check_node(state: ASAState) -> ASAState:
    """
    Score the top candidate against the query using the reranker.
    If score < threshold AND we haven't expanded yet → set needs_expansion=True.

    This is the key innovation:
      - Good results (score >= threshold): skip expansion → save 1 LLM call + extra retrieval
      - Poor results (score < threshold):  trigger expansion → better answers
    """
    docs = state.get("retrieved_docs", [])
    if not docs:
        return {**state, "needs_expansion": not state.get("expansion_done", False), "top_rerank_score": 0.0}

    query = state["query"]
    # Quick score using top-3 candidates only (fast)
    sample_texts = [doc.parent_text or doc.text for doc in docs[:3]]

    top_score = await asyncio.get_event_loop().run_in_executor(
        None, get_top_score, query, sample_texts
    )

    threshold = _settings.expansion_score_threshold
    expansion_done = state.get("expansion_done", False)
    needs_expansion = (top_score < threshold) and (not expansion_done)

    if needs_expansion:
        logger.info(f"Low confidence (score={top_score:.3f} < {threshold}) → triggering query expansion")
    else:
        logger.info(f"Confidence OK (score={top_score:.3f}) → skipping expansion")

    return {**state, "top_rerank_score": top_score, "needs_expansion": needs_expansion}


# ─── Node 6: Expand and Retrieve ──────────────────────────────────────────────

async def expand_and_retrieve_node(state: ASAState) -> ASAState:
    """
    Only reached when confidence is low.
    Generate query variations → re-run retrieval with all variations.
    """
    query = state["query"]
    logger.info("Expanding query with variations for better retrieval…")

    variations = await generate_query_variations(query)
    mf = state.get("metadata_filters", {})
    qdrant_filter = build_filter(
        product_category=mf.get("product_category"),
        doc_type=mf.get("doc_type"),
    )

    tasks = [_retrieve_one(q, qdrant_filter, _settings.retrieval_top_k) for q in variations]
    results_per_variation = await asyncio.gather(*tasks, return_exceptions=True)

    # Merge with existing results (union)
    seen: Dict[str, RetrievedDocument] = {d.doc_id: d for d in (state.get("retrieved_docs") or [])}
    for result in results_per_variation:
        if isinstance(result, Exception):
            continue
        for doc in result:
            if doc.doc_id not in seen or doc.score > seen[doc.doc_id].score:
                seen[doc.doc_id] = doc

    merged = sorted(seen.values(), key=lambda d: d.score, reverse=True)
    logger.info(f"After expansion: {len(merged)} total candidates")

    return {
        **state,
        "query_variations": variations,
        "retrieved_docs": merged,
        "expansion_done": True,
        "needs_expansion": False,
        "error": None if merged else "no_results",
    }


# ─── Node 7: Reranking ────────────────────────────────────────────────────────

async def rerank_node(state: ASAState) -> ASAState:
    """
    TinyBERT reranking on top-8 candidates.
    ~10-15ms on CPU (was 10s with MiniLM-L6 on 20 candidates).
    """
    candidates = state.get("retrieved_docs") or []
    if not candidates:
        return {**state, "reranked_docs": []}

    query = state["query"]
    texts = [doc.parent_text or doc.text for doc in candidates]

    ranked = await asyncio.get_event_loop().run_in_executor(
        None, rerank, query, texts, _settings.reranker_top_k
    )

    reranked: List[RetrievedDocument] = []
    for idx, score in ranked:
        doc = candidates[idx]
        doc.rerank_score = score
        reranked.append(doc)

    logger.info(f"Reranked to top {len(reranked)} documents")
    return {**state, "reranked_docs": reranked}


# ─── Node 8: Context Builder ──────────────────────────────────────────────────

async def context_builder_node(state: ASAState) -> ASAState:
    """Format reranked docs into a context block for the LLM."""
    docs = state.get("reranked_docs") or state.get("retrieved_docs") or []
    if not docs:
        return {**state, "context_str": "", "citations": []}

    parts, citations = [], []
    for i, doc in enumerate(docs):
        label = f"doc_{i + 1}"
        src = f"{doc.source}" + (f" § {doc.section}" if doc.section else "")
        parts.append(f"[{label}] Source: {src}\n{doc.parent_text or doc.text}")
        citations.append(f"[{label}] {src}")

    return {**state, "context_str": "\n\n---\n\n".join(parts), "citations": citations}


# ─── Node 9: Generation ──────────────────────────────────────────────────────

async def generation_node(state: ASAState) -> ASAState:
    """Generate final answer with LLM (non-streaming path)."""
    context = state.get("context_str", "")
    if not context.strip():
        return {
            **state,
            "answer": "I don't have enough information to answer that. Please contact support.",
        }

    user_msg = f"Context:\n{context}\n\nQuestion: {state['query']}\n\nAnswer (cite [doc_id]):"
    try:
        response = await _get_llm().ainvoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])
        answer = response.content.strip()
        logger.info(f"Generated answer ({len(answer)} chars)")
        return {**state, "answer": answer}
    except Exception as exc:
        logger.error(f"LLM generation failed: {exc}")
        return {**state, "answer": "An error occurred. Please try again.", "error": str(exc)}


# ─── Node 10: Cache Store ─────────────────────────────────────────────────────

async def cache_store_node(state: ASAState) -> ASAState:
    if state.get("cached") or state.get("error"):
        return state
    try:
        from src.cache.semantic_cache import store_cached_answer
        await store_cached_answer(state["query"], state.get("answer", ""))
    except Exception as exc:
        logger.warning(f"Cache store error: {exc}")
    return state


# ─── Node 11: Chitchat ────────────────────────────────────────────────────────

async def chitchat_node(state: ASAState) -> ASAState:
    """Zero-retrieval response for greetings. Returns in <5ms."""
    query = state["query"].lower().strip().rstrip("!.,?")
    response = _CHITCHAT_RESPONSES.get(query, "Hello! How can I assist you today?")
    return {**state, "answer": response, "citations": [], "cached": False}


# ─── Node 12: Fallback ────────────────────────────────────────────────────────

async def fallback_node(state: ASAState) -> ASAState:
    logger.warning(f"Fallback triggered. Error: {state.get('error')}")
    return {
        **state,
        "answer": (
            "I'm sorry, I couldn't find relevant information for your question. "
            "Please try rephrasing, or contact support@company.com for direct assistance."
        ),
        "citations": [],
        "reranked_docs": [],
    }