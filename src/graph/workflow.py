"""
src/graph/workflow.py  –  LangGraph StateGraph with lazy query expansion.

Happy path (most queries — NO LLM call before retrieval):
  START → cache_lookup → intent_router → retrieval
        → confidence_check → rerank → context_builder
        → generation → cache_store → END

Expansion path (only when confidence low — complex/ambiguous queries):
  ... → confidence_check → expand_and_retrieve (filters + variations here)
        → rerank → context_builder → generation → cache_store → END

Latency savings:
  - query_transform REMOVED from happy path → saves 4s per simple query
  - Metadata filter extraction only happens when confidence is low
  - Expansion triggered only when top rerank score < threshold
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph
from loguru import logger

from src.graph.nodes import (
    cache_lookup_node,
    cache_store_node,
    chitchat_node,
    confidence_check_node,
    context_builder_node,
    expand_and_retrieve_node,
    fallback_node,
    generation_node,
    intent_router_node,
    rerank_node,
    retrieval_node,
)
from src.graph.state import ASAState


# ─── Routing Functions ────────────────────────────────────────────────────────

def route_after_cache(state: ASAState) -> str:
    return END if state.get("cached") else "intent_router"


def route_after_intent(state: ASAState) -> str:
    return "chitchat" if state.get("intent") == "chitchat" else "retrieval"


def route_after_retrieval(state: ASAState) -> str:
    """After retrieval: if results → check confidence. Else fallback."""
    if state.get("retrieved_docs"):
        return "confidence_check"
    return "fallback"


def route_after_confidence(state: ASAState) -> str:
    """
    Key routing decision:
      - needs_expansion=True  → run expand_and_retrieve (LLM variation call)
      - needs_expansion=False → go straight to rerank (fast path)
    """
    return "expand_and_retrieve" if state.get("needs_expansion") else "rerank"


def route_after_expansion(state: ASAState) -> str:
    """After expansion retrieval: go to rerank or fallback if still no results."""
    return "rerank" if state.get("retrieved_docs") else "fallback"


# ─── Graph Builder ────────────────────────────────────────────────────────────

def build_graph() -> Any:
    builder = StateGraph(ASAState)

    # ── Nodes ──────────────────────────────────────────────────────────────────
    builder.add_node("cache_lookup",        cache_lookup_node)
    builder.add_node("intent_router",       intent_router_node)
    builder.add_node("retrieval",           retrieval_node)
    builder.add_node("confidence_check",    confidence_check_node)
    builder.add_node("expand_and_retrieve", expand_and_retrieve_node)
    builder.add_node("rerank",              rerank_node)
    builder.add_node("context_builder",     context_builder_node)
    builder.add_node("generation",          generation_node)
    builder.add_node("cache_store",         cache_store_node)
    builder.add_node("chitchat",            chitchat_node)
    builder.add_node("fallback",            fallback_node)

    # ── Edges ──────────────────────────────────────────────────────────────────
    builder.add_edge(START, "cache_lookup")

    builder.add_conditional_edges(
        "cache_lookup", route_after_cache,
        {END: END, "intent_router": "intent_router"},
    )
    builder.add_conditional_edges(
        "intent_router", route_after_intent,
        {"chitchat": "chitchat", "retrieval": "retrieval"},   # ← direct to retrieval
    )

    builder.add_edge("chitchat", END)

    builder.add_conditional_edges(
        "retrieval", route_after_retrieval,
        {"confidence_check": "confidence_check", "fallback": "fallback"},
    )
    builder.add_conditional_edges(
        "confidence_check", route_after_confidence,
        {"expand_and_retrieve": "expand_and_retrieve", "rerank": "rerank"},
    )
    builder.add_conditional_edges(
        "expand_and_retrieve", route_after_expansion,
        {"rerank": "rerank", "fallback": "fallback"},
    )

    builder.add_edge("rerank",          "context_builder")
    builder.add_edge("context_builder", "generation")
    builder.add_edge("generation",      "cache_store")
    builder.add_edge("cache_store",     END)
    builder.add_edge("fallback",        END)

    compiled = builder.compile()
    logger.success("LangGraph ASA workflow compiled — query_transform removed from happy path.")
    return compiled


@lru_cache(maxsize=1)
def get_workflow() -> Any:
    return build_graph()