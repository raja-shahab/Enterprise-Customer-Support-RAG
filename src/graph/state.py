"""
src/graph/state.py  –  LangGraph state definition.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class RetrievedDocument:
    __slots__ = ("doc_id", "text", "parent_text", "source", "section",
                 "doc_type", "product_category", "score", "rerank_score")

    def __init__(self, doc_id, text, parent_text, source, score,
                 section=None, doc_type=None, product_category=None, rerank_score=0.0):
        self.doc_id = doc_id
        self.text = text
        self.parent_text = parent_text
        self.source = source
        self.section = section
        self.doc_type = doc_type
        self.product_category = product_category
        self.score = score
        self.rerank_score = rerank_score

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


class ASAState(TypedDict, total=False):
    """
    Full state flowing through the LangGraph nodes.

    query              – original user query
    intent             – faq | troubleshoot | policy | chitchat
    query_variations   – list of queries used for retrieval
    metadata_filters   – extracted product/doc_type filters
    retrieved_docs     – candidates from hybrid search
    reranked_docs      – top-K after cross-encoder reranking
    top_rerank_score   – highest reranker score (used for expansion decision)
    needs_expansion    – True if confidence too low → trigger query expansion
    expansion_done     – True if we already expanded (prevent infinite loop)
    citations          – citation strings for final answer
    context_str        – formatted context block for LLM
    answer             – final generated answer
    cached             – True if served from cache
    error              – error message (triggers fallback)
    retry_count        – retrieval retry counter
    stream_tokens      – accumulated streamed tokens
    """
    query: str
    intent: str
    query_variations: List[str]
    metadata_filters: Dict[str, Optional[str]]
    retrieved_docs: List[RetrievedDocument]
    reranked_docs: List[RetrievedDocument]
    top_rerank_score: float
    needs_expansion: bool
    expansion_done: bool
    citations: List[str]
    context_str: str
    answer: str
    cached: bool
    error: Optional[str]
    retry_count: int
    stream_tokens: List[str]
