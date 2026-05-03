"""
src/retrieval/query_transform.py  –  Smart query expansion.

Strategy (latency-optimised):
  - Phase 1: Try retrieval with ORIGINAL query only (fast, 0 extra LLM calls)
  - Phase 2: Only if reranker top score < threshold OR no results found,
             generate 2 query variations and retry (costs 1 LLM call)

This avoids paying LLM variation costs for queries that already retrieve
good results — which is the majority of support queries.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from src.config import get_settings

_settings = get_settings()


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=_settings.openai_model,
        temperature=0.3,
        api_key=_settings.openai_api_key,
        max_tokens=128,
    )


_VARIATION_PROMPT = """\
Generate 2 alternative phrasings of the query that preserve its meaning \
but use different words to help retrieve relevant documents.
Output ONLY a JSON array of 2 strings. No explanation.
Example: ["How to reset password?", "Steps to change account password"]"""


async def generate_query_variations(query: str) -> List[str]:
    """
    Generate 2 query variations. Returns [original] + variations.
    Only called when initial retrieval confidence is low.
    """
    llm = _get_llm()
    try:
        messages = [
            SystemMessage(content=_VARIATION_PROMPT),
            HumanMessage(content=f"Query: {query}"),
        ]
        response = await llm.ainvoke(messages)
        raw = response.content.strip().strip("```json").strip("```").strip()
        variations: List[str] = json.loads(raw)
        logger.info(f"Generated {len(variations)} query variations for expansion")
        return [query] + [v for v in variations if v != query]
    except Exception as exc:
        logger.warning(f"Variation generation failed: {exc}")
        return [query]


_FILTER_PROMPT = """\
Extract metadata filters from this customer support query.
Available fields: product_category (string or null), doc_type (one of: faq, manual, policy, ticket, or null).
Output ONLY a JSON object. Example: {"product_category": "Pro", "doc_type": "manual"}"""


async def extract_metadata_filters(query: str) -> Dict[str, Optional[str]]:
    """Extract structured metadata filters from query."""
    llm = _get_llm()
    try:
        messages = [
            SystemMessage(content=_FILTER_PROMPT),
            HumanMessage(content=f"Query: {query}"),
        ]
        response = await llm.ainvoke(messages)
        raw = response.content.strip().strip("```json").strip("```").strip()
        return json.loads(raw)
    except Exception as exc:
        logger.warning(f"Filter extraction failed: {exc}")
        return {"product_category": None, "doc_type": None}
