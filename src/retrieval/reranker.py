"""
src/retrieval/reranker.py  –  Cross-encoder reranking.

Model: cross-encoder/ms-marco-TinyBERT-L-2-v2
  - 2-layer TinyBERT — ~3x faster than MiniLM-L-6
  - Runs on CPU in ~10-15ms for 8 candidates (was 10s with L-6 on 20 candidates)
  - Slight quality trade-off but excellent for production latency targets
"""
from __future__ import annotations

from typing import List, Tuple

from loguru import logger
from sentence_transformers import CrossEncoder

from src.config import get_settings

_settings = get_settings()
_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info(f"Loading reranker: {_settings.reranker_model}")
        _reranker = CrossEncoder(
            _settings.reranker_model,
            max_length=256,    # ← reduced from 512 for speed
            device="cpu",
        )
    return _reranker


def rerank(
    query: str,
    candidates: List[str],
    top_k: int | None = None,
) -> List[Tuple[int, float]]:
    """
    Score (query, candidate) pairs. Returns sorted (index, score) tuples.
    With TinyBERT + 8 candidates: ~10-15ms on CPU.
    """
    if not candidates:
        return []

    top_k = top_k or _settings.reranker_top_k
    model = get_reranker()
    # NEW — use more context for medical/long docs
    pairs = [(query, doc[:800]) for doc in candidates]
    scores: List[float] = model.predict(pairs, show_progress_bar=False).tolist()
    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return indexed[:top_k]


def get_top_score(
    query: str,
    candidates: List[str],
) -> float:
    """
    Return the highest reranker score for a set of candidates.
    Used to decide whether query expansion is needed.
    """
    if not candidates:
        return 0.0
    results = rerank(query, candidates, top_k=1)
    return results[0][1] if results else 0.0
