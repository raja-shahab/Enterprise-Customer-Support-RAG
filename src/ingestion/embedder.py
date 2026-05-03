"""
src/ingestion/embedder.py  –  Dense and sparse embedding models.

Dense:  all-MiniLM-L6-v2 (384-dim, CPU-friendly, ~5ms per query)
Sparse: BM25-style term weighting (no extra model needed)

FIX: embed_text_dense always returns List[float] — no .tolist() needed downstream.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Dict, List

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.config import get_settings

_settings = get_settings()
_dense_model: SentenceTransformer | None = None


def get_dense_model() -> SentenceTransformer:
    global _dense_model
    if _dense_model is None:
        logger.info(f"Loading dense embedding model: {_settings.dense_embedding_model}")
        _dense_model = SentenceTransformer(_settings.dense_embedding_model)
    return _dense_model


def embed_texts_dense(texts: List[str]) -> np.ndarray:
    """Returns float32 numpy array of shape (N, dim), normalised."""
    model = get_dense_model()
    return model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)


def embed_text_dense(text: str) -> List[float]:
    """
    Single-text embedding. Always returns a plain Python List[float].
    Safe to pass directly to JSON serialisation or Redis.
    """
    arr = embed_texts_dense([text])[0]
    # Always convert to plain list — fixes 'tolist' error everywhere
    return arr.tolist() if hasattr(arr, "tolist") else list(arr)


# ─── Sparse (BM25-style) ──────────────────────────────────────────────────────

_BM25_K1 = 1.5
_BM25_B = 0.75
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "and", "or", "but", "if", "not",
    "no", "so", "as", "it", "its", "this", "that", "we", "you", "he",
    "she", "they", "what", "which", "who", "how", "when", "where", "why",
}

_IDF: Dict[str, float] = {}
_AVG_DOC_LEN: float = 0.0
_DOC_COUNT: int = 0


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def update_idf_stats(corpus: List[str]) -> None:
    global _IDF, _AVG_DOC_LEN, _DOC_COUNT
    import math
    _DOC_COUNT = len(corpus)
    if not _DOC_COUNT:
        return
    df: Counter = Counter()
    total_len = 0
    for doc in corpus:
        tokens = set(_tokenize(doc))
        df.update(tokens)
        total_len += len(_tokenize(doc))
    _AVG_DOC_LEN = total_len / _DOC_COUNT
    _IDF = {
        term: math.log(1 + (_DOC_COUNT - freq + 0.5) / (freq + 0.5))
        for term, freq in df.items()
    }
    logger.info(f"IDF table built: {len(_IDF)} terms, avg_len={_AVG_DOC_LEN:.1f}")


def embed_text_sparse(text: str) -> Dict[int, float]:
    """BM25-style sparse vector as {token_hash: weight}."""
    tokens = _tokenize(text)
    if not tokens:
        return {}
    tf: Counter = Counter(tokens)
    doc_len = len(tokens)
    avg_len = _AVG_DOC_LEN if _AVG_DOC_LEN > 0 else doc_len
    result: Dict[int, float] = {}
    for term, count in tf.items():
        idf = _IDF.get(term, 1.0)
        tf_norm = count * (_BM25_K1 + 1) / (
            count + _BM25_K1 * (1 - _BM25_B + _BM25_B * doc_len / avg_len)
        )
        idx = int(hashlib.md5(term.encode()).hexdigest(), 16) % (2 ** 31)
        result[idx] = float(idf * tf_norm)
    return result
