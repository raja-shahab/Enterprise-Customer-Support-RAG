"""
src/cache/semantic_cache.py  –  Semantic cache using Redis.

FIX: embed_text_dense now always returns List[float], so no .tolist() needed.
Cache lookup uses cosine similarity — similar questions hit the cache.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Optional

import redis.asyncio as aioredis
from loguru import logger

from src.config import get_settings
from src.ingestion.embedder import embed_text_dense

_settings = get_settings()
_redis_client: Optional[aioredis.Redis] = None
_CACHE_PREFIX = "asa:cache:"


def _get_redis() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        kwargs = {"host": _settings.redis_host, "port": _settings.redis_port, "decode_responses": True}
        if _settings.redis_password:
            kwargs["password"] = _settings.redis_password
        _redis_client = aioredis.Redis(**kwargs)
    return _redis_client


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


async def get_cached_answer(query: str) -> Optional[str]:
    """Return cached answer if a semantically similar query was cached. None on miss."""
    try:
        redis = _get_redis()
        # embed_text_dense always returns List[float] — no .tolist() needed
        query_emb: list = embed_text_dense(query)
        keys = await redis.keys(f"{_CACHE_PREFIX}*")
        if not keys:
            return None

        threshold = _settings.cache_similarity_threshold
        for key in keys:
            raw = await redis.get(key)
            if not raw:
                continue
            try:
                entry = json.loads(raw)
                sim = _cosine_similarity(query_emb, entry["embedding"])
                if sim >= threshold:
                    logger.info(f"Cache HIT (sim={sim:.3f}): {entry['query'][:60]}")
                    return entry["answer"]
            except (json.JSONDecodeError, KeyError):
                continue
        return None
    except Exception as exc:
        logger.warning(f"Cache get error: {exc}")
        return None


async def store_cached_answer(query: str, answer: str) -> None:
    """Store query-answer pair in Redis with TTL."""
    if not answer:
        return
    try:
        redis = _get_redis()
        # embed_text_dense always returns List[float] — safe to store directly
        emb: list = embed_text_dense(query)
        key = f"{_CACHE_PREFIX}{_sha256(query)}"
        value = json.dumps({"query": query, "embedding": emb, "answer": answer})
        await redis.set(key, value, ex=_settings.cache_ttl_seconds)
        logger.debug(f"Cached: {query[:60]}")
    except Exception as exc:
        logger.warning(f"Cache store error: {exc}")


async def invalidate_cache() -> int:
    try:
        redis = _get_redis()
        keys = await redis.keys(f"{_CACHE_PREFIX}*")
        if keys:
            deleted = await redis.delete(*keys)
            logger.info(f"Cache cleared: {deleted} entries")
            return deleted
        return 0
    except Exception as exc:
        logger.error(f"Cache invalidation error: {exc}")
        return 0


async def cache_stats() -> dict:
    try:
        redis = _get_redis()
        keys = await redis.keys(f"{_CACHE_PREFIX}*")
        return {"cached_entries": len(keys)}
    except Exception:
        return {"cached_entries": 0}
