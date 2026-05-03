"""
src/retrieval/qdrant_client.py  –  Qdrant setup, upsert, and hybrid search.
Uses two separate queries (dense + sparse) merged via RRF.
Compatible with all qdrant-client versions.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from src.config import get_settings

_settings = get_settings()
_sync_client: Optional[QdrantClient] = None
_async_client: Optional[AsyncQdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    global _sync_client
    if _sync_client is None:
        kwargs: Dict[str, Any] = {
            "host": _settings.qdrant_host,
            "port": _settings.qdrant_port,
            "timeout": 60.0,  # 🔥 critical fix
        }
        if _settings.qdrant_api_key:
            kwargs.update({
                "api_key": _settings.qdrant_api_key,
                "https": True
            })

        _sync_client = QdrantClient(**kwargs)
        logger.info(f"Qdrant sync client: {_settings.qdrant_host}:{_settings.qdrant_port}")

    return _sync_client

def get_async_qdrant_client() -> AsyncQdrantClient:
    global _async_client
    if _async_client is None:
        kwargs: Dict[str, Any] = {
            "host": _settings.qdrant_host,
            "port": _settings.qdrant_port,
            "timeout": 60.0,  # 🔥 critical fix
        }
        if _settings.qdrant_api_key:
            kwargs.update({
                "api_key": _settings.qdrant_api_key,
                "https": True
            })

        _async_client = AsyncQdrantClient(**kwargs)
        logger.info("Qdrant async client connected")

    return _async_client


def ensure_collection(recreate: bool = False) -> None:
    client = get_qdrant_client()
    col = _settings.qdrant_collection_name

    if recreate:
        try:
            client.delete_collection(col)
            logger.warning(f"Deleted collection: {col}")
        except Exception:
            pass

    try:
        client.get_collection(col)
        logger.info(f"Collection '{col}' exists.")
        return
    except Exception:
        pass

    client.create_collection(
        collection_name=col,
        vectors_config={
            "dense": qmodels.VectorParams(
                size=_settings.dense_embedding_dim,
                distance=qmodels.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "sparse": qmodels.SparseVectorParams(
                index=qmodels.SparseIndexParams(on_disk=False),
            ),
        },
    )

    for field_name in ("source", "doc_type", "product_category", "section"):
        try:
            client.create_payload_index(
                collection_name=col,
                field_name=field_name,
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

    logger.success(f"Created collection '{col}'.")


def upsert_chunks(
    child_ids: List[str],
    dense_vectors: List[List[float]],
    sparse_vectors: List[Dict[int, float]],
    payloads: List[Dict[str, Any]],
    batch_size: int = 32,
) -> None:
    client = get_qdrant_client()
    col = _settings.qdrant_collection_name
    total = len(child_ids)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        points = []
        for i in range(start, end):
            sp = sparse_vectors[i]
            point = qmodels.PointStruct(
                id=child_ids[i],
                vector={
                    "dense": dense_vectors[i],
                    "sparse": qmodels.SparseVector(
                        indices=list(sp.keys()),
                        values=list(sp.values()),
                    ),
                },
                payload=payloads[i],
            )
            points.append(point)
        client.upsert(collection_name=col, points=points, wait=True)
        logger.info(f"  Upserted {end}/{total} chunks")

    logger.success(f"Upserted {total} chunks into '{col}'.")


def _rrf_merge(
    dense_hits: List[qmodels.ScoredPoint],
    sparse_hits: List[qmodels.ScoredPoint],
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    k: int = 60,
    top_k: int = 8,
) -> List[qmodels.ScoredPoint]:
    """Reciprocal Rank Fusion: merge dense + sparse ranked lists."""
    scores: Dict[str, float] = {}
    points_map: Dict[str, qmodels.ScoredPoint] = {}

    for rank, point in enumerate(dense_hits):
        pid = str(point.id)
        scores[pid] = scores.get(pid, 0.0) + dense_weight / (k + rank + 1)
        points_map[pid] = point

    for rank, point in enumerate(sparse_hits):
        pid = str(point.id)
        scores[pid] = scores.get(pid, 0.0) + sparse_weight / (k + rank + 1)
        if pid not in points_map:
            points_map[pid] = point

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
    results = []
    for pid in sorted_ids:
        pt = points_map[pid]
        results.append(qmodels.ScoredPoint(
            id=pt.id,
            version=pt.version,
            score=scores[pid],
            payload=pt.payload,
            vector=pt.vector,
        ))
    return results


async def hybrid_search(
    dense_vector: List[float],
    sparse_vector: Dict[int, float],
    top_k: int = 8,
    filters: Optional[qmodels.Filter] = None,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> List[qmodels.ScoredPoint]:
    """Dense + sparse search merged via RRF. Compatible with all qdrant-client versions."""
    client = get_async_qdrant_client()
    col = _settings.qdrant_collection_name
    fetch_k = top_k * 3

    # Dense search
    try:
        dense_results = await client.search(
            collection_name=col,
            query_vector=qmodels.NamedVector(name="dense", vector=dense_vector),
            limit=fetch_k,
            query_filter=filters,
            with_payload=True,
        )
    except Exception as exc:
        logger.error(f"Dense search failed: {exc}")
        dense_results = []

    # Sparse search
    try:
        sparse_results = await client.search(
            collection_name=col,
            query_vector=qmodels.NamedSparseVector(
                name="sparse",
                vector=qmodels.SparseVector(
                    indices=list(sparse_vector.keys()),
                    values=list(sparse_vector.values()),
                ),
            ),
            limit=fetch_k,
            query_filter=filters,
            with_payload=True,
        )
    except Exception as exc:
        logger.error(f"Sparse search failed: {exc}")
        sparse_results = []

    if not dense_results and not sparse_results:
        return []

    merged = _rrf_merge(dense_results, sparse_results, dense_weight, sparse_weight, top_k=top_k)
    logger.debug(f"Hybrid: {len(dense_results)} dense + {len(sparse_results)} sparse → {len(merged)} merged")
    return merged


def build_filter(
    product_category: Optional[str] = None,
    doc_type: Optional[str] = None,
    source: Optional[str] = None,
) -> Optional[qmodels.Filter]:
    conditions: List[qmodels.Condition] = []
    if product_category:
        conditions.append(qmodels.FieldCondition(key="product_category", match=qmodels.MatchValue(value=product_category)))
    if doc_type:
        conditions.append(qmodels.FieldCondition(key="doc_type", match=qmodels.MatchValue(value=doc_type)))
    if source:
        conditions.append(qmodels.FieldCondition(key="source", match=qmodels.MatchValue(value=source)))
    return qmodels.Filter(must=conditions) if conditions else None


def collection_info() -> Dict[str, Any]:
    client = get_qdrant_client()
    col = _settings.qdrant_collection_name
    info = client.get_collection(col)
    return {
        "name": col,
        "vectors_count": info.vectors_count,
        "points_count": info.points_count,
        "status": str(info.status),
    }
