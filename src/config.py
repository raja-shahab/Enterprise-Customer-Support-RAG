"""
src/config.py  –  Central configuration (pydantic-settings).
All values overridable via environment variables or .env file.

Latency-optimised defaults:
  - RETRIEVAL_TOP_K  : 8   (was 20) → reranker sees fewer candidates
  - RERANKER_MODEL   : TinyBERT-L-2 (was MiniLM-L-6) → 3x faster
  - QUERY_VARIATIONS : 1   (was 2)  → smart expansion only when needed
  - OPENAI_MAX_TOKENS: 512 (was 1024) → faster generation
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.1
    openai_max_tokens: int = 512          # ← reduced from 1024

    # ── Qdrant ───────────────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "asa_support_docs"

    # ── Embeddings ───────────────────────────────────────────────────────────
    dense_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dense_embedding_dim: int = 384
    reranker_model: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2"  # ← TinyBERT

    # ── Redis ────────────────────────────────────────────────────────────────
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    cache_ttl_seconds: int = 3600
    cache_similarity_threshold: float = 0.92

    # ── Retrieval (latency-tuned) ─────────────────────────────────────────────
    hybrid_dense_weight: float = 0.7
    hybrid_sparse_weight: float = 0.3
    retrieval_top_k: int = 5             # ← reduced from 20
    reranker_top_k: int = 3
    child_chunk_size: int = 200
    parent_chunk_size: int = 600
    query_variations: int = 1            # ← reduced from 2 (smart expansion)

    # ── Confidence threshold for smart query expansion ────────────────────────
    # If top reranker score < this, expand queries and retry
    expansion_score_threshold: float = 0.3

    # ── API ──────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://localhost:8000",   # Frontend served directly by FastAPI
        "http://127.0.0.1:8000",
    ]

    # ── LangSmith ────────────────────────────────────────────────────────────
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "asa-production"

    # ── Evaluation ───────────────────────────────────────────────────────────
    eval_dataset_path: str = "data/eval_dataset.json"

    @field_validator("hybrid_dense_weight", "hybrid_sparse_weight")
    @classmethod
    def weights_valid(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("Weights must be between 0 and 1")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()