"""src/api/schemas.py  –  Pydantic request/response models."""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    product_category: Optional[str] = None
    stream: bool = True


class QueryResponse(BaseModel):
    answer: str
    citations: List[str] = []
    intent: Optional[str] = None
    cached: bool = False
    query_variations: List[str] = []
    top_rerank_score: float = 0.0
    expansion_triggered: bool = False


class IngestRequest(BaseModel):
    source: str
    text: str
    doc_type: str = "faq"
    product_category: Optional[str] = None


class IngestResponse(BaseModel):
    chunks_indexed: int
    source: str
    status: str = "ok"


class HealthResponse(BaseModel):
    status: str
    qdrant: dict
    cache: dict
