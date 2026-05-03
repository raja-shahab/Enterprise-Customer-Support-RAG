"""
src/ingestion/pipeline.py  –  End-to-end ingestion orchestrator.
parse → chunk → embed (dense + sparse) → upsert to Qdrant
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from loguru import logger
from tqdm import tqdm

from src.config import get_settings
from src.ingestion.chunker import Chunk, DocumentChunker
from src.ingestion.embedder import embed_texts_dense, embed_text_sparse, update_idf_stats
from src.retrieval.qdrant_client import ensure_collection, upsert_chunks

_settings = get_settings()


class IngestPipeline:
    def __init__(self):
        self.chunker = DocumentChunker()
        ensure_collection()

    def ingest_file(
        self,
        path: Path,
        source: Optional[str] = None,
        doc_type: str = "manual",
        product_category: Optional[str] = None,
    ) -> int:
        source = source or path.name
        chunks = self.chunker.chunk_file(path, source, doc_type, product_category)
        return self._embed_and_upsert(chunks) if chunks else 0

    def ingest_text(
        self,
        text: str,
        source: str,
        doc_type: str = "faq",
        product_category: Optional[str] = None,
    ) -> int:
        chunks = self.chunker.chunk_text(text, source, doc_type, product_category)
        return self._embed_and_upsert(chunks) if chunks else 0

    def ingest_batch(self, paths: List[Path], doc_type: str = "manual", product_category: Optional[str] = None) -> int:
        all_chunks: List[Chunk] = []
        for path in tqdm(paths, desc="Chunking"):
            chunks = self.chunker.chunk_file(path, path.name, doc_type, product_category)
            all_chunks.extend(chunks)
        if not all_chunks:
            return 0
        update_idf_stats([c.text for c in all_chunks])
        return self._embed_and_upsert(all_chunks)

    def _embed_and_upsert(self, chunks: List[Chunk]) -> int:
        texts = [c.text for c in chunks]
        logger.info(f"Computing dense embeddings for {len(texts)} chunks…")
        dense_array: np.ndarray = embed_texts_dense(texts)
        dense_list = [dense_array[i].tolist() for i in range(len(chunks))]

        logger.info("Computing sparse embeddings…")
        sparse_list = [embed_text_sparse(t) for t in texts]

        payloads = [
            {
                "text": c.text,
                "parent_text": c.parent_text[:2000],
                "parent_id": c.parent_id,
                "source": c.source,
                "section": c.section,
                "doc_type": c.doc_type,
                "product_category": c.product_category,
                "token_count": c.token_count,
                **c.metadata,
            }
            for c in chunks
        ]

        upsert_chunks(
            child_ids=[c.child_id for c in chunks],
            dense_vectors=dense_list,
            sparse_vectors=sparse_list,
            payloads=payloads,
        )
        logger.success(f"Indexed {len(chunks)} chunks.")
        return len(chunks)
