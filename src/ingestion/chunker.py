"""
src/ingestion/chunker.py  –  Parent-child semantic chunking.

Parent chunks (~800 tokens) are returned to the LLM for rich context.
Child chunks (~200 tokens) are indexed in Qdrant for precise search.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from loguru import logger
from src.config import get_settings

_settings = get_settings()


@dataclass
class Chunk:
    child_id: str
    parent_id: str
    text: str
    parent_text: str
    source: str
    page: Optional[int] = None
    section: Optional[str] = None
    product_category: Optional[str] = None
    doc_type: Optional[str] = None
    token_count: int = 0
    metadata: dict = field(default_factory=dict)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


_SENT_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_PATTERN.split(text.strip()) if s.strip()]


def _split_by_tokens(text: str, chunk_size: int, overlap: int = 1) -> List[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return [text] if text.strip() else []
    chunks, current, current_tokens = [], [], 0
    for sent in sentences:
        t = _estimate_tokens(sent)
        if current_tokens + t > chunk_size and current:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
            current_tokens = sum(_estimate_tokens(s) for s in current)
        current.append(sent)
        current_tokens += t
    if current:
        chunks.append(" ".join(current))
    return chunks


def _parse_file(path: Path) -> List[dict]:
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return [{"text": path.read_text(encoding="utf-8", errors="replace"), "page": None}]

    if suffix == ".pdf":
        try:
            import fitz  # pymupdf
            doc = fitz.open(str(path))
            pages = []
            for i, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    pages.append({"text": text, "page": i + 1})
            doc.close()
            return pages if pages else [{"text": "", "page": None}]
        except Exception as exc:
            logger.warning(f"PDF parse failed: {exc}")
            return [{"text": "", "page": None}]

    if suffix == ".docx":
        try:
            from docx import Document
            doc = Document(str(path))
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            return [{"text": text, "page": None}]
        except Exception as exc:
            logger.warning(f"DOCX parse failed: {exc}")
            return [{"text": "", "page": None}]

    # fallback for .html etc
    return [{"text": path.read_bytes().decode("utf-8", errors="replace"), "page": None}]

_HEADING_RE = re.compile(r"^(#{1,3}\s+.+|[A-Z][A-Z\s]{4,}:?$)", re.MULTILINE)



def _detect_section(text: str) -> Optional[str]:
    m = _HEADING_RE.search(text)
    return m.group(0).strip("# ").strip() if m else None


class DocumentChunker:
    def __init__(
        self,
        parent_chunk_size: int = _settings.parent_chunk_size,
        child_chunk_size: int = _settings.child_chunk_size,
    ):
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size

    def chunk_file(
        self,
        path: Path,
        source: str,
        doc_type: str = "manual",
        product_category: Optional[str] = None,
        extra_metadata: Optional[dict] = None,
    ) -> List[Chunk]:
        logger.info(f"Chunking file: {path} | doc_type={doc_type}")
        pages = _parse_file(path)
        all_text = "\n\n".join(p["text"] for p in pages if p["text"].strip())
        if not all_text.strip():
            logger.warning(f"No text extracted from {path}")
            return []
        return self._build_chunks(all_text, source, doc_type, product_category, extra_metadata)

    def chunk_text(
        self,
        text: str,
        source: str,
        doc_type: str = "faq",
        product_category: Optional[str] = None,
    ) -> List[Chunk]:
        return self._build_chunks(text, source, doc_type, product_category)

    def _build_chunks(self, text, source, doc_type, product_category, extra_metadata=None):
        parent_texts = _split_by_tokens(text, self.parent_chunk_size, overlap=2)
        chunks: List[Chunk] = []
        for parent_text in parent_texts:
            parent_text = parent_text.strip()
            if not parent_text:
                continue
            parent_id = str(uuid.uuid4())
            section = _detect_section(parent_text)
            for child_text in _split_by_tokens(parent_text, self.child_chunk_size, overlap=1):
                child_text = child_text.strip()
                if not child_text:
                    continue
                chunks.append(Chunk(
                    child_id=str(uuid.uuid4()),
                    parent_id=parent_id,
                    text=child_text,
                    parent_text=parent_text,
                    source=source,
                    section=section,
                    product_category=product_category,
                    doc_type=doc_type,
                    token_count=_estimate_tokens(child_text),
                    metadata=extra_metadata or {},
                ))
        logger.info(f"  → {len(chunks)} child chunks from {source}")
        return chunks
