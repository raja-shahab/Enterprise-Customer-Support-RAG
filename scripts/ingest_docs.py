#!/usr/bin/env python3
"""
scripts/ingest_docs.py  –  CLI to index documents into Qdrant.

Usage:
    python scripts/ingest_docs.py --file data/sample_docs/support_faq.txt --doc-type faq
    python scripts/ingest_docs.py --dir data/sample_docs/ --recreate
    python scripts/ingest_docs.py --stats
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from src.ingestion.pipeline import IngestPipeline
from src.retrieval.qdrant_client import collection_info, ensure_collection

SUPPORTED = {".txt", ".md", ".pdf", ".docx", ".html"}


def main():
    parser = argparse.ArgumentParser(description="ASA ingestion CLI")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file",  help="Single file to ingest")
    group.add_argument("--dir",   help="Directory of files to ingest")
    group.add_argument("--stats", action="store_true", help="Show collection stats")
    parser.add_argument("--doc-type", default="manual", choices=["faq", "manual", "policy", "ticket"])
    parser.add_argument("--product",  default="")
    parser.add_argument("--recreate", action="store_true", help="Wipe and recreate collection")
    args = parser.parse_args()

    if args.stats:
        info = collection_info()
        print("\n── Qdrant Collection Stats ───────────────")
        for k, v in info.items():
            print(f"  {k:20s}: {v}")
        return

    if args.recreate:
        ensure_collection(recreate=True)

    pipeline = IngestPipeline()

    if args.file:
        path = Path(args.file)
        n = pipeline.ingest_file(path, path.name, args.doc_type, args.product or None)
        logger.success(f"Indexed {n} chunks from {path.name}")

    elif args.dir:
        directory = Path(args.dir)
        files = [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED]
        logger.info(f"Found {len(files)} files")
        total = 0
        for fpath in files:
            try:
                n = pipeline.ingest_file(fpath, fpath.name, args.doc_type, args.product or None)
                total += n
            except Exception as exc:
                logger.error(f"Failed {fpath.name}: {exc}")
        logger.success(f"Total indexed: {total} chunks")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
