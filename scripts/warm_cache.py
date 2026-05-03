#!/usr/bin/env python3
"""
scripts/warm_cache.py  –  Pre-warm semantic cache with frequent queries.

Usage:
    python scripts/warm_cache.py
    python scripts/warm_cache.py --queries data/top_queries.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from src.graph.workflow import get_workflow
from src.cache.semantic_cache import store_cached_answer, cache_stats

DEFAULT_QUERIES = [
    "How do I reset my password?",
    "How do I cancel my subscription?",
    "How can I upgrade my plan?",
    "Where can I find my invoices?",
    "How do I add a team member?",
    "What payment methods do you accept?",
    "How do I export my data?",
    "How do I contact support?",
    "How do I change my email address?",
    "What is your refund policy?",
    "How do I enable two-factor authentication?",
    "How do I delete my account?",
    "How do I integrate with Slack?",
    "What are the API rate limits?",
    "How do I generate an API key?",
    "How do I restore a deleted item?",
    "What file formats are supported?",
    "How do I set up SSO?",
    "How do I download the desktop app?",
    "What is the maximum file upload size?",
]


async def warm(queries: list[str], concurrency: int = 3):
    graph = get_workflow()
    sem = asyncio.Semaphore(concurrency)

    async def _one(q: str) -> bool:
        async with sem:
            try:
                state = await graph.ainvoke({"query": q, "retry_count": 0, "cached": False, "expansion_done": False})
                answer = state.get("answer", "")
                if answer:
                    await store_cached_answer(q, answer)
                    return True
            except Exception as exc:
                logger.error(f"Failed: {q[:50]} → {exc}")
            return False

    t0 = time.perf_counter()
    results = await asyncio.gather(*[_one(q) for q in queries])
    elapsed = time.perf_counter() - t0
    stats = await cache_stats()
    logger.success(f"Warmed {sum(results)}/{len(queries)} queries in {elapsed:.1f}s | Total cached: {stats['cached_entries']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default=None)
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()

    queries = DEFAULT_QUERIES
    if args.queries:
        with open(args.queries) as f:
            queries = json.load(f)

    asyncio.run(warm(queries, args.concurrency))


if __name__ == "__main__":
    main()
