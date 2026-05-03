"""
src/evaluation/harness.py  –  RAGAS offline evaluation.

Run:
    python -m src.evaluation.harness --dataset data/eval_dataset.json --out results/
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from src.graph.workflow import get_workflow


async def _run_query(query: str) -> Dict[str, Any]:
    graph = get_workflow()
    state = await graph.ainvoke({"query": query, "retry_count": 0, "cached": False, "expansion_done": False})
    return {
        "answer": state.get("answer", ""),
        "contexts": [doc.parent_text or doc.text for doc in state.get("reranked_docs", [])],
    }


async def _collect(questions: List[str], ground_truths: List[str]) -> Dict[str, List]:
    semaphore = asyncio.Semaphore(5)

    async def _guarded(q: str) -> Dict[str, Any]:
        async with semaphore:
            try:
                return await _run_query(q)
            except Exception as exc:
                logger.error(f"Query failed: {q[:60]} → {exc}")
                return {"answer": "", "contexts": []}

    results = await asyncio.gather(*[_guarded(q) for q in questions])
    return {
        "question": questions,
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": ground_truths,
    }


def run_evaluation(dataset_path: str, output_dir: str = "results", max_samples: Optional[int] = None) -> Dict[str, float]:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

    with open(dataset_path) as f:
        raw = json.load(f)
    if max_samples:
        raw = raw[:max_samples]

    questions = [item["question"] for item in raw]
    ground_truths = [item.get("ground_truth", "") for item in raw]
    logger.info(f"Evaluating {len(questions)} samples…")

    data = asyncio.get_event_loop().run_until_complete(_collect(questions, ground_truths))
    ds = Dataset.from_dict(data)
    result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
    scores = dict(result)
    logger.success(f"Results:\n{json.dumps(scores, indent=2)}")

    os.makedirs(output_dir, exist_ok=True)
    out = Path(output_dir) / "ragas_results.json"
    with open(out, "w") as f:
        json.dump({"metrics": scores, "n_samples": len(questions)}, f, indent=2)
    logger.info(f"Saved to {out}")
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/eval_dataset.json")
    parser.add_argument("--out", default="results")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    run_evaluation(args.dataset, args.out, args.max_samples)
