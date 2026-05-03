.PHONY: help install dev test ingest ingest-dir warm-cache evaluate docker-up docker-down clean

help:
	@echo ""
	@echo "  ASA v2 – Latency-Optimised RAG"
	@echo "  ─────────────────────────────────────────"
	@echo "  install      Install dependencies"
	@echo "  dev          Start API (hot reload)"
	@echo "  test         Run tests"
	@echo "  ingest       Index sample FAQ"
	@echo "  ingest-dir   Index all sample docs"
	@echo "  warm-cache   Pre-warm semantic cache"
	@echo "  evaluate     Run RAGAS evaluation"
	@echo "  docker-up    Start Qdrant + Redis + ASA"
	@echo "  docker-down  Stop all containers"
	@echo "  clean        Remove cache/logs"
	@echo ""

install:
	pip install --upgrade pip
	pip install -r requirements.txt

dev:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v --tb=short

ingest:
	python scripts/ingest_docs.py --file data/sample_docs/support_faq.txt --doc-type faq

ingest-dir:
	python scripts/ingest_docs.py --dir data/sample_docs/ --recreate

warm-cache:
	python scripts/warm_cache.py

evaluate:
	python -m src.evaluation.harness --dataset data/eval_dataset.json --out results/

docker-up:
	docker compose up -d --build
	@echo "API → http://localhost:8000/docs"

docker-down:
	docker compose down -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf logs/ results/ .pytest_cache/ .coverage
