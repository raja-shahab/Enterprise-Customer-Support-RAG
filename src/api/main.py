"""
src/api/main.py  –  FastAPI application factory.

IMPORTANT: load_dotenv() is called FIRST so LangSmith picks up env vars correctly.
"""
from __future__ import annotations

# ── Load .env into os.environ BEFORE any langsmith/langchain imports ──────────
import os
from dotenv import load_dotenv
load_dotenv()

import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from src.api.routes import router
from src.config import get_settings

_settings = get_settings()

# ─── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr,
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
           level="INFO")
os.makedirs("logs", exist_ok=True)
logger.add("logs/asa.log", rotation="10 MB", retention="7 days", level="DEBUG")

# ─── Resolve static directory (works both locally and in Docker) ──────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
# src/api/main.py → go up two levels to project root → static/
_STATIC_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "static"))
# Fallback: check relative to CWD (Docker sets WORKDIR /app)
if not os.path.isdir(_STATIC_DIR):
    _STATIC_DIR = os.path.join(os.getcwd(), "static")


# ─── App ──────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Agentic Support Assistant (ASA)",
        description=(
            "Production-grade agentic RAG — hybrid search, smart query expansion, "
            "TinyBERT reranking, semantic cache, SSE streaming."
        ),
        version="2.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],        # ← change this
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── API routes (registered BEFORE static mount so /api/v1/* takes priority) ──
    app.include_router(router, prefix="/api/v1", tags=["ASA"])

    # ── Serve the frontend UI ─────────────────────────────────────────────────
    if os.path.isdir(_STATIC_DIR):
        # Mount /static for any future static assets (CSS, JS, images, etc.)
        app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
        logger.info(f"Static files mounted from: {_STATIC_DIR}")

        @app.get("/", include_in_schema=False)
        async def serve_frontend():
            """Serve the ASA frontend UI."""
            return FileResponse(os.path.join(_STATIC_DIR, "index.html"))
    else:
        logger.warning(f"Static dir not found at {_STATIC_DIR}. Frontend UI will not be served.")

        @app.get("/")
        async def root():
            return {"name": "ASA", "version": "2.0.0", "docs": "/docs", "health": "/api/v1/health"}

    @app.on_event("startup")
    async def startup():
        logger.info("ASA startup – pre-loading models…")
        from src.ingestion.embedder import get_dense_model
        from src.retrieval.reranker import get_reranker
        get_dense_model()
        get_reranker()
        logger.success("Models loaded. ASA is ready.")

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host=_settings.api_host, port=_settings.api_port,
                workers=1, reload=False, log_level="info")