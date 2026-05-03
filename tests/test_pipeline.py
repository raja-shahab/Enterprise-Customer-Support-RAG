"""tests/test_pipeline.py – Unit tests for ASA v2."""
from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestConfig:
    def test_defaults(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            from src.config import get_settings
            get_settings.cache_clear()
            s = get_settings()
            assert s.retrieval_top_k == 8
            assert s.reranker_model == "cross-encoder/ms-marco-TinyBERT-L-2-v2"
            assert s.query_variations == 1
            assert s.openai_max_tokens == 512
            get_settings.cache_clear()


class TestEmbedder:
    def test_dense_returns_list(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            from src.ingestion.embedder import embed_text_dense
            result = embed_text_dense("How do I reset my password?")
            assert isinstance(result, list), "Must return plain list, not numpy array"
            assert len(result) == 384

    def test_sparse_returns_dict(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            from src.ingestion.embedder import embed_text_sparse
            result = embed_text_sparse("password reset help")
            assert isinstance(result, dict)


class TestChunker:
    def setup_method(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            from src.ingestion.chunker import DocumentChunker
            self.chunker = DocumentChunker(parent_chunk_size=200, child_chunk_size=80)

    def test_chunks_produced(self):
        text = " ".join([f"Sentence {i} about something." for i in range(30)])
        chunks = self.chunker.chunk_text(text, "test.txt")
        assert len(chunks) > 0
        for c in chunks:
            assert c.child_id and c.parent_id and c.text

    def test_empty_returns_none(self):
        assert self.chunker.chunk_text("  ", "empty.txt") == []


class TestRouter:
    @pytest.mark.asyncio
    async def test_chitchat(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            from src.graph.router import classify_intent
            assert await classify_intent("hello") == "chitchat"
            assert await classify_intent("thanks") == "chitchat"

    @pytest.mark.asyncio
    async def test_policy(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            from src.graph.router import classify_intent
            assert await classify_intent("What is your refund policy?") == "policy"

    @pytest.mark.asyncio
    async def test_troubleshoot(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            from src.graph.router import classify_intent
            assert await classify_intent("I am getting a 500 error") == "troubleshoot"


class TestReranker:
    def test_sorted_scores(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            from src.retrieval.reranker import rerank
            candidates = [
                "Reset password in Settings → Security.",
                "The weather is sunny today.",
                "Password reset link is sent to your email.",
            ]
            ranked = rerank("How to reset password?", candidates, top_k=2)
            assert len(ranked) == 2
            scores = [s for _, s in ranked]
            assert scores[0] >= scores[1]

    def test_empty_candidates(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            from src.retrieval.reranker import rerank
            assert rerank("query", [], top_k=5) == []


class TestSmartExpansion:
    @pytest.mark.asyncio
    async def test_no_expansion_when_confident(self):
        """High score should skip expansion."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            from src.graph.nodes import confidence_check_node
            from src.graph.state import RetrievedDocument

            doc = RetrievedDocument("id1", "Reset password in settings.", "Reset password in settings.", "faq.txt", 0.9)
            state = {
                "query": "How to reset password?",
                "retrieved_docs": [doc],
                "expansion_done": False,
                "top_rerank_score": 0.0,
            }
            with patch("src.graph.nodes.get_top_score", return_value=0.8):
                result = await confidence_check_node(state)
                assert result["needs_expansion"] is False

    @pytest.mark.asyncio
    async def test_expansion_when_low_confidence(self):
        """Low score should trigger expansion."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            from src.graph.nodes import confidence_check_node
            from src.graph.state import RetrievedDocument

            doc = RetrievedDocument("id1", "Some unrelated text.", "Some unrelated text.", "manual.txt", 0.1)
            state = {
                "query": "How do I configure webhooks?",
                "retrieved_docs": [doc],
                "expansion_done": False,
                "top_rerank_score": 0.0,
            }
            with patch("src.graph.nodes.get_top_score", return_value=0.1):
                result = await confidence_check_node(state)
                assert result["needs_expansion"] is True


class TestWorkflow:
    def test_compiles(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            from src.graph.workflow import get_workflow
            get_workflow.cache_clear()
            assert get_workflow() is not None
            get_workflow.cache_clear()

    @pytest.mark.asyncio
    async def test_chitchat_fast_path(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            from src.graph.workflow import get_workflow
            get_workflow.cache_clear()
            with patch("src.cache.semantic_cache.get_cached_answer", AsyncMock(return_value=None)):
                state = await get_workflow().ainvoke({"query": "hello", "retry_count": 0, "cached": False, "expansion_done": False})
                assert state.get("answer")
            get_workflow.cache_clear()


class TestSchemas:
    def test_query_request(self):
        from src.api.schemas import QueryRequest
        req = QueryRequest(query="How do I reset my password?")
        assert req.query == "How do I reset my password?"

    def test_empty_fails(self):
        import pydantic
        from src.api.schemas import QueryRequest
        with pytest.raises(pydantic.ValidationError):
            QueryRequest(query="")