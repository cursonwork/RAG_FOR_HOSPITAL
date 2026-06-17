"""Tests for configuration settings."""

from src.config import Settings


def test_settings_defaults():
    """Verify all Settings fields have sensible defaults."""
    s = Settings()
    assert s.chunk_size == 2000
    assert s.chunk_overlap == 100
    assert s.retrieval_top_k == 5
    assert s.hybrid_enabled is True
    assert s.hybrid_retrieval_top_k == 20
    assert s.reranker_enabled is True
    assert s.reranker_model == "ms-marco-MiniLM-L-6-v2"
    assert s.reranker_top_n == 5
    assert s.query_rewriting_enabled is True
    assert s.enable_image_understanding is True
    assert s.image_max_size == 800
    assert s.image_max_concurrent == 8
    assert s.image_min_bytes == 5120
    assert s.milvus_collection_name == "hospital_knowledge"
    assert s.embedding_dimension == 1024
    assert s.pdf_parser == "opendataloader"


def test_env_override(monkeypatch):
    """Verify env vars override defaults."""
    monkeypatch.setenv("CHUNK_SIZE", "512")
    monkeypatch.setenv("HYBRID_ENABLED", "false")
    monkeypatch.setenv("RERANKER_TOP_N", "10")
    s = Settings()
    assert s.chunk_size == 512
    assert s.hybrid_enabled is False
    assert s.reranker_top_n == 10


def test_required_fields_have_defaults():
    """Ensure no Settings field lacks a default — prevents .env-missing startup crash."""
    s = Settings()
    for field_name in Settings.model_fields:
        # All fields should be set (no None values that would crash)
        assert getattr(s, field_name) is not None, f"Field {field_name} is None"
