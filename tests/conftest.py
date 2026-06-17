"""Pytest fixtures for RAG_for_hospital tests."""

import os
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def mock_env():
    """Ensure tests never leak to real external services."""
    os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
    os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")
    os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    os.environ.setdefault("MILVUS_HOST", "localhost")
    os.environ.setdefault("MILVUS_PORT", "19530")
    os.environ.setdefault("PG_HOST", "localhost")
    os.environ.setdefault("PG_PORT", "5432")
    os.environ.setdefault("PG_DATABASE", "hospital_rag")
    os.environ.setdefault("PG_USER", "postgres")
    os.environ.setdefault("PG_PASSWORD", "postgres")


@pytest.fixture
def sample_doc():
    """A minimal LangChain Document for testing."""
    from langchain_core.documents import Document

    return Document(
        page_content="The patient presented with elevated troponin levels indicating myocardial infarction.",
        metadata={"source": "test.pdf", "page": 1, "section_title": "Cardiac Markers"},
    )


@pytest.fixture
def sample_chunks():
    """A list of sample Document chunks."""
    from langchain_core.documents import Document

    return [
        Document(
            page_content="Deep stroma score is a novel prognostic marker for colorectal cancer.",
            metadata={"source": "paper1.pdf", "page": 3, "section_title": "Results", "chunk_id": "c1"},
        ),
        Document(
            page_content="The CMS4 subtype is associated with poor prognosis.",
            metadata={"source": "paper1.pdf", "page": 5, "section_title": "Discussion", "chunk_id": "c2"},
        ),
        Document(
            page_content="VGG19 was used as the backbone for feature extraction.",
            metadata={"source": "paper1.pdf", "page": 2, "section_title": "Methods", "chunk_id": "c3"},
        ),
    ]


@pytest.fixture
def mock_llm():
    """A mock LLM that returns fixed responses."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content="Test answer")
    return mock


@pytest.fixture
def mock_embeddings():
    """A mock embeddings that returns fixed vectors."""
    mock = MagicMock()
    mock.embed_documents.return_value = [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]
    mock.embed_query.return_value = [0.1] * 1024
    return mock
