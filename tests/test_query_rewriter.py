"""Tests for query rewriting strategies."""

from unittest.mock import MagicMock, patch

from src.query_rewriter import rewrite_query


def test_rewrite_disabled():
    """When query rewriting is disabled, returns original question in list."""
    with patch("src.query_rewriter.settings.query_rewriting_enabled", False):
        result = rewrite_query("What is cancer?")
        assert result == ["What is cancer?"]


def test_rewrite_normal_query():
    """Normal length query (25-120 chars) should pass through."""
    question = "What are the prognostic factors for colorectal cancer survival?"
    result = rewrite_query(question, history=[])
    assert question in result


def test_rewrite_short_query():
    """Short query (<25 chars) should generate variants."""
    question = "Cancer"
    result = rewrite_query(question, history=[])
    # Should return original + variants
    assert question in result
    assert len(result) > 1


def test_rewrite_long_query():
    """Long query (>120 chars) should do step-back."""
    question = (
        "What are the detailed mechanisms by which stromal microenvironment "
        "influences colorectal cancer prognosis through immune cell infiltration "
        "and extracellular matrix remodeling in the tumor microenvironment?"
    )
    result = rewrite_query(question, history=[])
    assert question in result


def test_rewrite_llm_failure():
    """When LLM fails, return original question as fallback."""
    with patch("src.query_rewriter.create_llm") as mock_create:
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM down")
        mock_create.return_value = mock_llm
        result = rewrite_query("short", history=[])
        assert result == ["short"]


def test_rewrite_with_history():
    """When history is provided, should resolve anaphora/completion in query."""
    question = "What about the side effects?"
    history = [
        MagicMock(content="Tell me about metformin", type="human"),
        MagicMock(content="Metformin is a first-line diabetes drug...", type="ai"),
    ]
    result = rewrite_query(question, history=history)
    # Multi-turn rewrite resolves "the side effects" → specific drug context
    # The rewritten query should mention metformin, not the vague original
    assert len(result) >= 1
    assert "metformin" in result[0].lower()


def test_rewrite_empty_result():
    """When LLM returns empty string, return original question."""
    with patch("src.query_rewriter.create_llm") as mock_create:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="")
        mock_create.return_value = mock_llm
        result = rewrite_query("short query", history=[])
        assert result == ["short query"]
