"""Tests for intent classification."""

from unittest.mock import MagicMock, patch


def test_classify_medical_qa():
    from src.intent import classify_intent

    with patch("src.intent.create_llm") as mock_create:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="medical_qa")
        mock_create.return_value = mock_llm
        assert classify_intent("What is myocardial infarction?") == "medical_qa"


def test_classify_drug_query():
    from src.intent import classify_intent

    with patch("src.intent.create_llm") as mock_create:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="drug_query")
        mock_create.return_value = mock_llm
        assert classify_intent("What are the side effects of metformin?") == "drug_query"


def test_classify_diagnosis():
    from src.intent import classify_intent

    with patch("src.intent.create_llm") as mock_create:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="diagnosis")
        mock_create.return_value = mock_llm
        result = classify_intent("Patient has chest pain and shortness of breath")
        assert result == "diagnosis"


def test_classify_invalid_response_defaults_to_medical_qa():
    from src.intent import classify_intent

    with patch("src.intent.create_llm") as mock_create:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="garbage_output")
        mock_create.return_value = mock_llm
        result = classify_intent("Some random question")
        assert result == "medical_qa"


def test_classify_llm_failure_fallback():
    from src.intent import classify_intent

    with patch("src.intent.create_llm") as mock_create:
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API down")
        mock_create.return_value = mock_llm
        result = classify_intent("What is sepsis?")
        assert result == "medical_qa"
