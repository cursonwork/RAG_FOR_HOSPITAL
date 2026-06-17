"""Tests for evaluation metrics — pure math, no I/O."""

from src.evaluation.metrics import _dcg, _is_relevant, _ndcg


def test_is_relevant_phrase_match():
    from langchain_core.documents import Document

    doc = Document(page_content="Deep stroma score predicts survival in colorectal cancer.")
    assert _is_relevant(doc, ["deep stroma score"], []) is True


def test_is_relevant_no_match():
    from langchain_core.documents import Document

    doc = Document(page_content="Deep stroma score predicts survival.")
    assert _is_relevant(doc, ["VGG19"], []) is False


def test_is_relevant_section_match():
    from langchain_core.documents import Document

    doc = Document(
        page_content="Some content.",
        metadata={"section": "Results and Discussion"},
    )
    assert _is_relevant(doc, [], ["Discussion"]) is True


def test_is_relevant_section_no_match():
    from langchain_core.documents import Document

    doc = Document(page_content="Content.", metadata={"section": "Methods"})
    assert _is_relevant(doc, [], ["Results"]) is False


def test_dcg_basic():
    rels = [1, 0, 1, 0]
    # DCG@4 = 1/log2(2) + 0/log2(3) + 1/log2(4) + 0/log2(5)
    # = 1 + 0.5 = 1.5
    import math

    expected = 1 / math.log2(2) + 1 / math.log2(4)
    assert abs(_dcg(rels, 4) - expected) < 0.001


def test_dcg_empty():
    assert _dcg([], 5) == 0.0


def test_ndcg_perfect():
    rels = [1, 1, 1]
    assert abs(_ndcg(rels, 3) - 1.0) < 0.001


def test_ndcg_mixed():
    rels = [1, 0, 1]
    ndcg = _ndcg(rels, 3)
    assert 0 < ndcg < 1.0


def test_ndcg_empty():
    assert _ndcg([], 5) == 0.0
