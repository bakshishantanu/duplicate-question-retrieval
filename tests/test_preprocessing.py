"""Unit tests for text preprocessing."""
from src.preprocessing import preprocess_text, preprocess_for_sbert


def test_lowercase():
    assert "python" in preprocess_text("How to learn PYTHON?")


def test_contraction_expansion():
    result = preprocess_text("I can't do this", apply_stemming=False)
    assert "not" in result or "cannot" in result


def test_negation_retained():
    tokens = preprocess_text("Is Python not easy?", return_tokens=True, apply_stemming=False)
    assert "not" in tokens


def test_question_words_retained():
    tokens = preprocess_text("How do I learn Python?", return_tokens=True, apply_stemming=False)
    assert "how" in tokens


def test_stemming():
    tokens = preprocess_text("learning programming languages", return_tokens=True)
    assert "learn" in tokens


def test_empty_input():
    assert preprocess_text("") == ""
    assert preprocess_text(None) == ""
    assert preprocess_text("   ") == ""


def test_sbert_preserves_text():
    original = "How can I learn Python quickly?"
    assert preprocess_for_sbert(original) == original
