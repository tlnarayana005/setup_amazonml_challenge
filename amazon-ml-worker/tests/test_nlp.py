"""Test NLP text utilities."""
import pytest
import pandas as pd
from src.nlp.text_utils import clean_text, normalize_text, compute_text_stats, batch_clean_text, add_text_features


def test_clean_text():
    assert clean_text("  hello   world  ") == "hello world"
    assert clean_text("café résumé") == "café résumé"
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_normalize_text():
    assert normalize_text("  Hello WORLD  ") == "hello world"
    assert normalize_text("Hello", lowercase=False) == "Hello"


def test_compute_text_stats():
    texts = pd.Series(["hello world", "foo bar baz", "a", None, ""])
    stats = compute_text_stats(texts)
    assert stats["count"] == 4  # non-null
    assert stats["null_count"] == 1
    assert stats["empty_strings"] == 1
    assert stats["mean_word_count"] > 0


def test_batch_clean_text():
    df = pd.DataFrame({"id": [1, 2], "text": ["  HELLO  ", "  World  "]})
    result = batch_clean_text(df, "text")
    assert "text_clean" in result.columns
    assert result["text_clean"].iloc[0] == "hello"


def test_add_text_features():
    df = pd.DataFrame({"id": [1, 2], "text": ["hello world", "foo bar baz qux"]})
    result = add_text_features(df, "text")
    assert "text_char_len" in result.columns
    assert "text_word_count" in result.columns
    assert result["text_word_count"].iloc[0] == 2
    assert result["text_word_count"].iloc[1] == 4
