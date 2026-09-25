"""
Tests for Entity Resolution pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.er.pipeline import (
    normalize_field,
    normalize_df,
    extract_tokens,
    extract_numeric_tokens,
    _jaccard,
    _edit_similarity,
    _length_ratio,
    compute_pair_features,
    assign_labels,
    macro_f05_score,
    generate_candidates,
    get_feature_columns,
)


class TestNormalization:
    def test_normalize_basic(self):
        assert normalize_field("  Hello  World  ") == "hello world"

    def test_normalize_unicode(self):
        # NFKC normalization preserves accents but normalizes compatibility chars
        result = normalize_field("Café")
        assert "caf" in result
        assert result == "cafe" or result == "caf\xe9"  # either form is acceptable

    def test_normalize_empty(self):
        assert normalize_field("") == ""
        assert normalize_field(None) == ""

    def test_normalize_punctuation(self):
        result = normalize_field("Smith & Partners, LLC.")
        assert "&" not in result  # punctuation replaced with space
        assert "," not in result

    def test_normalize_df(self):
        df = pd.DataFrame({
            "entity_id": ["e1", "e2"],
            "business_name": ["ACME Corp", "  Test LLC  "],
            "business_address": ["123 Main St", "456 Oak Ave"],
            "country": ["US", "India"],
        })
        result = normalize_df(df)
        assert "business_name_norm" in result.columns
        assert "business_address_norm" in result.columns
        assert "country_norm" in result.columns
        assert result["business_name_norm"].iloc[0] == "acme corp"


class TestTokenExtraction:
    def test_extract_tokens(self):
        tokens = extract_tokens("hello world test")
        assert tokens == {"hello", "world", "test"}

    def test_extract_empty(self):
        assert extract_tokens("") == set()

    def test_extract_numeric_tokens(self):
        nums = extract_numeric_tokens("123 Main St, Suite 456, NY 10001")
        assert "123" in nums
        assert "456" in nums
        assert "10001" in nums


class TestSimilarityFunctions:
    def test_jaccard_identical(self):
        assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_jaccard_disjoint(self):
        assert _jaccard({"a"}, {"b"}) == 0.0

    def test_jaccard_partial(self):
        assert _jaccard({"a", "b", "c"}, {"b", "c", "d"}) == 0.5

    def test_jaccard_empty(self):
        assert _jaccard(set(), set()) == 1.0
        assert _jaccard({"a"}, set()) == 0.0

    def test_edit_sim_identical(self):
        assert _edit_similarity("hello", "hello") == 1.0

    def test_edit_sim_empty(self):
        assert _edit_similarity("", "hello") == 0.0
        assert _edit_similarity("", "") == 1.0

    def test_edit_sim_similar(self):
        sim = _edit_similarity("hello", "hallo")
        assert 0.5 < sim < 1.0

    def test_length_ratio(self):
        assert _length_ratio("hello", "hello") == 1.0
        assert _length_ratio("hi", "hello") == 2.0 / 5.0
        assert _length_ratio("", "") == 1.0


class TestPairFeatures:
    def test_exact_match(self):
        row1 = {"business_name_norm": "acme corp", "business_address_norm": "123 main st", "country_norm": "us"}
        row2 = {"business_name_norm": "acme corp", "business_address_norm": "123 main st", "country_norm": "us"}
        feats = compute_pair_features(row1, row2)
        assert feats["name_exact"] == 1.0
        assert feats["addr_exact"] == 1.0
        assert feats["country_match"] == 1.0
        assert feats["name_jaccard"] == 1.0

    def test_different_entities(self):
        row1 = {"business_name_norm": "acme corp", "business_address_norm": "123 main st", "country_norm": "us"}
        row2 = {"business_name_norm": "global tech", "business_address_norm": "456 oak ave", "country_norm": "india"}
        feats = compute_pair_features(row1, row2)
        assert feats["name_exact"] == 0.0
        assert feats["country_match"] == 0.0
        assert feats["name_jaccard"] < 0.5


class TestLabelAssignment:
    def test_assign_labels(self):
        pairs = pd.DataFrame({
            "source1_entity_id": ["s1_001", "s1_001", "s1_002"],
            "source2_entity_id": ["s2_001", "s2_002", "s2_003"],
        })
        gt = pd.DataFrame({
            "source1_entity_id": ["s1_001"],
            "matched_entity_ids": ["s2_001"],
        })
        result = assign_labels(pairs, gt)
        assert list(result["label"]) == [1, 0, 0]


class TestMacroF05:
    def test_perfect_score(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        groups = np.array(["g1", "g1", "g2", "g2"])
        score = macro_f05_score(y_true, y_pred, groups)
        assert score == 1.0

    def test_zero_score(self):
        y_true = np.array([1, 1])
        y_pred = np.array([0, 0])
        groups = np.array(["g1", "g1"])
        score = macro_f05_score(y_true, y_pred, groups)
        assert score == 0.0

    def test_all_negative(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        groups = np.array(["g1", "g1", "g2"])
        score = macro_f05_score(y_true, y_pred, groups)
        assert score == 1.0  # No positives, perfect prediction


class TestGetFeatureCols:
    def test_excludes_ids(self):
        df = pd.DataFrame({
            "source1_entity_id": ["a"],
            "source2_entity_id": ["b"],
            "label": [1],
            "name_jaccard": [0.5],
            "addr_edit_sim": [0.3],
        })
        cols = get_feature_columns(df)
        assert "source1_entity_id" not in cols
        assert "source2_entity_id" not in cols
        assert "label" not in cols
        assert "name_jaccard" in cols
        assert "addr_edit_sim" in cols
