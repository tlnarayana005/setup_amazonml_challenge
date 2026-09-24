"""Test ensemble utilities."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.models.ensemble import (
    _validate_id_alignment,
    average_predictions,
    weighted_average_predictions,
    ensemble_with_threshold,
    compare_individual_vs_ensemble,
    load_predictions,
)


def _make_pred_df(ids, preds):
    return pd.DataFrame({"id": ids, "prediction": preds})


class TestIDAlignment:
    def test_valid_alignment(self):
        df1 = _make_pred_df(["a", "b", "c"], [0.1, 0.2, 0.3])
        df2 = _make_pred_df(["a", "b", "c"], [0.4, 0.5, 0.6])
        result = _validate_id_alignment([df1, df2])
        assert result["valid"]

    def test_missing_ids(self):
        df1 = _make_pred_df(["a", "b", "c"], [0.1, 0.2, 0.3])
        df2 = _make_pred_df(["a", "b"], [0.4, 0.5])
        result = _validate_id_alignment([df1, df2])
        assert not result["valid"]
        assert any("missing" in d.lower() for d in result["details"])

    def test_extra_ids(self):
        df1 = _make_pred_df(["a", "b"], [0.1, 0.2])
        df2 = _make_pred_df(["a", "b", "c"], [0.4, 0.5, 0.6])
        result = _validate_id_alignment([df1, df2])
        assert not result["valid"]

    def test_duplicate_ids(self):
        df1 = _make_pred_df(["a", "b", "c"], [0.1, 0.2, 0.3])
        df2 = _make_pred_df(["a", "a", "c"], [0.4, 0.5, 0.6])
        result = _validate_id_alignment([df1, df2])
        assert not result["valid"]

    def test_empty_list(self):
        result = _validate_id_alignment([])
        assert not result["valid"]


class TestAveraging:
    def test_simple_average(self):
        ids = ["a", "b", "c"]
        df1 = _make_pred_df(ids, [0.2, 0.4, 0.6])
        df2 = _make_pred_df(ids, [0.4, 0.6, 0.8])
        result = average_predictions([df1, df2])
        assert len(result) == 3
        np.testing.assert_allclose(
            result.sort_values("id")["prediction"].values,
            [0.3, 0.5, 0.7],
            atol=1e-6,
        )

    def test_weighted_average(self):
        ids = ["a", "b"]
        df1 = _make_pred_df(ids, [0.0, 1.0])
        df2 = _make_pred_df(ids, [1.0, 0.0])
        result = weighted_average_predictions([df1, df2], weights=[3.0, 1.0])
        sorted_r = result.sort_values("id").reset_index(drop=True)
        np.testing.assert_allclose(sorted_r["prediction"].values, [0.25, 0.75], atol=1e-6)

    def test_weight_mismatch_raises(self):
        df1 = _make_pred_df(["a"], [0.5])
        with pytest.raises(ValueError):
            weighted_average_predictions([df1], weights=[1.0, 2.0])

    def test_misaligned_raises(self):
        df1 = _make_pred_df(["a", "b"], [0.1, 0.2])
        df2 = _make_pred_df(["a", "c"], [0.3, 0.4])
        with pytest.raises(ValueError):
            average_predictions([df1, df2])


class TestEnsembleWithThreshold:
    def test_threshold_search(self):
        np.random.seed(42)
        ids = [f"id_{i}" for i in range(20)]
        y_true = np.array([0]*10 + [1]*10)
        df1 = _make_pred_df(ids, np.concatenate([np.random.uniform(0, 0.4, 10), np.random.uniform(0.6, 1.0, 10)]))
        df2 = _make_pred_df(ids, np.concatenate([np.random.uniform(0, 0.4, 10), np.random.uniform(0.6, 1.0, 10)]))
        result = ensemble_with_threshold([df1, df2], y_true)
        assert "ensemble_df" in result
        assert "best_threshold" in result
        assert "best_score" in result
        assert 0 < result["best_threshold"] < 1
        assert result["best_score"] > 0


class TestCompareIndividualVsEnsemble:
    def test_comparison(self):
        np.random.seed(42)
        ids = [f"id_{i}" for i in range(50)]
        y_true = np.array([0]*25 + [1]*25)
        df1 = _make_pred_df(ids, np.concatenate([np.random.uniform(0, 0.5, 25), np.random.uniform(0.5, 1.0, 25)]))
        df2 = _make_pred_df(ids, np.concatenate([np.random.uniform(0, 0.5, 25), np.random.uniform(0.5, 1.0, 25)]))
        result = compare_individual_vs_ensemble(
            [df1, df2], y_true, metric="f1_binary", classify=True
        )
        assert "individual" in result
        assert "ensemble" in result
        assert "ensemble_improves" in result
        assert len(result["individual"]) == 2


class TestLoadPredictions:
    def test_load_csv(self, tmp_path):
        df = _make_pred_df(["x", "y"], [0.1, 0.9])
        p = tmp_path / "preds.csv"
        df.to_csv(p, index=False)
        loaded = load_predictions([str(p)])
        assert len(loaded) == 1
        assert len(loaded[0]) == 2

    def test_load_parquet(self, tmp_path):
        df = _make_pred_df(["x", "y"], [0.1, 0.9])
        p = tmp_path / "preds.parquet"
        df.to_parquet(p, index=False)
        loaded = load_predictions([str(p)])
        assert len(loaded) == 1

    def test_missing_file(self):
        loaded = load_predictions(["nonexistent.csv"])
        assert len(loaded) == 0
