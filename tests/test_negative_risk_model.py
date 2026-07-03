"""Tests for negative_risk_model module."""
import numpy as np
import pandas as pd
import pytest

from models.deep_sgdf_delta.negative_risk_model import (
    NegativeRiskConfig,
    NegativeRiskModel,
    NegativeRiskPredictionResult,
)


def _make_training_data(n=200, n_features=5, seed=42):
    """Create synthetic training data."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        rng.randn(n, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    # Create labels with some signal
    scores = X["feat_0"] * 0.5 + X["feat_1"] * 0.3 + rng.randn(n) * 0.2
    y_negative = (scores < -0.5).astype(int)
    y_deep_negative = (scores < -1.0).astype(int)
    y_relative_down = (scores < -0.3).astype(int)
    return X, y_negative, y_deep_negative, y_relative_down


class TestFitPredict:
    def test_fit_predict_runs(self):
        X, y_neg, y_deep, y_rel = _make_training_data()
        model = NegativeRiskModel()
        model.fit(X, y_neg, y_deep, y_rel)
        result = model.predict(X)
        assert isinstance(result, NegativeRiskPredictionResult)
        assert len(result.df) == len(X)

    def test_predict_before_fit_raises(self):
        model = NegativeRiskModel()
        X = pd.DataFrame(np.random.randn(10, 3), columns=["a", "b", "c"])
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(X)


class TestProbabilityRange:
    def test_probabilities_in_01(self):
        X, y_neg, y_deep, y_rel = _make_training_data()
        model = NegativeRiskModel()
        model.fit(X, y_neg, y_deep, y_rel)
        result = model.predict(X)

        for col in ["negative_prob", "deep_negative_prob", "relative_down_prob"]:
            assert (result.df[col] >= 0).all(), f"{col} has values < 0"
            assert (result.df[col] <= 1).all(), f"{col} has values > 1"


class TestRiskScore:
    def test_risk_score_in_01(self):
        X, y_neg, y_deep, y_rel = _make_training_data()
        model = NegativeRiskModel()
        model.fit(X, y_neg, y_deep, y_rel)
        result = model.predict(X)

        assert (result.df["negative_risk_score"] >= 0).all()
        assert (result.df["negative_risk_score"] <= 1).all()

    def test_confidence_in_01(self):
        X, y_neg, y_deep, y_rel = _make_training_data()
        model = NegativeRiskModel()
        model.fit(X, y_neg, y_deep, y_rel)
        result = model.predict(X)

        assert (result.df["confidence"] >= 0).all()
        assert (result.df["confidence"] <= 1).all()


class TestFeatureImportance:
    def test_feature_importance_output(self):
        X, y_neg, y_deep, y_rel = _make_training_data()
        model = NegativeRiskModel()
        model.fit(X, y_neg, y_deep, y_rel)
        fi = model.get_feature_importance()

        assert "feature" in fi.columns
        assert "mean_importance" in fi.columns
        assert len(fi) == X.shape[1]

    def test_feature_importance_before_fit_raises(self):
        model = NegativeRiskModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.get_feature_importance()


class TestInsufficientData:
    def test_too_few_samples_raises(self):
        X = pd.DataFrame(np.random.randn(5, 3), columns=["a", "b", "c"])
        y_neg = np.array([0, 1, 0, 1, 0])
        y_deep = np.array([0, 0, 1, 0, 0])
        y_rel = np.array([0, 0, 0, 1, 0])

        model = NegativeRiskModel()
        with pytest.raises(ValueError, match="Not enough valid samples"):
            model.fit(X, y_neg, y_deep, y_rel)


class TestInvalidTargets:
    def test_minus1_labels_excluded(self):
        """Rows with label=-1 should be excluded from training."""
        X, y_neg, y_deep, y_rel = _make_training_data(n=100)
        # Set some labels to -1 (invalid)
        y_neg[:10] = -1
        y_deep[:10] = -1

        model = NegativeRiskModel()
        model.fit(X, y_neg, y_deep, y_rel)
        assert model.is_fitted_
