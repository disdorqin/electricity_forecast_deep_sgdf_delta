"""Tests for spike_risk_model module."""
import numpy as np
import pandas as pd
import pytest

from models.deep_sgdf_delta.spike_risk_model import (
    SpikeRiskConfig,
    SpikeRiskModel,
    SpikeRiskPredictionResult,
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
    y_spike = (scores > 0.5).astype(int)
    y_extreme_spike = (scores > 1.0).astype(int)
    y_relative_spike = (scores > 0.3).astype(int)
    return X, y_spike, y_extreme_spike, y_relative_spike


class TestFitPredict:
    def test_fit_predict_runs(self):
        X, y_spike, y_extreme, y_relative = _make_training_data()
        model = SpikeRiskModel()
        model.fit(X, y_spike, y_extreme, y_relative)
        result = model.predict(X)
        assert isinstance(result, SpikeRiskPredictionResult)
        assert len(result.df) == len(X)

    def test_predict_before_fit_raises(self):
        model = SpikeRiskModel()
        X = pd.DataFrame(np.random.randn(10, 3), columns=["a", "b", "c"])
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(X)


class TestProbabilityRange:
    def test_probabilities_in_01(self):
        X, y_spike, y_extreme, y_relative = _make_training_data()
        model = SpikeRiskModel()
        model.fit(X, y_spike, y_extreme, y_relative)
        result = model.predict(X)

        for col in ["spike_prob", "extreme_spike_prob", "relative_spike_prob"]:
            assert (result.df[col] >= 0).all(), f"{col} has values < 0"
            assert (result.df[col] <= 1).all(), f"{col} has values > 1"


class TestRiskScore:
    def test_risk_score_in_01(self):
        X, y_spike, y_extreme, y_relative = _make_training_data()
        model = SpikeRiskModel()
        model.fit(X, y_spike, y_extreme, y_relative)
        result = model.predict(X)

        assert (result.df["spike_risk_score"] >= 0).all()
        assert (result.df["spike_risk_score"] <= 1).all()

    def test_confidence_in_01(self):
        X, y_spike, y_extreme, y_relative = _make_training_data()
        model = SpikeRiskModel()
        model.fit(X, y_spike, y_extreme, y_relative)
        result = model.predict(X)

        assert (result.df["confidence"] >= 0).all()
        assert (result.df["confidence"] <= 1).all()


class TestFeatureImportance:
    def test_feature_importance_output(self):
        X, y_spike, y_extreme, y_relative = _make_training_data()
        model = SpikeRiskModel()
        model.fit(X, y_spike, y_extreme, y_relative)
        fi = model.get_feature_importance()

        assert "feature" in fi.columns
        assert "mean_importance" in fi.columns
        assert len(fi) == X.shape[1]

    def test_feature_importance_before_fit_raises(self):
        model = SpikeRiskModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.get_feature_importance()


class TestInsufficientData:
    def test_too_few_samples_handled_gracefully(self):
        """When sample count < 10, model should handle gracefully (not crash)."""
        X = pd.DataFrame(np.random.randn(5, 3), columns=["a", "b", "c"])
        y_spike = np.array([0, 1, 0, 1, 0])
        y_extreme = np.array([0, 0, 1, 0, 0])
        y_relative = np.array([0, 0, 0, 1, 0])

        model = SpikeRiskModel()
        model.fit(X, y_spike, y_extreme, y_relative)
        
        # Model should be fitted but marked as insufficient
        assert model.is_fitted_
        assert hasattr(model, '_insufficient_events_')
        assert model._insufficient_events_ is True
        
        # Prediction should return NaN values
        pred = model.predict(X)
        assert pred.status == "INSUFFICIENT_EVENTS"
        assert pred.df["spike_prob"].isna().all()


class TestInvalidTargets:
    def test_minus1_labels_excluded(self):
        """Rows with label=-1 should be excluded from training."""
        X, y_spike, y_extreme, y_relative = _make_training_data(n=100)
        # Set some labels to -1 (invalid)
        y_spike[:10] = -1
        y_extreme[:10] = -1

        model = SpikeRiskModel()
        model.fit(X, y_spike, y_extreme, y_relative)
        assert model.is_fitted_
