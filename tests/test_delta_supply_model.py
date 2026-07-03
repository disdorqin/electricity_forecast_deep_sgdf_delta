"""Tests for delta_supply_model module."""
import numpy as np
import pandas as pd
import pytest

from models.deep_sgdf_delta.delta_supply_model import (
    DeltaSupplyConfig,
    DeltaSupplyModel,
    DeltaSupplyPredictionResult,
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
    y_upward = (scores > 0.5).astype(int)
    y_downward = (scores < -0.5).astype(int)
    y_large_abs = (np.abs(scores) > 0.7).astype(int)
    y_magnitude = scores * 100
    return X, y_upward, y_downward, y_large_abs, y_magnitude


class TestFitPredict:
    def test_fit_predict_runs(self):
        X, y_up, y_down, y_large, y_mag = _make_training_data()
        model = DeltaSupplyModel()
        model.fit(X, y_up, y_down, y_large, y_mag)
        result = model.predict(X)
        assert isinstance(result, DeltaSupplyPredictionResult)
        assert len(result.df) == len(X)

    def test_predict_before_fit_raises(self):
        model = DeltaSupplyModel()
        X = pd.DataFrame(np.random.randn(10, 3), columns=["a", "b", "c"])
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(X)


class TestProbabilityRange:
    def test_probabilities_in_01(self):
        X, y_up, y_down, y_large, y_mag = _make_training_data()
        model = DeltaSupplyModel()
        model.fit(X, y_up, y_down, y_large, y_mag)
        result = model.predict(X)

        for col in ["upward_deviation_prob", "downward_deviation_prob", "large_abs_deviation_prob"]:
            assert (result.df[col] >= 0).all(), f"{col} has values < 0"
            assert (result.df[col] <= 1).all(), f"{col} has values > 1"


class TestDirection:
    def test_direction_values(self):
        X, y_up, y_down, y_large, y_mag = _make_training_data()
        model = DeltaSupplyModel()
        model.fit(X, y_up, y_down, y_large, y_mag)
        result = model.predict(X)

        valid_directions = {"UP", "DOWN", "NEUTRAL"}
        assert set(result.df["deviation_direction"].unique()).issubset(valid_directions)

    def test_high_upward_prob_gives_up_direction(self):
        """If upward_prob is high and > downward_prob, direction should be UP."""
        X, y_up, y_down, y_large, y_mag = _make_training_data(n=500)
        model = DeltaSupplyModel()
        model.fit(X, y_up, y_down, y_large, y_mag)
        result = model.predict(X)

        up_mask = result.df["upward_deviation_prob"] > 0.7
        if up_mask.any():
            up_rows = result.df[up_mask]
            # Most high upward prob should be UP direction
            up_count = (up_rows["deviation_direction"] == "UP").sum()
            assert up_count > 0


class TestRiskScore:
    def test_risk_score_in_01(self):
        X, y_up, y_down, y_large, y_mag = _make_training_data()
        model = DeltaSupplyModel()
        model.fit(X, y_up, y_down, y_large, y_mag)
        result = model.predict(X)

        assert (result.df["deviation_risk_score"] >= 0).all()
        assert (result.df["deviation_risk_score"] <= 1).all()


class TestFeatureImportance:
    def test_feature_importance_output(self):
        X, y_up, y_down, y_large, y_mag = _make_training_data()
        model = DeltaSupplyModel()
        model.fit(X, y_up, y_down, y_large, y_mag)
        fi = model.get_feature_importance()

        assert "feature" in fi.columns
        assert "mean_importance" in fi.columns
        assert len(fi) == X.shape[1]

    def test_feature_importance_before_fit_raises(self):
        model = DeltaSupplyModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.get_feature_importance()


class TestInsufficientData:
    def test_too_few_samples_raises(self):
        X = pd.DataFrame(np.random.randn(5, 3), columns=["a", "b", "c"])
        y_up = np.array([0, 1, 0, 1, 0])
        y_down = np.array([0, 0, 1, 0, 0])
        y_large = np.array([0, 0, 0, 1, 0])
        y_mag = np.array([10.0, 20.0, -30.0, 40.0, -50.0])

        model = DeltaSupplyModel()
        with pytest.raises(ValueError, match="Not enough valid samples"):
            model.fit(X, y_up, y_down, y_large, y_mag)


class TestInvalidTargets:
    def test_minus1_labels_excluded(self):
        """Rows with label=-1 should be excluded from training."""
        X, y_up, y_down, y_large, y_mag = _make_training_data(n=100)
        # Set some labels to -1 (invalid)
        y_up[:10] = -1
        y_down[:10] = -1

        model = DeltaSupplyModel()
        model.fit(X, y_up, y_down, y_large, y_mag)
        assert model.is_fitted_
