"""Spike risk model: HGB classifiers for price spike prediction.

Models:
  1. HGB classifier for spike_label (rt_actual >= 500)
  2. HGB classifier for extreme_spike_label (rt_actual >= 800)
  3. HGB classifier for relative_spike_label (rt_actual - da_anchor >= 200)

Output:
  - spike_prob
  - extreme_spike_prob
  - relative_spike_prob
  - spike_risk_score
  - confidence
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try importing hist_gradient_boosting (sklearn >= 0.21)
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_HGB = True
except ImportError:
    HAS_HGB = False

from sklearn.linear_model import LogisticRegression


@dataclass
class SpikeRiskConfig:
    """Configuration for SpikeRiskModel."""
    prob_threshold: float = 0.5
    hgb_max_iter: int = 200
    hgb_max_depth: int = 6
    hgb_learning_rate: float = 0.1
    risk_score_weights: Optional[dict] = None  # weights for combining probs


@dataclass
class SpikeRiskPredictionResult:
    """Prediction output from SpikeRiskModel."""
    df: pd.DataFrame
    feature_importance: Optional[pd.DataFrame] = None

    # Output columns:
    # spike_prob, extreme_spike_prob, relative_spike_prob,
    # spike_risk_score, confidence


class SpikeRiskModel:
    """Lightweight spike risk model using HGB classifiers.

    Fits separate classifiers for each spike target:
      - spike_label (rt_actual >= 500)
      - extreme_spike_label (rt_actual >= 800)
      - relative_spike_label (rt_actual - da_anchor >= 200)
    """

    def __init__(self, config: Optional[SpikeRiskConfig] = None):
        self.config = config or SpikeRiskConfig()
        self.models_: dict = {}
        self.is_fitted_: bool = False
        self.feature_columns_: list = []
        self._train_X_: Optional[pd.DataFrame] = None
        self._train_targets_: dict = {}

    def fit(
        self,
        X: pd.DataFrame,
        y_spike: np.ndarray,
        y_extreme_spike: np.ndarray,
        y_relative_spike: np.ndarray,
    ) -> "SpikeRiskModel":
        """Fit all sub-models.

        Args:
            X: Feature DataFrame.
            y_spike: Binary labels for spike (rt >= 500).
            y_extreme_spike: Binary labels for extreme spike (rt >= 800).
            y_relative_spike: Binary labels for relative spike (delta >= 200).

        Returns:
            self
        """
        self.feature_columns_ = list(X.columns)

        # Filter out rows where any target is -1 (invalid)
        valid_mask = (
            (y_spike >= 0) & (y_extreme_spike >= 0) & (y_relative_spike >= 0)
        )
        X_valid = X.loc[valid_mask].copy()
        y_sp = y_spike[valid_mask]
        y_ext = y_extreme_spike[valid_mask]
        y_rel = y_relative_spike[valid_mask]

        if len(X_valid) < 10:
            raise ValueError(
                f"Not enough valid samples to fit: {len(X_valid)}. Need >= 10."
            )

        logger.info("Fitting SpikeRiskModel on %d samples, %d features",
                     len(X_valid), X_valid.shape[1])

        # HGB models
        if HAS_HGB:
            # Spike classifier
            self.models_["hgb_spike"] = HistGradientBoostingClassifier(
                max_iter=self.config.hgb_max_iter,
                max_depth=self.config.hgb_max_depth,
                learning_rate=self.config.hgb_learning_rate,
                random_state=42,
            )
            self.models_["hgb_spike"].fit(X_valid, y_sp)

            # Extreme spike classifier
            self.models_["hgb_extreme_spike"] = HistGradientBoostingClassifier(
                max_iter=self.config.hgb_max_iter,
                max_depth=self.config.hgb_max_depth,
                learning_rate=self.config.hgb_learning_rate,
                random_state=42,
            )
            self.models_["hgb_extreme_spike"].fit(X_valid, y_ext)

            # Relative spike classifier
            self.models_["hgb_relative_spike"] = HistGradientBoostingClassifier(
                max_iter=self.config.hgb_max_iter,
                max_depth=self.config.hgb_max_depth,
                learning_rate=self.config.hgb_learning_rate,
                random_state=42,
            )
            self.models_["hgb_relative_spike"].fit(X_valid, y_rel)

        # Baseline logistic regression (for spike as representative)
        self.models_["lr_spike"] = LogisticRegression(
            max_iter=1000, random_state=42, C=1.0,
        )
        self.models_["lr_spike"].fit(X_valid, y_sp)

        self._train_X_ = X_valid.copy()
        self._train_targets_ = {
            "hgb_spike": y_sp,
            "hgb_extreme_spike": y_ext,
            "hgb_relative_spike": y_rel,
        }

        self.is_fitted_ = True
        logger.info("SpikeRiskModel fitted successfully.")
        return self

    def predict(self, X: pd.DataFrame) -> SpikeRiskPredictionResult:
        """Generate spike risk predictions.

        Args:
            X: Feature DataFrame with same columns as training.

        Returns:
            SpikeRiskPredictionResult with risk scores and confidence.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")

        result_df = X.copy()

        # HGB predictions (primary)
        if HAS_HGB and "hgb_spike" in self.models_:
            result_df["spike_prob"] = self.models_["hgb_spike"].predict_proba(X)[:, 1]
            result_df["extreme_spike_prob"] = self.models_["hgb_extreme_spike"].predict_proba(X)[:, 1]
            result_df["relative_spike_prob"] = self.models_["hgb_relative_spike"].predict_proba(X)[:, 1]
        else:
            # Fallback to baseline
            result_df["spike_prob"] = self.models_["lr_spike"].predict_proba(X)[:, 1]
            result_df["extreme_spike_prob"] = 0.0
            result_df["relative_spike_prob"] = 0.0

        # Clip probabilities to [0, 1]
        for prob_col in ["spike_prob", "extreme_spike_prob", "relative_spike_prob"]:
            result_df[prob_col] = result_df[prob_col].clip(0.0, 1.0)

        # Risk score: weighted combination of probabilities
        weights = self.config.risk_score_weights or {
            "spike_prob": 0.3,
            "extreme_spike_prob": 0.4,
            "relative_spike_prob": 0.3,
        }
        result_df["spike_risk_score"] = (
            sum(result_df[k] * v for k, v in weights.items())
        ).clip(0.0, 1.0)

        # Confidence: max of classification probabilities
        probs = result_df[["spike_prob", "extreme_spike_prob", "relative_spike_prob"]]
        max_prob = probs.max(axis=1)
        result_df["confidence"] = max_prob.clip(0.0, 1.0)

        return SpikeRiskPredictionResult(df=result_df)

    def get_feature_importance(self, n_repeats: int = 5) -> pd.DataFrame:
        """Extract feature importance via permutation importance.

        Uses sklearn.inspection.permutation_importance since HGB models
        do not expose feature_importances_ directly.

        Returns:
            DataFrame with columns: feature, spike_importance,
            extreme_spike_importance, relative_spike_importance, mean_importance.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted.")

        from sklearn.inspection import permutation_importance

        if self._train_X_ is None or len(self._train_X_) == 0:
            return pd.DataFrame(columns=["feature", "mean_importance"])

        X_ref = self._train_X_
        importance_cols = {}

        for key, label in [
            ("hgb_spike", "spike_importance"),
            ("hgb_extreme_spike", "extreme_spike_importance"),
            ("hgb_relative_spike", "relative_spike_importance"),
        ]:
            if key in self.models_ and key in self._train_targets_:
                y_ref = self._train_targets_[key]
                try:
                    perm_result = permutation_importance(
                        self.models_[key], X_ref, y_ref,
                        n_repeats=n_repeats, random_state=42,
                        scoring="accuracy",
                    )
                    importance_cols[label] = perm_result.importances_mean
                except Exception:
                    importance_cols[label] = np.zeros(X_ref.shape[1])

        if not importance_cols:
            return pd.DataFrame(columns=["feature", "mean_importance"])

        fi_df = pd.DataFrame({"feature": self.feature_columns_})
        for col, vals in importance_cols.items():
            fi_df[col] = vals

        # Mean importance
        imp_cols = [c for c in fi_df.columns if c != "feature"]
        fi_df["mean_importance"] = fi_df[imp_cols].mean(axis=1)
        fi_df = fi_df.sort_values("mean_importance", ascending=False).reset_index(drop=True)

        return fi_df
