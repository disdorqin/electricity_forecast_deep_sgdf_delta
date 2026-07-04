"""Negative risk model: HGB classifiers for negative price prediction.

Models:
  1. HGB classifier for negative_label (rt_actual < 0)
  2. HGB classifier for deep_negative_label (rt_actual <= -100)
  3. HGB classifier for relative_down_label (rt_actual - da_anchor <= -200)

Output:
  - negative_prob
  - deep_negative_prob
  - relative_down_prob
  - negative_risk_score
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
class NegativeRiskConfig:
    """Configuration for NegativeRiskModel."""
    prob_threshold: float = 0.5
    hgb_max_iter: int = 200
    hgb_max_depth: int = 6
    hgb_learning_rate: float = 0.1
    risk_score_weights: Optional[dict] = None  # weights for combining probs


@dataclass
class NegativeRiskPredictionResult:
    """Prediction output from NegativeRiskModel."""
    df: pd.DataFrame
    feature_importance: Optional[pd.DataFrame] = None
    status: str = "OK"
    reason: str = ""

    # Output columns:
    # negative_prob, deep_negative_prob, relative_down_prob,
    # negative_risk_score, confidence


class NegativeRiskModel:
    """Lightweight negative risk model using HGB classifiers.

    Fits separate classifiers for each negative price target:
      - negative_label (rt_actual < 0)
      - deep_negative_label (rt_actual <= -100)
      - relative_down_label (rt_actual - da_anchor <= -200)
    """

    def __init__(self, config: Optional[NegativeRiskConfig] = None):
        self.config = config or NegativeRiskConfig()
        self.models_: dict = {}
        self.is_fitted_: bool = False
        self.feature_columns_: list = []
        self._train_X_: Optional[pd.DataFrame] = None
        self._train_targets_: dict = {}

    def fit(
        self,
        X: pd.DataFrame,
        y_negative: np.ndarray,
        y_deep_negative: np.ndarray,
        y_relative_down: np.ndarray,
    ) -> "NegativeRiskModel":
        """Fit all sub-models.

        Args:
            X: Feature DataFrame.
            y_negative: Binary labels for negative price (rt < 0).
            y_deep_negative: Binary labels for deep negative (rt <= -100).
            y_relative_down: Binary labels for relative down (delta <= -200).

        Returns:
            self
        """
        self.feature_columns_ = list(X.columns)

        # Filter out rows where any target is -1 (invalid)
        valid_mask = (
            (y_negative >= 0) & (y_deep_negative >= 0) & (y_relative_down >= 0)
        )
        X_valid = X.loc[valid_mask].copy()
        y_neg = y_negative[valid_mask]
        y_deep = y_deep_negative[valid_mask]
        y_rel = y_relative_down[valid_mask]

        if len(X_valid) < 10:
            logger.warning(
                "Insufficient samples to fit NegativeRiskModel: %d. Need >= 10. "
                "Model will be marked as insufficient and predict NaN.",
                len(X_valid),
            )
            self.is_fitted_ = True
            self._insufficient_events_ = True
            self._skip_reason_ = f"INSUFFICIENT_EVENTS: only {len(X_valid)} valid samples"
            return self

        # Check for single-class targets
        single_class_targets = []
        for name, y in [("negative", y_neg), ("deep_negative", y_deep), ("relative_down", y_rel)]:
            unique_classes = np.unique(y)
            if len(unique_classes) < 2:
                single_class_targets.append(f"{name}={unique_classes}")
        
        if single_class_targets:
            logger.warning(
                "Single-class target detected: %s. "
                "Model will be marked as insufficient and predict NaN.",
                ", ".join(single_class_targets),
            )
            self.is_fitted_ = True
            self._insufficient_events_ = True
            self._skip_reason_ = f"SINGLE_CLASS_TARGET: {', '.join(single_class_targets)}"
            return self

        logger.info("Fitting NegativeRiskModel on %d samples, %d features",
                     len(X_valid), X_valid.shape[1])

        # HGB models
        if HAS_HGB:
            # Negative classifier
            self.models_["hgb_negative"] = HistGradientBoostingClassifier(
                max_iter=self.config.hgb_max_iter,
                max_depth=self.config.hgb_max_depth,
                learning_rate=self.config.hgb_learning_rate,
                random_state=42,
            )
            self.models_["hgb_negative"].fit(X_valid, y_neg)

            # Deep negative classifier
            self.models_["hgb_deep_negative"] = HistGradientBoostingClassifier(
                max_iter=self.config.hgb_max_iter,
                max_depth=self.config.hgb_max_depth,
                learning_rate=self.config.hgb_learning_rate,
                random_state=42,
            )
            self.models_["hgb_deep_negative"].fit(X_valid, y_deep)

            # Relative down classifier
            self.models_["hgb_relative_down"] = HistGradientBoostingClassifier(
                max_iter=self.config.hgb_max_iter,
                max_depth=self.config.hgb_max_depth,
                learning_rate=self.config.hgb_learning_rate,
                random_state=42,
            )
            self.models_["hgb_relative_down"].fit(X_valid, y_rel)

        # Baseline logistic regression (for negative as representative)
        self.models_["lr_negative"] = LogisticRegression(
            max_iter=1000, random_state=42, C=1.0,
        )
        self.models_["lr_negative"].fit(X_valid, y_neg)

        self._train_X_ = X_valid.copy()
        self._train_targets_ = {
            "hgb_negative": y_neg,
            "hgb_deep_negative": y_deep,
            "hgb_relative_down": y_rel,
        }

        self.is_fitted_ = True
        logger.info("NegativeRiskModel fitted successfully.")
        return self

    def predict(self, X: pd.DataFrame) -> NegativeRiskPredictionResult:
        """Generate negative risk predictions.

        Args:
            X: Feature DataFrame with same columns as training.

        Returns:
            NegativeRiskPredictionResult with risk scores and confidence.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Handle insufficient events case
        if hasattr(self, '_insufficient_events_') and self._insufficient_events_:
            logger.warning(
                "Predicting with insufficient events model. Returning NaN values. "
                "Reason: %s",
                getattr(self, '_skip_reason_', 'UNKNOWN'),
            )
            result_df = X.copy()
            result_df["negative_prob"] = np.nan
            result_df["deep_negative_prob"] = np.nan
            result_df["relative_down_prob"] = np.nan
            result_df["negative_risk_score"] = np.nan
            result_df["confidence"] = 0.0
            return NegativeRiskPredictionResult(
                df=result_df,
                status="INSUFFICIENT_EVENTS",
                reason=getattr(self, '_skip_reason_', 'INSUFFICIENT_EVENTS'),
            )

        result_df = X.copy()

        # HGB predictions (primary)
        if HAS_HGB and "hgb_negative" in self.models_:
            result_df["negative_prob"] = self.models_["hgb_negative"].predict_proba(X)[:, 1]
            result_df["deep_negative_prob"] = self.models_["hgb_deep_negative"].predict_proba(X)[:, 1]
            result_df["relative_down_prob"] = self.models_["hgb_relative_down"].predict_proba(X)[:, 1]
        else:
            # Fallback to baseline
            result_df["negative_prob"] = self.models_["lr_negative"].predict_proba(X)[:, 1]
            result_df["deep_negative_prob"] = 0.0
            result_df["relative_down_prob"] = 0.0

        # Clip probabilities to [0, 1]
        for prob_col in ["negative_prob", "deep_negative_prob", "relative_down_prob"]:
            result_df[prob_col] = result_df[prob_col].clip(0.0, 1.0)

        # Risk score: weighted combination of probabilities
        weights = self.config.risk_score_weights or {
            "negative_prob": 0.3,
            "deep_negative_prob": 0.4,
            "relative_down_prob": 0.3,
        }
        result_df["negative_risk_score"] = (
            sum(result_df[k] * v for k, v in weights.items())
        ).clip(0.0, 1.0)

        # Confidence: max of classification probabilities
        probs = result_df[["negative_prob", "deep_negative_prob", "relative_down_prob"]]
        max_prob = probs.max(axis=1)
        result_df["confidence"] = max_prob.clip(0.0, 1.0)

        return NegativeRiskPredictionResult(df=result_df)

    def get_feature_importance(self, n_repeats: int = 5) -> pd.DataFrame:
        """Extract feature importance via permutation importance.

        Uses sklearn.inspection.permutation_importance since HGB models
        do not expose feature_importances_ directly.

        Returns:
            DataFrame with columns: feature, negative_importance,
            deep_negative_importance, relative_down_importance, mean_importance.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted.")

        from sklearn.inspection import permutation_importance

        if self._train_X_ is None or len(self._train_X_) == 0:
            return pd.DataFrame(columns=["feature", "mean_importance"])

        X_ref = self._train_X_
        importance_cols = {}

        for key, label in [
            ("hgb_negative", "negative_importance"),
            ("hgb_deep_negative", "deep_negative_importance"),
            ("hgb_relative_down", "relative_down_importance"),
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
