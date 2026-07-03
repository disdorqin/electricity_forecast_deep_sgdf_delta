"""DeltaSupply model: lightweight classifiers + regressors for deviation risk.

Models:
  1. HGB classifier for upward_deviation_label
  2. HGB classifier for downward_deviation_label
  3. HGB classifier for large_abs_deviation_label
  4. HGB regressor for deviation_magnitude_target
  5. Logistic regression baseline classifier
  6. Ridge regressor baseline

Output:
  - upward_deviation_prob
  - downward_deviation_prob
  - large_abs_deviation_prob
  - deviation_magnitude_pred
  - deviation_risk_score
  - deviation_direction
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
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )
    HAS_HGB = True
except ImportError:
    HAS_HGB = False

from sklearn.linear_model import LogisticRegression, Ridge


@dataclass
class DeltaSupplyConfig:
    """Configuration for DeltaSupplyModel."""
    prob_threshold: float = 0.5
    hgb_max_iter: int = 200
    hgb_max_depth: int = 6
    hgb_learning_rate: float = 0.1
    ridge_alpha: float = 1.0
    magnitude_scale: float = 300.0  # for risk score normalization


@dataclass
class DeltaSupplyPredictionResult:
    """Prediction output from DeltaSupplyModel."""
    df: pd.DataFrame
    feature_importance: Optional[pd.DataFrame] = None

    # Output columns
    # upward_deviation_prob, downward_deviation_prob, large_abs_deviation_prob,
    # deviation_magnitude_pred, deviation_risk_score, deviation_direction, confidence


class DeltaSupplyModel:
    """Lightweight deviation risk model using HGB + baselines.

    Fits separate models for each target:
      - upward_deviation_label (classification)
      - downward_deviation_label (classification)
      - large_abs_deviation_label (classification)
      - deviation_magnitude_target (regression)
    """

    def __init__(self, config: Optional[DeltaSupplyConfig] = None):
        self.config = config or DeltaSupplyConfig()
        self.models_: dict = {}
        self.is_fitted_: bool = False
        self.feature_columns_: list = []
        self._train_X_: Optional[pd.DataFrame] = None
        self._train_targets_: dict = {}

    def fit(
        self,
        X: pd.DataFrame,
        y_upward: np.ndarray,
        y_downward: np.ndarray,
        y_large_abs: np.ndarray,
        y_magnitude: np.ndarray,
    ) -> "DeltaSupplyModel":
        """Fit all sub-models.

        Args:
            X: Feature DataFrame.
            y_upward: Binary labels for upward deviation.
            y_downward: Binary labels for downward deviation.
            y_large_abs: Binary labels for large absolute deviation.
            y_magnitude: Continuous target for deviation magnitude.

        Returns:
            self
        """
        self.feature_columns_ = list(X.columns)

        # Filter out rows where any target is -1 (invalid)
        valid_mask = (
            (y_upward >= 0) & (y_downward >= 0) & (y_large_abs >= 0)
            & ~np.isnan(y_magnitude)
        )
        X_valid = X.loc[valid_mask].copy()
        y_up = y_upward[valid_mask]
        y_down = y_downward[valid_mask]
        y_large = y_large_abs[valid_mask]
        y_mag = y_magnitude[valid_mask]

        if len(X_valid) < 10:
            raise ValueError(
                f"Not enough valid samples to fit: {len(X_valid)}. Need >= 10."
            )

        logger.info("Fitting DeltaSupplyModel on %d samples, %d features",
                     len(X_valid), X_valid.shape[1])

        # HGB models
        if HAS_HGB:
            # Upward classifier
            self.models_["hgb_upward"] = HistGradientBoostingClassifier(
                max_iter=self.config.hgb_max_iter,
                max_depth=self.config.hgb_max_depth,
                learning_rate=self.config.hgb_learning_rate,
                random_state=42,
            )
            self.models_["hgb_upward"].fit(X_valid, y_up)

            # Downward classifier
            self.models_["hgb_downward"] = HistGradientBoostingClassifier(
                max_iter=self.config.hgb_max_iter,
                max_depth=self.config.hgb_max_depth,
                learning_rate=self.config.hgb_learning_rate,
                random_state=42,
            )
            self.models_["hgb_downward"].fit(X_valid, y_down)

            # Large abs classifier
            self.models_["hgb_large_abs"] = HistGradientBoostingClassifier(
                max_iter=self.config.hgb_max_iter,
                max_depth=self.config.hgb_max_depth,
                learning_rate=self.config.hgb_learning_rate,
                random_state=42,
            )
            self.models_["hgb_large_abs"].fit(X_valid, y_large)

            # Magnitude regressor
            self.models_["hgb_magnitude"] = HistGradientBoostingRegressor(
                max_iter=self.config.hgb_max_iter,
                max_depth=self.config.hgb_max_depth,
                learning_rate=self.config.hgb_learning_rate,
                random_state=42,
            )
            self.models_["hgb_magnitude"].fit(X_valid, y_mag)

        # Baseline models
        # Logistic regression for upward (as representative baseline)
        self.models_["lr_upward"] = LogisticRegression(
            max_iter=1000, random_state=42, C=1.0,
        )
        self.models_["lr_upward"].fit(X_valid, y_up)

        # Ridge regressor baseline
        self.models_["ridge_magnitude"] = Ridge(alpha=self.config.ridge_alpha)
        self.models_["ridge_magnitude"].fit(X_valid, y_mag)

        self._train_X_ = X_valid.copy()
        self._train_targets_ = {
            "hgb_upward": y_up,
            "hgb_downward": y_down,
            "hgb_large_abs": y_large,
            "hgb_magnitude": y_mag,
        }

        self.is_fitted_ = True
        logger.info("DeltaSupplyModel fitted successfully.")
        return self

    def predict(self, X: pd.DataFrame) -> DeltaSupplyPredictionResult:
        """Generate deviation risk predictions.

        Args:
            X: Feature DataFrame with same columns as training.

        Returns:
            DeltaSupplyPredictionResult with risk scores and directions.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")

        result_df = X.copy()

        # HGB predictions (primary)
        if HAS_HGB and "hgb_upward" in self.models_:
            result_df["upward_deviation_prob"] = self.models_["hgb_upward"].predict_proba(X)[:, 1]
            result_df["downward_deviation_prob"] = self.models_["hgb_downward"].predict_proba(X)[:, 1]
            result_df["large_abs_deviation_prob"] = self.models_["hgb_large_abs"].predict_proba(X)[:, 1]
            result_df["deviation_magnitude_pred"] = self.models_["hgb_magnitude"].predict(X)
        else:
            # Fallback to baseline
            result_df["upward_deviation_prob"] = self.models_["lr_upward"].predict_proba(X)[:, 1]
            result_df["downward_deviation_prob"] = 0.0  # No baseline for downward
            result_df["large_abs_deviation_prob"] = 0.0  # No baseline for large_abs
            result_df["deviation_magnitude_pred"] = self.models_["ridge_magnitude"].predict(X)

        # Clip probabilities to [0, 1]
        for prob_col in ["upward_deviation_prob", "downward_deviation_prob", "large_abs_deviation_prob"]:
            result_df[prob_col] = result_df[prob_col].clip(0.0, 1.0)

        # Risk score
        magnitude_score = (
            result_df["deviation_magnitude_pred"].abs() / self.config.magnitude_scale
        ).clip(0.0, 1.0)

        result_df["deviation_risk_score"] = (
            result_df[["upward_deviation_prob", "downward_deviation_prob", "large_abs_deviation_prob"]]
            .max(axis=1) * magnitude_score
        ).clip(0.0, 1.0)

        # Direction
        thr = self.config.prob_threshold
        conditions = [
            (result_df["upward_deviation_prob"] >= thr)
            & (result_df["upward_deviation_prob"] > result_df["downward_deviation_prob"]),
            (result_df["downward_deviation_prob"] >= thr)
            & (result_df["downward_deviation_prob"] > result_df["upward_deviation_prob"]),
        ]
        result_df["deviation_direction"] = np.select(
            conditions, ["UP", "DOWN"], default="NEUTRAL"
        )

        # Confidence: average of classification model agreement
        if HAS_HGB and "hgb_upward" in self.models_:
            probs = result_df[["upward_deviation_prob", "downward_deviation_prob", "large_abs_deviation_prob"]]
            max_prob = probs.max(axis=1)
            result_df["confidence"] = max_prob.clip(0.0, 1.0)
        else:
            result_df["confidence"] = result_df["upward_deviation_prob"].clip(0.0, 1.0)

        return DeltaSupplyPredictionResult(df=result_df)

    def get_feature_importance(self, n_repeats: int = 5) -> pd.DataFrame:
        """Extract feature importance via permutation importance.

        Uses sklearn.inspection.permutation_importance since HGB models
        do not expose feature_importances_ directly.

        Returns:
            DataFrame with columns: feature, upward_importance, downward_importance,
            large_abs_importance, magnitude_importance, mean_importance.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted.")

        from sklearn.inspection import permutation_importance

        if self._train_X_ is None or len(self._train_X_) == 0:
            return pd.DataFrame(columns=["feature", "mean_importance"])

        X_ref = self._train_X_
        importance_cols = {}

        for key, label in [
            ("hgb_upward", "upward_importance"),
            ("hgb_downward", "downward_importance"),
            ("hgb_large_abs", "large_abs_importance"),
            ("hgb_magnitude", "magnitude_importance"),
        ]:
            if key in self.models_ and key in self._train_targets_:
                y_ref = self._train_targets_[key]
                scorer = "accuracy" if "upward" in key or "downward" in key or "large_abs" in key else "r2"
                try:
                    perm_result = permutation_importance(
                        self.models_[key], X_ref, y_ref,
                        n_repeats=n_repeats, random_state=42,
                        scoring=scorer,
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
