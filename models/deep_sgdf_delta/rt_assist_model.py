"""
RT-Assist Model Pack | Production export.

Final model定位:
  Primary: rt_pred = da_anchor
  Optional: safe_correction (disabled by default)
  Assist outputs: error probs, direction, uncertainty, reason_codes

This module implements the production model pack for RT-Assist-1.
No deep residual correction is applied by default.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Output Schema (hour-level) ────────────────────────────────────────
RT_ASSIST_OUTPUT_COLUMNS = [
    "business_day",
    "hour_business",
    "ds",
    "da_anchor",
    "rt_pred",
    "safe_correction",
    "final_pred_source",
    "da_error_prob_50",
    "da_error_prob_100",
    "da_error_prob_150",
    "da_error_prob_200",
    "prob_residual_up",
    "prob_residual_down",
    "prob_residual_neutral",
    "expected_abs_residual",
    "uncertainty_score",
    "correction_permission",
    "reason_codes",
    "model_version",
]


# ── Model Pack Config ──────────────────────────────────────────────────
class RTAssistConfig:
    """Configuration for RT-Assist model pack."""

    def __init__(
        self,
        model_version: str = "RT-Assist-1",
        enable_safe_correction: bool = False,
        alpha: float = 1.0,
        clip_correction: float = 0.0,
        uncertainty_threshold: float = 0.5,
    ):
        """
        Args:
            model_version: Model version string.
            enable_safe_correction: Whether to enable safe correction.
                                           Default=False (DA-only).
            alpha: Correction strength (only used if enable_safe_correction=True).
            clip_correction: Max absolute correction per hour (0 = no clip).
            uncertainty_threshold: Uncertainty threshold for correction permission.
        """
        self.model_version = model_version
        self.enable_safe_correction = enable_safe_correction
        self.alpha = alpha
        self.clip_correction = clip_correction
        self.uncertainty_threshold = uncertainty_threshold


# ── RTAssistModel ─────────────────────────────────────────────────────
class RTAssistModel:
    """RT-Assist production model pack.

    Primary prediction: rt_pred = da_anchor
    Optional safe correction: da_anchor + alpha * residual_pred
    Assist outputs: error probs, direction, uncertainty, reason_codes
    """

    def __init__(self, config: RTAssistConfig, manifest: Optional[Dict] = None):
        """
        Args:
            config: RTAssistConfig instance.
            manifest: Optional manifest dict (from export).
        """
        self.config = config
        self.manifest = manifest or {}
        self.models_loaded = False

    def load_models(self, model_dir: str):
        """Load trained models from exported model directory.

        Args:
            model_dir: Path to exported model directory.
        """
        model_path = Path(model_dir)
        self.manifest_path = model_path / "manifest.json"
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                self.manifest = json.load(f)

        # Try to load residual model (if exists)
        residual_model_path = model_path / "residual_model.pkl"
        self.residual_model = None
        if residual_model_path.exists():
            try:
                import pickle
                with open(residual_model_path, "rb") as f:
                    self.residual_model = pickle.load(f)
                logger.info(f"Loaded residual model from {residual_model_path}")
            except Exception as e:
                logger.warning(f"Could not load residual model: {e}")
                self.residual_model = None

        # Try to load classifier models (if exist)
        self.classifier_models = {}
        for name in ["da_error_prob", "residual_direction", "uncertainty"]:
            clf_path = model_path / f"{name}_model.pkl"
            if clf_path.exists():
                try:
                    import pickle
                    with open(clf_path, "rb") as f:
                        self.classifier_models[name] = pickle.load(f)
                    logger.info(f"Loaded {name} model from {clf_path}")
                except Exception as e:
                    logger.warning(f"Could not load {name} model: {e}")

        self.models_loaded = True
        logger.info(f"RTAssistModel loaded from {model_dir}")
        logger.info(f"  residual_model: {'LOADED' if self.residual_model else 'MISSING'}")
        logger.info(f"  classifiers: {list(self.classifier_models.keys())}")

    def predict(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Run prediction on input DataFrame.

        Args:
            df: Input DataFrame with columns:
                - ds (timestamp)
                - da_anchor (DA price)
                - <feature_columns> (if residual model is loaded)
            feature_columns: List of feature columns for residual model.
                             If None, auto-detect from manifest.

        Returns:
            DataFrame with RT_ASSIST_OUTPUT_COLUMNS.
        """
        df = df.copy()
        df["ds"] = pd.to_datetime(df["ds"])

        # Ensure business_time columns
        from models.deep_sgdf_delta.business_time import add_business_time_columns
        if "business_day" not in df.columns:
            df = add_business_time_columns(df, timestamp_col="ds")

        # ── Primary prediction: DA-only ───────────────────────────────
        da_anchor = df["da_anchor"].values.astype(float)

        # ── Optional safe correction ───────────────────────────────────
        safe_correction = np.zeros(len(df), dtype=float)
        residual_pred = np.zeros(len(df), dtype=float)

        if self.config.enable_safe_correction and self.residual_model is not None:
            # Predict residual
            if feature_columns is None:
                feature_columns = self.manifest.get("feature_columns", [])
            feature_columns = [c for c in feature_columns if c in df.columns]

            if len(feature_columns) > 0:
                X = df[feature_columns].values.astype(float)
                residual_pred = self.residual_model.predict(X) * self.config.alpha

                # Clip correction
                if self.config.clip_correction > 0:
                    residual_pred = np.clip(
                        residual_pred,
                        -self.config.clip_correction,
                        self.config.clip_correction,
                    )

                safe_correction = residual_pred

        rt_pred = da_anchor + safe_correction

        # ── Assist outputs (placeholder if models missing) ────────────
        # DA error probability (placeholder: use heuristic)
        da_error_prob_50 = self._heuristic_da_error_prob(df, threshold=50)
        da_error_prob_100 = self._heuristic_da_error_prob(df, threshold=100)
        da_error_prob_150 = self._heuristic_da_error_prob(df, threshold=150)
        da_error_prob_200 = self._heuristic_da_error_prob(df, threshold=200)

        # Residual direction (placeholder)
        prob_up = np.ones(len(df)) * 0.33
        prob_down = np.ones(len(df)) * 0.33
        prob_neutral = np.ones(len(df)) * 0.34

        # Uncertainty (placeholder: based on DA price volatility)
        uncertainty = self._heuristic_uncertainty(df)

        # Correction permission
        correction_permission = (
            (uncertainty < self.config.uncertainty_threshold).astype(int)
            if self.config.enable_safe_correction
            else np.zeros(len(df), dtype=int)
        )

        # Reason codes
        reason_codes = np.where(
            self.config.enable_safe_correction,
            "SAFE_CORRECTION_ENABLED",
            "DA_ONLY",
        )

        # ── Build output DataFrame ─────────────────────────────────────
        output = pd.DataFrame({
            "business_day": df["business_day"].values,
            "hour_business": df["hour_business"].values,
            "ds": df["ds"].values,
            "da_anchor": da_anchor,
            "rt_pred": rt_pred,
            "safe_correction": safe_correction,
            "final_pred_source": "DA_ONLY" if not self.config.enable_safe_correction else "DA+RESIDUAL",
            "da_error_prob_50": da_error_prob_50,
            "da_error_prob_100": da_error_prob_100,
            "da_error_prob_150": da_error_prob_150,
            "da_error_prob_200": da_error_prob_200,
            "prob_residual_up": prob_up,
            "prob_residual_down": prob_down,
            "prob_residual_neutral": prob_neutral,
            "expected_abs_residual": np.abs(safe_correction),
            "uncertainty_score": uncertainty,
            "correction_permission": correction_permission,
            "reason_codes": reason_codes,
            "model_version": self.config.model_version,
        })

        return output[RT_ASSIST_OUTPUT_COLUMNS]

    def _heuristic_da_error_prob(self, df: pd.DataFrame, threshold: float) -> np.ndarray:
        """Heuristic DA error probability based on DA price level.

        Args:
            df: Input DataFrame.
            threshold: Error threshold (e.g., 50 = 50 CNY/MWh).

        Returns:
            Array of probabilities (0-1).
        """
        da = df["da_anchor"].values.astype(float)
        # Higher DA price → higher chance of large negative residual (price drops)
        # Lower DA price → higher chance of large positive residual (price spikes)
        prob = np.where(
            da > 500,
            0.3,  # High DA → moderate error prob
            np.where(
                da < 0,
                0.4,  # Negative DA → high error prob
                np.where(
                    np.abs(da) < 100,
                    0.1,  # Mid DA → low error prob
                    0.2,  # Default
                ),
            ),
        )
        return prob.astype(float)

    def _heuristic_uncertainty(self, df: pd.DataFrame) -> np.ndarray:
        """Heuristic uncertainty based on DA price volatility.

        Args:
            df: Input DataFrame.

        Returns:
            Array of uncertainty scores (0-1, higher = more uncertain).
        """
        da = df["da_anchor"].values.astype(float)
        # High uncertainty for extreme DA prices
        uncertainty = np.where(
            (da > 500) | (da < 0),
            0.7,  # Extreme prices → high uncertainty
            np.where(
                np.abs(da - 300) < 100,
                0.3,  # Near-normal prices → low uncertainty
                0.5,  # Default
            ),
        )
        return uncertainty.astype(float)

    def save(self, output_dir: str):
        """Save model pack to directory.

        Args:
            output_dir: Output directory path.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save manifest
        manifest = {
            "model_version": self.config.model_version,
            "enable_safe_correction": self.config.enable_safe_correction,
            "alpha": self.config.alpha,
            "clip_correction": self.config.clip_correction,
            "uncertainty_threshold": self.config.uncertainty_threshold,
            "models_loaded": self.models_loaded,
            "residual_model_loaded": self.residual_model is not None,
            "classifier_models_loaded": list(self.classifier_models.keys()),
        }

        with open(output_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        # Save residual model (if loaded)
        if self.residual_model is not None:
            import pickle
            with open(output_path / "residual_model.pkl", "wb") as f:
                pickle.dump(self.residual_model, f)

        # Save classifier models (if loaded)
        for name, model in self.classifier_models.items():
            import pickle
            with open(output_path / f"{name}_model.pkl", "wb") as f:
                pickle.dump(model, f)

        logger.info(f"Model pack saved to {output_dir}")


# ── Factory function ──────────────────────────────────────────────────
def create_rt_assist_model(
    model_dir: Optional[str] = None,
    enable_safe_correction: bool = False,
    **kwargs,
) -> RTAssistModel:
    """Create RT-Assist model pack.

    Args:
        model_dir: Path to exported model directory (optional).
        enable_safe_correction: Whether to enable safe correction.
        **kwargs: Additional arguments for RTAssistConfig.

    Returns:
        RTAssistModel instance.
    """
    config = RTAssistConfig(
        enable_safe_correction=enable_safe_correction,
        **kwargs,
    )
    model = RTAssistModel(config)

    if model_dir is not None:
        model.load_models(model_dir)

    return model
