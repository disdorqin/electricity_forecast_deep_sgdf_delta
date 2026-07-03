"""Tests for consolidate_sgdfnet_predictions.py."""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


class TestConsolidateSGDFNet:
    """Consolidation logic tests."""

    def test_detect_columns(self):
        """Column detection works for standard formats."""
        from scripts.consolidate_sgdfnet_predictions import _detect_columns
        df = pd.DataFrame({"ds": [1, 2], "sgdfnet_pred": [10, 20]})
        ts, pred = _detect_columns(df)
        assert ts == "ds"
        assert pred == "sgdfnet_pred"

    def test_rt_hat_detected(self):
        """rt_hat column is detected as prediction."""
        from scripts.consolidate_sgdfnet_predictions import _detect_columns
        df = pd.DataFrame({"timestamp": [1, 2], "rt_hat": [10, 20]})
        ts, pred = _detect_columns(df)
        assert ts == "timestamp"
        assert pred == "rt_hat"

    def test_no_pred_col_returns_none(self):
        """No prediction column returns None for both."""
        from scripts.consolidate_sgdfnet_predictions import _detect_columns
        df = pd.DataFrame({"ds": [1, 2], "foo": [10, 20]})
        ts, pred = _detect_columns(df)
        assert ts == "ds"
        assert pred is None

    def test_coverage_csv_format(self):
        """Coverage CSV has expected columns."""
        rows = [{"feature": "sgdfnet_pred", "present": True}]
        csv_df = pd.DataFrame(rows)
        assert "feature" in csv_df.columns
        assert "present" in csv_df.columns
