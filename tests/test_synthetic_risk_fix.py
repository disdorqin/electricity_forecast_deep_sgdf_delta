"""Quick test: Verify synthetic risk is properly rejected."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.deep_sgdf_delta.deep_rt_sota_features import build_deep_rt_sota_features

def test_synthetic_risk():
    """Test that synthetic risk features are properly labeled."""

    print("Test: Synthetic risk features labeling")
    print("=" * 80)

    # Create dummy data
    dates = pd.date_range("2026-01-01", periods=48, freq="h")
    df = pd.DataFrame({
        "ds": dates,
        "rt_actual": np.random.rand(48) * 100 + 300,
        "da_anchor": np.random.rand(48) * 100 + 300,
    })

    from models.deep_sgdf_delta.business_time import add_business_time_columns
    df = add_business_time_columns(df, timestamp_col="ds")

    # Test 1: risk_features="off"
    print("\nTest 1: risk_features='off'")
    df1, manifest1 = build_deep_rt_sota_features(
        df.copy(),
        risk_features="off",
        forecast_features=False,
    )
    print(f"  risk_features_source: {manifest1.get('risk_features_source', 'NOT SET')}")
    assert manifest1.get("risk_features_source") == "off"
    print("  ✅ PASS")

    # Test 2: risk_features="synthetic"
    print("\nTest 2: risk_features='synthetic'")
    df2, manifest2 = build_deep_rt_sota_features(
        df.copy(),
        risk_features="synthetic",
        forecast_features=False,
    )
    print(f"  risk_features_source: {manifest2.get('risk_features_source', 'NOT SET')}")
    assert manifest2.get("risk_features_source") == "synthetic"

    # Check that synthetic risk features are generated
    risk_cols = ["negative_prob", "negative_risk_score", "spike_prob"]
    for col in risk_cols:
        if col in df2.columns:
            print(f"  {col}: {df2[col].mean():.4f} (should be ~0.5)")
            assert 0 <= df2[col].mean() <= 1.0
    print("  ✅ PASS")

    # Test 3: Verify run_data_audit rejects synthetic
    print("\nTest 3: run_data_audit rejects synthetic risk...")
    from scripts.train_deep_realtime_sota_v2 import run_data_audit
    audit_result = run_data_audit(
        df=df2,
        target_month="2026-02",
        seq_len_days=7,
        risk_features="synthetic",
    )
    print(f"  Audit verdict: {audit_result['verdict']}")
    assert audit_result["verdict"] == "FAIL"
    assert any("SYNTHETIC" in err for err in audit_result["errors"])
    print("  ✅ PASS")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    test_synthetic_risk()
