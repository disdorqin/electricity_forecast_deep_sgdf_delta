"""
Generate base predictions from existing OOF files.

This script converts existing OOF prediction files to standardized format
for shadow replay. Coverage may be incomplete.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

def convert_sgdfnet_oof_to_standardized(output_dir: Path):
    """Convert SGDFNet OOF predictions to standardized format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all SGDFNet OOF prediction files
    sgdfnet_preds = list(Path("../electricity_forecast_model2.0_exp/oof_runs").rglob("SGDFNet_*/predictions.csv"))
    
    print(f"Found {len(sgdfnet_preds)} SGDFNet OOF prediction files")
    
    all_preds = []
    for pred_file in sgdfnet_preds:
        try:
            df = pd.read_csv(pred_file)
            
            # Check required columns
            if 'business_day' not in df.columns or 'rt_actual' not in df.columns:
                print(f"  Skipping {pred_file.name}: missing required columns")
                continue
            
            # Standardize format
            df['business_day'] = pd.to_datetime(df['business_day'])
            df['hour_business'] = df['hour']
            df['target_month'] = df['business_day'].dt.strftime('%Y-%m')
            df['ds'] = df['business_day']
            df['base_pred'] = df['rt_actual']  # Use actual as placeholder (will be updated)
            df['base_model_name'] = 'sgdfnet'
            df['base_source'] = 'OOF_PREDICTION'
            df['y_true'] = df['rt_actual']
            
            all_preds.append(df)
            print(f"  Loaded {pred_file.name}: {len(df)} rows, date range: {df['business_day'].min()} to {df['business_day'].max()}")
            
        except Exception as e:
            print(f"  Error loading {pred_file}: {e}")
            continue
    
    if not all_preds:
        print("No valid SGDFNet OOF predictions found")
        return None
    
    # Combine all predictions
    combined = pd.concat(all_preds, ignore_index=True)
    
    # Remove duplicates
    key_cols = ['business_day', 'hour_business', 'target_month']
    combined = combined.drop_duplicates(subset=key_cols)
    
    # Save standardized predictions
    output_file = output_dir / 'sgdfnet_base_predictions.csv'
    combined.to_csv(output_file, index=False)
    
    print(f"\nStandardized predictions saved to: {output_file}")
    print(f"  Total rows: {len(combined)}")
    print(f"  Date range: {combined['business_day'].min()} to {combined['business_day'].max()}")
    print(f"  Months covered: {sorted(combined['target_month'].unique())}")
    
    # Check oracle baseline
    if 'base_pred' in combined.columns and 'y_true' in combined.columns:
        if np.allclose(combined['base_pred'].fillna(0), combined['y_true'].fillna(0)):
            print("\n⚠️ WARNING: Oracle baseline detected (base_pred == y_true)")
            print("  Price metrics evaluation will be INVALID")
    
    return output_file

if __name__ == '__main__':
    output_dir = Path("reports/local/ledger_2/base_predictions_standardized/sgdfnet")
    output_file = convert_sgdfnet_oof_to_standardized(output_dir)
    
    if output_file:
        print(f"\n✅ Base predictions generated: {output_file}")
    else:
        print("\n❌ Failed to generate base predictions")
