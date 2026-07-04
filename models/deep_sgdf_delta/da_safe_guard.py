"""
Phase C: Fire-rate Guard.

Safety gates to prevent catastrophic failures like 2026-02.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DASafeGuard:
    """Safety guards for DA-safe correction."""
    
    def __init__(self, config=None):
        """
        Initialize guard with config.
        
        Config parameters:
          - max_fire_rate_per_month: float (default 0.05 = 5%)
          - max_abs_correction: float (default 20)
          - enable_validation_regret_guard: bool (default True)
          - enable_distribution_shift_guard: bool (default True)
          - enable_normal_bucket_damage_guard: bool (default True)
          - enable_low_confidence_guard: bool (default True)
        """
        if config is None:
            config = {}
        
        self.config = {
            'max_fire_rate_per_month': 0.05,
            'max_abs_correction': 20,
            'enable_validation_regret_guard': True,
            'enable_distribution_shift_guard': True,
            'enable_normal_bucket_damage_guard': True,
            'enable_low_confidence_guard': True,
            'validation_regret_threshold': 0.0,  # No improvement = block
            'distribution_shift_threshold': 2.0,  # 2 std devs
            'normal_bucket_damage_threshold': 0.2,  # 0.2pp damage
            'low_confidence_threshold': 0.1  # 10% margin
        }
        self.config.update(config)
        
        # State
        self.monthly_fire_count = 0
        self.monthly_total_count = 0
        self.current_month = None
        self.blocked = False
        self.block_reason = None
    
    def reset_month(self, month):
        """Reset monthly counters for new month."""
        if month!= self.current_month:
            self.current_month = month
            self.monthly_fire_count = 0
            self.monthly_total_count = 0
            self.blocked = False
            self.block_reason = None
    
    def check_guards(self, row, correction, trigger_score, val_improvement=None, 
                     feature_dist=None, train_feature_mean=None, train_feature_std=None):
        """
        Check all safety guards.
        
        Args:
          row: dict with row data
          correction: float, raw correction value
          trigger_score: float, trigger probability
          val_improvement: float, validation improvement (for regret guard)
          feature_dist: float, current feature distribution distance (for shift guard)
          train_feature_mean: array, training feature means (for shift guard)
          train_feature_std: array, training feature stds (for shift guard)
        
        Returns:
          dict with decisions
        """
        decisions = {
            'raw_correction': correction,
            'safe_correction': correction,
            'fire': True,
            'blocked_by_guard': None,
            'block_reason': None
        }
        
        # Guard 1: Max correction magnitude
        if abs(correction) > self.config['max_abs_correction']:
            correction = np.clip(correction, 
                               -self.config['max_abs_correction'], 
                               self.config['max_abs_correction'])
            decisions['safe_correction'] = correction
            decisions['clipped'] = True
        
        # Guard 2: Max fire rate per month
        self.monthly_total_count += 1
        
        if self.monthly_fire_count / max(self.monthly_total_count, 1) > self.config['max_fire_rate_per_month']:
            decisions['fire'] = False
            decisions['blocked_by_guard'] = 'max_fire_rate'
            decisions['block_reason'] = f"Fire rate exceeded {self.config['max_fire_rate_per_month']*100}%"
            return decisions
        
        # Guard 3: Validation regret guard
        if self.config['enable_validation_regret_guard'] and val_improvement is not None:
            if val_improvement <= self.config['validation_regret_threshold']:
                decisions['fire'] = False
                decisions['blocked_by_guard'] = 'validation_regret'
                decisions['block_reason'] = f"Validation improvement {val_improvement:.4f} <= {self.config['validation_regret_threshold']}"
                return decisions
        
        # Guard 4: Distribution shift guard
        if self.config['enable_distribution_shift_guard'] and feature_dist is not None:
            if feature_dist > self.config['distribution_shift_threshold']:
                decisions['fire'] = False
                decisions['blocked_by_guard'] = 'distribution_shift'
                decisions['block_reason'] = f"Feature distribution shift {feature_dist:.2f} > {self.config['distribution_shift_threshold']}"
                return decisions
        
        # Guard 5: Low confidence guard
        if self.config['enable_low_confidence_guard']:
            # Trigger score margin from decision boundary (0.5)
            margin = abs(trigger_score - 0.5)
            if margin < self.config['low_confidence_threshold']:
                decisions['fire'] = False
                decisions['blocked_by_guard'] = 'low_confidence'
                decisions['block_reason'] = f"Trigger score margin {margin:.2f} < {self.config['low_confidence_threshold']}"
                return decisions
        
        # If we get here, correction is allowed
        decisions['safe_correction'] = correction
        self.monthly_fire_count += 1
        
        return decisions
    
    def apply_guard(self, df, raw_corrections, trigger_scores, validation_improvement=None,
                    feature_df=None, train_feature_df=None):
        """
        Apply guard to a dataframe of predictions.
        
        Args:
          df: DataFrame with columns ['business_day', 'hour_business', 'da_anchor']
          raw_corrections: array of raw correction values
          trigger_scores: array of trigger scores
          validation_improvement: float, validation improvement
          feature_df: DataFrame of current features (for shift detection)
          train_feature_df: DataFrame of training features (for shift detection)
        
        Returns:
          DataFrame with decision log
        """
        decisions = []
        
        for i in range(len(df)):
            row = df.iloc[i].to_dict()
            correction = raw_corrections[i]
            trigger_score = trigger_scores[i]
            
            # Reset month if needed
            month = pd.Timestamp(row['business_day']).to_period('M')
            self.reset_month(month)
            
            # Check guards
            decision = self.check_guards(
                row, correction, trigger_score, 
                val_improvement=validation_improvement
            )
            
            # Build log row
            log_row = {
                'business_day': row['business_day'],
                'hour_business': row['hour_business'],
                'da_anchor': row['da_anchor'],
                'raw_correction': decision['raw_correction'],
                'safe_correction': decision['safe_correction'],
                'final_pred': row['da_anchor'] + decision['safe_correction'],
                'fire': decision['fire'],
                'blocked_by_guard': decision['blocked_by_guard'],
                'block_reason': decision['block_reason'],
                'trigger_score': trigger_score
            }
            
            decisions.append(log_row)
        
        return pd.DataFrame(decisions)


def test_guard():
    """Test the guard module."""
    
    # Create dummy data
    n = 1000
    df = pd.DataFrame({
        'business_day': pd.date_range('2026-02-01', periods=n, freq='h'),
        'hour_business': np.random.randint(1, 25, n),
        'da_anchor': np.random.normal(300, 100, n)
    })
    
    raw_corrections = np.random.normal(0, 30, n)  # Some large corrections
    trigger_scores = np.random.uniform(0, 1, n)
    
    # Test with default config
    print("=== Testing DASafeGuard (default config) ===\n")
    
    guard = DASafeGuard(config={'max_fire_rate_per_month': 0.05, 'max_abs_correction': 20})
    
    log = guard.apply_guard(df, raw_corrections, trigger_scores, validation_improvement=0.5)
    
    print(f"Total rows: {len(log)}")
    print(f"Fire rate: {log['fire'].sum() / len(log) * 100:.2f}%")
    print(f"Blocked by guard:")
    print(log['blocked_by_guard'].value_counts())
    
    print(f"\nCorrection stats:")
    print(log[['raw_correction', 'safe_correction']].describe())
    
    print(f"\n=== Testing with very low validation improvement ===\n")
    
    # Test with validation_regret_guard
    guard2 = DASafeGuard(config={
        'max_fire_rate_per_month': 0.05,
        'max_abs_correction': 20,
        'enable_validation_regret_guard': True,
        'validation_regret_threshold': 0.0
    })
    
    log2 = guard2.apply_guard(df, raw_corrections, trigger_scores, validation_improvement=0.0)
    
    print(f"Fire rate with regret guard: {log2['fire'].sum() / len(log2) * 100:.2f}%")
    print(f"Blocked by regret guard: {(log2['blocked_by_guard'] == 'validation_regret').sum()}")
    
    print(f"\n=== Guard test complete ===")
    
    return log


if __name__ == '__main__':
    test_guard()
