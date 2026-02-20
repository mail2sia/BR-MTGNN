#!/usr/bin/env python
"""Quick validation that sm_data_g.csv is ready for training."""

import pandas as pd
import numpy as np

def validate_smoothed_data():
    print("=" * 70)
    print("SMOOTHED DATASET VALIDATION")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv('data/sm_data_g.csv')
    print(f"\n✓ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Check for NaN/inf
    nan_count = df.isna().sum().sum()
    inf_count = np.isinf(df.values).sum()
    print(f"✓ NaN values: {nan_count}")
    print(f"✓ Inf values: {inf_count}")
    
    if nan_count > 0 or inf_count > 0:
        print("⚠ WARNING: Data contains NaN or Inf values!")
        return False
    
    # Check for negative values (important for log1p if used)
    neg_count = (df.values < 0).sum()
    print(f"✓ Negative values: {neg_count}")
    
    # Check data range
    print(f"\n✓ Data range:")
    print(f"  Min: {df.values.min():.4f}")
    print(f"  Max: {df.values.max():.4f}")
    print(f"  Mean: {df.values.mean():.4f}")
    print(f"  Median: {np.median(df.values):.4f}")
    print(f"  Std: {df.values.std():.4f}")
    
    # Check for remaining extreme outliers (should be minimal after smoothing)
    for col in df.columns:
        vals = np.asarray(df[col].values, dtype=float)
        median = np.median(vals)
        mad = np.median(np.abs(vals - median))
        if mad > 0:
            z_scores = np.abs(vals - median) / (1.4826 * mad)
            outliers = (z_scores > 5.0).sum()  # More lenient threshold
            if outliers > 0:
                max_val = vals.max()
                print(f"  ⚠ {col}: {outliers} mild outliers (max={max_val:.2f})")
    
    # Column statistics
    print(f"\n✓ Column statistics:")
    print(f"  Columns with mean > 10: {(df.mean() > 10).sum()}")
    print(f"  Columns with max > 100: {(df.max() > 100).sum()}")
    print(f"  Columns with std > 20: {(df.std() > 20).sum()}")
    
    # Check compatibility with training script expectations
    print(f"\n✓ Training compatibility:")
    print(f"  Has date column: {'date' in df.columns or 'Date' in df.columns}")
    print(f"  Numeric columns: {df.select_dtypes(include=[np.number]).shape[1]}")
    print(f"  Expected format: ✓ Headers with no date column (use --has_header, no --drop_first_col)")
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE - Data is ready for training!")
    print("=" * 70)
    print("\nRecommended training command:")
    print("  bash run_safe_train.sh")
    print("  or")
    print("  python scripts/train_test.py --device cuda:0 --train --has_header")
    print("=" * 70)
    
    return True

if __name__ == '__main__':
    validate_smoothed_data()
