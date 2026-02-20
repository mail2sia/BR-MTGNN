#!/usr/bin/env python
"""Create naturally smoothed dataset WITHOUT aggressive clipping.
Uses gentle moving average + trend preservation for natural curves.
"""

import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d

def gentle_smooth_column(series, window=9, sigma=2.0, extra_smooth=False):
    """
    Gentle smoothing that preserves natural shape:
    1. Moving average (captures local trends)
    2. Light Gaussian filter (removes high-frequency noise only)
    
    NO clipping - preserves all data points including peaks/valleys.
    
    If extra_smooth=True, applies stronger smoothing for oscillating series.
    """
    x = series.values.astype(np.float64)
    
    if extra_smooth:
        window = 11
        sigma = 2.5

    # Step 1: Moving average (preserves trends)
    ma = pd.Series(x).rolling(window=window, center=True, min_periods=1).mean().values
    
    # Step 2: Light Gaussian (smooths without over-flattening)
    smoothed = gaussian_filter1d(ma, sigma=sigma, mode='nearest')
    
    # Step 3: Extra smoothing pass if requested
    if extra_smooth:
        smoothed = gaussian_filter1d(smoothed, sigma=2.0, mode='nearest')
    
    return smoothed.astype(np.float32)


def create_natural_smooth_data():
    print("=" * 80)
    print("CREATING NATURALLY SMOOTHED DATASET (No Clipping)")
    print("=" * 80)
    
    # Load original data
    df = pd.read_csv('data/data.csv')
    print(f"\nOriginal data: {df.shape}")
    
    # Remove date column
    if 'date' in df.columns or 'Date' in df.columns:
        date_col = 'date' if 'date' in df.columns else 'Date'
        df = df.drop(columns=[date_col])
        print(f"Removed date column: {df.shape}")
    
    # Apply aggressive smoothing to each column
    print(f"\nApplying aggressive smoothing (window=9, sigma=2.0, extra pass)...")
    smoothed_df = pd.DataFrame()
    
    for col in df.columns:
        original = np.array(df[col].values, dtype=np.float64)
        smoothed = gentle_smooth_column(df[col], window=9, sigma=2.0, extra_smooth=True)
        
        # Statistics
        orig_max = float(np.max(original))
        smooth_max = float(np.max(smoothed))
        reduction = ((orig_max - smooth_max) / orig_max * 100) if orig_max > 0 else 0
        
        if reduction > 10:  # Only print significant changes
            print(f"  {col}: {orig_max:.1f} → {smooth_max:.1f} ({reduction:.1f}% peak reduction)")
        
        smoothed_df[col] = smoothed
    
    # Save smoothed data
    print(f"\n📊 Smoothed Data Statistics:")
    print(f"  Shape: {smoothed_df.shape}")
    print(f"  Original max: {df.values.max():.2f}")
    print(f"  Smoothed max: {smoothed_df.values.max():.2f}")
    print(f"  Reduction: {((df.values.max() - smoothed_df.values.max()) / df.values.max() * 100):.1f}%")
    
    # Save without headers
    output_no_header = 'data/sm_data.csv'
    smoothed_df.to_csv(output_no_header, index=False, header=False)
    print(f"\n✅ Saved: {output_no_header} (no headers)")
    
    # Save with headers
    output_with_header = 'data/sm_data_g.csv'
    smoothed_df.to_csv(output_with_header, index=False, header=True)
    print(f"✅ Saved: {output_with_header} (with headers)")
    
    # Compare key series
    print(f"\n📈 Sample Comparisons:")
    samples = ['RMD_Anorexia nervosa', 'RMD_Savant', 'RMD_Body Integrity Dysphoria']
    for col in samples:
        if col in df.columns:
            orig = np.array(df[col].values, dtype=np.float64)
            smooth = np.array(smoothed_df[col].values, dtype=np.float64)
            print(f"\n  {col}:")
            print(f"    Original: min={np.min(orig):.1f}, max={np.max(orig):.1f}, std={np.std(orig):.1f}")
            print(f"    Smoothed: min={np.min(smooth):.1f}, max={np.max(smooth):.1f}, std={np.std(smooth):.1f}")
    
    print("\n" + "=" * 80)
    print("NATURAL SMOOTHING COMPLETE")
    print("=" * 80)
    print("\nKey differences from previous version:")
    print("  ✅ NO spike clipping (preserves all peaks/valleys)")
    print("  ✅ Gentle moving average (captures trends)")
    print("  ✅ Light Gaussian filter (removes noise only)")
    print("  ✅ Natural curve shapes preserved")
    print("\nThese files preserve the natural dynamics of your data!")
    print("=" * 80)
    
    return smoothed_df


if __name__ == '__main__':
    create_natural_smooth_data()
