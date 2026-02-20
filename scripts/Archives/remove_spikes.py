#!/usr/bin/env python3
"""
Remove noise spikes from time series using median filter + percentile clipping.
This creates cleaner training data and prettier plots.

Usage:
    python scripts/remove_spikes.py --input data/sm_aggr_v4.csv --output data/sm_aggr_v4_despike.csv
"""

import numpy as np
import pandas as pd
import argparse
from scipy.ndimage import median_filter


def remove_spikes(data: np.ndarray, 
                  median_window: int = 3,
                  percentile_clip: float = 95.0,
                  max_jump_std: float = 3.0) -> np.ndarray:
    """
    Remove spikes using three complementary techniques:
    1. Median filter to remove sharp isolated spikes
    2. Percentile clipping to cap extreme outliers
    3. Max-jump detection to smooth sudden transitions
    
    Args:
        data: 2D array (time, nodes)
        median_window: Window size for median filter (odd number, e.g., 3 or 5)
        percentile_clip: Clip values above this percentile per column
        max_jump_std: Maximum allowed jump in units of column std dev
    
    Returns:
        Cleaned 2D array
    """
    cleaned = data.copy()
    T, N = cleaned.shape
    
    # Step 1: Median filter per column to remove isolated spikes
    for n in range(N):
        col = cleaned[:, n]
        # Apply median filter (handles edges by reflection)
        cleaned[:, n] = median_filter(col, size=median_window, mode='reflect')
    
    # Step 2: Percentile clipping per column
    for n in range(N):
        col = cleaned[:, n]
        threshold = np.percentile(col[col > 0], percentile_clip) if np.any(col > 0) else np.max(col)
        cleaned[:, n] = np.clip(col, 0, threshold)
    
    # Step 3: Smooth sudden jumps (optional, more conservative)
    if max_jump_std > 0:
        for n in range(N):
            col = cleaned[:, n]
            if len(col) < 2:
                continue
            col_std = np.std(col) if np.std(col) > 0 else 1.0
            
            # Forward pass: detect and dampen upward jumps
            for t in range(1, T):
                jump = col[t] - col[t-1]
                if jump > max_jump_std * col_std:
                    # Dampen the jump to max_jump_std
                    cleaned[t, n] = col[t-1] + max_jump_std * col_std
            
            # Backward pass: detect and dampen downward jumps
            col = cleaned[:, n]  # Get updated values
            for t in range(T-2, -1, -1):
                jump = col[t] - col[t+1]
                if jump > max_jump_std * col_std:
                    cleaned[t, n] = col[t+1] + max_jump_std * col_std
    
    return cleaned


def main():
    parser = argparse.ArgumentParser(description='Remove noise spikes from time series data')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file')
    parser.add_argument('--median_window', type=int, default=5, 
                       help='Median filter window size (odd number, e.g., 3, 5, 7)')
    parser.add_argument('--percentile_clip', type=float, default=92.0,
                       help='Clip values above this percentile per column (0-100)')
    parser.add_argument('--max_jump_std', type=float, default=2.5,
                       help='Maximum allowed jump in units of std dev (0 to disable)')
    
    args = parser.parse_args()
    
    # Read data
    try:
        df = pd.read_csv(args.input, header=None)
        print(f"Loaded {args.input}: {df.shape[0]} rows × {df.shape[1]} cols")
    except Exception:
        # Try with header
        df = pd.read_csv(args.input)
        print(f"Loaded {args.input} with header: {df.shape[0]} rows × {df.shape[1]} cols")
    
    data = df.values.astype(float)
    
    # Remove spikes
    print(f"\nRemoving spikes:")
    print(f"  - Median filter window: {args.median_window}")
    print(f"  - Percentile clip: {args.percentile_clip}%")
    print(f"  - Max jump: {args.max_jump_std} std dev")
    
    cleaned = remove_spikes(
        data,
        median_window=args.median_window,
        percentile_clip=args.percentile_clip,
        max_jump_std=args.max_jump_std
    )
    
    # Statistics
    total_vals = data.size
    changed = np.sum(np.abs(cleaned - data) > 1e-6)
    pct_changed = 100.0 * changed / total_vals
    
    print(f"\nResults:")
    print(f"  - Values modified: {changed:,} / {total_vals:,} ({pct_changed:.2f}%)")
    print(f"  - Original range: [{np.min(data):.2f}, {np.max(data):.2f}]")
    print(f"  - Cleaned range: [{np.min(cleaned):.2f}, {np.max(cleaned):.2f}]")
    print(f"  - Max reduction: {np.max(data) - np.max(cleaned):.2f}")
    
    # Save
    pd.DataFrame(cleaned).to_csv(args.output, index=False, header=False)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
