#!/usr/bin/env python3
"""
Create smoothed dataset from data.csv WITHOUT clipping.
Generates:
- sm_data.csv (no headers, no date column)
- sm_data_g.csv (with headers, no date column)
"""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import os


def moving_avg(a: np.ndarray, window: int = 3) -> np.ndarray:
    """Apply moving average filter."""
    if window <= 1:
        return a
    ret = np.convolve(a, np.ones(window)/window, mode='same')
    return ret


def smooth_column(col: np.ndarray, gauss_sigma: float = 1.5, ma_window: int = 5) -> np.ndarray:
    """
    Smooth a single column WITHOUT clipping.
    
    Steps:
    1. Apply Gaussian filter
    2. Apply moving average
    
    Args:
        col: Input column
        gauss_sigma: Gaussian filter sigma
        ma_window: Moving average window size
    
    Returns:
        Smoothed column
    """
    # Step 1: Gaussian smoothing
    if gauss_sigma > 0:
        smoothed = gaussian_filter1d(col, sigma=gauss_sigma, mode='reflect')
    else:
        smoothed = col
    
    # Step 2: Moving average
    if ma_window > 1:
        smoothed = moving_avg(smoothed, ma_window)
    
    return smoothed


def analyze_spikes(df: pd.DataFrame, k: float = 3.5):
    """
    Analyze and report spikes in the dataset.
    
    Args:
        df: DataFrame with numerical columns
        k: MAD threshold
    """
    print("\n" + "="*80)
    print("SPIKE ANALYSIS REPORT")
    print("="*80)
    print(f"Threshold: {k} * MAD (Median Absolute Deviation)\n")
    
    spike_report = []
    
    for col_name in df.columns:
        col = df[col_name].to_numpy(dtype=float)
        median = np.median(col)
        mad = np.median(np.abs(col - median))
        
        if mad < 1e-6:
            continue
        
        threshold = k * mad
        upper_bound = median + threshold
        lower_bound = median - threshold
        
        # Find spike indices
        spike_mask = (col > upper_bound) | (col < lower_bound)
        num_spikes = np.sum(spike_mask)
        
        if num_spikes > 0:
            spike_indices = np.where(spike_mask)[0]
            spike_values = col[spike_mask]
            max_spike = np.max(np.abs(spike_values - median))
            
            spike_report.append({
                'column': col_name,
                'num_spikes': num_spikes,
                'median': median,
                'mad': mad,
                'upper_bound': upper_bound,
                'lower_bound': lower_bound,
                'max_spike': max_spike,
                'spike_indices': spike_indices[:10]  # First 10 indices
            })
    
    # Sort by number of spikes (descending)
    spike_report = sorted(spike_report, key=lambda x: x['num_spikes'], reverse=True)
    
    print(f"Total columns with spikes: {len(spike_report)}\n")
    
    # Print top 20 columns with most spikes
    print("Top 20 columns with most spikes:")
    print("-" * 80)
    print(f"{'Column Name':<50} {'Spikes':>8} {'Median':>10} {'Max Spike':>12}")
    print("-" * 80)
    
    for item in spike_report[:20]:
        print(f"{item['column']:<50} {item['num_spikes']:>8} {item['median']:>10.2f} {item['max_spike']:>12.2f}")
    
    print("\n" + "="*80)
    print("Detailed spike locations (first 10 occurrences):")
    print("="*80)
    
    for item in spike_report[:10]:
        print(f"\n{item['column']}:")
        print(f"  Median: {item['median']:.2f}, MAD: {item['mad']:.2f}")
        print(f"  Bounds: [{item['lower_bound']:.2f}, {item['upper_bound']:.2f}]")
        print(f"  Spike row indices: {item['spike_indices'].tolist()}")
    
    print("\n" + "="*80)


def main():
    input_file = 'data/data.csv'
    output_file_no_header = 'data/sm_data.csv'
    output_file_with_header = 'data/sm_data_g.csv'
    
    print("Loading data.csv...")
    df = pd.read_csv(input_file)
    
    # Separate date column
    if 'date' in df.columns:
        date_col = df['date']
        data_cols = df.drop(columns=['date'])
    else:
        data_cols = df
    
    print(f"Data shape: {data_cols.shape[0]} rows × {data_cols.shape[1]} columns")
    
    # Convert to numpy array
    X = data_cols.to_numpy(dtype=float)
    T, N = X.shape
    
    print(f"\nProcessing {N} columns...")
    
    # Apply smoothing to each column
    S = np.empty_like(X, dtype=float)
    for j in range(N):
        col = X[:, j]
        S[:, j] = smooth_column(col, gauss_sigma=1.5, ma_window=5)
        
        if (j + 1) % 20 == 0:
            print(f"  Processed {j + 1}/{N} columns...")
    
    print(f"  Processed {N}/{N} columns.")
    
    # Save sm_data.csv (no headers, no date)
    print(f"\nSaving {output_file_no_header} (no headers)...")
    pd.DataFrame(S).to_csv(output_file_no_header, index=False, header=False)
    
    # Save sm_data_g.csv (with headers, no date)
    print(f"Saving {output_file_with_header} (with column names)...")
    pd.DataFrame(S, columns=data_cols.columns).to_csv(output_file_with_header, index=False)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Original data: {input_file}")
    print(f"  Shape: {T} rows × {N} columns")
    print(f"  Range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"\nSmoothed data:")
    print(f"  {output_file_no_header} (no headers)")
    print(f"  {output_file_with_header} (with headers)")
    print(f"  Shape: {S.shape[0]} rows × {S.shape[1]} columns")
    print(f"  Range: [{S.min():.2f}, {S.max():.2f}]")
    print(f"\nSmoothing parameters:")
    print(f"  Gaussian sigma: 1.5")
    print(f"  Moving average window: 5")
    print("="*80)


if __name__ == '__main__':
    main()
