#!/usr/bin/env python3
"""
Detailed spike investigation across all columns in data.csv
"""
import pandas as pd
import numpy as np


def analyze_column_spikes(col_data, col_name, k=3.5):
    """Analyze spikes in a single column."""
    median = np.median(col_data)
    mad = np.median(np.abs(col_data - median))
    
    if mad < 1e-6:
        return None
    
    threshold = k * mad
    upper_bound = median + threshold
    lower_bound = median - threshold
    
    spike_mask = (col_data > upper_bound) | (col_data < lower_bound)
    num_spikes = np.sum(spike_mask)
    
    if num_spikes == 0:
        return None
    
    spike_indices = np.where(spike_mask)[0]
    spike_values = col_data[spike_mask]
    
    return {
        'column': col_name,
        'num_spikes': num_spikes,
        'pct_spikes': 100 * num_spikes / len(col_data),
        'median': median,
        'mean': np.mean(col_data),
        'std': np.std(col_data),
        'mad': mad,
        'min': np.min(col_data),
        'max': np.max(col_data),
        'upper_bound': upper_bound,
        'lower_bound': lower_bound,
        'max_spike_value': np.max(spike_values),
        'min_spike_value': np.min(spike_values),
        'max_deviation': np.max(np.abs(spike_values - median)),
        'spike_indices': spike_indices
    }


def main():
    # Load data
    df = pd.read_csv('data/data.csv')
    df_smooth = pd.read_csv('data/sm_data_g.csv')
    
    # Remove date column
    dates = None
    if 'date' in df.columns:
        dates = df['date']
        df = df.drop(columns=['date'])
    else:
        dates = pd.Series(range(len(df)))
    
    print("="*100)
    print("COMPREHENSIVE SPIKE INVESTIGATION - data.csv")
    print("="*100)
    print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Date range: {dates.iloc[0]} to {dates.iloc[-1]}")
    print(f"MAD threshold: 3.5")
    print("="*100)
    
    # Analyze all columns
    all_results = []
    for col in df.columns:
        result = analyze_column_spikes(df[col].values, col)
        if result:
            all_results.append(result)
    
    # Sort by different metrics
    print(f"\n{'='*100}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*100}")
    print(f"Total columns: {df.shape[1]}")
    print(f"Columns with spikes: {len(all_results)}")
    print(f"Columns without spikes: {df.shape[1] - len(all_results)}")
    print(f"Percentage with spikes: {100 * len(all_results) / df.shape[1]:.1f}%")
    
    # Extreme spikes (>100x median)
    extreme = [r for r in all_results if r['max_deviation'] > 100 * r['median']]
    print(f"\nExtreme outliers (>100x median): {len(extreme)}")
    
    # High spike frequency (>30% of data points)
    high_freq = [r for r in all_results if r['pct_spikes'] > 30]
    print(f"High spike frequency (>30% rows): {len(high_freq)}")
    
    print(f"\n{'='*100}")
    print(f"TOP 30 COLUMNS BY NUMBER OF SPIKES")
    print(f"{'='*100}")
    print(f"{'Rank':<5} {'Column':<50} {'Spikes':>7} {'% Data':>7} {'Max Val':>9} {'Median':>8}")
    print("-"*100)
    
    sorted_by_count = sorted(all_results, key=lambda x: x['num_spikes'], reverse=True)
    for i, r in enumerate(sorted_by_count[:30], 1):
        print(f"{i:<5} {r['column']:<50} {r['num_spikes']:>7} {r['pct_spikes']:>6.1f}% {r['max']:>9.1f} {r['median']:>8.2f}")
    
    print(f"\n{'='*100}")
    print(f"TOP 20 COLUMNS BY SPIKE SEVERITY (Max Deviation from Median)")
    print(f"{'='*100}")
    print(f"{'Rank':<5} {'Column':<50} {'Max Val':>9} {'Median':>8} {'Deviation':>11}")
    print("-"*100)
    
    sorted_by_severity = sorted(all_results, key=lambda x: x['max_deviation'], reverse=True)
    for i, r in enumerate(sorted_by_severity[:20], 1):
        print(f"{i:<5} {r['column']:<50} {r['max']:>9.1f} {r['median']:>8.2f} {r['max_deviation']:>11.1f}")
    
    print(f"\n{'='*100}")
    print(f"SPIKE PATTERNS BY TIME PERIOD")
    print(f"{'='*100}")
    
    # Analyze when spikes occur most frequently
    spike_timeline = np.zeros(len(df))
    for r in all_results:
        for idx in r['spike_indices']:
            spike_timeline[idx] += 1
    
    # Find top 10 time periods with most spikes
    top_spike_times = np.argsort(spike_timeline)[-10:][::-1]
    print(f"\nTop 10 time periods with most concurrent spikes:")
    print(f"{'Row':<5} {'Date':<12} {'Columns with Spikes':>20}")
    print("-"*40)
    for idx in top_spike_times:
        if spike_timeline[idx] > 0:
            print(f"{idx:<5} {dates.iloc[idx]:<12} {int(spike_timeline[idx]):>20}")
    
    # Compare original vs smoothed
    print(f"\n{'='*100}")
    print(f"SMOOTHING EFFECTIVENESS - Top 10 Most Reduced Columns")
    print(f"{'='*100}")
    print(f"{'Column':<50} {'Orig Max':>10} {'Smooth Max':>11} {'Reduction':>11}")
    print("-"*100)
    
    reductions = []
    for col in df.columns:
        orig_max = df[col].max()
        smooth_max = df_smooth[col].max()
        reduction_pct = 100 * (1 - smooth_max / orig_max) if orig_max > 0 else 0
        reductions.append({
            'column': col,
            'orig_max': orig_max,
            'smooth_max': smooth_max,
            'reduction_pct': reduction_pct
        })
    
    reductions_sorted = sorted(reductions, key=lambda x: x['reduction_pct'], reverse=True)
    for r in reductions_sorted[:10]:
        print(f"{r['column']:<50} {r['orig_max']:>10.1f} {r['smooth_max']:>11.2f} {r['reduction_pct']:>10.1f}%")
    
    # Detailed analysis of worst offenders
    print(f"\n{'='*100}")
    print(f"DETAILED ANALYSIS - TOP 5 WORST SPIKE OFFENDERS")
    print(f"{'='*100}")
    
    for i, r in enumerate(sorted_by_severity[:5], 1):
        col_name = r['column']
        print(f"\n{i}. {col_name}")
        print("-"*100)
        print(f"   Total spikes: {r['num_spikes']} ({r['pct_spikes']:.1f}% of data)")
        print(f"   Range: [{r['min']:.2f}, {r['max']:.2f}]")
        print(f"   Median: {r['median']:.2f}, Mean: {r['mean']:.2f}, Std: {r['std']:.2f}")
        print(f"   MAD: {r['mad']:.2f}, Bounds: [{r['lower_bound']:.2f}, {r['upper_bound']:.2f}]")
        print(f"   Max deviation: {r['max_deviation']:.2f} ({r['max_deviation']/r['median'] if r['median'] > 0 else 0:.1f}x median)")
        
        # Show top 5 spike locations
        col_data = df[col_name].values
        spike_mask = (col_data > r['upper_bound']) | (col_data < r['lower_bound'])
        spike_vals = col_data[spike_mask]
        spike_idxs = np.where(spike_mask)[0]
        
        # Sort by deviation
        deviations = np.abs(spike_vals - r['median'])
        top_5_idx = np.argsort(deviations)[-5:][::-1]
        
        print(f"   Top 5 spike locations:")
        for j, idx in enumerate(top_5_idx[:5], 1):
            row_idx = spike_idxs[idx]
            val = spike_vals[idx]
            date = dates.iloc[row_idx]
            print(f"      {j}. Row {row_idx:3d} ({date}): {val:8.2f} (deviation: {abs(val-r['median']):8.2f})")
    
    print(f"\n{'='*100}")
    print(f"INVESTIGATION COMPLETE")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
