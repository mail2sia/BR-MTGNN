#!/usr/bin/env python3
"""
Quick comparison of original vs smoothed data for RMD_Autosarcophagy
"""
import pandas as pd
import numpy as np

# Read original data
df_orig = pd.read_csv('data/data.csv')
# Read smoothed data (with headers)
df_smooth = pd.read_csv('data/sm_data_g.csv')

# Column to analyze
col_name = 'RMD_Autosarcophagy'

print("="*80)
print(f"COMPARISON: {col_name}")
print("="*80)

orig_values = df_orig[col_name].to_numpy()
smooth_values = df_smooth[col_name].to_numpy()

print(f"\nOriginal Data Statistics:")
print(f"  Min:    {orig_values.min():.2f}")
print(f"  Max:    {orig_values.max():.2f}")
print(f"  Mean:   {orig_values.mean():.2f}")
print(f"  Median: {np.median(orig_values):.2f}")
print(f"  Std:    {orig_values.std():.2f}")

print(f"\nSmoothed Data Statistics:")
print(f"  Min:    {smooth_values.min():.2f}")
print(f"  Max:    {smooth_values.max():.2f}")
print(f"  Mean:   {smooth_values.mean():.2f}")
print(f"  Median: {np.median(smooth_values):.2f}")
print(f"  Std:    {smooth_values.std():.2f}")

# Find the spike location
spike_idx = int(np.argmax(orig_values))
spike_date = df_orig.loc[spike_idx, 'date']

print(f"\nSpike at row {spike_idx} ({spike_date}):")
print(f"  Original value:  {orig_values[spike_idx]:.2f}")
print(f"  Smoothed value:  {smooth_values[spike_idx]:.2f}")
print(f"  Reduction:       {orig_values[spike_idx] - smooth_values[spike_idx]:.2f} ({100*(1-smooth_values[spike_idx]/orig_values[spike_idx]):.1f}%)")

# Show neighboring values
print(f"\nNeighboring values around spike (rows {max(0,spike_idx-3)} to {min(len(orig_values)-1,spike_idx+3)}):")
print(f"{'Row':<6} {'Date':<12} {'Original':>10} {'Smoothed':>10} {'Diff':>10}")
print("-"*55)
for i in range(max(0, spike_idx-3), min(len(orig_values), spike_idx+4)):
    date_val = df_orig.loc[i, 'date']
    orig_val = orig_values[i]
    smooth_val = smooth_values[i]
    diff = orig_val - smooth_val
    marker = " <-- SPIKE" if i == spike_idx else ""
    print(f"{i:<6} {date_val:<12} {orig_val:>10.2f} {smooth_val:>10.2f} {diff:>10.2f}{marker}")

print("\n" + "="*80)
