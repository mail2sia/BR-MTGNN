#!/usr/bin/env python3
"""
Compare multiple columns before and after smoothing
"""
import pandas as pd
import numpy as np

# Load data
df_orig = pd.read_csv('data/data.csv')
df_smooth = pd.read_csv('data/sm_data_g.csv')

# Top worst offenders
columns_to_check = [
    'PT_Metaverse',
    'PT_Behavioral Activation Therapy',
    'PT_Mood Stabilizers',
    'PT_Animal-Assisted Interventions',
    'RMD_Delusional',
    'RMD_Autosarcophagy',
    'PT_Electroencephalography',
    'PT_Digital Health Tools'
]

print("="*120)
print("BEFORE vs AFTER SMOOTHING - Multiple Columns Analysis")
print("="*120)

for col_name in columns_to_check:
    if col_name not in df_orig.columns:
        continue
        
    orig = np.asarray(df_orig[col_name].values, dtype=float)
    smooth = np.asarray(df_smooth[col_name].values, dtype=float)
    
    print(f"\n{col_name}")
    print("-"*120)
    
    # Statistics
    print(f"{'Metric':<25} {'Original':>15} {'Smoothed':>15} {'Change':>15}")
    print("-"*70)
    print(f"{'Min':<25} {orig.min():>15.2f} {smooth.min():>15.2f} {smooth.min()-orig.min():>15.2f}")
    print(f"{'Max':<25} {orig.max():>15.2f} {smooth.max():>15.2f} {smooth.max()-orig.max():>15.2f}")
    print(f"{'Mean':<25} {orig.mean():>15.2f} {smooth.mean():>15.2f} {smooth.mean()-orig.mean():>15.2f}")
    print(f"{'Median':<25} {np.median(orig):>15.2f} {np.median(smooth):>15.2f} {np.median(smooth)-np.median(orig):>15.2f}")
    print(f"{'Std Dev':<25} {orig.std():>15.2f} {smooth.std():>15.2f} {smooth.std()-orig.std():>15.2f}")
    
    # Find spikes
    max_idx = int(np.argmax(np.asarray(orig)))
    max_date = df_orig.loc[max_idx, 'date']
    
    print(f"\n{'Peak Spike Location':}")
    print(f"  Date: {max_date} (row {max_idx})")
    print(f"  Original value: {orig[max_idx]:.2f}")
    print(f"  Smoothed value: {smooth[max_idx]:.2f}")
    print(f"  Reduction: {orig[max_idx] - smooth[max_idx]:.2f} ({100*(1-smooth[max_idx]/orig[max_idx]):.1f}%)")
    
    # Count values > median + 3*MAD
    orig_median = np.median(np.asarray(orig))
    mad_orig = np.median(np.abs(np.asarray(orig) - orig_median))
    threshold = np.median(np.asarray(orig)) + 3.5 * mad_orig
    spikes_orig = int(np.sum(np.asarray(orig) > threshold))
    spikes_smooth = int(np.sum(np.asarray(smooth) > threshold))
    
    print(f"\n{'Spike Count':} (values > median + 3.5*MAD)")
    print(f"  Original: {spikes_orig} spikes")
    print(f"  Smoothed: {spikes_smooth} spikes")
    print(f"  Reduction: {spikes_orig - spikes_smooth} spikes removed ({100*(spikes_orig-spikes_smooth)/spikes_orig if spikes_orig>0 else 0:.1f}%)")

print("\n" + "="*120)
print("SUMMARY: Data Quality Improvement")
print("="*120)

# Overall statistics
total_reduction_pct = []
for col in columns_to_check:
    if col in df_orig.columns:
        orig_max = df_orig[col].max()
        smooth_max = df_smooth[col].max()
        if orig_max > 0:
            reduction = 100 * (1 - smooth_max / orig_max)
            total_reduction_pct.append(reduction)

print(f"\nAverage peak reduction across {len(total_reduction_pct)} columns: {np.mean(total_reduction_pct):.1f}%")
print(f"Median peak reduction: {np.median(total_reduction_pct):.1f}%")
print(f"Range: {np.min(total_reduction_pct):.1f}% to {np.max(total_reduction_pct):.1f}%")

print("\n" + "="*120)
print("✓ Smoothed datasets ready for training:")
print("  • data/sm_data.csv (no headers)")
print("  • data/sm_data_g.csv (with headers)")
print("="*120)
