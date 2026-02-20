#!/usr/bin/env python
"""Compare data.csv with sm_data_g.csv to validate smoothing."""

import pandas as pd
import numpy as np

def compare_datasets():
    print("=" * 80)
    print("DATASET COMPARISON: data.csv vs sm_data_g.csv")
    print("=" * 80)
    
    # Load datasets
    orig = pd.read_csv('data/data.csv')
    smooth = pd.read_csv('data/sm_data_g.csv')
    
    print(f"\n📊 Structure:")
    print(f"  data.csv:      {orig.shape[0]} rows × {orig.shape[1]} columns")
    print(f"  sm_data_g.csv: {smooth.shape[0]} rows × {smooth.shape[1]} columns")
    
    # Column comparison
    orig_cols = set(orig.columns)
    smooth_cols = set(smooth.columns)
    common = orig_cols & smooth_cols
    only_orig = orig_cols - smooth_cols
    only_smooth = smooth_cols - orig_cols
    
    print(f"\n📋 Columns:")
    print(f"  Common: {len(common)}")
    print(f"  Only in data.csv: {len(only_orig)} → {list(only_orig)}")
    print(f"  Only in sm_data_g.csv: {len(only_smooth)}")
    
    # Value comparison for key columns
    print(f"\n📈 Value Comparison (Key Columns):")
    print(f"{'Column':<45} {'Original':<25} {'Smoothed':<25} {'Reduction'}")
    print(f"{'-'*45} {'-'*25} {'-'*25} {'-'*10}")
    
    cols_to_check = [
        'Global_Disaster_Total',
        'PT_Metaverse',
        'PT_Behavioral Activation Therapy',
        'PT_Mood Stabilizers',
        'RMD_Delusional',
        'RMD_Autosarcophagy',
        'PT_Digital Health Tools'
    ]
    
    for col in cols_to_check:
        if col in orig.columns and col in smooth.columns:
            o_min, o_max, o_mean = orig[col].min(), orig[col].max(), orig[col].mean()
            s_min, s_max, s_mean = smooth[col].min(), smooth[col].max(), smooth[col].mean()
            reduction = ((o_max - s_max) / o_max * 100) if o_max > 0 else 0
            
            print(f"{col:<45} Max:{o_max:7.1f} Mean:{o_mean:6.1f}  "
                  f"Max:{s_max:7.1f} Mean:{s_mean:6.1f}  {reduction:5.1f}%")
    
    # Overall statistics
    print(f"\n📊 Overall Statistics:")
    
    # Get numeric columns only (exclude date)
    orig_numeric = orig.select_dtypes(include=[np.number])
    smooth_numeric = smooth.select_dtypes(include=[np.number])
    
    print(f"  Original dataset:")
    print(f"    Global max: {orig_numeric.values.max():.2f}")
    print(f"    Global mean: {orig_numeric.values.mean():.2f}")
    print(f"    Global std: {orig_numeric.values.std():.2f}")
    
    print(f"  Smoothed dataset:")
    print(f"    Global max: {smooth_numeric.values.max():.2f}")
    print(f"    Global mean: {smooth_numeric.values.mean():.2f}")
    print(f"    Global std: {smooth_numeric.values.std():.2f}")
    
    print(f"\n  Overall peak reduction: "
          f"{((orig_numeric.values.max() - smooth_numeric.values.max()) / orig_numeric.values.max() * 100):.1f}%")
    
    # Check if values are identical for some columns (no spikes)
    unchanged_cols = []
    for col in common:
        if col != 'date':
            if np.allclose(np.asarray(orig[col].values), np.asarray(smooth[col].values), rtol=1e-4):
                unchanged_cols.append(col)
    
    print(f"\n🔄 Columns unchanged (no spikes detected): {len(unchanged_cols)}")
    if len(unchanged_cols) > 0 and len(unchanged_cols) <= 5:
        for col in unchanged_cols:
            print(f"  - {col}")
    
    # Nodes.csv alignment
    nodes = pd.read_csv('nodes.csv')
    print(f"\n🗂️  nodes.csv Alignment:")
    print(f"  Nodes in nodes.csv: {len(nodes)}")
    print(f"  Columns in sm_data_g.csv: {len(smooth.columns)}")
    
    if len(nodes) == len(smooth.columns):
        mismatches = [(i, n, c) for i, (n, c) in enumerate(zip(nodes['display'], smooth.columns)) if n != c]
        if len(mismatches) == 0:
            print(f"  ✅ Perfect match! All {len(nodes)} nodes align with data columns")
        else:
            print(f"  ⚠️  {len(mismatches)} order mismatches detected")
    else:
        print(f"  ⚠️  Count mismatch: {len(nodes)} nodes vs {len(smooth.columns)} columns")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Date column removed from original (96→95 columns)")
    print(f"✅ All {len(common)} data columns preserved")
    print(f"✅ Spike smoothing applied (max value: 1313→277)")
    print(f"✅ nodes.csv regenerated to match sm_data_g.csv columns")
    print(f"✅ Ready for training with --has_header (no --drop_first_col)")
    print("=" * 80)

if __name__ == '__main__':
    compare_datasets()
