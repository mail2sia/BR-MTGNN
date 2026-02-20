#!/usr/bin/env python
"""View actual curves for selected disorders from Jan 2016 to Jan 2022."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load data with dates
data_orig = pd.read_csv('data/data.csv')
data_smooth = pd.read_csv('data/sm_data_g.csv')

# Extract date column
dates = pd.to_datetime(data_orig['date'])

# Selected columns
selected = [
    'RMD_Schizoaffective',
    'RMD_Avoidant Personality',
    'RMD_Circadian Rhythm Sleep-Wake',
    'RMD_Anorexia nervosa',
    'RMD_Body Integrity Dysphoria'
]

# Use full timeline (no filtering)
dates_filtered = dates
data_filtered = data_smooth

# Create plot
fig, axes = plt.subplots(5, 1, figsize=(14, 12))
fig.suptitle('Actual Curves: Full Timeline (Naturally Smoothed Data)', 
             fontsize=16, fontweight='bold')

for i, col in enumerate(selected):
    ax = axes[i]
    values = np.asarray(data_filtered[col].values, dtype=float)
    
    ax.plot(dates_filtered, values, linewidth=2, color='#2E86AB', marker='o', 
            markersize=4, markerfacecolor='white', markeredgewidth=1.5)
    
    ax.set_title(col, fontsize=12, fontweight='bold', pad=10)
    ax.set_ylabel('Value', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Format x-axis as MM-YYYY
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
    
    # Add statistics
    stats_text = f'Min: {np.min(values):.1f} | Max: {np.max(values):.1f} | Mean: {np.mean(values):.1f} | Std: {np.std(values):.1f}'
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('outputs/selected_curves_full.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved plot to: outputs/selected_curves_full.png")
plt.show()

# Print detailed statistics
print("\n" + "=" * 80)
print("DETAILED STATISTICS (Full Timeline)")
print("=" * 80)

for col in selected:
    values = np.asarray(data_filtered[col].values, dtype=float)
    print(f"\n{col}:")
    print(f"  Min:  {np.min(values):.2f}")
    print(f"  Max:  {np.max(values):.2f}")
    print(f"  Mean: {np.mean(values):.2f}")
    print(f"  Std:  {np.std(values):.2f}")
    print(f"  Range: {np.max(values) - np.min(values):.2f}")
