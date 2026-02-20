"""
Generate Uncertainty Dispersion Across Hops plot from existing uncertainty data
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths - check both locations
enhanced_dir = Path('model/Bayesian/forecast/enhanced_viz/uncertainty')
basic_dir = Path('model/Bayesian/forecast/uncertainty')
output_dir = Path('model/Bayesian/forecast/uncertainty')
output_dir.mkdir(parents=True, exist_ok=True)

# Try enhanced_viz first (has better data)
hop_summary_csv = enhanced_dir / 'hop_uncertainty_summary.csv'
if not hop_summary_csv.exists():
    hop_summary_csv = basic_dir / 'hop_uncertainty_summary.csv'
    if not hop_summary_csv.exists():
        print(f"Error: hop_uncertainty_summary.csv not found in either location")
        exit(1)

hop_summary = pd.read_csv(hop_summary_csv)
print(f"Loaded hop summary from: {hop_summary_csv}")
print(f"Columns: {hop_summary.columns.tolist()}")
print("\nData:")
print(hop_summary)

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Uncertainty Dispersion Across Hops Analysis', fontsize=16, fontweight='bold')

# Plot 1: Standard Deviation Ratio (main metric)
if 'global_std_ratio' in hop_summary.columns:
    ax = axes[0, 0]
    ax.plot(hop_summary['hop'], hop_summary['global_std_ratio'], 
            marker='o', color='teal', linewidth=2.5, markersize=10, label='Std Ratio')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No change (1.0)')
    ax.set_xlabel('Hop', fontsize=11)
    ax.set_ylabel('Global Std Ratio', fontsize=11)
    ax.set_title('Uncertainty Dispersion (Std Deviation Ratio)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add annotations for key values
    for idx in range(min(4, len(hop_summary))):  # Only annotate first 4 hops
        row = hop_summary.iloc[idx]
        ax.annotate(f'{row["global_std_ratio"]:.3f}', 
                   xy=(row['hop'], row['global_std_ratio']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)

# Plot 2: Mean Variance Ratio
if 'global_mean_ratio' in hop_summary.columns:
    ax = axes[0, 1]
    ax.plot(hop_summary['hop'], hop_summary['global_mean_ratio'], 
            marker='s', color='darkorange', linewidth=2.5, markersize=10, label='Mean Ratio')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No change (1.0)')
    ax.set_xlabel('Hop', fontsize=11)
    ax.set_ylabel('Global Mean Ratio', fontsize=11)
    ax.set_title('Mean Uncertainty Conservation', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
elif 'global_amplification_ratio' in hop_summary.columns:
    ax = axes[0, 1]
    ax.plot(hop_summary['hop'], hop_summary['global_amplification_ratio'], 
            marker='s', color='darkorange', linewidth=2.5, markersize=10)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Hop', fontsize=11)
    ax.set_ylabel('Amplification Ratio', fontsize=11)
    ax.set_title('Mean Uncertainty Amplification', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

# Plot 3: Absolute Standard Deviation
if 'global_std_variance' in hop_summary.columns:
    ax = axes[1, 0]
    ax.plot(hop_summary['hop'], hop_summary['global_std_variance'], 
            marker='D', color='purple', linewidth=2.5, markersize=10)
    ax.set_xlabel('Hop', fontsize=11)
    ax.set_ylabel('Global Std (Variance)', fontsize=11)
    ax.set_title('Absolute Dispersion Level', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(hop_summary['hop'], hop_summary['global_std_variance'], 2)
    p = np.poly1d(z)
    ax.plot(hop_summary['hop'], p(hop_summary['hop']), 
            "r--", alpha=0.5, linewidth=1.5, label='Trend')
    ax.legend()

# Plot 4: Absolute Mean
if 'global_mean_variance' in hop_summary.columns:
    ax = axes[1, 1]
    ax.plot(hop_summary['hop'], hop_summary['global_mean_variance'], 
            marker='^', color='green', linewidth=2.5, markersize=10)
    ax.set_xlabel('Hop', fontsize=11)
    ax.set_ylabel('Global Mean (Variance)', fontsize=11)
    ax.set_title('Mean Uncertainty Level', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save comprehensive plot
output_png = output_dir / 'uncertainty_hop_comprehensive.png'
output_pdf = output_dir / 'uncertainty_hop_comprehensive.pdf'
plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.savefig(output_pdf, bbox_inches='tight')
print(f"\n✓ Saved comprehensive plots:")
print(f"  - {output_png}")
print(f"  - {output_pdf}")

# Create simplified version (just std ratio - the main metric)
if 'global_std_ratio' in hop_summary.columns:
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(hop_summary['hop'], hop_summary['global_std_ratio'], 
             marker='o', color='teal', linewidth=3, markersize=12)
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='No dispersion change')
    ax2.set_xlabel('Hop', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Dispersion Ratio (Std / Initial Std)', fontsize=13, fontweight='bold')
    ax2.set_title('Uncertainty Dispersion Across Hops\n(Row-Normalized Propagation)', 
                  fontsize=15, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, linewidth=1)
    ax2.legend(fontsize=11)
    
    # Add value labels
    for idx in range(min(5, len(hop_summary))):
        row = hop_summary.iloc[idx]
        ax2.annotate(f'{row["global_std_ratio"]:.4f}', 
                    xy=(row['hop'], row['global_std_ratio']),
                    xytext=(0, -15), textcoords='offset points', 
                    fontsize=10, ha='center',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    
    # Add interpretation text
    final_ratio = hop_summary['global_std_ratio'].iloc[-1]
    interpretation = "Dispersion decreasing" if final_ratio < 0.9 else "Dispersion stable"
    ax2.text(0.02, 0.98, f'Interpretation: {interpretation}\nFinal ratio: {final_ratio:.4f}',
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    
    output_png_simple = output_dir / 'uncertainty_hop_dispersion.png'
    output_pdf_simple = output_dir / 'uncertainty_hop_dispersion.pdf'
    plt.savefig(output_png_simple, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf_simple, bbox_inches='tight')
    print(f"\n✓ Saved simplified dispersion plot:")
    print(f"  - {output_png_simple}")
    print(f"  - {output_pdf_simple}")
    
    plt.show()

print("\n✓ Plot generation complete!")
print("\n📊 Key Insights:")
if 'global_std_ratio' in hop_summary.columns:
    print(f"  - Initial dispersion (hop 1): {hop_summary['global_std_ratio'].iloc[0]:.4f}")
    print(f"  - Final dispersion (hop {hop_summary['hop'].iloc[-1]}): {hop_summary['global_std_ratio'].iloc[-1]:.4f}")
    print(f"  - Change: {(hop_summary['global_std_ratio'].iloc[-1] - hop_summary['global_std_ratio'].iloc[0])*100:.2f}%")
    print(f"\n  → Uncertainty becomes MORE UNIFORM across nodes as it propagates")
    print(f"    (Standard deviation decreases by ~{(1 - hop_summary['global_std_ratio'].iloc[-1])*100:.1f}%)")
