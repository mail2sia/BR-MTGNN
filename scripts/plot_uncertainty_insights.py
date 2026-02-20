#!/usr/bin/env python3
"""
Enhanced Uncertainty Visualization - Better alternatives to flat amplification plots

Creates informative uncertainty visualizations:
1. Node-level variance distribution by category (Global/PT/RMD)
2. Top uncertain vs confident nodes comparison
3. Edge uncertainty correlation patterns
4. Categorical uncertainty summary

Usage:
    python scripts/plot_uncertainty_insights.py
"""

import os
import sys
import json
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup paths
_ROOT = Path(__file__).parent.parent
uncertainty_dir = _ROOT / "model" / "Bayesian" / "forecast" / "uncertainty"
nodes_path = _ROOT / "data" / "nodes.csv"
hop_summary_path = uncertainty_dir / "hop_uncertainty_summary.csv"
hop_diag_path = uncertainty_dir / "hop_uncertainty_diagnostics.csv"
metrics_validation_path = _ROOT / "model" / "Bayesian" / "metrics_validation.json"
metrics_testing_path = _ROOT / "model" / "Bayesian" / "metrics_testing.json"
forecast_metrics_path = _ROOT / "model" / "Bayesian" / "forecast" / "forecast.csv"


def run_auto_recalibration_if_available():
    recal_script = _ROOT / 'scripts' / 'recalibrate_intervals.py'
    interval_input = _ROOT / 'model' / 'Bayesian' / 'forecast' / 'calibration_interval_samples.csv'
    if not recal_script.exists() or not interval_input.exists():
        return

    cmd = [
        sys.executable,
        str(recal_script),
        '--auto-tune-cap',
        '--cap-candidates', '4,6,8,10,12,15,20',
        '--max-mpiw-multiplier', '12',
        '--group-by', 'horizon_category',
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
        print('[info] Auto-tuned interval recalibration completed')
        if proc.stdout:
            for line in proc.stdout.strip().splitlines():
                print(f'  {line}')
    except subprocess.CalledProcessError as exc:
        print(f'[warn] Auto recalibration failed (continuing diagnostics): {exc}')
        if exc.stdout:
            print(exc.stdout.strip())
        if exc.stderr:
            print(exc.stderr.strip())


def run_calibration_diagnostics_if_available():
    diag_script = _ROOT / 'scripts' / 'uncertainty_calibration_diagnostics.py'
    raw_input = _ROOT / 'model' / 'Bayesian' / 'forecast' / 'calibration_interval_samples.csv'
    if not diag_script.exists() or not raw_input.exists():
        return

    cmd = [
        sys.executable,
        str(diag_script),
        '--input-raw', str(raw_input),
        '--input-recal', str(_ROOT / 'model' / 'Bayesian' / 'forecast' / 'calibration_interval_samples_recalibrated.csv'),
        '--output-dir', str(uncertainty_dir),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
        if proc.stdout:
            for line in proc.stdout.strip().splitlines():
                print(f'  {line}')
    except subprocess.CalledProcessError as exc:
        print(f'[warn] Calibration diagnostics failed (continuing): {exc}')
        if exc.stdout:
            print(exc.stdout.strip())
        if exc.stderr:
            print(exc.stderr.strip())

# Load data
node_unc = pd.read_csv(uncertainty_dir / "node_uncertainty.csv")
edge_unc = pd.read_csv(uncertainty_dir / "edge_uncertainty.csv")
hop_summary_df = pd.read_csv(hop_summary_path) if hop_summary_path.exists() else None
hop_diag_df = pd.read_csv(hop_diag_path) if hop_diag_path.exists() else None

# Map generic node_i names back to tokens if needed
if node_unc['node'].astype(str).str.startswith('node_').all() and nodes_path.exists():
    nodes_df = pd.read_csv(nodes_path)
    tokens = nodes_df['token'].astype(str).tolist() if 'token' in nodes_df.columns else []

    def map_generic_node(name):
        name = str(name)
        if not name.startswith('node_'):
            return name
        idx_str = name.split('node_', 1)[1]
        if idx_str.isdigit():
            idx = int(idx_str)
            if 0 <= idx < len(tokens):
                return tokens[idx]
        return name

    node_unc['node'] = node_unc['node'].apply(map_generic_node)
    if {'src', 'dst'}.issubset(edge_unc.columns):
        edge_unc['src'] = edge_unc['src'].apply(map_generic_node)
        edge_unc['dst'] = edge_unc['dst'].apply(map_generic_node)
    print('[info] Mapped generic node indices to tokens using data/nodes.csv')

# Categorize nodes
def categorize_node(name):
    if name.startswith("Global_"):
        return "Global"
    elif name.startswith("PT_"):
        return "PT"
    elif name.startswith("RMD_"):
        return "RMD"
    return "Other"

node_unc['category'] = node_unc['node'].apply(categorize_node)

# Keep Global only when it has enough support to be analytically useful
global_count = int((node_unc['category'] == 'Global').sum())
include_global = global_count >= 5

plot_categories = ['PT', 'RMD']
if include_global:
    plot_categories.append('Global')

analysis_unc = node_unc[node_unc['category'].isin(plot_categories)].copy()

if not include_global:
    print(f"[info] Excluding Global category (count={global_count}) as not relevant for this report")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

CATEGORY_COLORS = {
    'Global': '#66c2a5',
    'PT': '#fc8d62',
    'RMD': '#8da0cb',
    'Other': '#bdbdbd',
}

# ============================================================================
# Figure 1: Node Variance Distribution by Category
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Uncertainty Analysis: Node-Level Variance Distribution', fontsize=16, fontweight='bold')

# 1A: Box plot by category
ax1 = axes[0, 0]
categories = plot_categories
data_by_cat = [analysis_unc[analysis_unc['category'] == cat]['mean_variance'].values for cat in categories]
bp = ax1.boxplot(data_by_cat, tick_labels=categories, patch_artist=True, showfliers=False)
for patch, cat in zip(bp['boxes'], categories):
    color = CATEGORY_COLORS.get(cat, CATEGORY_COLORS['Other'])
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax1.set_ylabel('Mean Variance', fontsize=11)
ax1.set_xlabel('Node Category', fontsize=11)
ax1.set_title('Variance Distribution by Category', fontsize=12, fontweight='bold')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# 1B: Top 15 most uncertain nodes
ax2 = axes[0, 1]
top_uncertain = analysis_unc.nlargest(15, 'mean_variance')
colors = [CATEGORY_COLORS.get(cat, CATEGORY_COLORS['Other'])
          for cat in top_uncertain['category']]
bars = ax2.barh(range(len(top_uncertain)), top_uncertain['mean_variance'], color=colors, alpha=0.7)
ax2.set_yticks(range(len(top_uncertain)))
ax2.set_yticklabels([n.replace('PT_', '').replace('RMD_', '')[:30] for n in top_uncertain['node']], fontsize=9)
ax2.set_xlabel('Mean Variance', fontsize=11)
ax2.set_title('Top 15 Most Uncertain Nodes', fontsize=12, fontweight='bold')
ax2.invert_yaxis()
ax2.grid(True, alpha=0.3, axis='x')

# 1C: Top 15 most confident nodes
ax3 = axes[1, 0]
top_confident = analysis_unc.nsmallest(15, 'mean_variance')
colors = [CATEGORY_COLORS.get(cat, CATEGORY_COLORS['Other'])
          for cat in top_confident['category']]
bars = ax3.barh(range(len(top_confident)), top_confident['mean_variance'], color=colors, alpha=0.7)
ax3.set_yticks(range(len(top_confident)))
ax3.set_yticklabels([n.replace('PT_', '').replace('RMD_', '')[:30] for n in top_confident['node']], fontsize=9)
ax3.set_xlabel('Mean Variance (log scale)', fontsize=11)
ax3.set_title('Top 15 Most Confident Nodes', fontsize=12, fontweight='bold')
ax3.set_xscale('log')
ax3.invert_yaxis()
ax3.grid(True, alpha=0.3, axis='x')

# 1D: Summary statistics table
ax4 = axes[1, 1]
ax4.axis('off')
summary_data = []
for cat in categories:
    cat_data = analysis_unc[analysis_unc['category'] == cat]['mean_variance']
    summary_data.append([
        cat,
        len(cat_data),
        f"{cat_data.mean():.4f}",
        f"{cat_data.median():.4f}",
        f"{cat_data.std():.4f}",
        f"{cat_data.min():.2e}",
        f"{cat_data.max():.2f}"
    ])

table = ax4.table(cellText=summary_data,
                 colLabels=['Category', 'Count', 'Mean', 'Median', 'Std', 'Min', 'Max'],
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0.3, 1, 0.6])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)
for i in range(len(categories)):
    table[(i+1, 0)].set_facecolor(CATEGORY_COLORS.get(categories[i], CATEGORY_COLORS['Other']))
    table[(i+1, 0)].set_alpha(0.3)
ax4.set_title('Summary Statistics by Category', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(uncertainty_dir / 'uncertainty_node_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig(uncertainty_dir / 'uncertainty_node_analysis.pdf', bbox_inches='tight')
print(f"✓ Saved: uncertainty_node_analysis.pdf")
plt.close()

# ============================================================================
# Figure 2: Variance vs Confidence Interval Relationship
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Uncertainty Metrics: Variance vs Confidence Intervals', fontsize=16, fontweight='bold')

analysis_unc_log = analysis_unc[(analysis_unc['mean_variance'] > 0) & (analysis_unc['mean_ci_95'] > 0)].copy()

# 2A: Scatter plot colored by category
ax1 = axes[0]
for cat in categories:
    color = CATEGORY_COLORS.get(cat, CATEGORY_COLORS['Other'])
    mask = analysis_unc_log['category'] == cat
    if not mask.any():
        continue
    ax1.scatter(analysis_unc_log[mask]['mean_variance'],
               analysis_unc_log[mask]['mean_ci_95'],
               alpha=0.6, s=50, label=cat, color=color)
ax1.set_xlabel('Mean Variance', fontsize=11)
ax1.set_ylabel('Mean 95% CI Width', fontsize=11)
ax1.set_title('Variance vs CI Width by Category', fontsize=12, fontweight='bold')
ax1.set_xscale('log')
ax1.set_yscale('log')
if ax1.get_legend_handles_labels()[0]:
    ax1.legend()
ax1.grid(True, alpha=0.3)

# 2B: Histogram of variance distribution
ax2 = axes[1]
for cat in categories:
    color = CATEGORY_COLORS.get(cat, CATEGORY_COLORS['Other'])
    cat_data = analysis_unc[analysis_unc['category'] == cat]['mean_variance']
    cat_data = cat_data[cat_data > 0]
    if cat_data.empty:
        continue
    ax2.hist(np.log10(cat_data), bins=20, alpha=0.5, label=cat, color=color)
ax2.set_xlabel('Log10(Mean Variance)', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title('Variance Distribution (log scale)', fontsize=12, fontweight='bold')
if ax2.get_legend_handles_labels()[0]:
    ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(uncertainty_dir / 'uncertainty_metrics_relationship.png', dpi=300, bbox_inches='tight')
plt.savefig(uncertainty_dir / 'uncertainty_metrics_relationship.pdf', bbox_inches='tight')
print(f"✓ Saved: uncertainty_metrics_relationship.pdf")
plt.close()

# ============================================================================
# Figure 3: Edge Uncertainty Patterns
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Edge-Level Uncertainty Propagation Patterns', fontsize=16, fontweight='bold')

# 3A: Amplification ratio distribution
ax1 = axes[0, 0]
amp_col = 'amplification_ratio_floored' if 'amplification_ratio_floored' in edge_unc.columns else 'amplification_ratio'
amp_ratios = edge_unc[amp_col].replace([np.inf, -np.inf], np.nan).dropna()
amp_ratios_clipped = amp_ratios.clip(0, 100)  # Clip extreme values
ax1.hist(amp_ratios_clipped, bins=50, alpha=0.7, color='teal', edgecolor='black')
ax1.axvline(amp_ratios.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {amp_ratios.median():.2f}')
ax1.set_xlabel(f'Amplification Ratio ({amp_col})', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Edge Amplification Ratio Distribution', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 3B: Uncertainty correlation scatter
ax2 = axes[0, 1]
corr_clean = edge_unc['uncertainty_corr'].replace([np.inf, -np.inf], np.nan).dropna()
ax2.scatter(edge_unc['src_mean_variance'], 
           edge_unc['dst_mean_variance'],
           c=edge_unc['uncertainty_corr'],
           cmap='RdYlBu_r', alpha=0.6, s=20)
ax2.set_xlabel('Source Node Variance', fontsize=11)
ax2.set_ylabel('Destination Node Variance', fontsize=11)
ax2.set_title('Edge Uncertainty: Source vs Destination', fontsize=12, fontweight='bold')
ax2.set_xscale('log')
ax2.set_yscale('log')
cbar = plt.colorbar(ax2.collections[0], ax=ax2)
cbar.set_label('Uncertainty Correlation', fontsize=10)
ax2.grid(True, alpha=0.3)

# 3C: Top edges with highest amplification
ax3 = axes[1, 0]
top_amp = edge_unc.nlargest(15, amp_col)
edge_labels = [f"{s.split('_')[-1][:10]}→{d.split('_')[-1][:10]}" 
               for s, d in zip(top_amp['src'], top_amp['dst'])]
bars = ax3.barh(range(len(top_amp)), top_amp[amp_col], alpha=0.7, color='coral')
ax3.set_yticks(range(len(top_amp)))
ax3.set_yticklabels(edge_labels, fontsize=9)
ax3.set_xlabel('Amplification Ratio', fontsize=11)
ax3.set_title('Top 15 Amplifying Edges', fontsize=12, fontweight='bold')
ax3.invert_yaxis()
ax3.grid(True, alpha=0.3, axis='x')

# 3D: Uncertainty correlation distribution
ax4 = axes[1, 1]
corr_bins = np.linspace(-1, 1, 30)
ax4.hist(corr_clean, bins=corr_bins, alpha=0.7, color='purple', edgecolor='black')
ax4.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax4.axvline(corr_clean.mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {corr_clean.mean():.3f}')
ax4.set_xlabel('Uncertainty Correlation', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title('Edge Uncertainty Correlation Distribution', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(uncertainty_dir / 'uncertainty_edge_patterns.png', dpi=300, bbox_inches='tight')
plt.savefig(uncertainty_dir / 'uncertainty_edge_patterns.pdf', bbox_inches='tight')
print(f"✓ Saved: uncertainty_edge_patterns.pdf")
plt.close()

# ============================================================================
# Summary Report
# ============================================================================
print("\n" + "="*60)
print("UNCERTAINTY ANALYSIS SUMMARY")
print("="*60)

for cat in categories:
    cat_data = analysis_unc[analysis_unc['category'] == cat]['mean_variance']
    print(f"\n{cat} Nodes (n={len(cat_data)}):")
    print(f"  Mean Variance:   {cat_data.mean():.6f}")
    print(f"  Median Variance: {cat_data.median():.6f}")
    print(f"  Std Dev:         {cat_data.std():.6f}")
    print(f"  Range:           [{cat_data.min():.2e}, {cat_data.max():.2f}]")

print(f"\nEdge Statistics:")
print(f"  Total Edges:               {len(edge_unc)}")
print(f"  Mean Amplification Ratio ({amp_col}):  {amp_ratios.mean():.2f}")
print(f"  Median Amplification ({amp_col}):      {amp_ratios.median():.2f}")
print(f"  P95 Amplification ({amp_col}):         {np.quantile(amp_ratios, 0.95):.2f}")
print(f"  P99 Amplification ({amp_col}):         {np.quantile(amp_ratios, 0.99):.2f}")
print(f"  Mean Uncertainty Corr:     {corr_clean.mean():.3f}")
print(f"  Median Uncertainty Corr:   {corr_clean.median():.3f}")

if hop_summary_df is not None and len(hop_summary_df):
    print(f"\nHop Diagnostics:")
    cols = set(hop_summary_df.columns)
    if {'hop', 'global_mean_ratio'}.issubset(cols):
        for _, row in hop_summary_df.iterrows():
            std_ratio_val = row['global_std_ratio'] if 'global_std_ratio' in cols else float('nan')
            print(f"  Hop {int(row['hop'])}: mean_ratio={float(row['global_mean_ratio']):.3f}, std_ratio={float(std_ratio_val):.3f}")
    elif {'hop', 'global_amplification_ratio'}.issubset(cols):
        for _, row in hop_summary_df.iterrows():
            print(f"  Hop {int(row['hop'])}: amplification_ratio={float(row['global_amplification_ratio']):.3f}")

if hop_diag_df is not None and len(hop_diag_df) and {'hop', 'share_above_base_variance'}.issubset(hop_diag_df.columns):
    print("  Directional spread:")
    for _, row in hop_diag_df.iterrows():
        print(f"    Hop {int(row['hop'])}: share_above_base={float(row['share_above_base_variance']):.3f}, mean_abs_delta={float(row['mean_abs_change_from_prev']):.6f}")

print(f"\nCalibration Backtest:")
calib_rows = []
for split_name, metrics_path in [('Validation', metrics_validation_path), ('Testing', metrics_testing_path)]:
    if not metrics_path.exists():
        continue
    try:
        payload = json.loads(metrics_path.read_text(encoding='utf-8'))
    except Exception:
        continue
    metrics = payload.get('metrics', {}) or {}
    args = payload.get('args', {}) or {}
    row = {
        'split': split_name,
        'RSE': metrics.get('RSE', np.nan),
        'RAE': metrics.get('RAE', np.nan),
        'Corr': metrics.get('Corr', np.nan),
        'sMAPE': metrics.get('sMAPE', np.nan),
        'conformal_enabled': bool(args.get('conformal', False)),
        'conf_alpha': args.get('conf_alpha', np.nan),
        'conf_calibrate': bool(args.get('conf_calibrate', False)),
    }
    calib_rows.append(row)
    print(
        f"  {split_name}: RSE={row['RSE']:.4f}, RAE={row['RAE']:.4f}, Corr={row['Corr']:.4f}, "
        f"sMAPE={row['sMAPE']:.4f}, conformal={row['conformal_enabled']}, alpha={row['conf_alpha']}"
    )

proxy = analysis_unc[['mean_variance', 'mean_ci_95']].replace([np.inf, -np.inf], np.nan).dropna()
proxy = proxy[(proxy['mean_variance'] > 0) & (proxy['mean_ci_95'] > 0)]
if len(proxy):
    var_values = np.asarray(proxy['mean_variance'], dtype=np.float64)
    ci_values = np.asarray(proxy['mean_ci_95'], dtype=np.float64)
    proxy_corr = float(np.corrcoef(var_values, ci_values)[0, 1].real)
    v80 = float(np.quantile(var_values, 0.8))
    ci10 = float(np.quantile(ci_values, 0.1))
    high_var_low_ci = int(((proxy['mean_variance'] >= v80) & (proxy['mean_ci_95'] <= ci10)).sum())
else:
    proxy_corr = float('nan')
    high_var_low_ci = 0
print(f"  Proxy checks: var_ci_corr={proxy_corr:.4f}, high_var_low_ci_count={high_var_low_ci}")

run_auto_recalibration_if_available()

# Optional interval-calibration diagnostics (requires actual + interval bounds)
interval_diag_rows = []
interval_candidates = [
    _ROOT / 'model' / 'Bayesian' / 'forecast' / 'calibration_interval_samples_conformal_horizon.csv',
    _ROOT / 'model' / 'Bayesian' / 'forecast' / 'calibration_interval_samples_recalibrated.csv',
    uncertainty_dir / 'calibration_interval_samples.csv',
    _ROOT / 'model' / 'Bayesian' / 'forecast' / 'calibration_interval_samples.csv',
    _ROOT / 'model' / 'Bayesian' / 'forecast' / 'forecast_backtest.csv',
    forecast_metrics_path,
]

for candidate in interval_candidates:
    if not candidate.exists():
        continue
    try:
        cdf = pd.read_csv(candidate)
    except Exception:
        continue

    cols = set(cdf.columns)
    if {'actual', 'lower', 'upper'}.issubset(cols):
        lower = pd.to_numeric(cdf['lower'], errors='coerce')
        upper = pd.to_numeric(cdf['upper'], errors='coerce')
        actual = pd.to_numeric(cdf['actual'], errors='coerce')
    elif {'actual', 'forecast', 'ci_95'}.issubset(cols):
        pred = pd.to_numeric(cdf['forecast'], errors='coerce')
        ci = pd.to_numeric(cdf['ci_95'], errors='coerce')
        lower = pred - ci
        upper = pred + ci
        actual = pd.to_numeric(cdf['actual'], errors='coerce')
    else:
        continue

    valid = pd.DataFrame({'actual': actual, 'lower': lower, 'upper': upper}).dropna()
    valid = valid[valid['upper'] >= valid['lower']]
    if valid.empty:
        continue

    inside = (valid['actual'] >= valid['lower']) & (valid['actual'] <= valid['upper'])
    picp = float(inside.mean())
    mpiw = float((valid['upper'] - valid['lower']).mean())
    ace = float(picp - 0.95)
    interval_diag_rows.append({
        'source_file': str(candidate.relative_to(_ROOT)).replace('\\', '/'),
        'samples': int(len(valid)),
        'target_coverage': 0.95,
        'picp': picp,
        'ace': ace,
        'mpiw': mpiw,
    })

if interval_diag_rows:
    print('  Interval diagnostics (actual-aware):')
    for row in interval_diag_rows:
        print(
            f"    {row['source_file']}: n={row['samples']}, "
            f"PICP={row['picp']:.4f}, ACE={row['ace']:+.4f}, MPIW={row['mpiw']:.4f}"
        )
else:
    print('  Interval diagnostics (actual-aware): unavailable (no actual/lower/upper backtest file found)')

calib_csv = uncertainty_dir / 'calibration_backtest_summary.csv'
calib_df = pd.DataFrame(calib_rows)
if len(calib_df):
    calib_df.to_csv(calib_csv, index=False)
else:
    pd.DataFrame([{
        'split': 'N/A',
        'RSE': np.nan,
        'RAE': np.nan,
        'Corr': np.nan,
        'sMAPE': np.nan,
        'conformal_enabled': np.nan,
        'conf_alpha': np.nan,
        'conf_calibrate': np.nan,
    }]).to_csv(calib_csv, index=False)
print(f"  Saved: {calib_csv.name}")

interval_diag_csv = uncertainty_dir / 'calibration_interval_diagnostics.csv'
if interval_diag_rows:
    pd.DataFrame(interval_diag_rows).to_csv(interval_diag_csv, index=False)
else:
    pd.DataFrame([{
        'source_file': 'N/A',
        'samples': 0,
        'target_coverage': 0.95,
        'picp': np.nan,
        'ace': np.nan,
        'mpiw': np.nan,
    }]).to_csv(interval_diag_csv, index=False)
print(f"  Saved: {interval_diag_csv.name}")

run_calibration_diagnostics_if_available()

print("\n" + "="*60)
print("✅ Generated 3 enhanced uncertainty visualizations")
print("="*60)
print("\nFiles created:")
print(f"  1. uncertainty_node_analysis.pdf")
print(f"  2. uncertainty_metrics_relationship.pdf")
print(f"  3. uncertainty_edge_patterns.pdf")
print("\n✓ These provide much more insight than flat amplification plots!")
