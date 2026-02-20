"""
Generate uncertainty propagation analysis for all propagation modes
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import csv

# Load data
nodes_df = pd.read_csv('data/nodes.csv')
forecast_metrics = pd.read_csv('model/Bayesian/forecast/forecast.csv')
graph_path = Path('data/graph_topk_k12.csv')

# Extract node uncertainty (variance) - group by node and average
if 'variance' not in forecast_metrics.columns or 'node' not in forecast_metrics.columns:
    print("Error: Required columns not found in forecast.csv")
    exit(1)

# Group by node and compute mean variance
node_variance = forecast_metrics.groupby('node')['variance'].mean()
node_names = node_variance.index.tolist()
variances = node_variance.values.tolist()

print(f"Loaded {len(node_names)} nodes")

# Load adjacency
adj = np.loadtxt(str(graph_path), delimiter=',')
threshold = 0.1
max_hops = 6
adj = np.where(adj > threshold, adj, 0.0)

size = min(len(node_names), adj.shape[0], adj.shape[1])
node_names = node_names[:size]
adj = adj[:size, :size]
u0 = np.array(variances[:size], dtype=float)

base_mean = float(np.mean(u0))
base_std = float(np.std(u0))

# Test different propagation modes
modes = {
    'row': 'Row-Normalized (Conservative)',
    'spectral': 'Spectral (Controlled Amplification)',
    'none': 'Raw Adjacency (Aggressive)'
}

output_dir = Path('model/Bayesian/forecast/uncertainty')
output_dir.mkdir(parents=True, exist_ok=True)

results = {}

for mode_key, mode_label in modes.items():
    print(f"\n{'='*60}")
    print(f"Processing: {mode_label}")
    print(f"{'='*60}")
    
    # Prepare adjacency based on mode
    if mode_key == 'row':
        row_sum = adj.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum < 1e-12, 1.0, row_sum)
        adj_prop = adj / row_sum
    elif mode_key == 'spectral':
        eigvals = np.linalg.eigvals(adj)
        rho = float(np.max(np.abs(eigvals))) if eigvals.size else 1.0
        rho = rho if rho > 1e-12 else 1.0
        gain = 1.0
        adj_prop = (gain / rho) * adj
        print(f"  Spectral radius: {rho:.4f}")
    else:  # 'none'
        adj_prop = adj
    
    # Propagate
    hop_data = []
    u_prev = u0
    
    for hop in range(1, max_hops + 1):
        u_next = adj_prop.T @ u_prev
        global_mean = float(np.mean(u_next))
        global_std = float(np.std(u_next))
        
        hop_data.append({
            'hop': hop,
            'mean': global_mean,
            'std': global_std,
            'mean_ratio': global_mean / (base_mean + 1e-12),
            'std_ratio': global_std / (base_std + 1e-12)
        })
        
        u_prev = u_next
    
    results[mode_key] = {
        'label': mode_label,
        'data': hop_data
    }
    
    # Print summary
    print(f"  Initial std: {base_std:.4f}")
    print(f"  Final std (hop {max_hops}): {hop_data[-1]['std']:.4f}")
    print(f"  Std ratio: {hop_data[-1]['std_ratio']:.4f}")
    if hop_data[-1]['std_ratio'] > 1.0:
        print(f"  → Amplification: {(hop_data[-1]['std_ratio'] - 1)*100:.1f}% increase")
    else:
        print(f"  → Attenuation: {(1 - hop_data[-1]['std_ratio'])*100:.1f}% decrease")

# Create comparison plot
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Uncertainty Dispersion Across Hops: Propagation Mode Comparison', 
             fontsize=16, fontweight='bold')

colors = {'row': 'teal', 'spectral': 'darkorange', 'none': 'crimson'}

# Plot 1: Std Ratio Comparison
ax = axes[0, 0]
for mode_key, result in results.items():
    hops = [d['hop'] for d in result['data']]
    std_ratios = [d['std_ratio'] for d in result['data']]
    ax.plot(hops, std_ratios, marker='o', linewidth=2.5, markersize=8,
            label=result['label'], color=colors.get(mode_key, 'blue'))

ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='No change')
ax.set_xlabel('Hop', fontsize=12, fontweight='bold')
ax.set_ylabel('Global Std Ratio', fontsize=12, fontweight='bold')
ax.set_title('Standard Deviation Ratio (Dispersion)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

# Plot 2: Absolute Std
ax = axes[0, 1]
for mode_key, result in results.items():
    hops = [d['hop'] for d in result['data']]
    stds = [d['std'] for d in result['data']]
    ax.plot(hops, stds, marker='s', linewidth=2.5, markersize=8,
            label=result['label'], color=colors.get(mode_key, 'blue'))

ax.axhline(y=base_std, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='Initial')
ax.set_xlabel('Hop', fontsize=12, fontweight='bold')
ax.set_ylabel('Global Std (Absolute)', fontsize=12, fontweight='bold')
ax.set_title('Absolute Standard Deviation', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

# Plot 3: Mean Ratio
ax = axes[1, 0]
for mode_key, result in results.items():
    hops = [d['hop'] for d in result['data']]
    mean_ratios = [d['mean_ratio'] for d in result['data']]
    ax.plot(hops, mean_ratios, marker='D', linewidth=2.5, markersize=8,
            label=result['label'], color=colors.get(mode_key, 'blue'))

ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax.set_xlabel('Hop', fontsize=12, fontweight='bold')
ax.set_ylabel('Global Mean Ratio', fontsize=12, fontweight='bold')
ax.set_title('Mean Uncertainty Conservation', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

# Plot 4: Summary table
ax = axes[1, 1]
ax.axis('off')

table_data = [['Mode', 'Initial Std', 'Final Std', 'Ratio', 'Change']]
for mode_key, result in results.items():
    final = result['data'][-1]
    change = f"+{(final['std_ratio']-1)*100:.1f}%" if final['std_ratio'] > 1 else f"{(final['std_ratio']-1)*100:.1f}%"
    table_data.append([
        result['label'].split('(')[0].strip(),
        f"{base_std:.3f}",
        f"{final['std']:.3f}",
        f"{final['std_ratio']:.3f}",
        change
    ])

table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                colWidths=[0.3, 0.15, 0.15, 0.15, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(5):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color code rows
for i, mode_key in enumerate(['row', 'spectral', 'none'], 1):
    for j in range(5):
        table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')

ax.set_title('Summary Statistics', fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()

# Save
output_png = output_dir / 'uncertainty_modes_comparison.png'
output_pdf = output_dir / 'uncertainty_modes_comparison.pdf'
plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.savefig(output_pdf, bbox_inches='tight')

print(f"\n{'='*60}")
print(f"✓ Saved comparison plots:")
print(f"  - {output_png}")
print(f"  - {output_pdf}")
print(f"{'='*60}")

plt.show()

print("\n✓ Complete!")
