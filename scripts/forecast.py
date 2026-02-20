#!/usr/bin/env python3
"""
Future Forecast Script - Corrected for B-MTGNN v0.2.0

Generates future forecasts (2026-2028) using a trained model.
Loads trained checkpoint, processes recent historical data, and produces:
  - Forecast plots for each node/group
  - Numerical forecast data with confidence intervals  
  - Gap analysis between related trends

Usage:
    python scripts/forecast.py --checkpoint model/Bayesian/model.pt \
        --data data/sm_data_g.csv --forecast_months 36 --mc_runs 50

Aligned with optimized defaults from train_test.py v0.2.0
"""
import pickle
import numpy as np
import os
import sys
import csv
import argparse
from pathlib import Path
from collections import defaultdict
from matplotlib import pyplot
import torch

# Ensure project root is on sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Import from src package
from src.net import gtnet
from scripts.o_util import build_model_from_checkpoint, filter_state_dict_for_model
from src.util import (
    StandardScaler, DataLoaderS,
    parse_dates_from_csv, load_graph
)

pyplot.rcParams['savefig.dpi'] = 1200


def consistent_name(name):
    """Clean and format node names for display"""
    name = (
        name.replace("RMD_", "")
        .replace("RMD", "")
        .replace("PT_", "")
        .replace("PT", "")
    )
    
    # Special cases
    if 'HIDDEN MARKOV MODEL' in name:
        return 'Statistical HMM'
    if name in ('CAPTCHA', 'DNSSEC', 'RRAM'):
        return name
    if 'IZ' in name:
        name = name.replace('IZ', 'IS')
    if 'IOR' in name:
        name = name.replace('IOR', 'IOUR')

    # Title case handling
    if not name.isupper():
        words = name.split(' ')
        result = ''
        for i, word in enumerate(words):
            if len(word) <= 2:
                result += word
            else:
                result += word[0].upper() + word[1:]
            if i < len(words) - 1:
                result += ' '
        return result

    # Uppercase abbreviation handling
    words = name.split(' ')
    result = ''
    for i, word in enumerate(words):
        if len(word) <= 3 or '/' in word or word in ('MITM', 'SIEM'):
            result += word
        else:
            result += word[0] + word[1:].lower()
        if i < len(words) - 1:
            result += ' '
    return result


def zero_negative_curves(data, forecast):
    """Set negative values to zero (artifact of smoothing)"""
    data = torch.clamp(data, min=0.0)
    forecast = torch.clamp(forecast, min=0.0)
    return data, forecast


def load_nodes_metadata(nodes_file):
    """
    Load node metadata from nodes.csv
    Returns: (column_names, column_index_dict, category_dict)
    """
    col_names = []
    col_index = {}
    col_category = {}
    
    try:
        with open(nodes_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                token = row.get('token', f'node_{i}')
                display = row.get('display', token)
                category = row.get('category', 'Unknown')
                col_names.append(token)
                col_index[token] = i
                col_category[token] = category
    except FileNotFoundError:
        print(f"[warn] {nodes_file} not found, using column indices")
        return None, None, None
    
    return col_names, col_index, col_category


def resolve_node_names(preferred_nodes_file, n_nodes):
    candidate_paths = []
    if preferred_nodes_file:
        candidate_paths.append(preferred_nodes_file)
    candidate_paths.append(os.path.join('data', 'nodes.csv'))
    candidate_paths.append('nodes.csv')

    tried = set()
    for path in candidate_paths:
        if not path:
            continue
        norm_path = os.path.normpath(path)
        if norm_path in tried:
            continue
        tried.add(norm_path)
        col_names, _, _ = load_nodes_metadata(path)
        if col_names and len(col_names) >= int(n_nodes):
            return col_names[:n_nodes], path

    return [f'node_{i}' for i in range(int(n_nodes))], None


def build_graph_from_adjacency(graph_file, col_names, threshold=0.1):
    """
    Build graph from adjacency matrix (e.g., graph_topk_k12.csv)
    Returns dict: {node: [connected_nodes]}
    """
    graph = defaultdict(list)
    
    try:
        adj_matrix = np.loadtxt(graph_file, delimiter=',')
        n_nodes = min(adj_matrix.shape[0], len(col_names) if col_names else adj_matrix.shape[0])
        
        for i in range(n_nodes):
            node_name = col_names[i] if col_names else f'node_{i}'
            for j in range(n_nodes):
                if i != j and adj_matrix[i, j] > threshold:
                    connected_name = col_names[j] if col_names else f'node_{j}'
                    graph[node_name].append(connected_name)
        
        print(f'[forecast] Graph loaded with {len(graph)} nodes')
    except Exception as e:
        print(f'[forecast] Warning: Could not load graph from {graph_file}: {e}')
    
    return graph


def _load_adjacency_matrix(graph_file, n_nodes):
    try:
        adj_matrix = np.loadtxt(graph_file, delimiter=',')
    except Exception as e:
        raise RuntimeError(f"Failed to load graph adjacency: {e}")

    if adj_matrix.ndim != 2:
        raise ValueError(f"Adjacency must be 2D, got shape={adj_matrix.shape}")

    size = min(adj_matrix.shape[0], adj_matrix.shape[1], n_nodes)
    return adj_matrix[:size, :size]


def _row_normalize(adj, eps=1e-12):
    row_sum = adj.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum < eps, 1.0, row_sum)
    return adj / row_sum


def export_uncertainty_propagation(
    forecast,
    confidence,
    variance,
    col_names,
    graph_file,
    output_dir,
    threshold=0.1,
    max_hops=3,
):
    """
    Export uncertainty propagation metrics:
      - per-node uncertainty (mean variance + mean CI)
      - edge-wise coupling (corr + amplification ratio)
      - hop-wise propagation (diffusion on adjacency)
    """
    os.makedirs(output_dir, exist_ok=True)

    var_np = variance.cpu().numpy() if torch.is_tensor(variance) else np.asarray(variance)
    ci_np = confidence.cpu().numpy() if torch.is_tensor(confidence) else np.asarray(confidence)

    if var_np.ndim != 2:
        raise ValueError(f"variance must be [T,N], got {var_np.shape}")

    n_nodes = var_np.shape[1]
    if col_names is None:
        col_names = [f"node_{i}" for i in range(n_nodes)]

    adj = _load_adjacency_matrix(graph_file, n_nodes)
    adj = np.where(adj > float(threshold), adj, 0.0)

    mean_var = np.mean(var_np, axis=0)
    mean_ci = np.mean(ci_np, axis=0)

    # --- Node uncertainty summary ---
    node_path = os.path.join(output_dir, "node_uncertainty.csv")
    with open(node_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node", "mean_variance", "mean_ci_95"])
        for i, name in enumerate(col_names[:n_nodes]):
            writer.writerow([name, float(mean_var[i]), float(mean_ci[i])])

    # --- Edge-wise coupling ---
    edge_path = os.path.join(output_dir, "edge_uncertainty.csv")
    edge_summary_path = os.path.join(output_dir, "edge_uncertainty_summary.csv")

    positive_var = mean_var[np.isfinite(mean_var) & (mean_var > 0.0)]
    var_floor = float(np.percentile(positive_var, 5)) if positive_var.size else 1e-8
    var_floor = max(var_floor, 1e-8)

    amp_values_raw = []
    amp_values_floored = []
    corr_values = []

    with open(edge_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "src",
            "dst",
            "weight",
            "uncertainty_corr",
            "amplification_ratio",
            "amplification_ratio_floored",
            "log10_amplification_ratio_floored",
            "src_variance_floored",
            "amplification_floor_applied",
            "src_mean_variance",
            "dst_mean_variance",
        ])

        eps = 1e-12
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j:
                    continue
                w = float(adj[i, j])
                if w <= 0.0:
                    continue
                src_series = var_np[:, i]
                dst_series = var_np[:, j]
                src_std = float(np.std(src_series))
                dst_std = float(np.std(dst_series))
                if src_std < eps or dst_std < eps:
                    corr = 0.0
                else:
                    corr = float(np.corrcoef(src_series, dst_series)[0, 1])
                amp = float(mean_var[j] / (mean_var[i] + eps))
                src_var_floored = float(max(float(mean_var[i]), var_floor))
                amp_floored = float(mean_var[j] / (src_var_floored + eps))
                log_amp_floored = float(np.log10(max(amp_floored, 1e-12)))
                floor_applied = 1 if float(mean_var[i]) < var_floor else 0
                writer.writerow([
                    col_names[i],
                    col_names[j],
                    w,
                    corr,
                    amp,
                    amp_floored,
                    log_amp_floored,
                    src_var_floored,
                    floor_applied,
                    float(mean_var[i]),
                    float(mean_var[j]),
                ])
                amp_values_raw.append(amp)
                amp_values_floored.append(amp_floored)
                corr_values.append(corr)

    with open(edge_summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "metric",
            "value",
        ])

        def _q(arr, q):
            if not arr:
                return float('nan')
            return float(np.quantile(np.asarray(arr, dtype=float), q))

        def _mean(arr):
            return float(np.mean(np.asarray(arr, dtype=float))) if arr else float('nan')

        def _median(arr):
            return float(np.median(np.asarray(arr, dtype=float))) if arr else float('nan')

        writer.writerow(["total_edges", len(amp_values_raw)])
        writer.writerow(["variance_floor", var_floor])
        writer.writerow(["raw_amp_mean", _mean(amp_values_raw)])
        writer.writerow(["raw_amp_median", _median(amp_values_raw)])
        writer.writerow(["raw_amp_q95", _q(amp_values_raw, 0.95)])
        writer.writerow(["raw_amp_q99", _q(amp_values_raw, 0.99)])
        writer.writerow(["floored_amp_mean", _mean(amp_values_floored)])
        writer.writerow(["floored_amp_median", _median(amp_values_floored)])
        writer.writerow(["floored_amp_q95", _q(amp_values_floored, 0.95)])
        writer.writerow(["floored_amp_q99", _q(amp_values_floored, 0.99)])
        writer.writerow(["uncertainty_corr_mean", _mean(corr_values)])
        writer.writerow(["uncertainty_corr_median", _median(corr_values)])

    # --- Hop-wise propagation (diffusion) ---
    hop_path = os.path.join(output_dir, "hop_uncertainty.csv")
    hop_summary_path = os.path.join(output_dir, "hop_uncertainty_summary.csv")
    hop_diag_path = os.path.join(output_dir, "hop_uncertainty_diagnostics.csv")
    adj_norm = _row_normalize(adj)
    u0 = mean_var.astype(float)

    with open(hop_path, "w", newline="") as f_hop, open(hop_summary_path, "w", newline="") as f_sum, open(hop_diag_path, "w", newline="") as f_diag:
        hop_writer = csv.writer(f_hop)
        sum_writer = csv.writer(f_sum)
        diag_writer = csv.writer(f_diag)
        hop_writer.writerow(["node", "hop", "propagated_mean_variance", "amplification_ratio"])
        sum_writer.writerow([
            "hop",
            "global_mean_variance",
            "global_amplification_ratio",
            "global_mean_ratio",
            "global_std_variance",
            "global_std_ratio",
            "global_q10_variance",
            "global_q50_variance",
            "global_q90_variance",
            "global_iqr_variance",
        ])
        diag_writer.writerow([
            "hop",
            "nodes_above_base_variance",
            "nodes_below_base_variance",
            "share_above_base_variance",
            "mean_abs_change_from_prev",
            "max_abs_change_from_prev",
        ])

        base_mean = float(np.mean(u0))
        base_std = float(np.std(u0))
        u_prev = u0
        for hop in range(1, int(max_hops) + 1):
            u_next = adj_norm.T @ u_prev
            global_mean = float(np.mean(u_next))
            global_std = float(np.std(u_next))
            q10 = float(np.quantile(u_next, 0.10))
            q50 = float(np.quantile(u_next, 0.50))
            q90 = float(np.quantile(u_next, 0.90))
            sum_writer.writerow([
                hop,
                global_mean,
                float(global_mean / (base_mean + 1e-12)),
                float(global_mean / (base_mean + 1e-12)),
                global_std,
                float(global_std / (base_std + 1e-12)),
                q10,
                q50,
                q90,
                float(q90 - q10),
            ])
            delta = np.abs(u_next - u_prev)
            above_base = int(np.sum(u_next > u0))
            below_base = int(np.sum(u_next < u0))
            diag_writer.writerow([
                hop,
                above_base,
                below_base,
                float(above_base / max(1, n_nodes)),
                float(np.mean(delta)),
                float(np.max(delta)),
            ])
            for i in range(n_nodes):
                hop_writer.writerow([
                    col_names[i],
                    hop,
                    float(u_next[i]),
                    float(u_next[i] / (u0[i] + 1e-12)),
                ])
            u_prev = u_next

    print(f"[forecast] Uncertainty propagation exports saved to {output_dir}")

def plot_forecast(data, forecast, confidence, node_idx, node_name, 
                  start_year=2004, start_month=1, steps_per_year=12, 
                  output_dir='model/Bayesian/forecast/plots'):
    """
    Plot historical data + forecast for a single node with confidence intervals
    
    Args:
        data: Historical data (T_hist, N)
        forecast: Future forecast (T_future, N)
        confidence: Confidence interval width (T_future, N)
        node_idx: Index of node to plot
        node_name: Display name of node
        start_year, start_month: Data start date
        steps_per_year: Time steps per year (12 for monthly)
        output_dir: Directory to save plots
    """
    data, forecast = zero_negative_curves(data, forecast)
    
    pyplot.style.use('seaborn-v0_8-darkgrid') 
    fig = pyplot.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)
    
    # Extract data for this node
    hist_data = data[:, node_idx].cpu().numpy() if torch.is_tensor(data) else data[:, node_idx]
    fore_data = forecast[:, node_idx].cpu().numpy() if torch.is_tensor(forecast) else forecast[:, node_idx]
    conf_data = confidence[:, node_idx].cpu().numpy() if torch.is_tensor(confidence) else confidence[:, node_idx]
    
    T_hist = len(hist_data)
    T_fore = len(fore_data)
    
    # Generate date labels
    def offset_year_month(base_year, base_month, offset, steps_per_year):
        """Convert offset steps to (year, month)"""
        total_months = (base_year - 1) * 12 + (base_month - 1) + offset
        year = total_months // 12 + 1
        month = total_months % 12 + 1
        return year, month
    
    # Create x-axis positions
    x_hist = np.arange(T_hist)
    x_fore = np.arange(T_hist - 1, T_hist + T_fore)  # Include connection point
    
    # Plot historical data
    ax.plot(x_hist, hist_data, '-', color='RoyalBlue', label='Historical', linewidth=2)
    
    # Plot forecast (with connection to last historical point)
    ax.plot(x_fore, np.concatenate([[hist_data[-1]], fore_data]), 
            '-', color='Crimson', label='Forecast', linewidth=2)
    
    # Add confidence interval
    fore_with_connection = np.concatenate([[hist_data[-1]], fore_data])
    conf_with_connection = np.concatenate([[0], conf_data])
    ax.fill_between(x_fore, 
                     fore_with_connection - conf_with_connection,
                     fore_with_connection + conf_with_connection,
                     color='Crimson', alpha=0.3, label='95% CI')
    
    # Generate year labels for x-axis (every 12 months)
    total_steps = T_hist + T_fore
    tick_positions = []
    tick_labels = []
    for step in range(0, total_steps, steps_per_year):
        yr, mn = offset_year_month(start_year, start_month, step, steps_per_year)
        tick_positions.append(step)
        tick_labels.append(str(yr))
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=11)
    
    # Mark forecast region
    ax.axvspan(T_hist, T_hist + T_fore, color='skyblue', alpha=0.15, label='Forecast Period')
    
    ax.set_ylabel('Trend Value', fontsize=14)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_title(consistent_name(node_name), fontsize=16, pad=15)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    safe_name = node_name.replace('/', '_').replace('\\', '_')
    pyplot.savefig(os.path.join(output_dir, f'{safe_name}.png'), 
                   bbox_inches='tight', dpi=300)
    pyplot.savefig(os.path.join(output_dir, f'{safe_name}.pdf'), 
                   bbox_inches='tight', format='pdf')
    pyplot.close()
    print(f'[forecast] Saved plot for {node_name}')


def save_forecast_data(data, forecast, confidence, variance, col_names, 
                       output_dir='model/Bayesian/forecast/data'):
    """Save numerical forecast data to text files"""
    os.makedirs(output_dir, exist_ok=True)
    
    n_nodes = data.shape[1]
    for i in range(n_nodes):
        node_name = col_names[i] if col_names else f'node_{i}'
        safe_name = node_name.replace('/', '_').replace('\\', '_')
        
        hist = data[:, i].cpu().numpy() if torch.is_tensor(data) else data[:, i]
        fore = forecast[:, i].cpu().numpy() if torch.is_tensor(forecast) else forecast[:, i]
        conf = confidence[:, i].cpu().numpy() if torch.is_tensor(confidence) else confidence[:, i]
        var = variance[:, i].cpu().numpy() if torch.is_tensor(variance) else variance[:, i]
        
        filepath = os.path.join(output_dir, f'{safe_name}.txt')
        with open(filepath, 'w') as f:
            f.write(f'Node: {node_name}\n')
            f.write(f'Historical Data: {hist.tolist()}\n')
            f.write(f'Forecast: {fore.tolist()}\n')
            f.write(f'95% Confidence: {conf.tolist()}\n')
            f.write(f'Variance: {var.tolist()}\n')
    
    print(f'[forecast] Saved data for {n_nodes} nodes to {output_dir}')


def export_forecast_csv(forecast, confidence, variance, col_names, start_year, start_month,
                        steps_per_year=12, output_path='model/Bayesian/forecast/forecast.csv'):
    """Export forecast-only data to a single CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def offset_year_month(base_year, base_month, offset, steps_per_year):
        total_months = (base_year - 1) * 12 + (base_month - 1) + offset
        year = total_months // 12 + 1
        month = total_months % 12 + 1
        return year, month

    fore_np = forecast.cpu().numpy() if torch.is_tensor(forecast) else forecast
    conf_np = confidence.cpu().numpy() if torch.is_tensor(confidence) else confidence
    var_np = variance.cpu().numpy() if torch.is_tensor(variance) else variance

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['node', 'year', 'month', 'forecast', 'ci_95', 'variance'])
        for step in range(fore_np.shape[0]):
            year, month = offset_year_month(start_year, start_month, step, steps_per_year)
            for i, node_name in enumerate(col_names):
                writer.writerow([
                    node_name,
                    year,
                    month,
                    float(fore_np[step, i]),
                    float(conf_np[step, i]),
                    float(var_np[step, i]),
                ])

    print(f'[forecast] CSV export saved to {output_path}')


def export_range_forecast_csvs(forecast, confidence, col_names, output_dir, start_year, start_month,
                               steps_per_year=12):
    """Export 2026-2028 CSVs expected by plot_graph_forecast.py."""
    os.makedirs(output_dir, exist_ok=True)

    def offset_year_month(base_year, base_month, offset, steps_per_year):
        total_months = (base_year - 1) * 12 + (base_month - 1) + offset
        year = total_months // 12 + 1
        month = total_months % 12 + 1
        return year, month

    fore_np = forecast.cpu().numpy() if torch.is_tensor(forecast) else forecast
    conf_np = confidence.cpu().numpy() if torch.is_tensor(confidence) else confidence

    point_path = os.path.join(output_dir, 'forecast_2026_2028.csv')
    lo_path = os.path.join(output_dir, 'forecast_2026_2028_pi_95_lower.csv')
    hi_path = os.path.join(output_dir, 'forecast_2026_2028_pi_95_upper.csv')

    with open(point_path, 'w', newline='') as f_point, \
        open(lo_path, 'w', newline='') as f_lo, \
        open(hi_path, 'w', newline='') as f_hi:
        point_writer = csv.writer(f_point)
        lo_writer = csv.writer(f_lo)
        hi_writer = csv.writer(f_hi)

        header = ['t'] + list(col_names)
        point_writer.writerow(header)
        lo_writer.writerow(header)
        hi_writer.writerow(header)

        for step in range(fore_np.shape[0]):
            year, month = offset_year_month(start_year, start_month, step, steps_per_year)
            t_label = f'{year:04d}-{month:02d}'
            row_point: list[object] = [t_label]
            row_lo: list[object] = [t_label]
            row_hi: list[object] = [t_label]
            for i in range(fore_np.shape[1]):
                mean_val = float(fore_np[step, i])
                ci_val = float(conf_np[step, i])
                row_point.append(mean_val)
                row_lo.append(mean_val - ci_val)
                row_hi.append(mean_val + ci_val)
            point_writer.writerow(row_point)
            lo_writer.writerow(row_lo)
            hi_writer.writerow(row_hi)

    print(f'[forecast] Forecast CSVs saved to {output_dir}')


def run_grouped_plots() -> None:
    try:
        from scripts import plot_graph_forecast
        print('[forecast] Generating grouped condition plots...')
        plot_graph_forecast.main()
        print('[forecast] ✓ Grouped plots generated successfully')
    except Exception as exc:
        import traceback
        print(f'[forecast] ⚠ WARNING: grouped plot generation failed!')
        print(f'[forecast] Error: {exc}')
        print(f'[forecast] Common fixes:')
        print(f'[forecast]   1. Run: python scripts/create_graph.py')
        print(f'[forecast]   2. Check data/graph.csv exists')
        print(f'[forecast]   3. Verify matplotlib backend: matplotlib.use("Agg")')
        if '--verbose' in sys.argv or '-v' in sys.argv:
            print('[forecast] Full traceback:')
            traceback.print_exc()


def load_trained_model(checkpoint_path, device='cpu', args=None):
    """
    Load trained model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        device: Device to load model onto
        args: Optional arguments dict with model hyperparameters
    
    Returns:
        model: Loaded model in eval mode
    """
    print(f'[forecast] Loading model from {checkpoint_path}')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model, seq_length, in_dim, state_dict = build_model_from_checkpoint(checkpoint, device)
        filtered, missing, unexpected, shape_mismatch = filter_state_dict_for_model(state_dict, model)
        if shape_mismatch:
            print(f'[forecast] Warning: {len(shape_mismatch)} parameter shape mismatches filtered')
        if unexpected:
            print(f'[forecast] Warning: {len(unexpected)} unexpected keys in checkpoint')
        if missing:
            print(f'[forecast] Warning: {len(missing)} missing keys after filtering')
        model.load_state_dict(filtered, strict=False)
        model.eval()
        print(f'[forecast] Model loaded successfully with {model.num_nodes if hasattr(model, "num_nodes") else "unknown"} nodes')
        return model, seq_length, in_dim
    
    # Extract model or handle direct model save
    if isinstance(checkpoint, dict):
        # Try different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            model_args = checkpoint.get('args', args)
        elif 'state_dict' in checkpoint:
            # Lightning/newer checkpoint format
            state_dict = checkpoint['state_dict']
            model_args = checkpoint.get('hparams', checkpoint.get('args', args))
        else:
            # Assume the dict itself is the state dict
            state_dict = checkpoint
            model_args = args
    elif isinstance(checkpoint, torch.nn.Module):
        # Checkpoint is the model itself
        checkpoint.eval()
        return checkpoint.to(device), getattr(checkpoint, 'seq_length', None), getattr(checkpoint, 'in_dim', None)
    else:
        state_dict = checkpoint
        model_args = args
    
    # Infer architecture from state_dict if args not provided
    if model_args is None:
        print('[forecast] Warning: Model args not found in checkpoint, using defaults')
        model_args = argparse.Namespace(
            gcn_true=True,
            buildA_true=True,
            gcn_depth=2,
            num_nodes=95,  # Will be overridden
            device=device,
            predefined_A=None,
            static_feat=None,
            dropout=0.0,
            subgraph_size=20,
            node_dim=40,
            dilation_exponential=2,
            conv_channels=96,
            residual_channels=96,
            skip_channels=192,
            end_channels=384,
            seq_length=24,
            in_dim=2,
            out_dim=36,
            layers=5,
            propalpha=0.05,
            tanhalpha=3,
            layer_norm_affline=True,
            temporal_attn=True,
            attn_heads=4,
            attn_dim=96,
            temporal_transformer=True,
            tt_layers=2,
        )
    elif isinstance(model_args, dict):
        # Convert dict to Namespace
        model_args = argparse.Namespace(**model_args)
    
    # Reconstruct model
    try:
        # Infer num_nodes from state_dict if possible
        for key in state_dict.keys():
            if 'nodevec1' in key:
                num_nodes = state_dict[key].shape[0]
                model_args.num_nodes = num_nodes
                break
        
        gauss_head = any(key.startswith('end_conv_gauss.') for key in state_dict.keys())
        model = gtnet(
            gcn_true=getattr(model_args, 'gcn_true', True),
            buildA_true=getattr(model_args, 'buildA_true', True),
            gcn_depth=getattr(model_args, 'gcn_depth', 2),
            num_nodes=getattr(model_args, 'num_nodes', 95),
            device=device,
            predefined_A=getattr(model_args, 'predefined_A', None),
            static_feat=getattr(model_args, 'static_feat', None),
            dropout=getattr(model_args, 'dropout', 0.0),
            subgraph_size=getattr(model_args, 'subgraph_size', 20),
            node_dim=getattr(model_args, 'node_dim', 40),
            dilation_exponential=getattr(model_args, 'dilation_exponential', 2),
            conv_channels=getattr(model_args, 'conv_channels', 96),
            residual_channels=getattr(model_args, 'residual_channels', 96),
            skip_channels=getattr(model_args, 'skip_channels', 192),
            end_channels=getattr(model_args, 'end_channels', 384),
            seq_length=getattr(model_args, 'seq_length', getattr(model_args, 'seq_in_len', 24)),
            in_dim=getattr(model_args, 'in_dim', 2),
            out_dim=getattr(model_args, 'out_dim', getattr(model_args, 'seq_out_len', 36)),
            layers=getattr(model_args, 'layers', 5),
            propalpha=getattr(model_args, 'propalpha', 0.05),
            tanhalpha=getattr(model_args, 'tanhalpha', 3),
            layer_norm_affline=getattr(model_args, 'layer_norm_affline', True),
            temporal_attn=getattr(model_args, 'temporal_attn', False),
            attn_heads=getattr(model_args, 'attn_heads', 4),
            attn_dim=getattr(model_args, 'attn_dim', 96),
            temporal_transformer=getattr(model_args, 'temporal_transformer', False),
            tt_layers=getattr(model_args, 'tt_layers', 0),
            gauss_head=bool(getattr(model_args, 'gauss_head', False)) or gauss_head,
        )
        
        # Load state dict with strict=False to allow missing optional components
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f'[forecast] Warning: Missing keys (optional components): {len(missing_keys)} keys')
        if unexpected_keys:
            print(f'[forecast] Warning: Unexpected keys: {len(unexpected_keys)} keys')
        
        model.to(device)
        model.eval()
        print(f'[forecast] Model loaded successfully with {model_args.num_nodes if hasattr(model_args, "num_nodes") else "unknown"} nodes')
        return model, getattr(model_args, 'seq_length', getattr(model_args, 'seq_in_len', None)), getattr(model_args, 'in_dim', None)
        
    except Exception as e:
        print(f'[forecast] Error loading model: {e}')
        raise


def generate_forecast(model, data_loader, forecast_months=36, mc_runs=50, device='cpu', force_in_dim=None):
    """
    Generate future forecast using trained model with MC Dropout
    
    Args:
        model: Trained model
        data_loader: DataLoader with historical data  
        forecast_months: Number of months to forecast
        mc_runs: Number of MC dropout iterations
        device: Device to run on
    
    Returns:
        forecast: Mean forecast (T_future, N)
        confidence: 95% CI width (T_future, N)
        variance: Forecast variance (T_future, N)
    """
    model.eval()
    
    # Get last sequence from data
    # Use the most recent seq_length time steps
    seq_length = getattr(model, 'seq_length', 24)
    
    # Get raw data from loader (original-unit levels, possibly offset-shifted)
    raw_data = data_loader.rawdat  # (T, N)
    scaler = StandardScaler(mean=data_loader.mu, std=data_loader.std)
    
    # Prepare input: last seq_length steps
    last_seq = raw_data[-seq_length:, :]  # (seq_length, N)

    # IMPORTANT: match DataLoaderS preprocessing.
    # normalize=3 in DataLoaderS means log1p (on non-negative) + z-score.
    use_log1p = bool(getattr(data_loader, 'use_log1p', False))
    y_offset = float(getattr(data_loader, 'y_offset', 0.0) or 0.0)

    if use_log1p:
        last_seq_proc = np.log1p(np.clip(last_seq, 0.0, None))
    else:
        last_seq_proc = last_seq

    # Normalize in the same space the model was trained on
    # Ensure tensor is on correct device before scaling
    last_seq_tensor = torch.from_numpy(last_seq_proc).float().to(device)
    last_seq_norm = scaler.transform(last_seq_tensor)
    
    # Reshape for model: (batch=1, features=1 or 2, N, T)
    # Check model in_dim to determine if dual channel is used
    in_dim = getattr(model, 'start_conv', None)
    if in_dim is not None:
        in_dim = in_dim.in_channels
    else:
        in_dim = 2  # default

    if force_in_dim is not None:
        force_in_dim = int(force_in_dim)
        if in_dim != force_in_dim:
            print(f'[forecast] Warning: override_in_dim={force_in_dim} does not match model in_dim={in_dim}; using model value')
        else:
            in_dim = force_in_dim
    
    if in_dim == 2:
        # Add percentage change channel
        pct = torch.zeros_like(last_seq_norm)
        pct[1:] = (last_seq_norm[1:] - last_seq_norm[:-1]) / (last_seq_norm[:-1].abs() + 1e-6)
        pct = torch.clamp(pct, -3.0, 3.0)  # clip
        X = torch.stack([last_seq_norm, pct], dim=0)  # (2, T, N)
    else:
        X = last_seq_norm.unsqueeze(0)  # (1, T, N)
    
    X = X.unsqueeze(0)  # (1, features, T, N)
    X = X.transpose(2, 3)  # (1, features, N, T)
    X = X.to(device)
    
    print(f'[forecast] Generating forecast with MC dropout ({mc_runs} runs)...')
    
    # MC Dropout iterations
    num_nodes = getattr(model, 'num_nodes', None)
    outputs = []
    logvar_outputs = []
    def _reshape_output(tensor, nodes_hint):
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
            if tensor.size(-1) == 1:
                tensor = tensor.squeeze(-1)
            if tensor.dim() == 2:
                if nodes_hint is None or tensor.shape[0] != nodes_hint:
                    tensor = tensor.T
            elif tensor.dim() == 3:
                if nodes_hint is not None and tensor.shape[0] != nodes_hint:
                    tensor = tensor.transpose(0, 1)
        elif tensor.dim() == 3:
            tensor = tensor.squeeze(0)
            if nodes_hint is not None and tensor.shape[0] != nodes_hint:
                tensor = tensor.T
        else:
            print(f'[forecast] Warning: unexpected output shape {tensor.shape}')
            tensor = tensor.squeeze()
        return tensor
    with torch.no_grad():
        for run in range(mc_runs):
            # Enable dropout during inference
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.train()
            
            output = model(X)  # Expected: tensor or dict with 'mean'
            logvar = None
            if isinstance(output, dict):
                logvar = output.get('logvar', None)
                output = output.get('mean', next(iter(output.values())))
            if run == 0:
                print(f'[forecast] Model output shape (raw): {output.shape}')
            
            # Handle different output formats
            output = _reshape_output(output, num_nodes)
            if logvar is not None:
                logvar = _reshape_output(logvar, num_nodes)
            
            if run == 0:
                print(f'[forecast] Output shape after processing: {output.shape} (expected: N x T_out)')
            
            outputs.append(output)  # (N, out_dim)
            if logvar is not None:
                logvar_outputs.append(logvar)
            
            if (run + 1) % 10 == 0:
                print(f'[forecast] MC run {run + 1}/{mc_runs}')
    
    # Stack outputs: (mc_runs, N, out_dim)
    outputs = torch.stack(outputs, dim=0)
    
    # Compute statistics
    mean_forecast = torch.mean(outputs, dim=0)  # (N, out_dim)
    var_epistemic = torch.var(outputs, dim=0)  # (N, out_dim)
    std_epistemic = torch.std(outputs, dim=0)  # (N, out_dim)
    z = 1.96
    if logvar_outputs:
        logvar_stack = torch.stack(logvar_outputs, dim=0)
        logvar_stack = torch.nan_to_num(logvar_stack, nan=0.0, posinf=10.0, neginf=-10.0)
        logvar_stack = logvar_stack.clamp(min=-10.0, max=10.0)
        aleatoric_var = torch.exp(logvar_stack).clamp_min(1e-6)
        mean_aleatoric = torch.mean(aleatoric_var, dim=0)
        var_forecast = var_epistemic + mean_aleatoric
        mean_var_for_ci = mean_aleatoric + (var_epistemic / max(1, mc_runs))
        ci_forecast = z * torch.sqrt(mean_var_for_ci)
    else:
        var_forecast = var_epistemic
        ci_forecast = z * std_epistemic / np.sqrt(mc_runs)  # (N, out_dim)
    
    # Inverse transform back to the loader's pre-normalization space
    # mean_forecast is (N, out_dim), transpose to (out_dim, N) for scaler
    mean_forecast_t = mean_forecast.transpose(0, 1)  # (out_dim, N)
    mean_forecast = scaler.inverse_transform(mean_forecast_t)  # (out_dim, N) in log1p-space if use_log1p
    
    # Variance and CI scale with std^2 and std respectively
    # Get std as tensor for proper broadcasting
    std_tensor = scaler.std if torch.is_tensor(scaler.std) else torch.tensor(scaler.std)
    std_tensor = std_tensor.to(var_forecast.device).unsqueeze(1)  # (N, 1)
    
    var_forecast = var_forecast * (std_tensor ** 2)  # (N, out_dim)
    ci_forecast = ci_forecast * std_tensor  # (N, out_dim)
    
    var_forecast = var_forecast.transpose(0, 1)  # (out_dim, N)
    ci_forecast = ci_forecast.transpose(0, 1)  # (out_dim, N)

    # Undo optional log1p transform so saved forecasts/CI are in original units.
    if use_log1p:
        # Convert symmetric CI in log-space into an approximate symmetric half-width in original space
        lower_log = mean_forecast - ci_forecast
        upper_log = mean_forecast + ci_forecast

        mean_orig = torch.expm1(mean_forecast)
        lower_orig = torch.expm1(lower_log)
        upper_orig = torch.expm1(upper_log)

        # Optional undo of any y-offset shift applied in loader
        if y_offset != 0.0:
            mean_orig = mean_orig - y_offset
            lower_orig = lower_orig - y_offset
            upper_orig = upper_orig - y_offset

        mean_orig = torch.clamp(mean_orig, min=0.0)
        lower_orig = torch.clamp(lower_orig, min=0.0)
        upper_orig = torch.clamp(upper_orig, min=0.0)

        ci_forecast = torch.clamp((upper_orig - lower_orig) / 2.0, min=0.0)
        # Delta-method variance approximation under expm1
        # If X = exp(Y) - 1, dX/dY = exp(Y)
        var_forecast = torch.clamp((torch.exp(mean_forecast) ** 2) * var_forecast, min=0.0)
        mean_forecast = mean_orig
    
    # Truncate or pad to requested forecast_months
    out_dim = mean_forecast.shape[0]
    if forecast_months < out_dim:
        mean_forecast = mean_forecast[:forecast_months]
        ci_forecast = ci_forecast[:forecast_months]
        var_forecast = var_forecast[:forecast_months]
    elif forecast_months > out_dim:
        # Pad with last values (simple persistence)
        pad_len = forecast_months - out_dim
        mean_forecast = torch.cat([mean_forecast, mean_forecast[-1:].repeat(pad_len, 1)], dim=0)
        ci_forecast = torch.cat([ci_forecast, ci_forecast[-1:].repeat(pad_len, 1)], dim=0)
        var_forecast = torch.cat([var_forecast, var_forecast[-1:].repeat(pad_len, 1)], dim=0)
    
    print(f'[forecast] Forecast generated: {mean_forecast.shape}')
    return mean_forecast, ci_forecast, var_forecast


def main():
    parser = argparse.ArgumentParser(description='Generate future forecasts from trained B-MTGNN model')
    parser.add_argument('--checkpoint', type=str, default='model/Bayesian/model.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--data', type=str, default='data/sm_data_g.csv',
                        help='Path to historical data CSV')
    parser.add_argument('--nodes', type=str, default='data/nodes.csv',
                        help='Path to nodes metadata CSV')
    parser.add_argument('--graph', type=str, default='data/graph_topk_k12.csv',
                        help='Path to graph adjacency matrix')
    parser.add_argument('--forecast_months', type=int, default=36,
                        help='Number of months to forecast (default: 36 for 3 years)')
    parser.add_argument('--mc_runs', type=int, default=50,
                        help='Number of MC dropout runs for uncertainty (default: 50)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run on (cpu or cuda:0)')
    parser.add_argument('--output_dir', type=str, default='model/Bayesian/forecast',
                        help='Base directory for outputs')
    parser.add_argument('--seq_in_len', type=int, default=24,
                        help='Input sequence length (must match trained model)')
    parser.add_argument('--override_seq_length', type=int, default=None,
                        help='Override checkpoint seq_length for data window if compatible')
    parser.add_argument('--override_in_dim', type=int, default=None,
                        help='Override checkpoint in_dim for input features if compatible')
    parser.add_argument('--normalize', type=int, default=3,
                        help='Normalization mode (3=per-node z-score, 2=global, etc.)')
    parser.add_argument('--has_header', action='store_true', default=True,
                        help='Data file has header row')
    parser.add_argument('--drop_first_col', action='store_true',
                        help='Drop first column (date column)')
    parser.add_argument('--no_node_plots', action='store_true', default=True,
                        help='Skip per-node plots (default: True)')
    parser.add_argument('--node_plots', action='store_false', dest='no_node_plots',
                        help='Enable per-node plots')
    parser.add_argument('--uncertainty_threshold', type=float, default=0.1,
                        help='Adjacency threshold for uncertainty propagation metrics')
    parser.add_argument('--uncertainty_max_hops', type=int, default=3,
                        help='Max hops for uncertainty propagation summary')
    
    args = parser.parse_args()
    
    print('='*80)
    print('B-MTGNN Future Forecast')
    print('='*80)
    
    # Load nodes metadata
    col_names, col_index, col_category = load_nodes_metadata(args.nodes)
    if col_names is None:
        print(f'[forecast] Warning: Could not load nodes metadata from {args.nodes}; fallback will be attempted after data load')
    
    # Load trained model first to align seq_length
    model, ckpt_seq_len, ckpt_in_dim = load_trained_model(args.checkpoint, device=args.device, args=None)

    if args.override_seq_length is not None:
        if ckpt_seq_len and int(args.override_seq_length) != int(ckpt_seq_len):
            print(f'[forecast] Warning: override_seq_length={args.override_seq_length} does not match checkpoint seq_length={ckpt_seq_len}; using checkpoint value')
        else:
            ckpt_seq_len = int(args.override_seq_length)

    if ckpt_seq_len and ckpt_seq_len != args.seq_in_len:
        print(f'[forecast] Using checkpoint seq_length={ckpt_seq_len} instead of args.seq_in_len={args.seq_in_len}')
        args.seq_in_len = int(ckpt_seq_len)

    # Load data
    print(f'[forecast] Loading data from {args.data}')
    data_loader = DataLoaderS(
        file_name=args.data,
        train=1.0,  # Use all historical data for analysis
        valid=0.0,  # No validation split needed for forecasting
        device=args.device,
        horizon=args.forecast_months,
        window=args.seq_in_len,
        normalize=args.normalize,
        has_header=args.has_header,
        drop_first_col=args.drop_first_col,
    )
    
    print(f'[forecast] Data shape: {data_loader.rawdat.shape}')
    print(f'[forecast] Date range: {data_loader.start_year}/{data_loader.start_month} - '
          f'{data_loader.start_year + (len(data_loader.rawdat)-1)//12}')
    
    # Override col_names if not loaded from nodes.csv
    if col_names is None:
        n_cols = data_loader.rawdat.shape[1]
        col_names, resolved_path = resolve_node_names(args.nodes, n_cols)
        if resolved_path:
            print(f'[forecast] Using fallback node metadata from {resolved_path}')
        else:
            print('[forecast] Warning: Falling back to generic node_i labels')
        col_index = {name: i for i, name in enumerate(col_names)}
    
    # Generate forecast
    forecast, confidence, variance = generate_forecast(
        model=model,
        data_loader=data_loader,
        forecast_months=args.forecast_months,
        mc_runs=args.mc_runs,
        device=args.device,
        force_in_dim=args.override_in_dim
    )
    
    # Convert historical data to tensor
    historical_data = torch.from_numpy(data_loader.rawdat).float()
    
    # Save numerical data
    save_forecast_data(
        data=historical_data,
        forecast=forecast,
        confidence=confidence,
        variance=variance,
        col_names=col_names,
        output_dir=os.path.join(args.output_dir, 'data')
    )

    export_forecast_csv(
        forecast=forecast,
        confidence=confidence,
        variance=variance,
        col_names=col_names,
        start_year=data_loader.start_year,
        start_month=data_loader.start_month,
        steps_per_year=data_loader.steps_per_year,
        output_path=os.path.join(args.output_dir, 'forecast.csv')
    )

    if args.graph and os.path.exists(args.graph):
        try:
            export_uncertainty_propagation(
                forecast=forecast,
                confidence=confidence,
                variance=variance,
                col_names=col_names,
                graph_file=args.graph,
                output_dir=os.path.join(args.output_dir, 'uncertainty'),
                threshold=args.uncertainty_threshold,
                max_hops=args.uncertainty_max_hops,
            )
        except Exception as e:
            print(f"[forecast] Warning: uncertainty propagation export failed: {e}")
    else:
        print(f"[forecast] Skipping uncertainty propagation: graph file not found: {args.graph}")

    if args.forecast_months == 36:
        export_range_forecast_csvs(
            forecast=forecast,
            confidence=confidence,
            col_names=col_names,
            output_dir=args.output_dir,
            start_year=data_loader.start_year,
            start_month=data_loader.start_month,
            steps_per_year=data_loader.steps_per_year,
        )
        run_grouped_plots()
    else:
        print('[forecast] Skipping grouped plots (requires 36-month forecast)')
    
    plot_dir = os.path.join(args.output_dir, 'plots')
    
    if not args.no_node_plots:
        # Generate plots for all nodes
        print(f'[forecast] Generating plots...')
        for i, node_name in enumerate(col_names):
            plot_forecast(
                data=historical_data,
                forecast=forecast,
                confidence=confidence,
                node_idx=i,
                node_name=node_name,
                start_year=data_loader.start_year,
                start_month=data_loader.start_month,
                steps_per_year=data_loader.steps_per_year,
                output_dir=plot_dir
            )
    else:
        print('[forecast] Skipping per-node plots (--no_node_plots)')
    
    print('='*80)
    print(f'[forecast] Forecast complete!')
    print(f'[forecast] Outputs saved to {args.output_dir}')
    if not args.no_node_plots:
        print(f'[forecast] - Plots: {plot_dir}')
    print(f'[forecast] - Data: {os.path.join(args.output_dir, "data")}')
    print('='*80)


if __name__ == '__main__':
    main()
    








