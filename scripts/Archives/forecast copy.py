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
from src.util import (
    StandardScaler, DataLoaderS,
    parse_dates_from_csv, load_graph
)

pyplot.rcParams['savefig.dpi'] = 1200


def consistent_name(name):
    """Clean and format node names for display"""
    name = name.replace('-ALL', '').replace('Mentions-', '').replace(' ALL', '')
    name = name.replace('Solution_', '').replace('_Mentions', '')
    
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
        return checkpoint.to(device)
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
        return model
        
    except Exception as e:
        print(f'[forecast] Error loading model: {e}')
        raise


def generate_forecast(model, data_loader, forecast_months=36, mc_runs=50, device='cpu'):
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
    last_seq_norm = scaler.transform(torch.from_numpy(last_seq_proc).float())
    
    # Reshape for model: (batch=1, features=1 or 2, N, T)
    # Check model in_dim to determine if dual channel is used
    in_dim = getattr(model, 'start_conv', None)
    if in_dim is not None:
        in_dim = in_dim.in_channels
    else:
        in_dim = 2  # default
    
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
    outputs = []
    with torch.no_grad():
        for run in range(mc_runs):
            # Enable dropout during inference
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.train()
            
            output = model(X)  # Expected: (1, N, out_dim) or (batch, T_out, N, features)
            if run == 0:
                print(f'[forecast] Model output shape (raw): {output.shape}')
            
            # Handle different output formats
            if output.dim() == 4:
                # (batch, T_out, N, features) = (1, 36, 95, 1)
                output = output[0, :, :, 0]  # (T_out, N) = (36, 95)
                output = output.T  # (N, T_out) = (95, 36)
            elif output.dim() == 3:
                # (batch, N, T_out) or (batch, T_out, N)
                output = output.squeeze(0)  # (N, T_out) or (T_out, N)
                # Assume (N, T_out) if first dim is num_nodes
                if output.shape[0] != 95:
                    output = output.T
            else:
                print(f'[forecast] Warning: unexpected output shape {output.shape}')
                output = output.squeeze()
            
            if run == 0:
                print(f'[forecast] Output shape after processing: {output.shape} (expected: N x T_out)')
            
            outputs.append(output)  # (N, out_dim)
            
            if (run + 1) % 10 == 0:
                print(f'[forecast] MC run {run + 1}/{mc_runs}')
    
    # Stack outputs: (mc_runs, N, out_dim)
    outputs = torch.stack(outputs, dim=0)
    
    # Compute statistics
    mean_forecast = torch.mean(outputs, dim=0)  # (N, out_dim)
    var_forecast = torch.var(outputs, dim=0)  # (N, out_dim)
    std_forecast = torch.std(outputs, dim=0)  # (N, out_dim)
    
    # 95% confidence interval: z * std / sqrt(n)
    z = 1.96
    ci_forecast = z * std_forecast / np.sqrt(mc_runs)  # (N, out_dim)
    
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
    parser.add_argument('--nodes', type=str, default='nodes.csv',
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
    parser.add_argument('--normalize', type=int, default=3,
                        help='Normalization mode (3=per-node z-score, 2=global, etc.)')
    parser.add_argument('--has_header', action='store_true', default=True,
                        help='Data file has header row')
    parser.add_argument('--drop_first_col', action='store_true',
                        help='Drop first column (date column)')
    
    args = parser.parse_args()
    
    print('='*80)
    print('B-MTGNN Future Forecast')
    print('='*80)
    
    # Load nodes metadata
    col_names, col_index, col_category = load_nodes_metadata(args.nodes)
    if col_names is None:
        print('[forecast] Warning: Using data columns directly')
    
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
        col_names = [f'node_{i}' for i in range(n_cols)]
        col_index = {name: i for i, name in enumerate(col_names)}
    
    # Load trained model
    model = load_trained_model(args.checkpoint, device=args.device, args=None)
    
    # Generate forecast
    forecast, confidence, variance = generate_forecast(
        model=model,
        data_loader=data_loader,
        forecast_months=args.forecast_months,
        mc_runs=args.mc_runs,
        device=args.device
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
    
    # Generate plots for all nodes
    print(f'[forecast] Generating plots...')
    plot_dir = os.path.join(args.output_dir, 'plots')
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
    
    print('='*80)
    print(f'[forecast] Forecast complete!')
    print(f'[forecast] Outputs saved to {args.output_dir}')
    print(f'[forecast] - Plots: {plot_dir}')
    print(f'[forecast] - Data: {os.path.join(args.output_dir, "data")}')
    print('='*80)


if __name__ == '__main__':
    main()
    








