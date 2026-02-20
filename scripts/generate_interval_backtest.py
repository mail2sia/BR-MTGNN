#!/usr/bin/env python3
"""
Generate rolling historical interval backtest samples for uncertainty calibration.

Outputs CSV with columns:
    node, category, horizon, window_index, t_index, forecast, ci_95, lower, upper, actual

How to run: python scripts/generate_interval_backtest.py --checkpoint model/Bayesian/model.pt --mc_runs 100 --seed 42 --max_horizon 6 --device cpu    
Designed to feed: scripts/plot_uncertainty_insights.py interval diagnostics
(PICP, ACE, MPIW).
"""

import argparse
import csv
import os
import sys
import random
from pathlib import Path

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.util import DataLoaderS, to_model_layout
from scripts.forecast import load_trained_model, resolve_node_names


def infer_category(node_name: object) -> str:
    if not isinstance(node_name, str) or not node_name:
        return 'Other'
    token = node_name.strip()
    if token.startswith('PT_'):
        return 'PT'
    if token.startswith('RMD_'):
        return 'RMD'
    return 'Other'


def _extract_output_nt(output, num_nodes):
    if isinstance(output, dict):
        output = output.get('mean', next(iter(output.values())))

    if output.dim() == 4:
        output = output.squeeze(0)
        if output.size(-1) == 1:
            output = output.squeeze(-1)
        if output.dim() == 2:
            if num_nodes is None or output.shape[0] != num_nodes:
                output = output.T
        elif output.dim() == 3:
            if num_nodes is not None and output.shape[0] != num_nodes:
                output = output.transpose(0, 1)
    elif output.dim() == 3:
        output = output.squeeze(0)
        if num_nodes is not None and output.shape[0] != num_nodes:
            output = output.T
    else:
        output = output.squeeze()

    if output.dim() != 2:
        raise RuntimeError(f"Unexpected model output shape after normalization: {tuple(output.shape)}")

    return output


def _to_original_space(data_loader, mean_z, half_z, y_true_z, window_idx):
    # mean_z, half_z, y_true_z are [N]
    mean_z_2d = mean_z.unsqueeze(0)
    half_z_2d = half_z.unsqueeze(0)
    y_true_z_2d = y_true_z.unsqueeze(0)

    if getattr(data_loader, 'rolling', False) and hasattr(data_loader, 'per_window_mu') and 'test' in data_loader.per_window_mu:
        in_range = 0 <= window_idx < len(data_loader.per_window_mu['test'])
        if in_range:
            mu = torch.as_tensor(
                data_loader.per_window_mu['test'][window_idx],
                dtype=mean_z_2d.dtype,
                device=mean_z_2d.device,
            ).unsqueeze(0)
            std = torch.as_tensor(
                data_loader.per_window_std['test'][window_idx],
                dtype=mean_z_2d.dtype,
                device=mean_z_2d.device,
            ).unsqueeze(0)
            mean_base = mean_z_2d * std + mu
            true_base = y_true_z_2d * std + mu
            lower_base = (mean_z_2d - half_z_2d) * std + mu
            upper_base = (mean_z_2d + half_z_2d) * std + mu

            if getattr(data_loader, 'use_log1p', False):
                mean_o = torch.expm1(mean_base)
                true_o = torch.expm1(true_base)
                lower_o = torch.expm1(lower_base)
                upper_o = torch.expm1(upper_base)
            else:
                mean_o = mean_base
                true_o = true_base
                lower_o = lower_base
                upper_o = upper_base

            ci_o = 0.5 * (upper_o - lower_o)
            return mean_o.squeeze(0), ci_o.squeeze(0), lower_o.squeeze(0), upper_o.squeeze(0), true_o.squeeze(0)

    # global inversion fallback
    if getattr(data_loader, 'use_log1p', False):
        lower_o = data_loader.inv_transform_like(mean_z_2d - half_z_2d)
        upper_o = data_loader.inv_transform_like(mean_z_2d + half_z_2d)
        mean_o = data_loader.inv_transform_like(mean_z_2d)
        true_o = data_loader.inv_transform_like(y_true_z_2d)
        ci_o = 0.5 * (upper_o - lower_o)
    else:
        mean_o = data_loader.inv_transform_like(mean_z_2d)
        true_o = data_loader.inv_transform_like(y_true_z_2d)
        ci_o = half_z_2d * data_loader.std_expand_like(half_z_2d)
        lower_o = mean_o - ci_o
        upper_o = mean_o + ci_o

    return mean_o.squeeze(0), ci_o.squeeze(0), lower_o.squeeze(0), upper_o.squeeze(0), true_o.squeeze(0)


def main():
    parser = argparse.ArgumentParser(description='Generate rolling interval backtest samples with actuals')
    parser.add_argument('--checkpoint', type=str, default='model/Bayesian/model.pt')
    parser.add_argument('--data', type=str, default='data/sm_data_g.csv')
    parser.add_argument('--nodes', type=str, default='data/nodes.csv')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--mc_runs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seq_in_len', type=int, default=120)
    parser.add_argument('--normalize', type=int, default=3)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--valid_ratio', type=float, default=0.2)
    parser.add_argument('--horizon', type=int, default=1, help='Target horizon for backtest samples (default one-step ahead)')
    parser.add_argument('--max_horizon', type=int, default=1, help='Number of forecast steps to export per window (true multi-horizon)')
    parser.add_argument('--max_windows', type=int, default=0, help='Limit number of latest test windows (0=all)')
    parser.add_argument('--output', type=str, default='model/Bayesian/forecast/calibration_interval_samples.csv')
    args = parser.parse_args()

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    args.max_horizon = int(max(1, args.max_horizon))

    model, ckpt_seq_len, _ = load_trained_model(args.checkpoint, device=args.device, args=None)
    if ckpt_seq_len:
        args.seq_in_len = int(ckpt_seq_len)

    data_loader = DataLoaderS(
        file_name=args.data,
        train=float(args.train_ratio),
        valid=float(args.valid_ratio),
        device=args.device,
        horizon=int(args.horizon),
        window=int(args.seq_in_len),
        normalize=int(args.normalize),
        has_header=True,
        drop_first_col=False,
        out=int(args.max_horizon),
    )

    n_nodes = int(data_loader.rawdat.shape[1])
    col_names, resolved_nodes = resolve_node_names(args.nodes, n_nodes)
    if resolved_nodes:
        print(f"[backtest] Using node metadata from {resolved_nodes}")
    else:
        print('[backtest] Warning: using generic node_i names')

    test_x, test_y = data_loader.test
    if test_x is None or len(test_x) == 0:
        raise RuntimeError('No test samples available for interval backtest')

    candidate_idx = list(range(len(test_x)))
    if args.max_windows and args.max_windows > 0:
        candidate_idx = candidate_idx[-int(args.max_windows):]

    test_start_indices = getattr(data_loader, '_test_idx_set', list(range(len(test_x))))

    z = 1.96
    model.eval()
    rows = []

    with torch.no_grad():
        for k, local_idx in enumerate(candidate_idx):
            x_input = test_x[local_idx].to(args.device)  # [P,N,C]
            y_true = test_y[local_idx].to(args.device)   # [1,N] or [T,N]

            X_raw = x_input.unsqueeze(0)  # [1,P,N,C]
            expected_in = int(getattr(getattr(model, 'start_conv', None), 'in_channels', X_raw.shape[-1]))
            if expected_in == 2 and X_raw.shape[-1] == 1:
                level = X_raw[..., 0]
                pct = torch.zeros_like(level)
                pct[:, 1:, :] = (level[:, 1:, :] - level[:, :-1, :]) / (level[:, :-1, :].abs() + 1e-6)
                pct = torch.clamp(pct, -3.0, 3.0)
                X_raw = torch.stack([level, pct], dim=-1)
            exp_len = int(getattr(model, 'seq_length', x_input.shape[0]))
            X = to_model_layout(X_raw, exp_len, debug=False).to(args.device)

            if y_true.dim() == 1:
                y_true_mat = y_true.unsqueeze(0)
            else:
                y_true_mat = y_true

            outputs = []
            for _ in range(int(args.mc_runs)):
                for module in model.modules():
                    if isinstance(module, torch.nn.Dropout):
                        module.train()
                out = model(X)
                out = _extract_output_nt(out, num_nodes=n_nodes)
                outputs.append(out)

            mc = torch.stack(outputs, dim=0)  # [mc, N, T_out]
            pred_h = int(mc.shape[2])
            true_h = int(y_true_mat.shape[0])
            h_count = int(min(args.max_horizon, pred_h, true_h))

            w_idx = int(local_idx)
            t_index_base = int(test_start_indices[local_idx]) if local_idx < len(test_start_indices) else int(local_idx)

            for horizon_idx in range(h_count):
                mean_z = mc[:, :, horizon_idx].mean(dim=0)
                std_z = mc[:, :, horizon_idx].std(dim=0) + 1e-8
                half_z = z * std_z
                y_true_z = y_true_mat[horizon_idx]

                mean_o, ci_o, lower_o, upper_o, true_o = _to_original_space(data_loader, mean_z, half_z, y_true_z, w_idx)

                mean_np = mean_o.detach().cpu().numpy()
                ci_np = ci_o.detach().cpu().numpy()
                lo_np = lower_o.detach().cpu().numpy()
                hi_np = upper_o.detach().cpu().numpy()
                true_np = true_o.detach().cpu().numpy()

                for node_idx, node_name in enumerate(col_names):
                    rows.append([
                        node_name,
                        infer_category(node_name),
                        int(horizon_idx + 1),
                        int(w_idx),
                        int(t_index_base + horizon_idx),
                        float(mean_np[node_idx]),
                        float(ci_np[node_idx]),
                        float(lo_np[node_idx]),
                        float(hi_np[node_idx]),
                        float(true_np[node_idx]),
                    ])

            if (k + 1) % 10 == 0:
                print(f"[backtest] Processed windows: {k + 1}/{len(candidate_idx)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['node', 'category', 'horizon', 'window_index', 't_index', 'forecast', 'ci_95', 'lower', 'upper', 'actual'])
        writer.writerows(rows)

    print(f"[backtest] Saved interval samples: {out_path}")
    print(f"[backtest] Rows: {len(rows)}")


if __name__ == '__main__':
    main()
