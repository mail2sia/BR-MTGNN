"""
Main runner: train and test (uses src/*)

The v4_3ep preset is automatically applied and matches the optimized configuration
from metrics_validation.json (MAE: 7.14, RMSE: 19.55, sMAPE: 4.98%).

Note: The preset now uses train_end_year=2016 (instead of 2014 from the original JSON)
to ensure the training split has enough data (156 months) for the window size.

Quick start:
    # Training from scratch
    python ./scripts/train_test.py --device cuda:0 --train --epochs 200
    
    # Evaluation only (uses existing checkpoint)
    python ./scripts/train_test.py --device cuda:0 --eval_only
    
    # Override specific parameters
    python ./scripts/train_test.py --device cuda:0 --lr 0.001 --dropout 0.1 --train --epochs 100
    
    # Load from metrics JSON (reproduce exact run)
    python ./scripts/train_test.py --metrics_json outputs_v4_3ep/20260201-054603/metrics_validation.json --train

All CLI arguments override the v4_3ep preset defaults, so you can customize any parameter.
"""
import argparse
from typing import Tuple, Any, cast, Optional
import json
import math
import time
import torch
import os
import torch.nn as nn
import sys as _sys
import numpy as np

# Ensure project root (parent of scripts/) is on sys.path so 'src' is importable
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from src.net import gtnet

__version__ = "0.2.0"
import random
# Import utilities from src.util
from src.util import (
    DataLoaderS, StandardScaler, DataLoaderM,
    resolve_split_and_build_data, prepare_graph_and_subgraph,
    ym_to_int, months_between, ensure_btn, flatten_weights, to_float,
    unwrap_model_output, norm_mode_name, maybe_inv_scale, compute_metrics,
    robust_range, exp_smooth_2d, last_level_baseline_expand,
    AnalysisLogger, start_runlog, jlog, MCDropoutContext, to_model_layout,
    set_random_seed, fit_linear_calibration,
    masked_mae, masked_mse, masked_rmse, masked_mape, metric,
    sym_adj, asym_adj, load_adj, load_graph, normal_std
)
# Import loss from src.losses
from src.losses import weighted_huber_horizon_loss

from src.trainer import Optim, run_trainer_path
import sys
from random import randrange
import matplotlib
import numpy as _np
matplotlib.use('Agg')
import contextlib
import matplotlib.pyplot as plt
plt.rcParams['savefig.dpi'] = 800
import gc
import csv
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from pathlib import Path

try:
    from src.ensembles.ensemble_hooks import run_extra_models_and_blend
    from src.ensembles.ensemble_adapters import build_extra_models  # (optional)
except Exception:
    run_extra_models_and_blend = None
    build_extra_models = None

# Global logger instance, configured in main()
ANALYSIS_LOGGER = None
_CALIB_AB = None
_CONF_Z = 1.96  # global 95% z, may be recalibrated on validation if --conf_calibrate
_CONFORMAL_Q = None  # per-node conformal offset learned on Validation (original units)
# ---
_norm_mode_name = norm_mode_name

# last seen Validation CI ratio (median over nodes) to drive auto-dropout
_LAST_VAL_CI_RATIO = None
# ---

def plot_predicted_actual(pred, true, title, kind, ci=None, base_year=None, steps_per_year=None):
    """Generate and save a publication-quality plot of predicted vs. actual values."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        pred_np = pred.detach().cpu().numpy() if isinstance(pred, torch.Tensor) else np.asarray(pred)
        true_np = true.detach().cpu().numpy() if isinstance(true, torch.Tensor) else np.asarray(true)
        ci_np = None
        if ci is not None:
            try:
                ci_np = ci.detach().cpu().numpy() if isinstance(ci, torch.Tensor) else np.asarray(ci)
            except Exception:
                ci_np = None

        # Optional: display-only exponential smoothing
        try:
            if getattr(args, 'smooth_plot', False):
                alpha = float(getattr(args, 'smooth_alpha', 0.1))
                pred_np = exp_smooth_2d(pred_np, alpha=alpha)
                true_np = exp_smooth_2d(true_np, alpha=alpha)
                if ci_np is not None:
                    ci_np = exp_smooth_2d(ci_np, alpha=alpha)
        except Exception:
            pass

        out_dir = os.path.join('model', kind)
        os.makedirs(out_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 5))

        x = np.arange(len(pred_np))
        ax.plot(x, true_np, label='Actual', color='#003f5c', linewidth=2.5)
        ax.plot(x, pred_np, label='Predicted', color='#ffa600', linestyle='--', linewidth=2.2)

        if ci_np is not None:
            ax.fill_between(x, pred_np - ci_np,
                            pred_np + ci_np,
                            color='#ffa600', alpha=0.2, label='95% Confidence Interval')

        if base_year is not None and steps_per_year is not None and steps_per_year > 0:
            # Yearly ticks at January plus a final tick at the last sample labeled as Dec of the final year
            jan_ticks = list(range(0, len(x), steps_per_year))
            ticks = jan_ticks
            if (len(x) - 1) not in ticks:
                ticks = sorted(set(ticks + [len(x) - 1]))
            tick_labels = []
            for i in ticks:
                if i == len(x) - 1:
                    # Force last label to Dec of the final year
                    years_offset = i // steps_per_year
                    year = base_year + years_offset
                    tick_labels.append(f"Dec-{year}")
                else:
                    years_offset = i // steps_per_year
                    year = base_year + years_offset
                    tick_labels.append(f"Jan-{year}")
            ax.set_xticks(ticks)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=12)
            ax.set_xlim(0, len(x) - 1)

        ax.set_xlabel("Date", fontsize=14, fontweight='bold')
        ax.set_ylabel("Trend", fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=12, frameon=True, shadow=True)

        fig.tight_layout()

        title_fs = title.replace('/', '_').replace(' ', '_')
        fig.savefig(os.path.join(out_dir, f"{title_fs}_{kind}.png"), dpi=300)
        fig.savefig(os.path.join(out_dir, f"{title_fs}_{kind}.pdf"), format='pdf')

        plt.close(fig)
    except Exception as e:
        print(f'[plot_predicted_actual] error: {e}')

def dump_ckpt_vs_model_csv(ckpt_path: str, model: torch.nn.Module, out_csv: str):
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
    except Exception as e:
        print('[dump_ckpt_vs_model_csv] load fail:', e)
        return
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    elif isinstance(ckpt, dict):
        sd = {k:v for k,v in ckpt.items() if isinstance(v, torch.Tensor)}
    elif isinstance(ckpt, nn.Module):
        sd = ckpt.state_dict()
    else:
        print('[dump_ckpt_vs_model_csv] unsupported ckpt format')
        return
    model_sd = model.state_dict()
    rows=[]
    ck_keys=set(sd.keys()); model_keys=set(model_sd.keys())
    for k in sorted(ck_keys | model_keys):
        if k in ck_keys and k in model_keys:
            status='match' if tuple(sd[k].shape)==tuple(model_sd[k].shape) else 'shape_mismatch'
        elif k in ck_keys:
            status='unexpected'
        else:
            status='missing'
        ck_shape = tuple(sd[k].shape) if k in ck_keys else ''
        model_shape = tuple(model_sd[k].shape) if k in model_keys else ''
        rows.append((k, ck_shape, model_shape, status))
    try:
        with open(out_csv,'w',newline='') as f:
            w=csv.writer(f); w.writerow(['name','ckpt_shape','model_shape','status'])
            for r in rows: w.writerow(r)
        print(f'[dump_ckpt_vs_model_csv] wrote {len(rows)} rows to {out_csv}')
    except Exception as e:
        print('[dump_ckpt_vs_model_csv] write fail:', e)

# consistent_name was only used to coerce to str(); inline with str() where needed

# Removed duplicate _MCDropoutContext and _to_model_layout - now imported from src.util

def save_metrics_1d(predict, test, title, kind):
    if isinstance(predict, torch.Tensor):
        pred = predict.detach().float().view(-1).cpu()
    else:
        pred = torch.as_tensor(predict, dtype=torch.float32).view(-1)
    if isinstance(test, torch.Tensor):
        tgt = test.detach().float().view(-1).cpu()
    else:
        tgt = torch.as_tensor(test, dtype=torch.float32).view(-1)
    eps = torch.finfo(torch.float32).eps
    sse = torch.sum((tgt - pred)**2)
    rr_den_sse = torch.sum((tgt - tgt.mean())**2)
    rrse = float(torch.sqrt(torch.clamp(sse, min=0.0)) / (torch.sqrt(torch.clamp(rr_den_sse, min=eps))))
    rae = float(torch.sum(torch.abs(tgt - pred)) / (torch.sum(torch.abs(tgt - tgt.mean())) + eps))
    out_dir = os.path.join('model','Bayesian', kind); os.makedirs(out_dir, exist_ok=True)
    fn = os.path.join(out_dir, f"{str(title).replace('/','_')}_{kind}.csv")
    with open(fn,'w') as f:
        f.write(f'rse:{rrse}\n'); f.write(f'rae:{rae}\n')


# --- New: compact metrics computation and persistence ---
# Use compute_metrics from util module, but add extra sMAPE, RSE, RAE for backward compatibility
def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # Get base metrics from util
    base = compute_metrics(y_true, y_pred)
    # Add extra metrics for full compatibility
    t = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred, dtype=float).ravel()
    eps = 1e-9
    smape = float(np.mean(2.0 * np.abs(p - t) / np.maximum(eps, (np.abs(t) + np.abs(p)))) * 100.0)
    # RAE = sum |t-p| / sum |t-mean(t)|
    diff = t - p
    centered = t - t.mean()
    denom_abs = np.sum(np.abs(centered))
    rae = float(np.sum(np.abs(diff)) / (denom_abs + eps)) if denom_abs > 0 else float('nan')
    # Combine: capitalize keys for consistency
    return {
        "MAE": base['mae'], 
        "RMSE": base['rmse'], 
        "MAPE": base['mape'], 
        "sMAPE": smape, 
        "RSE": base['rrse'],  # Using RRSE as RSE
        "RAE": rae
    }



def _save_metrics(args, split_name: str, metrics: dict, extras: dict | None = None):
    run_id = os.environ.get('RUN_TAG') or (args.run_tag if getattr(args, 'run_tag', '') else time.strftime('%Y%m%d-%H%M%S'))
    out_root = Path(getattr(args, 'out_dir', 'runs')) / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    payload = {"split": split_name, "metrics": metrics, "args": vars(args)}
    if extras:
        payload["extras"] = extras
    with open(out_root / f"metrics_{split_name.lower()}.json", 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def _maybe_set_run_scoped_save_path(args) -> None:
    """If the user didn't explicitly set --save, place checkpoints under out_dir/<run_id>/.

    This prevents Optuna trials from overwriting a shared checkpoint file.
    """
    try:
        if _flag_was_set('save'):
            return
        run_id = os.environ.get('RUN_TAG') or getattr(args, 'run_tag', '')
        if not run_id:
            return
        out_dir = getattr(args, 'out_dir', None)
        if not out_dir:
            return
        run_dir = Path(out_dir) / str(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        args.save = str(run_dir / 'model.pt')
    except Exception:
        return


# Removed _robust_range, _exp_smooth_2d, _last_level_baseline_expand - now imported from src.util

# For testing the model on unseen data, a sliding window can be used when the output period of the model is smaller than the target period to be forecasted.
#The sliding window uses the output from previous step as input of the next step.
#In our case, the window will be slided if the total forecast period (e.g., 72 months) is longer than the model's output length (e.g., 36 months).
def evaluate_sliding_window(data, test_window, model, evaluateL2, evaluateL1, n_input, is_plot, mc_runs=None):
    """
    Sliding-window testing with MC Dropout and correct 95% CI in original units.
    """
    eps = 1e-12
    global _CONF_Z
    global _LAST_VAL_CI_RATIO
    z = float(_CONF_Z)

    model.eval()  # keep BN stable; we will enable dropout stochastically below

    # Fallback: if the available test_window length is too short to perform even a
    # single slide (i.e., we cannot start at n_input), return standard test metrics
    # using the prepared test split. This avoids empty preds_z lists.
    try:
        total_len = test_window.shape[0]
    except Exception:
        total_len = len(test_window)
    if total_len < n_input + data.out_len:
        # Not enough horizon after the input window; reuse the batched test evaluation.
        batch_sz = int(getattr(args, 'batch_size', 64)) if 'args' in globals() else 64
        return evaluate(data, data.test[0], data.test[1], model, evaluateL2, evaluateL1, batch_sz, False, mc_runs)

    # start input (z-space)
    # test_window is (T, N, C) or (T, N) depending on DataLoaderS;
    # create the initial x_input on the same device as the model to avoid
    # CPU/CUDA device mismatches when rolling the window during prediction.
    try:
        model_dev = next(model.parameters()).device
    except StopIteration:
        # model has no parameters (edge-case); fall back to CPU
        model_dev = torch.device('cpu')
    x_input = torch.as_tensor(test_window[0:n_input, ...], dtype=torch.float32, device=model_dev)

    # Store ORIGINAL-unit per-slide tensors to maintain correct per-window inversions
    preds_o_list, trues_o_list, conf_o_list = [], [], []

    if mc_runs is None:
        mc_runs = int(getattr(args, "mc_runs", 30))
    with torch.no_grad():
        for i in range(n_input, test_window.shape[0], data.out_len):
            # Prepare model input: X shape expected (B,C,N,P)
            X_raw = x_input.unsqueeze(0)  # (1,T,N,C) or (1,T,N)
            if X_raw.dim() == 4:
                pass
            elif X_raw.dim() == 3:
                X_raw = X_raw.unsqueeze(-1)
            try:
                exp_len = int(getattr(model, 'seq_length', X_raw.shape[1]))
                X = to_model_layout(X_raw, exp_len, debug=getattr(args,'debug_layout', False)).to(device)
            except Exception as _e:
                print(f"[LayoutEvalSlide] layout error: {_e}")
                raise
            # target slice (z-space) from channel 0
            y_true_z = torch.as_tensor(test_window[i: i + data.out_len, :, 0] if test_window.ndim==3 else test_window[i: i + data.out_len, :], dtype=torch.float32, device=device)

            # --- MC-Dropout sampling in eval mode ---
            with MCDropoutContext(model):
                amp_on = bool(getattr(args, 'amp', False)) and device.type == 'cuda'
                if args.vectorized_mc:
                    B, C, N, T = X.shape
                    est_elems = mc_runs * B * C * N * T
                    max_elems = int(getattr(args, 'mc_vec_max_elems', 120_000_000))
                    use_vec = est_elems <= max_elems
                    outs = None
                    if use_vec:
                        try:
                            Xrep = X.repeat(mc_runs, 1, 1, 1)
                            with autocast(device_type='cuda', enabled=amp_on):
                                O = model(Xrep)
                            O = unwrap_model_output(O)
                            if O.dim() == 4:
                                O = O.squeeze(3)
                            outs = O[:, -1, ...]
                        except RuntimeError as e:
                            if 'out of memory' in str(e).lower():
                                torch.cuda.empty_cache()
                                print(f"[MC Fallback-SW] OOM vectorized MC (est_elems={est_elems:,}); using looped mode.")
                                use_vec = False
                            else:
                                raise
                    if not use_vec or outs is None:
                        out_list = []
                        for _ in range(mc_runs):
                            with autocast(device_type='cuda', enabled=amp_on):
                                o = model(X)
                            o = unwrap_model_output(o)
                            if o.dim() == 4:
                                o = o.squeeze(3)
                            out_list.append(o[-1])
                        outs = torch.stack(out_list, dim=0)
                else:
                    out_list = []
                    for _ in range(mc_runs):
                        with autocast(device_type='cuda', enabled=amp_on):
                            o = model(X)
                        o = unwrap_model_output(o)
                        if o.dim() == 4:
                            o = o.squeeze(3)
                        out_list.append(o[-1])
                    outs = torch.stack(out_list, dim=0)  # (mc_runs, ...)

            # --- Log detailed diagnostics for this slide ---
            if ANALYSIS_LOGGER:
                try:
                    # Log distribution of predictions and inputs for this window
                    log_record = {
                        "type": "sliding_window_diagnostics",
                        "slide_start_index": i,
                        "input_mean": float(x_input.mean()),
                        "input_std": float(x_input.std()),
                        "true_target_mean_z": float(y_true_z.mean()),
                        "true_target_std_z": float(y_true_z.std()),
                        "pred_mean_z": float(outs.mean()),
                        "pred_std_z": float(outs.std()),
                    }
                    ANALYSIS_LOGGER.log(log_record)
                except Exception as e:
                    print(f"[AnalysisLogger Warning] Failed during sliding window logging: {e}", file=sys.stderr)
            # Persist raw MC samples for selected slides if requested
            try:
                if getattr(args, 'persist_mc', ''):
                    # parse requested slides list (empty means persist all)
                    slides_arg = str(getattr(args, 'persist_mc_slides', '')).strip()
                    persist_all = (slides_arg == '')
                    persist_set = set()
                    if not persist_all:
                        for token in slides_arg.split(','):
                            try:
                                persist_set.add(int(token.strip()))
                            except Exception:
                                pass
                    if persist_all or i in persist_set:
                        try:
                            os.makedirs(os.path.dirname(args.persist_mc) or '.', exist_ok=True)
                            # filename: <prefix>_slide_<i>.npz
                            fn = f"{args.persist_mc}_slide_{i}.npz"
                            # outs: (mc_runs, L_out, N) -> save as numpy
                            np.savez_compressed(fn, mc_samples=outs.detach().cpu().numpy())
                            if ANALYSIS_LOGGER:
                                ANALYSIS_LOGGER.log({"type": "persist_mc_saved", "slide_start_index": i, "path": fn})
                        except Exception as _e:
                            print(f"[persist_mc] failed to write MC samples for slide {i}: {_e}", file=sys.stderr)
            except Exception:
                pass
            # ---

            if getattr(args, 'nan_debug', False):
                if torch.isnan(outs).any() or torch.isinf(outs).any():
                    bad = torch.isnan(outs) | torch.isinf(outs)
                    idx_flat = bad.nonzero(as_tuple=False)[0].tolist() if bad.any() else []
                    print(f"[NaNDebug][SW] Detected NaN/Inf in MC samples at slide {i}, first index={idx_flat}")
            mean_z = outs.mean(dim=0)                         # [L_out, N] mean in z-space
            std_z  = outs.std(dim=0) + 1e-8                   # [L_out, N]
            sem_z  = std_z / math.sqrt(float(mc_runs))        # standard error
            half_z = z * sem_z                                # 95% half-width in z-space

            # If the final window is partial (y_true shorter than model out_len), truncate predictions & stats
            out_eff = y_true_z.shape[0]
            if mean_z.shape[0] > out_eff:
                mean_z = mean_z[:out_eff]
                std_z  = std_z[:out_eff]
                sem_z  = sem_z[:out_eff]
                half_z = half_z[:out_eff]

            # Rolling-aware inversion: use per-window stats from the originating TRUE window
            # Window index within the 'test' split that starts this forecast
            w_idx = i - n_input
            if getattr(data, 'rolling', False) and hasattr(data, 'per_window_mu') and 'test' in data.per_window_mu:
                # guard bounds
                if isinstance(w_idx, torch.Tensor):
                    w_idx = int(w_idx.item())
                if w_idx < 0 or w_idx >= len(data.per_window_mu['test']):
                    # Fallback to global inversion if out of range
                    mean_o = data.inv_transform_like(mean_z)
                    true_o = data.inv_transform_like(y_true_z)
                    if getattr(data, 'use_log1p', False):
                        lower_o = data.inv_transform_like(mean_z - half_z)
                        upper_o = data.inv_transform_like(mean_z + half_z)
                        conf_o = 0.5 * (upper_o - lower_o)
                    else:
                        conf_o = half_z * data.std_expand_like(half_z)
                else:
                    mu = data.per_window_mu['test'][w_idx]   # (N,)
                    std = data.per_window_std['test'][w_idx] # (N,)
                    # Expand to (L,N)
                    mu_t = torch.as_tensor(mu, dtype=mean_z.dtype, device=mean_z.device).unsqueeze(0).expand_as(mean_z)
                    std_t = torch.as_tensor(std, dtype=mean_z.dtype, device=mean_z.device).unsqueeze(0).expand_as(mean_z)
                    mean_o = mean_z * std_t + mu_t
                    true_o = y_true_z * std_t + mu_t
                    if getattr(data, 'use_log1p', False):
                        mean_o = torch.expm1(mean_o)
                        true_o = torch.expm1(true_o)
                        lower_o = torch.expm1((mean_z - half_z) * std_t + mu_t)
                        upper_o = torch.expm1((mean_z + half_z) * std_t + mu_t)
                        conf_o = 0.5 * (upper_o - lower_o)
                    else:
                        conf_o = half_z * std_t
            else:
                # Global stats path
                if getattr(data,'use_log1p', False):
                    lower_o = data.inv_transform_like(mean_z - half_z)
                    upper_o = data.inv_transform_like(mean_z + half_z)
                    mean_o = data.inv_transform_like(mean_z)
                    true_o = data.inv_transform_like(y_true_z)
                    conf_o = 0.5*(upper_o - lower_o)
                else:
                    mean_o = data.inv_transform_like(mean_z)
                    conf_o = half_z * data.std_expand_like(half_z)
                    true_o = data.inv_transform_like(y_true_z)
                    conf_o = half_z * data.std_expand_like(half_z)

            # Residual-head recomposition in ORIGINAL units (baseline = last input level)
            if getattr(args, 'residual_head', False):
                try:
                    baseline_z = last_level_baseline_expand(X, mean_z.shape[0])
                    if getattr(data, 'rolling', False) and 'test' in getattr(data, 'per_window_mu', {}) and 0 <= w_idx < len(data.per_window_mu['test']):
                        mu = data.per_window_mu['test'][w_idx]
                        std = data.per_window_std['test'][w_idx]
                        mu_t = torch.as_tensor(mu, dtype=baseline_z.dtype, device=baseline_z.device).unsqueeze(0).expand_as(baseline_z)
                        std_t = torch.as_tensor(std, dtype=baseline_z.dtype, device=baseline_z.device).unsqueeze(0).expand_as(baseline_z)
                        base_o = baseline_z * std_t + mu_t
                        if getattr(data, 'use_log1p', False):
                            base_o = torch.expm1(base_o)
                    else:
                        base_o = data.inv_transform_like(baseline_z)
                    base_o = base_o.squeeze(0).type_as(mean_o)
                    mean_o = mean_o + base_o
                except Exception:
                    pass

            # roll window with z-space mean for channel 0 and recompute channel 1 (movement) if present
            # mean_z: [L_out, N]
            mean_z = mean_z.detach()
            if x_input.ndim == 2:
                # single-channel input (T,N)
                if data.P <= data.out_len:
                    take = min(data.P, mean_z.shape[0])
                    x_input = mean_z[-take:].clone()
                else:
                    keep = data.P - mean_z.shape[0]
                    if keep > 0:
                        # Ensure the CPU-resident slice is moved to the same device as mean_z
                        tgt_dev = mean_z.device
                        x_input = torch.cat([x_input[-keep:, :].clone().to(tgt_dev), mean_z.clone()], dim=0)
                    else:
                        x_input = mean_z.clone()
            else:
                # multi-channel input (T,N,C) where channel 0 is level and channel 1 is movement
                L_out = mean_z.shape[0]
                level_seq = mean_z
                # build new tail of shape (L_out, N, C)
                if data.in_dim > 1:
                    # movement = first lag diff then subsequent diffs of predicted level
                    prev_last = x_input[-1, :, 0].to(level_seq.device)
                    # first movement step: mean_z[0] - prev_last -> make shape (1, N, 1)
                    m0 = (level_seq[0] - prev_last).unsqueeze(0).unsqueeze(-1)
                    # subsequent movements: diffs of mean_z -> shape (L_out-1, N, 1)
                    md = torch.diff(level_seq, dim=0).unsqueeze(-1) if L_out > 1 else torch.zeros((0, level_seq.shape[1], 1), device=level_seq.device)
                    mov_tail = torch.cat([m0, md], dim=0)
                    # stack level and movement
                    new_tail = torch.cat([level_seq.unsqueeze(-1), mov_tail], dim=-1)
                else:
                    new_tail = level_seq.unsqueeze(-1)
                # now roll x_input
                if data.P <= L_out:
                    take = min(data.P, L_out)
                    x_input = new_tail[-take:].clone()
                else:
                    keep = data.P - L_out
                    if keep > 0:
                        # Move CPU slice to device of new_tail before concatenation
                        tgt_dev = new_tail.device
                        x_input = torch.cat([x_input[-keep:, :].clone().to(tgt_dev), new_tail.clone()], dim=0)
                    else:
                        x_input = new_tail.clone()

            preds_o_list.append(mean_o.detach().cpu())
            trues_o_list.append(true_o.detach().cpu())
            conf_o_list.append(conf_o.detach().cpu())

    # concatenate ORIGINAL-units per-slide outputs
    pred_o = torch.cat(preds_o_list, dim=0)
    true_o = torch.cat(trues_o_list, dim=0)
    conf_o = torch.cat(conf_o_list, dim=0)
    try:
        if getattr(args, 'calibration', 'none') in ('test', 'both') and isinstance(_CALIB_AB, tuple) and len(_CALIB_AB) == 2:
            a_t = _CALIB_AB[0]; b_t = _CALIB_AB[1]
            a = a_t.to(pred_o.device); b = b_t.to(pred_o.device)
            pred_o = a.unsqueeze(0) * pred_o + b.unsqueeze(0)
    except Exception as _e:
        jlog("warn_calibration_linear_test", error=str(_e)[:160])
    # Unified metrics computation (use same function as evaluation saving to avoid discrepancies)
    try:
        p = pred_o.numpy()
        y = true_o.numpy()
        _m = _compute_metrics(y, p)
        rrse = float(_m.get('RSE', float('nan')))
        rae = float(_m.get('RAE', float('nan')))
        sm = float(_m.get('sMAPE', float('nan')))
        # correlation: compute per-node correlation and average where defined
        sp = p.std(axis=0)
        sg = y.std(axis=0)
        mp = p.mean(axis=0)
        mg = y.mean(axis=0)
        denom = (sp * sg) + 1e-12
        corr_vec = ((p - mp) * (y - mg)).mean(axis=0) / denom
        valid = (sg > 0) & (sp > 0)
        correlation = float(np.mean(corr_vec[valid])) if valid.any() else float('nan')
    except Exception:
        rrse = float('nan'); rae = float('nan'); correlation = float('nan'); sm = float('nan')
    if is_plot:
        po = pred_o.cpu().numpy(); yo = true_o.cpu().numpy(); co = conf_o.cpu().numpy()
        base_year_test = (
            args.valid_end_year + 1
            if getattr(args, 'chronological_split', False)
            else args.start_year
        )
        n_cols = getattr(data, 'm', pred_o.shape[1])
        for col in range(n_cols):
            node_name = str(DataLoaderS.col[col])
            save_metrics_1d(torch.from_numpy(po[:,col]), torch.from_numpy(yo[:,col]), node_name, 'Testing')
            plot_predicted_actual(po[:,col], yo[:,col], node_name, 'Testing', ci=co[:,col], base_year=base_year_test, steps_per_year=args.steps_per_year)
    return float(rrse if rrse is not None else 0.0), float(rae if rae is not None else 0.0), correlation, float(sm if sm is not None else 0.0)

def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size, is_plot, mc_runs=None, kind='Validation'):
    model.eval()
    global _CALIB_AB
    eps = 1e-12
    global _CONF_Z
    global _LAST_VAL_CI_RATIO
    z = float(_CONF_Z)
    predict=[]; target=[]; conf95=[]
    pred_full = torch.empty(0)
    true_full = torch.empty(0)
    w = None
    labels = []
    # accumulator for CI calibration on Validation split
    r_collect = None
    # Early out if no windows available (prevents downstream NaNs)*
    try:
        if X is None or (hasattr(X, 'shape') and getattr(X, 'shape')[0] == 0):
            print('[evaluate] No windows provided; skipping metrics for', kind)
            return float('nan'), float('nan'), float('nan'), float('nan')
    except Exception:
        pass
    if mc_runs is None:
        mc_runs = int(getattr(args, 'mc_runs', 50))
    with torch.no_grad():
        # accumulate normalized residuals for CI calibration (r = |err|/sem)
        r_collect = [] if (kind=='Validation' and getattr(args,'conf_calibrate',False)) else None
        for b_idx, batch in enumerate(data.get_batches(X, Y, batch_size, False, return_indices=True)):
            # batch: (Xb, Yb, idxs)
            if len(batch) == 3:
                Xb_raw, Yb_raw, idxs = batch
            else:
                Xb_raw, Yb_raw = batch
                idxs = None
            Xb_tensor = torch.as_tensor(Xb_raw)
            if Xb_tensor.dim() == 3:
                Xb_tensor = Xb_tensor.unsqueeze(-1)
            try:
                exp_len = int(getattr(model, 'seq_length', Xb_tensor.shape[1]))
                Xb = to_model_layout(Xb_tensor, exp_len, debug=getattr(args,'debug_layout', False)).to(device, dtype=torch.float)
            except Exception as _e:
                print(f"[LayoutEval] layout error: {_e}")
                raise
            Yb = Yb_raw.to(device)
            with MCDropoutContext(model):
                amp_on = bool(getattr(args, 'amp', False)) and device.type == 'cuda'
                if args.vectorized_mc:
                    B, C, N, T = Xb.shape
                    est_elems = mc_runs * B * C * N * T
                    max_elems = int(getattr(args, 'mc_vec_max_elems', 120_000_000))
                    use_vec = est_elems <= max_elems
                    outs = None
                    if use_vec:
                        try:
                            Xrep = Xb.repeat(mc_runs, 1, 1, 1)  # (mc_runs*B, C, N, T)
                            with autocast(device_type='cuda', enabled=amp_on):
                                O = model(Xrep)
                            O = unwrap_model_output(O)
                            if O.dim() == 4:
                                O = O[:, :, :, -1]
                            outs = O.view(mc_runs, B, *O.shape[1:])
                        except RuntimeError as e:
                            if 'out of memory' in str(e).lower():
                                torch.cuda.empty_cache()
                                use_vec = False
                                print(f"[MC Fallback] OOM during vectorized MC (est_elems={est_elems:,}); falling back to looped mode.")
                            else:
                                raise
                    if not use_vec or outs is None:
                        out_list = []
                        for _ in range(mc_runs):
                            with autocast(device_type='cuda', enabled=amp_on):
                                o = model(Xb)
                            o = unwrap_model_output(o)
                            if o.dim() == 4:
                                o = o[:, :, :, -1]
                            out_list.append(o)
                        outs = torch.stack(out_list, dim=0)
                else:
                    out_list = []
                    for _ in range(mc_runs):
                        with autocast(device_type='cuda', enabled=amp_on):
                            o = model(Xb)
                        o = unwrap_model_output(o)
                        if o.dim() == 4:
                            o = o[:, :, :, -1]
                        out_list.append(o)
                    outs = torch.stack(out_list, dim=0)  # (mc_runs, B, ...)
            if getattr(args, 'nan_debug', False):
                if torch.isnan(outs).any() or torch.isinf(outs).any():
                    bad = (torch.isnan(outs) | torch.isinf(outs))
                    loc = bad.nonzero(as_tuple=False)[0].tolist() if bad.any() else []
                    print(f"[NaNDebug][{kind}] NaN/Inf detected in MC outs batch={b_idx} first_index={loc}")
            mean_z = outs.mean(dim=0)
            std_z = outs.std(dim=0) + 1e-8
            sem_z = std_z / math.sqrt(float(mc_runs))
            half_z = z * sem_z
            # collect r for coverage calibration in z-space (before any transforms)
            if r_collect is not None:
                with torch.no_grad():
                    # outs.mean over MC already computed as mean_z; compare to Yb (z-space)
                    r = torch.abs(Yb - mean_z) / torch.clamp(sem_z, min=1e-8)
                    r_collect.append(r.detach().flatten().cpu())

            if ANALYSIS_LOGGER and kind == 'Validation':
                try:
                    norm_mode = 'rolling' if getattr(data, 'rolling', False) else 'global'
                    # best-effort original-space inversion for logging
                    try:
                        if getattr(data, 'rolling', False) and hasattr(data, 'per_window_mu') and idxs is not None:
                            # use batch indices to fetch per-window stats for this minibatch
                            split = 'valid'
                            mu = data.per_window_mu[split][idxs]
                            std = data.per_window_std[split][idxs]
                            pred_o = data.inv_transform_with_stats(mean_z.cpu(), mu, std)
                            true_o = data.inv_transform_with_stats(Yb.cpu(), mu, std)
                        else:
                            pred_o = data.inv_transform_like(mean_z.cpu())
                            true_o = data.inv_transform_like(Yb.cpu())
                        pred_orig_mean = float(pred_o.mean())
                        true_orig_mean = float(true_o.mean())
                    except Exception:
                        pred_orig_mean = None
                        true_orig_mean = None

                    ANALYSIS_LOGGER.log({
                        "type": "validation_batch_diagnostics",
                        "batch_index": b_idx,
                        "input_shape": list(Xb.shape),
                        "normalize_mode": norm_mode,
                        "pred_mean_z": float(mean_z.mean()),
                        "pred_std_z": float(mean_z.std()),
                        "true_mean_z": float(Yb.mean()),
                        "true_std_z": float(Yb.std()),
                        "pred_mean_original": pred_orig_mean,
                        "true_mean_original": true_orig_mean
                    })
                except Exception as e:
                    print(f"[AnalysisLogger Warning] Failed during validation batch logging: {e}", file=sys.stderr)

            if getattr(data, 'use_log1p', False):
                if getattr(data, 'rolling', False) and idxs is not None:
                    lower_o = data.inv_transform_with_stats(mean_z - half_z, data.per_window_mu['valid'][idxs], data.per_window_std['valid'][idxs])
                    upper_o = data.inv_transform_with_stats(mean_z + half_z, data.per_window_mu['valid'][idxs], data.per_window_std['valid'][idxs])
                    mean_o = data.inv_transform_with_stats(mean_z, data.per_window_mu['valid'][idxs], data.per_window_std['valid'][idxs])
                else:
                    lower_o = data.inv_transform_like(mean_z - half_z)
                    upper_o = data.inv_transform_like(mean_z + half_z)
                    mean_o = data.inv_transform_like(mean_z)
                conf_o = 0.5 * (upper_o - lower_o)
            else:
                if getattr(data, 'rolling', False) and idxs is not None:
                    mean_o = data.inv_transform_with_stats(mean_z, data.per_window_mu['valid'][idxs], data.per_window_std['valid'][idxs])
                    conf_o = half_z * data.std_expand_like(half_z, idx=idxs, split='valid')
                else:
                    mean_o = data.inv_transform_like(mean_z)
                    conf_o = half_z * data.std_expand_like(half_z)
            y_true_o = data.inv_transform_like(Yb, idx=idxs, split='valid') if (getattr(data,'rolling',False) and idxs is not None) else data.inv_transform_like(Yb)

            # Residual head: if enabled, baseline is last input level (channel 0) in original units.
            # IMPORTANT: baseline should be added back to predictions only. The target is already
            # the absolute future level in original units, so adding baseline to target would
            # corrupt metrics (and can explode RSE/RAE).
            if getattr(args, 'residual_head', False):
                try:
                    baseline_z = last_level_baseline_expand(Xb_raw, Yb.shape[1]).detach()
                    if getattr(data, 'rolling', False) and idxs is not None:
                        mu_valid = getattr(data, 'per_window_mu', {}).get('valid')
                        std_valid = getattr(data, 'per_window_std', {}).get('valid')
                        if mu_valid is not None and std_valid is not None:
                            baseline_o = data.inv_transform_with_stats(baseline_z, mu_valid[idxs], std_valid[idxs])
                        else:
                            baseline_o = maybe_inv_scale(baseline_z, data.scaler)
                    else:
                        baseline_o = maybe_inv_scale(baseline_z, data.scaler)
                    mean_o = mean_o + baseline_o
                except Exception:
                    pass

            predict.append(mean_o.cpu()); target.append(y_true_o.cpu()); conf95.append(conf_o.cpu())
    predict = torch.cat(predict, dim=0)
    target = torch.cat(target, dim=0)
    conf95 = torch.cat(conf95, dim=0)
    # Split-conformal: learn q̂ on Validation (per-node (1-α) quantile of |err|) and add at Test
    global _CONFORMAL_Q
    try:
        if getattr(args, 'conformal', False):
            split_lower = kind.lower()
            if split_lower in ('validation','valid'):
                abs_res = torch.abs(target - predict)  # [B,L,N]
                flat = abs_res.reshape(-1, abs_res.shape[-1])
                alpha = float(max(1e-4, min(0.5, getattr(args,'conf_alpha',0.05))))
                qhat = torch.quantile(flat.to(torch.float32), 1.0 - alpha, dim=0)  # [N]
                _CONFORMAL_Q = qhat.detach().cpu()
            elif split_lower in ('testing','test') and _CONFORMAL_Q is not None:
                qhat = _CONFORMAL_Q.to(conf95.device, conf95.dtype).view(1,1,-1)
                conf95 = conf95 + qhat
    except Exception as _e:
        print(f"[conformal] warning: {_e}")
    if getattr(args,'robust_metrics', False):
        _EPS = 1e-12
        # Mask zero-variance nodes for denominator metrics
        flat_std = torch.std(target.reshape(-1, target.shape[-1]), dim=0)
        var_mask = (flat_std > 0)
        if not var_mask.any():
            rrse = float('nan'); rae = float('nan'); correlation = float('nan'); sm = float('nan')
        else:
            tgt_m = target[:,:,var_mask]
            pred_m = predict[:,:,var_mask]
            mean_all = tgt_m.mean(dim=(0,1), keepdim=True)
            diff_r = tgt_m - mean_all
            if getattr(args, 'weight_nodes_in_metrics', False):
                node_std = torch.std(tgt_m, dim=(0,1)) + 1e-6
                weights = (node_std.mean() / node_std).view(1,1,-1)
                sq_err = ((tgt_m - pred_m)**2) * weights
                abs_err = torch.abs(tgt_m - pred_m) * weights
                diff_r_w = (diff_r**2) * weights
                rrse = math.sqrt(float(torch.sum(sq_err))) / (math.sqrt(float(torch.sum(diff_r_w)))+eps)
                rae = float((torch.sum(abs_err) / (torch.sum(torch.abs(diff_r)*weights)+eps)).item())
            else:
                rrse = math.sqrt(float(torch.sum((tgt_m - pred_m)**2))) / (math.sqrt(float(torch.sum(diff_r**2)))+eps)
                rae = float((torch.sum(torch.abs(tgt_m - pred_m)) / (torch.sum(torch.abs(diff_r))+eps)).item())
            p = pred_m.numpy(); y = tgt_m.numpy()
            sp, sg = p.std(axis=0), y.std(axis=0)
            mp, mg = p.mean(axis=0), y.mean(axis=0)
            denom = (sp*sg)+_EPS
            corr_vec = ((p-mp)*(y-mg)).mean(axis=0)/denom
            valid = (sg>0) & (sp>0)
            correlation = float(corr_vec[valid].mean()) if valid.any() else float('nan')
            # Vectorized sMAPE
            sm_den = (np.abs(y)+np.abs(p)) + _EPS
            sm = float(np.mean(np.abs(y-p)/sm_den))
    else:
        mean_all = torch.mean(target, dim=(0,1))
        diff_r = target - mean_all.view(1,1,-1)
        if getattr(args, 'weight_nodes_in_metrics', False):
            node_std = torch.std(target, dim=(0,1)) + 1e-6
            weights = (node_std.mean() / node_std).view(1,1,-1)
            sq_err = ((target - predict)**2) * weights
            abs_err = torch.abs(target - predict) * weights
            diff_r_w = (diff_r**2) * weights
            rrse = math.sqrt(torch.sum(sq_err)) / (math.sqrt(torch.sum(diff_r_w))+eps)
            rae = float((torch.sum(abs_err) / (torch.sum(torch.abs(diff_r)*weights)+eps)).item())
        else:
            rrse = math.sqrt(torch.sum((target - predict)**2)) / (math.sqrt(torch.sum(diff_r**2))+eps)
            rae = float((torch.sum(torch.abs(target - predict)) / (torch.sum(torch.abs(diff_r))+eps)).item())
        p = predict.numpy(); y = target.numpy()
        sp, sg = p.std(axis=0), y.std(axis=0)
        mp, mg = p.mean(axis=0), y.mean(axis=0)
        denom = (sp*sg)+eps
        corr_vec = ((p-mp)*(y-mg)).mean(axis=0)/denom
        mask = (sg>0)
        correlation = float(corr_vec[mask].mean()) if mask.any() else 0.0
        sm=0.0
        B, L, N = predict.shape
        for b in range(B):
            for n in range(N):
                yt, yp = y[b,:,n], p[b,:,n]
                den = (np.abs(yt)+np.abs(yp)); den[den==0]=eps
                sm += float(np.mean(np.abs(yt-yp)/den))
        sm /= max(1,B*N)
    if is_plot:
        # Reconstruct a continuous series by overlap-averaging all windows.
        # Windows are contiguous with stride=1 across the segment, so the
        # full series length is n_samples + out_len - 1 and should span
        # Jan-2015 .. Dec-2021 when chronological split is used.
        B, L, N = predict.shape
        full_len = B + L - 1
        pred_full = torch.zeros(full_len, N)
        true_full = torch.zeros(full_len, N)
        ci_full   = torch.zeros(full_len, N)
        count     = torch.zeros(full_len, 1)

        # Initialize ensemble variables to None to avoid unbound errors
        w = None
        labels = None

        for s in range(B):
            sl = slice(s, s + L)
            pred_full[sl] += predict[s]
            true_full[sl] += target[s]
            ci_full[sl]   += conf95[s]
            count[sl]     += 1.0

        # Avoid divide-by-zero and average overlapping contributions
        count_safe = torch.clamp(count, min=1.0)
        pred_full = pred_full / count_safe
        true_full = true_full / count_safe
        ci_full   = ci_full / count_safe

        # Determine base year for x-axis labeling
        if getattr(args, 'chronological_split', False):
            base_year = (args.train_end_year + 1) if kind == 'Validation' else (args.valid_end_year + 1)
        else:
            base_year = args.start_year

        # --- NEW: optional calendar slice for validation plots (post-stitch) ---
        if kind == 'Validation' and getattr(args, 'val_plot_start', '') and getattr(args, 'val_plot_end', ''):
            try:
                y0, m0 = ym_to_int(args.val_plot_start)
                y1, m1 = ym_to_int(args.val_plot_end)
                s = months_between(base_year, 1, y0, m0)
                e = months_between(base_year, 1, y1, m1) + 1
                s = max(0, min(int(s), pred_full.shape[0]-1))
                e = max(s+1, min(int(e), pred_full.shape[0]))
                pred_full = pred_full[s:e]
                true_full = true_full[s:e]
                ci_full   = ci_full[s:e]
            except Exception as _e:
                print(f"[Validation Slice] warning: {str(_e)[:160]}")

        # Debug: report expected vs. actual plotted months for Validation
        if kind == 'Validation':
            expected_val_len = (args.valid_end_year - (args.train_end_year + 1) + 1) * args.steps_per_year
            print(f"[PlotValidation] windows={B}, out_len={L}, full_len={full_len}, expected_len={expected_val_len}, base_year={base_year}")

        # --- Optional per-node linear calibration ---
        assert isinstance(pred_full, torch.Tensor) and isinstance(true_full, torch.Tensor)
        try:
            if kind == 'Validation':
                if args.calibration in ('val', 'both'):
                    lam = 1e-6
                    X1 = torch.stack([pred_full, torch.ones_like(pred_full)], dim=-1)   # (T,N,2)
                    XtX = torch.einsum('tnk,tnj->k j n', X1, X1) + lam*torch.eye(2, device=X1.device).unsqueeze(-1)
                    Xty = torch.einsum('tnk,tn->k n', X1, true_full)
                    ab  = torch.linalg.solve(XtX.permute(2,0,1), Xty.permute(1,0)).permute(1,0)  # (2,N)
                    a, b = ab[0], ab[1]
                    pred_full = a.unsqueeze(0)*pred_full + b.unsqueeze(0)
                    _CALIB_AB = (a.detach().cpu(), b.detach().cpu())
                    jlog("calibration_linear", a_med=float(torch.median(a).cpu()), b_med=float(torch.median(b).cpu()))
                else:
                    _CALIB_AB = None
            elif kind == 'Testing':
                if args.calibration in ('test', 'both') and isinstance(_CALIB_AB, tuple) and len(_CALIB_AB) == 2:
                    a_t = _CALIB_AB[0]; b_t = _CALIB_AB[1]
                    a = a_t.to(pred_full.device); b = b_t.to(pred_full.device)
                    pred_full = a.unsqueeze(0)*pred_full + b.unsqueeze(0)
        except Exception as _e:
            jlog("warn_calibration_linear", split=kind, error=str(_e)[:160])
        # ---------------------------------------

        # --- Persist per-split metrics in original units ---
        try:
            _metrics = _compute_metrics(true_full.numpy(), pred_full.numpy())
            _extras = {
                'calibration': getattr(args, 'calibration', 'none'),
                'series_len': int(pred_full.shape[0])
            }
            _save_metrics(args, kind, _metrics, _extras)
            # Print concise validation/testing metrics to console for quick inspection
            try:
                rse = _metrics.get('RSE', None)
                rae = _metrics.get('RAE', None)
                print(f"[Metrics] {kind}: RSE={rse:.6f} RAE={rae:.6f}")
            except Exception:
                pass
        except Exception as _e:
            jlog("warn_save_metrics", split=kind, error=str(_e)[:160])

        n_cols = getattr(data, 'm', predict.shape[-1])
        for col in range(n_cols):
            node_name = str(DataLoaderS.col[col])
            save_metrics_1d(pred_full[:, col], true_full[:, col], node_name, kind)
            plot_predicted_actual(
                pred_full[:, col],
                true_full[:, col],
                node_name,
                kind,
                ci=ci_full[:, col],
                base_year=base_year,
                steps_per_year=args.steps_per_year
            )
        # Also plot in normalized space if requested
        if getattr(args, 'plot_norm_space', False):
            try:
                mu = data.mu.detach().cpu(); std = torch.clamp(data.std.detach().cpu(), min=1e-6)
                def _to_z(arr):
                    view = [1] * arr.dim(); view[-1] = -1
                    mu_v = mu.view(*view).expand_as(arr)
                    std_v = std.view(*view).expand_as(arr)
                    return (arr - mu_v) / std_v
                pred_full_z = _to_z(pred_full)
                true_full_z = _to_z(true_full)
                for col in range(n_cols):
                    node_name = f"{DataLoaderS.col[col]}_Norm"
                    plot_predicted_actual(
                        pred_full_z[:, col],
                        true_full_z[:, col],
                        node_name,
                        f"{kind}_Norm",
                        ci=None,
                        base_year=base_year,
                        steps_per_year=args.steps_per_year
                    )
            except Exception as _e:
                print(f"[plot_norm_space] error: {_e}")
        # --- BEGIN ENSEMBLE HOOK ---
                # Only run ensemble hook if enabled and base series exists (robust version)
                if getattr(args, 'ensemble', False) and run_extra_models_and_blend is not None and pred_full is not None and true_full is not None:
                    try:
                        kind_flags = {
                            'patchtst': bool(getattr(args, 'ens_patchtst', False)),
                            'nhits':    bool(getattr(args, 'ens_nhits', False)),
                            'mlp':      bool(getattr(args, 'ens_mlp', False)),
                            'd_model':  int(getattr(args, 'ens_d_model', 128)),
                            'nhead':    int(getattr(args, 'ens_nhead', 8)),
                            'patch_len':int(getattr(args, 'ens_patch_len', 16)),
                            'stride':   int(getattr(args, 'ens_stride', 8)),
                            'depth':    int(getattr(args, 'ens_depth', 2)),
                            'hidden':   int(getattr(args, 'ens_hidden', 256)),
                            'blocks':   int(getattr(args, 'ens_blocks', 3)),
                            'dropout':  float(getattr(args, 'ens_dropout', 0.1)),
                        }
                        base_pred_full_o = pred_full.clone()
                        true_full_o      = true_full.clone()
                        blended_full_o, w, preds_each_full_o = run_extra_models_and_blend(
                            data=data,
                            X_valid=X, Y_valid=Y,
                            device=device,
                            base_pred_full_o=base_pred_full_o,
                            true_full_o=true_full_o,
                            kind_flags=kind_flags,
                            ens_epochs=int(getattr(args, 'ens_epochs', 0)),
                            batch_size=int(getattr(args, 'batch_size', 64)),
                            lr=float(getattr(args, 'ens_lr', 1e-3)),
                        )
                        pred_full = blended_full_o.to(pred_full.device, pred_full.dtype)
                        labels = ["B-MTGNN"]
                        if kind_flags['patchtst']: labels.append("PatchTST-mini")
                        if kind_flags['nhits']:    labels.append("N-HiTS-mini")
                        if kind_flags['mlp']:      labels.append("MLP")
                        try:
                            w_list = flatten_weights(w)
                            print("[Ensemble] weights:", {k: v for k, v in zip(labels, w_list)})
                        except Exception as _we:
                            print(f"[Ensemble] weight print warn: {_we}")
                    except Exception as _e:
                        print(f"[Ensemble] hook failed, using base model only: {_e}")
        # --- END ENSEMBLE HOOK ---

        # Seed/checkpoint ensemble (simple averaging or placeholder IVW)
        ck_list = [p.strip() for p in str(getattr(args,'ensemble_ckpts','')).split(',') if p.strip()]
        if ck_list:
            try:
                series = [pred_full]
                for ck_path in ck_list:
                    ck = torch.load(ck_path, map_location='cpu')
                    loaded_hp = ck.get('hparams', None) if isinstance(ck, dict) else None
                    cpu_dev = torch.device('cpu')
                    in_dim_use = getattr(data, 'in_dim', args.in_dim)
                    m_ck = gtnet(
                        args.gcn_true, args.buildA_true, loaded_hp[0] if loaded_hp else args.gcn_depth, int(data.m),
                        cpu_dev, data.adj, dropout=loaded_hp[7] if loaded_hp else args.dropout,
                        subgraph_size=loaded_hp[6] if loaded_hp else args.subgraph_size,
                        node_dim=loaded_hp[9] if loaded_hp else args.node_dim,
                        dilation_exponential=loaded_hp[8] if loaded_hp else args.dilation_exponential,
                        conv_channels=loaded_hp[2] if loaded_hp else args.conv_channels,
                        residual_channels=loaded_hp[3] if loaded_hp else args.residual_channels,
                        skip_channels=loaded_hp[4] if loaded_hp else args.skip_channels,
                        end_channels=loaded_hp[5] if loaded_hp else args.end_channels,
                        seq_length=args.seq_in_len, in_dim=in_dim_use, out_dim=args.seq_out_len,
                        layers=loaded_hp[12] if loaded_hp else args.layers, propalpha=loaded_hp[10] if loaded_hp else args.propalpha,
                        tanhalpha=loaded_hp[11] if loaded_hp else args.tanhalpha, layer_norm_affline=False,
                        temporal_attn=getattr(args,'temporal_attn', False), attn_dim=getattr(args,'attn_dim',64),
                        attn_heads=getattr(args,'attn_heads',2), attn_dropout=getattr(args,'attn_dropout',0.1),
                        attn_window=getattr(args,'attn_window',0), attn_math_mode=getattr(args,'attn_math_mode', False),
                        attn_bn_chunk=int(getattr(args,'attn_bn_chunk',0)),
                        attn_gate_threshold=int(getattr(args,'attn_gate_threshold',0)),
                        temporal_transformer=bool(getattr(args,'temporal_transformer',0)),
                        tt_layers=int(getattr(args,'tt_layers',2)),
                        graph_mix=float(getattr(args,'graph_mix',0.0)),
                        dropedge_p=float(getattr(args,'dropedge_p',0.0)),
                        quantiles=q_list
                    )
                    sd = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else (ck if isinstance(ck, dict) else None)
                    if sd is None:
                        continue
                    sd = {k.replace('module.','',1): v for k,v in sd.items()}
                    m_ck.load_state_dict(sd, strict=False)
                    m_ck = m_ck.to(device); m_ck.eval()
                    # Minimal integration: reuse current pred_full as placeholder.
                    series.append(pred_full.clone())
                if len(series) > 1:
                    stack = torch.stack(series, dim=0)
                    if str(getattr(args,'ensemble_mode','ivw')) == 'ivw':
                        pred_full = stack.mean(dim=0)  # placeholder for inverse-variance weighting
                    else:
                        pred_full = stack.mean(dim=0)
            except Exception as _e:
                print(f"[seed_ensemble] warning: {_e}")

        # Recompute metrics based on the full overlap-averaged series (reflects blending if applied)
        try:
            assert isinstance(pred_full, torch.Tensor) and isinstance(true_full, torch.Tensor)
            mean_all = torch.mean(true_full, dim=0)
            diff_r = true_full - mean_all.view(1, -1)
            rrse = float((torch.sum((true_full - pred_full)**2).sqrt() / (torch.sum(diff_r**2).sqrt() + eps)).item())
            rae = float((torch.sum(torch.abs(true_full - pred_full)) / (torch.sum(torch.abs(diff_r)) + eps)).item())
            p = pred_full.numpy(); y = true_full.numpy()
            sp, sg = p.std(axis=0), y.std(axis=0)
            mp, mg = p.mean(axis=0), y.mean(axis=0)
            denom = (sp*sg)+eps
            corr_vec = ((p-mp)*(y-mg)).mean(axis=0)/denom
            mask = (sg>0)
            correlation = float(corr_vec[mask].mean()) if mask.any() else 0.0
            sm = 0.0
            for n in range(y.shape[1]):
                den = (np.abs(y[:,n])+np.abs(p[:,n])); den[den==0]=eps
                sm += float(np.mean(np.abs(y[:,n]-p[:,n])/den))
            sm /= max(1, y.shape[1])
        except Exception as _e:
            print(f"[metrics_recompute] warning: {_e}")
    # --- small calibration log (keeps file tiny) ---
    if getattr(args, "runlog", False) and is_plot and pred_full is not None and true_full is not None:
        try:
            err = (pred_full - true_full)
            bias_node = err.mean(dim=0).abs()
            std_p = pred_full.std(dim=0)
            std_t = true_full.std(dim=0) + 1e-12
            var_ratio = (std_p / std_t)
            def _q(t, q):
                import torch
                return float(torch.quantile(t, q).cpu().item())
            jlog("calibration",
                 split=kind,
                 bias_med=_q(bias_node, 0.5),
                 bias_p95=_q(bias_node, 0.95),
                 var_ratio_med=_q(var_ratio, 0.5),
                 var_ratio_p95=_q(var_ratio, 0.95),
                 corr_mean=float(correlation))
        except Exception as _e:
            jlog("warn_calibration", split=kind, error=str(_e)[:160])
        # Ensemble weights logging
        if getattr(args, 'ensemble', False) and w is not None and labels:
            try:
                jlog("ensemble_weights", **{labels[i]: float(w[i].cpu()) for i in range(len(labels))})
            except Exception as _e:
                jlog("warn_ensemble_weights", error=str(_e)[:160])
    # ------------------------------------------------
    # --- CI coverage calibration: pick z so that ~95% of |err|/sem <= z on Validation
    if kind == 'Validation' and r_collect:
        try:
            # Ignore outliers beyond 5-sigma before computing quantile
            r_all = torch.cat([torch.as_tensor(x) for x in r_collect], dim=0).numpy()
            z_new = float(_np.quantile(r_all, 0.95))
            # keep changes gentle to avoid exploding CI
            _CONF_Z = max(0.5, min(3.5, z_new))
            if getattr(args, "runlog", False):
                jlog("conf_calibrate", z_new=z_new, z_final=_CONF_Z)
        except Exception as _e:
            jlog("warn_conf_calibrate", err=str(_e)[:160])

    # ---- CI width statistic for auto-dropout (Validation only) ----
    if kind == 'Validation':
        try:
            # conf95, target already built: shapes (B, L, N)
            # robust amplitude per node from middle quantiles (80-20)
            targ_flat = target.reshape(-1, target.shape[-1])                 # (B*L, N)
            q80 = torch.quantile(targ_flat, 0.80, dim=0)
            q20 = torch.quantile(targ_flat, 0.20, dim=0)
            robust_amp = torch.clamp(q80 - q20, min=1e-6)                    # (N,)
            ci_half_per_node = conf95.mean(dim=(0,1))                        # (N,)
            ratio_nodes = ci_half_per_node / robust_amp                      # (N,)
            global _LAST_VAL_CI_RATIO
            # Compute median robustly and guard against unexpected None types
            median_val = torch.median(ratio_nodes).item()
            _LAST_VAL_CI_RATIO = float(median_val) if median_val is not None else None
            if getattr(args, "runlog", False):
                jlog("val_ci_ratio", ci_ratio=_LAST_VAL_CI_RATIO)
        except Exception as _e:
            _LAST_VAL_CI_RATIO = None

    # ------------------------------------------------
    # Safely convert possible None/unknown numeric values to float (NaN on None)
    def _to_safe_float(v):
        try:
            return float(v) if v is not None else float('nan')
        except Exception:
            try:
                return float(str(v))
            except Exception:
                return float('nan')

    return _to_safe_float(rrse), _to_safe_float(rae), _to_safe_float(correlation), _to_safe_float(sm)

# ===================== Full-Series MC Dropout Mean/Var + IVW Ensemble (New) =====================
@torch.no_grad()
def _predict_full_series_mean_var(model: nn.Module,
                                  X_windows: torch.Tensor,
                                  mc_runs: int,
                                  device: torch.device,
                                  residual_head: bool,
                                  data_obj) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute stitched full-series predictive mean & variance via MC dropout.

    Args:
        model: trained model (will be toggled to train() for dropout).
        X_windows: tensor of shape (B, Tin, N, C) or (B, Tin, N) representing input windows used during evaluation (Data.valid[0] or Data.test[0]).
        mc_runs: number of stochastic forward passes.
        device: torch device.
        residual_head: whether residual baseline (last level) should be added back in original units.
        data_obj: DataLoaderS instance for inversion helpers (uses global normalization or rolling disabled here).

    Returns:
        mean_full: (T_full, N) stitched mean in original units.
        var_full:  (T_full, N) stitched predictive variance in original units.
    Notes:
        - Assumes stride=1 between successive windows (as produced by DataLoaderS).
        - Uses law of total variance across overlapping windows: Var = E[var] + Var(mean_component).
    """
    if X_windows is None or X_windows.numel() == 0:
        raise ValueError("Empty X_windows passed to _predict_full_series_mean_var")
    # Ensure model and windows share the same device (fixes MC forward mismatch)
    model = model.to(device)
    if X_windows.device != device:
        X_windows = X_windows.to(device)
    # enable dropout
    was_training = model.training
    model.train()
    try:
        B = X_windows.shape[0]
        # Normalize layout to (B, C, N, Tin)
        if X_windows.dim() == 4 and X_windows.size(-1) <= 4:  # (B, Tin, N, C)
            Xn = X_windows.permute(0, 3, 2, 1).contiguous()
        elif X_windows.dim() == 4:  # already (B, C, N, Tin)
            Xn = X_windows
        elif X_windows.dim() == 3:  # (B, Tin, N)
            Xn = X_windows.permute(0, 2, 1).unsqueeze(1).contiguous()
        else:
            raise RuntimeError(f"Unexpected X_windows shape: {tuple(X_windows.shape)}")

        means_per_batch = []   # list of (L, N)
        vars_per_batch = []    # list of (L, N)

        for b in range(B):
            xb = Xn[b:b+1].repeat(mc_runs, 1, 1, 1)  # vectorized MC for single window
            if xb.device != device:
                xb = xb.to(device)
            try:
                outs = model(xb)  # might be dict
                outs = unwrap_model_output(outs)
            except Exception as e:
                raise RuntimeError(f"Model forward failed during MC eval: {e}")
            if outs.dim() == 4:  # (mc_runs, C?, N, T) -> adapt to repo conv (collapse last)
                # follow existing evaluate logic: last temporal step only if shape like (mc_runs, ?, ?, T)
                outs = outs[:, :, :, -1]
            # ensure shape (mc_runs, L, N)
            if outs.dim() == 3 and outs.shape[1] != data_obj.out_len and outs.shape[2] == data_obj.m:
                # assume (mc_runs, L, N)
                pass
            elif outs.dim() == 2:  # (L, N) without mc axis (unlikely)
                outs = outs.unsqueeze(0)
            elif outs.dim() != 3:
                raise RuntimeError(f"Unsupported model output shape for MC: {tuple(outs.shape)}")

            mu_b = outs.mean(dim=0)  # (L, N) in z-space
            var_b = outs.var(dim=0, unbiased=True) if mc_runs > 1 else torch.zeros_like(mu_b)

            # Invert to original units (handles rolling/global + log1p)
            # We cannot supply per-window idx here (not stored); treat as global inversion.
            mu_o = data_obj.inv_transform_like(mu_b)
            # variance scaling: use data_obj.std_expand_like to ensure correct
            # broadcasting, dtype, and device handling for std across shapes
            try:
                std_exp = data_obj.std_expand_like(var_b)
                var_o = var_b * (std_exp**2)
            except Exception:
                # Fallback to conservative reshape if std_expand_like is unavailable
                std_expand = data_obj.std.view(1, -1).to(mu_b.device)
                var_o = var_b * (std_expand**2)

            # residual head recomposition (baseline = last input level) in original units
            if residual_head:
                try:
                    if X_windows.dim() == 4 and X_windows.size(-1) <= 4:  # (B, Tin, N, C)
                        last_level = X_windows[b, -1, :, 0]
                    elif X_windows.dim() == 4:
                        last_level = X_windows[b, 0, :, -1]
                    else:  # (B, Tin, N)
                        last_level = X_windows[b, -1, :]
                    baseline_o = data_obj.inv_transform_like(last_level)
                    baseline_seq = baseline_o.unsqueeze(0).expand_as(mu_o)
                    mu_o = mu_o + baseline_seq
                    # variance unchanged (additive constant)
                except Exception:
                    pass

            means_per_batch.append(mu_o.detach().cpu())
            vars_per_batch.append(var_o.detach().cpu())

        # Stitch overlap: windows with stride=1 -> total length = B + L - 1
        L = means_per_batch[0].shape[0]
        N = means_per_batch[0].shape[1]
        T_full = B + L - 1
        sum_mu = torch.zeros(T_full, N)
        sum_var = torch.zeros(T_full, N)
        sum_mu_sq = torch.zeros(T_full, N)
        count = torch.zeros(T_full, 1)
        for s, (m_b, v_b) in enumerate(zip(means_per_batch, vars_per_batch)):
            sl = slice(s, s + L)
            sum_mu[sl] += m_b
            sum_var[sl] += v_b
            sum_mu_sq[sl] += m_b**2
            count[sl] += 1.0
        count = torch.clamp(count, min=1.0)
        mean_full = sum_mu / count
        e_x2 = sum_mu_sq / count
        var_mean_component = torch.clamp(e_x2 - mean_full**2, min=0.0)
        var_full = torch.clamp((sum_var / count) + var_mean_component, min=0.0)
        return mean_full, var_full
    finally:
        if not was_training:
            model.eval()

def _reconstruct_true_full_series(Y_windows: torch.Tensor, data_obj) -> torch.Tensor:
    """Overlap-average target windows (B, L, N) -> (T_full, N)."""
    if Y_windows is None or Y_windows.numel() == 0:
        raise ValueError("Empty Y_windows passed to _reconstruct_true_full_series")
    if Y_windows.dim() != 3:
        raise RuntimeError(f"Y_windows should be (B,L,N); got {tuple(Y_windows.shape)}")
    B, L, N = Y_windows.shape
    T_full = B + L - 1
    sum_y = torch.zeros(T_full, N)
    count = torch.zeros(T_full, 1)
    for s in range(B):
        sl = slice(s, s + L)
        sum_y[sl] += Y_windows[s]
        count[sl] += 1.0
    count = torch.clamp(count, min=1.0)
    y_full_z = sum_y / count
    y_full = data_obj.inv_transform_like(y_full_z)
    return y_full

def _ivw_blend_full(means: list[torch.Tensor], vars_: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if not means:
        raise ValueError("No means provided for IVW blend")
    if len(means) != len(vars_):
        raise ValueError("Means/vars length mismatch")
    base = means[0]
    W = [1.0 / (v + 1e-8) for v in vars_]
    Wsum = torch.zeros_like(base)
    for w in W:
        Wsum += w
    mu_ens = torch.zeros_like(base)
    for w, m in zip(W, means):
        mu_ens += (w / (Wsum + 1e-8)) * m
    var_ens = torch.zeros_like(base)
    for w, m, v in zip(W, means, vars_):
        w_norm = w / (Wsum + 1e-8)
        var_ens += w_norm * (v + (m - mu_ens)**2)
    var_ens = torch.clamp(var_ens, min=0.0)
    return mu_ens, var_ens

def run_full_series_ensemble(model, Data, device, args):
    """Perform full-series MC variance estimation and optional checkpoint IVW ensemble.

    Saves arrays to args.out_dir: forecast_mean.npy, forecast_var.npy, forecast_L.npy, forecast_U.npy, forecast_true.npy
    Only runs if there is at least one trigger flag (ensemble_ckpts, conformal).
    """
    triggers = any([
        bool(getattr(args, 'ensemble_ckpts', '')),
        bool(getattr(args, 'conformal', False)),
    ])
    if not triggers:
        return
    try:
        os.makedirs(getattr(args, 'out_dir', 'runs'), exist_ok=True)
    except Exception:
        pass
    mc_runs = int(getattr(args, 'mc_runs', 30))
    print(f"[full-series] Starting MC evaluation (mc_runs={mc_runs})…")
    # Guards: ensure windows tensors are present and non-empty
    if (Data.valid is None or Data.valid[0] is None or Data.valid[0].numel()==0 or
        Data.test is None or Data.test[0] is None or Data.test[0].numel()==0):
        print('[full-series][skip] Empty validation/test windows; skipping MC ensemble.')
        return
    val_mean, _ = _predict_full_series_mean_var(model, Data.valid[0], mc_runs, device, getattr(args,'residual_head', False), Data)
    val_true = _reconstruct_true_full_series(Data.valid[1], Data)
    # Conformal calibration (per-node absolute residual quantile)
    qhat = None
    if getattr(args, 'conformal', False):
        alpha = float(max(1e-4, min(0.5, getattr(args, 'conf_alpha', 0.05))))
        abs_res = torch.abs(val_true - val_mean)
        qhat = torch.quantile(abs_res, 1.0 - alpha, dim=0)
        print(f"[full-series][conformal] alpha={alpha:.4f} computed q̂ per node.")

    test_mean_base, test_var_base = _predict_full_series_mean_var(model, Data.test[0], mc_runs, device, getattr(args,'residual_head', False), Data)
    means = [test_mean_base]
    vars_ = [test_var_base]

    # Load and evaluate additional checkpoints (exact, not reusing base prediction)
    ck_list = [p.strip() for p in str(getattr(args,'ensemble_ckpts','')).split(',') if p.strip()]
    if ck_list:
        print(f"[full-series][ensemble] Evaluating {len(ck_list)} checkpoint(s)…")
        for ck in ck_list:
            try:
                ck_obj = torch.load(ck, map_location='cpu')
                state = ck_obj.get('state_dict', ck_obj) if isinstance(ck_obj, dict) else ck_obj
                # Rebuild fresh model with best available hyperparameters (fallback to current args)
                hp = ck_obj.get('hparams', None) if isinstance(ck_obj, dict) else None
                in_dim_use = getattr(Data, 'in_dim', args.in_dim)
                mdl = gtnet(
                    args.gcn_true, args.buildA_true, (hp[0] if hp else args.gcn_depth), int(Data.m),
                    torch.device('cpu'), Data.adj, dropout=(hp[7] if hp else args.dropout),
                    subgraph_size=(hp[6] if hp else args.subgraph_size), node_dim=(hp[9] if hp else args.node_dim),
                    dilation_exponential=(hp[8] if hp else args.dilation_exponential), conv_channels=(hp[2] if hp else args.conv_channels),
                    residual_channels=(hp[3] if hp else args.residual_channels), skip_channels=(hp[4] if hp else args.skip_channels),
                    end_channels=(hp[5] if hp else args.end_channels), seq_length=args.seq_in_len, in_dim=in_dim_use, out_dim=args.seq_out_len,
                    layers=(hp[12] if hp else args.layers), propalpha=(hp[10] if hp else args.propalpha), tanhalpha=(hp[11] if hp else args.tanhalpha),
                    layer_norm_affline=False,
                    temporal_attn=getattr(args,'temporal_attn', False), attn_dim=getattr(args,'attn_dim',64),
                    attn_heads=getattr(args,'attn_heads',2), attn_dropout=getattr(args,'attn_dropout',0.1),
                    attn_window=getattr(args,'attn_window',0), attn_math_mode=getattr(args,'attn_math_mode', False),
                    attn_bn_chunk=int(getattr(args,'attn_bn_chunk',0)), attn_gate_threshold=int(getattr(args,'attn_gate_threshold',0)),
                    temporal_transformer=bool(getattr(args,'temporal_transformer',0)), tt_layers=int(getattr(args,'tt_layers',2)),
                    graph_mix=float(getattr(args,'graph_mix',0.0)), dropedge_p=float(getattr(args,'dropedge_p',0.0)), quantiles=q_list,
                    nb_head=bool(int(getattr(args,'use_nb_head',0))==1), zinb=bool(int(getattr(args,'use_zinb',0))==1)
                )
                sd = {k.replace('module.','',1): v for k,v in state.items()} if isinstance(state, dict) else state
                missing, unexpected = mdl.load_state_dict(sd, strict=False)
                if missing or unexpected:
                    print(f"[full-series][ensemble][warn] {os.path.basename(ck)} missing={len(missing)} unexpected={len(unexpected)}")
                mdl = mdl.to(device)
                mu_i, var_i = _predict_full_series_mean_var(mdl, Data.test[0], mc_runs, device, getattr(args,'residual_head', False), Data)
                means.append(mu_i)
                vars_.append(var_i)
                print(f"[full-series][ensemble] {os.path.basename(ck)} done.")
            except Exception as e:
                print(f"[full-series][ensemble][warn] Failed {ck}: {e}")

    if len(means) > 1:
        if getattr(args,'ensemble_mode','ivw') == 'ivw':
            mean_ens, var_ens = _ivw_blend_full(means, vars_)
        else:
            mean_ens = torch.mean(torch.stack(means, dim=0), dim=0)
            var_ens = torch.mean(torch.stack(vars_, dim=0), dim=0)
    else:
        mean_ens, var_ens = means[0], vars_[0]

    test_true_full = _reconstruct_true_full_series(Data.test[1], Data)
    # Gaussian 95% band
    z = 1.96
    sigma = torch.sqrt(torch.clamp(var_ens, min=0.0))
    L = mean_ens - z * sigma
    U = mean_ens + z * sigma
    # Quick metric summary
    try:
        yt = test_true_full.cpu().numpy(); yp = mean_ens.cpu().numpy()
        rrse_denom = np.sqrt(((yt - yt.mean(axis=0, keepdims=True))**2).sum(axis=0) + 1e-12)
        rrse = np.sqrt(((yt - yp)**2).sum(axis=0) + 1e-12) / rrse_denom
        smape = 200.0 * np.mean(np.abs(yt - yp) / (np.abs(yt) + np.abs(yp) + 1e-12))
        print(f"[full-series] RRSE(mean)={float(rrse.mean()):.4f} sMAPE={smape:.2f}%")
    except Exception:
        pass
# ================================================================================================

def train(data, X, Y, model, criterion, optim, batch_size,
          data_scaler,
          alpha: float = 1.0, beta: float = 0.4, gamma: float = 0.8, clip: float = 10.0,
          mae_weight: float = 0.2,
          grad_scaler: "torch.amp.grad_scaler.GradScaler|None" = None,
          scheduler=None):
    """
    Composite loss:
      total = α * L1(z-space) + β * (1 - corr) + γ * sMAPE(original)
    This strongly encourages the *shape* of predictions to follow ground truth.
    """
    model.train()
    total_loss = 0.0
    n_elems = 0
    it = 0
    perm = np.arange(args.num_nodes)
    # defensive defaults so analysis doesn't flag 'out' or 'loss' as possibly unbound
    out = None
    loss = None

    # AMP should only ever run when explicitly requested.
    # (Some environments still pass a disabled GradScaler object around; treat that as None.)
    if not bool(getattr(args, 'amp', False)):
        grad_scaler = None

    # AMP should only be considered enabled when the scaler is enabled.
    # Passing a disabled GradScaler here previously caused us to take the
    # scaler/backward path while still running in fp32, triggering
    # "Found dtype Float but expected Half".
    try:
        use_amp = (grad_scaler is not None) and (not hasattr(grad_scaler, 'is_enabled') or grad_scaler.is_enabled())
    except Exception:
        use_amp = (grad_scaler is not None)
    # Precompute inverse-std node weights (z-space) once per epoch from training set stats
    try:
        with torch.no_grad():
            # Data.train[1] shape: (n_samples, out_len, m) -> std over samples & time
            tr_std = data.train[1].std(dim=(0,1)) + 1e-6
            node_weights = (tr_std.mean() / tr_std).to(device)
    except Exception:
        node_weights = None

    # Request batch indices so we can look up per-window train stats in rolling mode
    grad_accum = max(1, int(getattr(args, 'grad_accum_steps', 1)))
    accum_step = 0
    for batch in data.get_batches(X, Y, batch_size, True, return_indices=True):
        # batch may be (Xb, Yb) or (Xb, Yb, idxs)
        if len(batch) == 3:
            Xb_raw, Yb_raw, idxs = batch
        else:
            Xb_raw, Yb_raw = batch; idxs = None
        if accum_step == 0:
            if hasattr(optim, "zero_grad"):
                optim.zero_grad()
            else:
                optim.optimizer.zero_grad()
        Xb_tensor = torch.as_tensor(Xb_raw)
        if Xb_tensor.dim() == 3:
            # (B,T,N) -> (B,T,N,1)
            Xb_tensor = Xb_tensor.unsqueeze(-1)
        try:
            expected_len = int(getattr(model, 'seq_length', Xb_tensor.shape[1]))
            Xb = to_model_layout(Xb_tensor, expected_len, debug=getattr(args, 'debug_layout', False))
        except Exception as _e:
            print(f"[LayoutTrain] Failed to reconcile layout: {_e}")
            raise
        Xb = Xb.to(device, dtype=torch.float)
        Yb = Yb_raw.to(device)

        if it % args.step_size == 0:
            perm = np.random.permutation(range(args.num_nodes))
        num_sub = max(1, int(args.num_nodes / max(1, args.num_split)))

        for j in range(args.num_split):
            if args.num_split > 1:
                # Not supported in this version, but placeholder for logic
                pass

            if use_amp:
                with autocast(device_type='cuda', enabled=use_amp):
                    out_raw = model(Xb)
                    q_pred = None; taus_tensor = None
                    if isinstance(out_raw, dict):
                        out = out_raw['mean']
                        q_pred = out_raw.get('quantiles', None)
                        taus_tensor = out_raw.get('taus', None)
                    else:
                        out = out_raw
                    if out.dim() == 4:
                        out = out.squeeze(3)

                # --- Loss calculation ---
                # Check if using weighted horizon loss mode
                if getattr(args, 'use_weighted_horizon_loss', False):
                    # Use weighted Huber horizon loss in original units
                    out_o = maybe_inv_scale(out, data_scaler)
                    y_o   = maybe_inv_scale(Yb, data_scaler)
                    
                    if getattr(args, 'residual_head', False):
                        baseline_z = last_level_baseline_expand(Xb_raw, out.shape[1]).detach()
                        baseline_o = maybe_inv_scale(baseline_z, data_scaler)
                        out_o = out_o + baseline_o
                        y_o   = y_o + baseline_o
                    
                    loss = weighted_huber_horizon_loss(
                        out_o,
                        y_o,
                        delta=float(getattr(args, 'huber_delta', 1.0)),
                        nonzero_weight=float(getattr(args, 'nonzero_weight', 4.0)),
                        horizon_gamma=float(getattr(args, 'horizon_gamma', 1.5)),
                    )
                else:
                    # Original composite loss
                    # Always calculate metrics in ORIGINAL units for consistency.
                    # 1. Invert predictions and targets to original scale
                    out_o = maybe_inv_scale(out, data_scaler)
                    y_o   = maybe_inv_scale(Yb, data_scaler)

                    # 2. If using residual head, add back the baseline in ORIGINAL units
                    if getattr(args, 'residual_head', False):
                        baseline_z = last_level_baseline_expand(Xb_raw, out.shape[1]).detach()
                        baseline_o = maybe_inv_scale(baseline_z, data_scaler)
                        out_o = out_o + baseline_o
                        y_o   = y_o + baseline_o                    # 3. Calculate shape-based loss (sMAPE) in original units
                    # Using a robust version that avoids division by zero
                    smape_denom = torch.clamp(torch.abs(y_o) + torch.abs(out_o), min=1e-6)
                    s_mape = torch.mean(torch.abs(y_o - out_o) / smape_denom)

                    # 4. Calculate L1 loss in z-space (encourages matching the normalized distribution)
                    l1_loss = criterion(out, Yb)

                    # 5. Calculate correlation loss in z-space
                    # Centering both tensors before computing correlation
                    out_centered = out - out.mean(dim=[1, 2], keepdim=True)
                    yb_centered = Yb - Yb.mean(dim=[1, 2], keepdim=True)
                    corr_num = torch.mean(out_centered * yb_centered, dim=[1, 2])
                    corr_den = torch.sqrt(
                        torch.mean(out_centered**2, dim=[1, 2]) * torch.mean(yb_centered**2, dim=[1, 2])
                    )
                    corr_loss = 1.0 - torch.mean(corr_num / torch.clamp(corr_den, min=1e-6))

                    # 6. Combine losses
                    loss = (alpha * l1_loss + beta * corr_loss + gamma * s_mape)
                    # Pinball (quantile) loss in original units if quantile outputs present
                    if q_pred is not None and taus_tensor is not None and getattr(args,'lambda_quantile',0.0) > 0.0:
                        try:
                            Bq, KC, Nq, _ = q_pred.shape
                            K = len(taus_tensor)
                            out_len = KC // max(1,K)
                            q_view = q_pred.view(Bq, K, out_len, Nq, 1).permute(0,1,3,2,4).squeeze(-1)  # [B,K,N,T]
                            y_target = y_o  # original units
                            taus = taus_tensor.to(q_view.device, q_view.dtype).view(1,-1,1,1)
                            err = y_target.unsqueeze(1) - q_view
                            pinball = torch.maximum(taus * err, (taus - 1.0) * err).mean()
                            loss = loss + float(getattr(args,'lambda_quantile',0.0)) * pinball
                        except Exception as _e:
                            if getattr(args,'runlog',False):
                                jlog('warn_pinball', err=str(_e)[:120])
                    # Gaussian NLL (homoscedastic) in original units
                    if getattr(args,'use_gauss',0) and getattr(args,'lambda_nll',0.0) > 0.0:
                        try:
                            log_var_param = None
                            core = model.module if hasattr(model,'module') else model
                            if hasattr(core, '_log_var'):
                                log_var_param = core._log_var
                            if log_var_param is not None:
                                log_var = log_var_param.view(1,1)
                                diff = y_o - out_o
                                nll = 0.5 * ((diff**2)/torch.exp(log_var) + log_var)
                                loss = loss + float(getattr(args,'lambda_nll',0.0)) * nll.mean()
                        except Exception as _e:
                            if getattr(args,'runlog',False):
                                jlog('warn_gauss_nll', err=str(_e)[:120])
                    # Optional: add a simple MAE term on original-unit values
                    if mae_weight > 0:
                        loss = loss + mae_weight * torch.mean(torch.abs(y_o - out_o))
                if loss is not None:
                    if grad_scaler is not None:
                        grad_scaler.scale(loss / grad_accum).backward()
                    else:
                        (loss / grad_accum).backward()
            else:
                out_raw = model(Xb)
                q_pred = None; taus_tensor = None
                if isinstance(out_raw, dict):
                    out = out_raw['mean']
                    q_pred = out_raw.get('quantiles', None)
                    taus_tensor = out_raw.get('taus', None)
                else:
                    out = out_raw
                if out.dim() == 4:
                    out = out.squeeze(3)

                # --- Loss calculation (non-AMP path) ---
                out_o = maybe_inv_scale(out, data_scaler)
                y_o   = maybe_inv_scale(Yb, data_scaler)

                if getattr(args, 'residual_head', False):
                    baseline_z = last_level_baseline_expand(Xb_raw, out.shape[1]).detach()
                    baseline_o = maybe_inv_scale(baseline_z, data_scaler)
                    out_o = out_o + baseline_o
                    y_o   = y_o + baseline_o

                smape_denom = torch.clamp(torch.abs(y_o) + torch.abs(out_o), min=1e-6)
                s_mape = torch.mean(torch.abs(y_o - out_o) / smape_denom)
                l1_loss = criterion(out, Yb)
                out_centered = out - out.mean(dim=[1, 2], keepdim=True)
                yb_centered = Yb - Yb.mean(dim=[1, 2], keepdim=True)
                corr_num = torch.mean(out_centered * yb_centered, dim=[1, 2])
                corr_den = torch.sqrt(
                    torch.mean(out_centered**2, dim=[1, 2]) * torch.mean(yb_centered**2, dim=[1, 2])
                )
                corr_loss = 1.0 - torch.mean(corr_num / torch.clamp(corr_den, min=1e-6))
                loss = (alpha * l1_loss + beta * corr_loss + gamma * s_mape)
                if q_pred is not None and taus_tensor is not None and getattr(args,'lambda_quantile',0.0) > 0.0:
                    try:
                        Bq, KC, Nq, _ = q_pred.shape
                        K = len(taus_tensor)
                        out_len = KC // max(1,K)
                        q_view = q_pred.view(Bq, K, out_len, Nq, 1).permute(0,1,3,2,4).squeeze(-1)
                        y_target = y_o
                        taus = taus_tensor.to(q_view.device, q_view.dtype).view(1,-1,1,1)
                        err = y_target.unsqueeze(1) - q_view
                        pinball = torch.maximum(taus * err, (taus - 1.0) * err).mean()
                        loss = loss + float(getattr(args,'lambda_quantile',0.0)) * pinball
                    except Exception as _e:
                        if getattr(args,'runlog',False):
                            jlog('warn_pinball', err=str(_e)[:120])
                if getattr(args,'use_gauss',0) and getattr(args,'lambda_nll',0.0) > 0.0:
                    try:
                        core = model.module if hasattr(model,'module') else model
                        if hasattr(core,'_log_var'):
                            log_var = core._log_var.view(1,1)
                            diff = y_o - out_o
                            nll = 0.5 * ((diff**2)/torch.exp(log_var) + log_var)
                            loss = loss + float(getattr(args,'lambda_nll',0.0)) * nll.mean()
                    except Exception as _e:
                        if getattr(args,'runlog',False):
                            jlog('warn_gauss_nll', err=str(_e)[:120])
                if mae_weight > 0:
                    loss = loss + mae_weight * torch.mean(torch.abs(y_o - out_o))
                (loss / grad_accum).backward()


            if loss is not None:
                if (accum_step + 1) % grad_accum == 0:
                    if clip is not None and clip > 0:
                        if grad_scaler is not None:
                            grad_scaler.unscale_(optim.optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    if grad_scaler is not None:
                        grad_scaler.step(optim.optimizer)
                        grad_scaler.update()
                    else:
                        optim.step()
                    if scheduler is not None and scheduler.__class__.__name__ == 'OneCycleLR':
                        scheduler.step()
                    accum_step = 0
                else:
                    accum_step += 1
                    # Defer optimizer step until accumulation complete

            if loss is not None:
                total_loss += float(loss.item()) * out.numel()
            n_elems += out.numel()

            if it % 20 == 0 and out is not None and loss is not None:
                denom = max(1, out.size(0) * out.size(1) * data.m)
                print(f'iter:{it:3d} | loss: {float(loss.item())/denom:.6f}')
            it += 1

    # Flush leftover accumulated gradients (if any) after loop
    if 'grad_accum' in locals() and 'accum_step' in locals():
        if accum_step != 0:  # we performed some micro-batches without stepping
            try:
                clip = float(clip)
            except Exception:
                pass
            if clip is not None and clip > 0:
                if grad_scaler is not None:
                    grad_scaler.unscale_(optim.optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            if grad_scaler is not None:
                grad_scaler.step(optim.optimizer)
                grad_scaler.update()
            else:
                optim.step()
            # OneCycle scheduler needs a step per optimizer step
            if scheduler is not None and scheduler.__class__.__name__ == 'OneCycleLR':
                scheduler.step()
    return total_loss / max(1.0, n_elems)



parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default='./data/sm_aggr_v4.csv',
                    help='Path to the time-series data file (CSV or TSV).')
parser.add_argument('--metrics_json', type=str, default=None,
                    help='Path to a metrics_*.json file whose embedded "args" dict will be applied to this run.')
parser.add_argument('--skip_ckpt_infer', action='store_true',
                    help='If set, do not infer hyperparameters from checkpoint shapes (sets BMTGNN_SKIP_CKPT_INFER=1).')
parser.add_argument('--train', action='store_true',
                    help='Force training mode even when metrics_json has eval_only=true (sets eval_only=False).')
parser.add_argument('--no_trainer_mode', action='store_true',
                    help='Disable Trainer path even if trainer_mode is set in JSON/args (forces trainer_mode=False).')
# CLI switch for runlog
parser.add_argument('--runlog', action='store_true', help='Write compact JSONL diagnostics while training/evaluating')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model/Bayesian/model.pt',
                    help='path to save the final model')
parser.add_argument('--out_dir', type=str, default='runs', help='Root folder to save metrics/plots for each run')
parser.add_argument('--run_tag', type=str, default='', help='Optional unique tag for this run; used to place outputs')
parser.add_argument('--val_plot_start', type=str, default='',
                    help="Validation plot start 'YYYY-MM' (after window stitching; e.g. 2018-01)")
parser.add_argument('--val_plot_end', type=str, default='',
                    help="Validation plot end 'YYYY-MM' (after window stitching; e.g. 2021-12)")
parser.add_argument('--optim', type=str, default='adamw')
parser.add_argument('--L1Loss', action='store_true', help='use L1 loss in training')
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--gcn_true', action='store_true',
                    help='Enable graph convolution layer')
parser.add_argument('--buildA_true', action='store_true',
                    help='Enable construction of adaptive adjacency matrix')
parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--num_nodes',type=int,default=0,help='number of nodes/variables (auto-detected)')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--subgraph_size',type=int,default=0,help='Size of subgraphs (0 to disable). Recommended: 20-40 for large graphs.')
parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
parser.add_argument('--dilation_exponential',type=int,default=2,help='dilation exponential')
parser.add_argument('--conv_channels',type=int,default=16,help='convolution channels')
parser.add_argument('--exclude-names', type=str, default='',
                    help='Comma-separated list of series names to exclude from training/eval')
parser.add_argument('--residual_channels',type=int,default=16,help='residual channels')
parser.add_argument('--skip_channels',type=int,default=32,help='skip channels')
parser.add_argument('--end_channels',type=int,default=64,help='end channels')
parser.add_argument('--allow_wide_end', action='store_true',
                    help='Allow 1024 end channels in search space (may increase memory).')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--seq_in_len',type=int,default=10,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=12,help='output sequence length')
parser.add_argument('--horizon', type=int, default=1) 
parser.add_argument('--layers',type=int,default=5,help='number of layers')

parser.add_argument('--batch_size',type=int,default=8,help='batch size')
parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')

parser.add_argument('--clip', type=float, default=10.0, help='Gradient norm clip (float)')
parser.add_argument('--grad_clip', type=float, default=None, help='Alias for --clip (if set, overrides --clip value)')

parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='tanh alpha')

parser.add_argument('--epochs',type=int,default=250,help='')
parser.add_argument('--seed', type=int, default=None, help='random seed (overrides fixed_seed)')
parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')
parser.add_argument('--step_size',type=int,default=100,help='step_size')
parser.add_argument('--amp', action='store_true', help='Enable mixed precision (AMP)')
# parser.add_argument('--dataparallel', action='store_true', default=True, help='Wrap model with nn.DataParallel if >1 GPU')
parser.add_argument('--no_dataparallel', action='store_true', help='Disable DataParallel even if multiple GPUs are visible')
parser.add_argument('--cudnn_benchmark', action='store_true', help='Enable cuDNN benchmark (faster, non-deterministic)')
parser.add_argument('--mc_runs', type=int, default=50, help='MC-Dropout samples at eval')
parser.add_argument('--vectorized_mc', action='store_true', default=False,
                    help='Vectorize MC-Dropout by repeating the batch (ON = faster, higher peak memory). OFF = looped passes (safer).')
parser.add_argument('--no_vectorized_mc', action='store_true',
                    help='Explicitly disable vectorized MC (overrides --vectorized_mc).')

# --- Probabilistic training knob (routing) -----------------------------------
# (Actual Gaussian NLL flags live in the quantile block below to avoid duplicates)
parser.add_argument('--trainer_mode', action='store_true',
                    help='Use the Trainer-based path (skips random search) and wire probabilistic losses if requested.')
parser.add_argument('--mc_vec_max_elems', type=int, default=120_000_000,
                    help='If vectorized MC would create (mc_runs*B*C*N*T) > this element count, fallback to looped MC automatically.')
parser.add_argument('--grad_accum_steps', type=int, default=1,
                    help='Accumulate gradients over N mini-batches before optimizer step (simulated larger batch).')
parser.add_argument('--fused_optim', action='store_true',
                    help='Use fused AdamW optimizer (CUDA fused=True) instead of the custom Optim wrapper')
parser.add_argument('--graph', type=str, default='./data/graph_topk_k12.csv',
                    help='Path to the predefined adjacency matrix (square CSV).')
parser.add_argument('--persist_mc', type=str, default='', help='Path prefix to persist per-slide MC samples (npz files)')
parser.add_argument('--persist_mc_slides', type=str, default='', help='Comma-separated slide start indices to persist (e.g. "10,46"). Empty = persist all when --persist_mc supplied')
parser.add_argument('--preset', type=str, default='v4_3ep', choices=['v4_3ep'], help='Apply the v4_3ep preset (default)')
parser.add_argument('--debug_rf', action='store_true', help='Print receptive field and kernel widths per layer and exit')
parser.add_argument('--ckpt_to_compare', type=str, default='', help='Path to checkpoint file to compare shapes against current model')
parser.add_argument('--ckpt_compare_csv', type=str, default='', help='If provided, write CSV comparing checkpoint vs model parameter shapes')
parser.add_argument('--use_cached_hp', action='store_true', default=True,
                    help='Skip hyperparameter search; use cached best hyperparameters from model/Bayesian/hp.txt and existing checkpoint')
parser.add_argument('--eval_only', action='store_true',
                    help='With --use_cached_hp, only evaluate the cached checkpoint without any training/search')
parser.add_argument('--search_trials', type=int, default=120,
                    help='Number of random hyperparameter samples during search')
parser.add_argument('--early_stop_patience', type=int, default=150,
                    help='Epoch patience per trial before aborting that trial early (0 disables)')
parser.add_argument('--no_plots', action='store_true', help='Disable plotting during evaluation')
parser.add_argument('--enable_test_plots', action='store_true', default=True, help='Generate validation & testing plots whenever a new global best is found during training (ENABLED by default)')
parser.add_argument('--no_test_plots', dest='enable_test_plots', action='store_false', help='Disable test plot generation')
parser.add_argument('--smooth_l1_only', action='store_true', help='Use only SmoothL1Loss in normalized space (debug)')
parser.add_argument('--use_weighted_horizon_loss', action='store_true', 
                    help='Use weighted Huber horizon loss (upweights non-zeros & far horizon) instead of composite loss')
parser.add_argument('--plot_norm_space', action='store_true', help='Also plot predictions vs truth in normalized (z) space')
parser.add_argument('--movement_debug', action='store_true', help='Print movement diagnostics in normalized space (variance, corr)')
parser.add_argument('--start_year', type=int, default=2004, help='Starting calendar year for x-axis ticks in plots')
parser.add_argument('--steps_per_year', type=int, default=12, help='Number of time steps that correspond to one year (e.g., 12 for monthly)')
parser.add_argument('--chronological_split', action='store_true', help='Enable fixed calendar-year split (train 2004-2018, val 2019-2022, test 2023-2025 by default)')
parser.add_argument('--use_chronological_split', type=bool, default=True, help='Enable fixed calendar-year split by default')
parser.add_argument('--train_end_year', type=int, default=2018, help='Final year (inclusive) of training segment when using chronological split')
parser.add_argument('--valid_end_year', type=int, default=2022, help='Final year (inclusive) of validation segment when using chronological split')
parser.add_argument('--test_end_year', type=int, default=2025, help='Final year (inclusive) of test segment when using chronological split')
parser.add_argument('--auto_disable_chrono', type=int, default=1,
                    help='If 1 (default), auto-disable chronological split when val/test spans are too short for seq_in_len+seq_out_len; fall back to ratio splits.')
parser.add_argument('--train_ratio', type=float, default=0.60,
                    help='Train ratio for ratio-based split (used if chronological is disabled/unsupported).')
parser.add_argument('--valid_ratio', type=float, default=0.20,
                    help='Validation ratio for ratio-based split (used if chronological is disabled/unsupported).')
parser.add_argument(
    '--calibration',
    choices=['none', 'val', 'test', 'both'],
    default='none',
    help=(
        'Per-node linear calibration a*pred+b. '
        'none=disabled (default); '
        'val=fit on validation only; '
        'test=apply cached params on test; '
        'both=fit on val and apply on test.'
    )
)
# Removed duplicate _fit_linear_calibration - now imported from src.util
parser.add_argument('--analysis_log', type=str, default=None, help='Path to a file for detailed JSON analysis logging.')
parser.add_argument('--y_transform', type=str, default='log1p', choices=['none','log1p'],
                    help='Target transform to stabilize heavy-tailed sparse series.')
parser.add_argument('--nonzero_weight', type=float, default=4.0,
                    help='Upweight loss when y_true > 0 (handles sparse signals).')
parser.add_argument('--horizon_gamma', type=float, default=1.5,
                    help='Exponent for horizon weighting; higher => more weight on far horizon.')
parser.add_argument('--huber_delta', type=float, default=1.0,
                    help='Huber delta in transformed space.')
# Robust metrics toggle: enable guarded RRSE/RAE/Corr/sMAPE with zero-variance masking
parser.add_argument('--robust_metrics', action='store_true', help='Use variance-masked, epsilon-guarded RRSE/RAE/Correlation/sMAPE computations (safer with constant nodes).')
# Residual head option: train/predict residuals relative to last input level
parser.add_argument('--residual_head', action='store_true', help='Train/predict residuals relative to last input level per-sample (original units)')
# Dual-channel input option: none/diff/pct (must be defined BEFORE parse_args)
parser.add_argument('--dual_channel', choices=['none', 'diff', 'pct'], default='diff',
                    help='Second input channel: diff (z[t]-z[t-1]) or pct; none for single-channel')
parser.add_argument('--pct_clip', type=float, default=0.0,
                    help='Clip absolute value of pct channel to stabilize training (0 disables).')
parser.add_argument('--ensemble', action='store_true', help='Enable validation-time blending')
parser.add_argument('--ensemble_ckpts', type=str, default='',
                    help='Comma-separated list of extra checkpoints to ensemble at eval (seed/ckpt ensemble).')
parser.add_argument('--ensemble_mode', choices=['ivw','mean'], default='ivw',
                    help='How to fuse seed/ckpt ensemble predictions: ivw (inverse variance placeholder) or mean.')
parser.add_argument('--conformal', action='store_true',
                    help='Enable split-conformal adjustment on MC intervals (adds per-node q̂ half-width).')
parser.add_argument('--conf_alpha', type=float, default=0.05,
                    help='Miscoverage level α for conformal PI (0.05 => ~95% nominal).')
parser.add_argument('--ens_patchtst', action='store_true', help='Add PatchTST-mini to ensemble')
parser.add_argument('--ens_nhits', action='store_true', help='Add N-HiTS-mini to ensemble')
parser.add_argument('--ens_mlp', action='store_true', help='Add simple MLP head to ensemble')
parser.add_argument('--ens_d_model', type=int, default=128)
parser.add_argument('--ens_nhead', type=int, default=8)
parser.add_argument('--ens_patch_len', type=int, default=16)
parser.add_argument('--auto_tune_dropout', action='store_true',
                    help='Gently auto-lower dropout when validation CI is too wide.')
parser.add_argument('--auto_dropout_target', type=float, default=0.35,
                    help='Target median(CI_half / robust_amp) on Validation.')
parser.add_argument('--auto_dropout_min', type=float, default=0.02,
                    help='Minimum allowed dropout when auto-tuning.')
parser.add_argument('--auto_dropout_step', type=float, default=0.10,
                    help='Fractional decrease per adjust step, e.g. 0.10 = -10 percent.')
parser.add_argument('--ens_stride', type=int, default=8)
parser.add_argument('--ens_depth', type=int, default=2)
parser.add_argument('--ens_hidden', type=int, default=256)
parser.add_argument('--ens_blocks', type=int, default=3)
parser.add_argument('--ens_dropout', type=float, default=0.1)
parser.add_argument('--ens_epochs', type=int, default=0, help='Quick-fit epochs for extra models on TRAIN')
parser.add_argument('--ens_lr', type=float, default=1e-3)

# Amplitude-aware loss and display smoothing
parser.add_argument('--amp_loss_weight', type=float, default=0.3,
                    help='Weight for amplitude (range) penalty in original units')
parser.add_argument('--smooth_plot', action='store_true', help='Apply light exponential smoothing to plots only')
parser.add_argument('--smooth_alpha', type=float, default=0.1, help='Smoothing alpha for plots (0-1)')


# ---- NEW: expose composite-loss weights & CI calibration & scheduler ----
parser.add_argument('--loss_alpha', type=float, default=1.0, help='Weight of L1 term')
parser.add_argument('--loss_beta',  type=float, default=0.4, help='Weight of 1-corr term')
parser.add_argument('--loss_gamma', type=float, default=0.8, help='Weight of sMAPE term')
parser.add_argument('--mae_weight', type=float, default=0.2, help='Extra MAE (original units)')
parser.add_argument('--conf_calibrate', action='store_true',
                    help='Calibrate dropout CI width on validation for ~95% coverage')


parser.add_argument('--scheduler', choices=['none','cosine','onecycle'], default='cosine',
                    help='LR scheduler (epoch-wise).')
parser.add_argument('--sched_T0', type=int, default=10, help='CosineWarmRestarts T0')
parser.add_argument('--sched_Tmult', type=int, default=2, help='CosineWarmRestarts T_mult')
parser.add_argument('--onecycle_pct', type=float, default=0.3, help='OneCycle warmup pct')

# --- Temporal attention flags -----------------------------------------------
parser.add_argument('--temporal_attn', action='store_true',
                    help='Enable lightweight temporal self-attention block (after temporal convs).')
parser.add_argument('--attn_heads', type=int, default=2,
                    help='Number of attention heads for temporal MHSA.')
parser.add_argument('--attn_dim', type=int, default=64,
                    help='Projection dimension for temporal MHSA (Q/K/V).')
parser.add_argument('--attn_dropout', type=float, default=0.10,
                    help='Dropout inside temporal MHSA block.')
parser.add_argument('--attn_window', type=int, default=0,
                    help='If >0, restrict attention to last W timesteps (local window).')
parser.add_argument('--attn_math_mode', action='store_true',
                    help='Force PyTorch to use math (non-flash) scaled dot product attention kernels for stability.')
parser.add_argument('--attn_bn_chunk', type=int, default=0,
                    help='If >0, chunk the (batch*nodes) axis for temporal attention to save memory.')
parser.add_argument('--attn_gate_threshold', type=int, default=1500000,
                    help='Auto-disable temporal attention when (mc_runs * batch_size * num_nodes) exceeds this threshold (0 disables gating).')

# --- Temporal Transformer (per-node) ----------------------------------------
parser.add_argument('--temporal_transformer', type=int, default=0,
                    help='Enable Transformer encoder over time (0/1).')
parser.add_argument('--tt_layers', type=int, default=2,
                    help='Number of Transformer encoder layers when enabled.')

# --- Graph fusion + DropEdge -------------------------------------------------
parser.add_argument('--graph_mix', type=float, default=0.0,
                    help='Blend ratio alpha for predefined vs learned adjacency (alpha*A_pre + (1-alpha)*A_learned).')
parser.add_argument('--dropedge_p', type=float, default=0.0,
                    help='DropEdge probability during training (0.0-0.5 typical).')

# --- Quantile / Pinball Loss -------------------------------------------------
parser.add_argument('--quantiles', type=str, default='',
                    help='Comma-separated list of quantiles, e.g. "0.1,0.5,0.9" (empty to disable).')
parser.add_argument('--lambda_quantile', type=float, default=0.0,
                    help='Weight for quantile (pinball) loss.')
# NOTE: lambda_quantile (CLI) maps to Trainer(lambda_q).
#       use_gauss/lambda_nll map directly to Trainer(use_gauss/lambda_nll).
#       The Trainer path is only engaged when --trainer_mode or any probabilistic loss is effectively ON.
parser.add_argument('--use_gauss', type=int, default=0,
                    help='Enable Gaussian NLL auxiliary head/loss (1=on,0=off).')
parser.add_argument('--lambda_nll', type=float, default=0.0,
                    help='Weight for Gaussian NLL term when --use_gauss=1.')
parser.add_argument('--use_nb_head', type=int, default=0,
                    help='Enable Negative Binomial head/loss (1=on,0=off). Optional: --use_zinb 1 adds zero-inflation.')
parser.add_argument('--use_zinb', type=int, default=0,
                    help='With --use_nb_head: use Zero-Inflated NB (ZINB).')

# --- Graph normalization + weighting + compile controls ----------------------
parser.add_argument('--graph_normalize', choices=['none','sym','square','row'], default='none',
                    help='Apply normalization to loaded adjacency: sym (D^-1/2 A D^-1/2), square (A@A, then row norm), row (row stochastic), none.')
parser.add_argument('--weight_nodes_in_loss', action='store_true',
                    help='Weight node contributions in loss by inverse robust amplitude (per-node).')
parser.add_argument('--weight_nodes_in_metrics', action='store_true',
                    help='Apply same node weights when aggregating metrics (MAE/RMSE/etc).')
parser.add_argument('--compile', type=str, choices=['off','eager','inductor','auto'], default='auto',
                    help='Control torch.compile usage: off (disable), eager (backend=eager), inductor (default backend), auto (best effort).')
parser.add_argument('--nan_debug', action='store_true',
                    help='Enable verbose NaN/inf diagnostics during evaluation and sliding-window MC.')
parser.add_argument('--auto_window_adjust', action='store_true',
                    help='If no training windows exist, automatically shrink seq_in_len (and if needed seq_out_len) until at least --auto_window_min_train windows are available.')
parser.add_argument('--auto_window_min_train', type=int, default=8,
                    help='Target minimum number of training windows when using --auto_window_adjust (stop early if reached).')
parser.add_argument('--debug_layout', action='store_true',
                    help='Print detailed tensor layout transformations for debugging sequence length mismatches.')
parser.add_argument('--log_gpu_mem', action='store_true',
                    help='Log CUDA allocated/peak memory after each epoch and before/after evaluation.')
parser.add_argument('--log_peak_mem', action='store_true',
                    help='Alias for --log_gpu_mem (deprecated name).')

parser.add_argument(
    '--strong_rmdpt',
    action='store_true',
    help='Enable a strong default recipe for sparse/heavy-tailed RMD/PT forecasting (better RAE & curve hugging).'
)

def _flag_was_set(name: str) -> bool:
    """
    True if user explicitly set --<name> or --<name>=... in CLI.
    Keeps strong mode from overriding your manual choices.
    """
    key = f'--{name}'
    for a in sys.argv[1:]:
        if a == key or a.startswith(key + '='):
            return True
    return False


def _apply_strong_rmdpt_defaults(args):
    # Data/representation
    if not _flag_was_set('normalize'):      args.normalize = 4         # rolling/window norm
    if not _flag_was_set('dual_channel'):   args.dual_channel = 'pct'  # better “movement following”
    if not _flag_was_set('pct_clip'):       args.pct_clip = 3.0        # stabilizes pct spikes
    if not _flag_was_set('y_transform'):    args.y_transform = 'log1p' # stabilize heavy tail

    # Loss shaping (key for RAE)
    if not _flag_was_set('use_weighted_horizon_loss'): args.use_weighted_horizon_loss = True
    if not _flag_was_set('nonzero_weight'): args.nonzero_weight = 6.0
    if not _flag_was_set('horizon_gamma'):  args.horizon_gamma = 2.0
    if not _flag_was_set('huber_delta'):    args.huber_delta = 2.0

    # Model knobs that usually improve hugging
    if not _flag_was_set('residual_head'):  args.residual_head = True
    if not _flag_was_set('graph_mix'):      args.graph_mix = max(float(getattr(args,'graph_mix',0.0)), 0.6)
    if not _flag_was_set('dropedge_p'):     args.dropedge_p = max(float(getattr(args,'dropedge_p',0.0)), 0.10)
    if not _flag_was_set('graph_normalize'): args.graph_normalize = 'sym'
    if not _flag_was_set('weight_nodes_in_loss'): args.weight_nodes_in_loss = True

    # Temporal attention helps long horizon on your monthly series
    if not _flag_was_set('temporal_attn'):  args.temporal_attn = True
    if not _flag_was_set('temporal_transformer'): args.temporal_transformer = 1
    if not _flag_was_set('tt_layers'):      args.tt_layers = 3
    if not _flag_was_set('attn_heads'):     args.attn_heads = 4
    if not _flag_was_set('attn_dim'):       args.attn_dim = 128
    if not _flag_was_set('attn_dropout'):   args.attn_dropout = 0.08

    # Training stability
    if not _flag_was_set('lr'):            args.lr = 4e-4
    if not _flag_was_set('weight_decay'):  args.weight_decay = 1e-5
    if not _flag_was_set('dropout'):       args.dropout = 0.05
    if not _flag_was_set('clip'):          args.clip = 1.5
    if not _flag_was_set('early_stop_patience'): args.early_stop_patience = 80

    # Safer metrics aggregation
    if not _flag_was_set('robust_metrics'): args.robust_metrics = True
    if not _flag_was_set('weight_nodes_in_metrics'): args.weight_nodes_in_metrics = True

args = parser.parse_args()

# If the user passed --metrics_json, apply the embedded args dict to override defaults.
if getattr(args, 'metrics_json', None):
    try:
        with open(args.metrics_json, 'r', encoding='utf-8') as _f:
            _metrics = json.load(_f)
        if 'args' in _metrics and isinstance(_metrics['args'], dict):
            # Preserve explicit CLI overrides for any key the user passed.
            _cli = sys.argv[1:]
            _cli_keys: set[str] = set()
            for _tok in _cli:
                if _tok.startswith('--'):
                    _cli_keys.add(_tok[2:].replace('-', '_'))

            for _k, _v in _metrics['args'].items():
                if _v is None:
                    continue
                if isinstance(_v, str) and _v == '':
                    continue
                # don't override explicit CLI args
                if _k in _cli_keys:
                    continue
                # set attribute if parser knows about it
                if hasattr(args, _k):
                    try:
                        setattr(args, _k, _v)
                    except Exception:
                        pass

            print(f'[metrics_json] Applied args from {args.metrics_json} (preserved CLI overrides)')
    except Exception as _e:
        print('[metrics_json] Failed to apply metrics_json:', _e)

# If requested, set env var to skip checkpoint-shape-based inference
if getattr(args, 'skip_ckpt_infer', False):
    os.environ['BMTGNN_SKIP_CKPT_INFER'] = '1'

# Force training mode even if metrics_json came from an eval-only run.
if getattr(args, 'train', False):
    try:
        args.eval_only = False
    except Exception:
        pass
    try:
        args.use_cached_hp = False
    except Exception:
        pass

# Allow explicit disabling of trainer_mode when needed (so random search runs are performed)
if getattr(args, 'no_trainer_mode', False):
    try:
        args.trainer_mode = False
        print('[cli] --no_trainer_mode: forcing trainer_mode=False so search_trials will run')
    except Exception:
        pass
if getattr(args, 'strong_rmdpt', False):
    print('[strong_rmdpt] applying strong defaults...')
    _apply_strong_rmdpt_defaults(args)
# ---- Optional: force-disable hyperparameter search (debug only) ----
try:
    if os.environ.get('BMTGNN_FORCE_SEARCH_TRIALS_0', '0') == '1':
        if getattr(args, 'search_trials', 0) and int(getattr(args, 'search_trials', 0)) > 0:
            print('[warn] BMTGNN_FORCE_SEARCH_TRIALS_0=1; forcing search_trials=0')
            args.search_trials = 0
except Exception:
    pass
# Map grad_clip alias if provided
if getattr(args, 'grad_clip', None) is not None:
    try:
        if args.grad_clip is not None:
            args.clip = float(args.grad_clip)
    except Exception:
        pass
    print('[Deprecation] --grad_clip is deprecated; use --clip instead.')

# --- Auto-detect number of nodes (variables) if not provided or <=0 ---
if getattr(args, 'num_nodes', 0) <= 0:
    try:
        import numpy as _np, os as _os
        _candidates = [args.data]
        if not _os.path.exists(args.data):
            _candidates.append(_os.path.join('data', _os.path.basename(args.data)))
            _candidates.append('data/sm_data.csv')
            _candidates.append('data/sm_data.txt')
        for _p in _candidates:
            if _p and _os.path.exists(_p):
                try:
                    _delim = ',' if str(_p).lower().endswith('.csv') else '\t'
                    _row = _np.loadtxt(_p, delimiter=_delim, max_rows=1)
                    if _row.ndim == 1:
                        args.num_nodes = int(_row.shape[0])
                        print(f"[AutoDetect] Inferred num_nodes={args.num_nodes} from '{_p}'")
                        break
                except Exception:
                    pass
        if getattr(args, 'num_nodes', 0) <= 0:
            print("[AutoDetect] Warning: could not infer num_nodes; will attempt after data load")
    except Exception as _e:
        print(f"[AutoDetect] Warning: num_nodes inference failed early: {_e}")

# Parse quantiles (if provided)
q_list = []
if getattr(args, 'quantiles', ''):
    raw_q = str(getattr(args, 'quantiles')).strip()
    if raw_q:
        try:
            parts = [p for p in raw_q.split(',') if p.strip()]
            q_list = [float(p) for p in parts]
            q_list = [q for q in q_list if 0.0 < q < 1.0]
        except Exception:
            q_list = []

# Normalize legacy/alias flags
if getattr(args, 'log_peak_mem', False) and not getattr(args, 'log_gpu_mem', False):
    args.log_gpu_mem = True

# Harmonize vectorized MC flags (explicit --no_vectorized_mc overrides)
if getattr(args, 'no_vectorized_mc', False):
    args.vectorized_mc = False

# Decide whether to use the Trainer path (probabilistic features ON or explicit switch)
_has_quantiles = bool(getattr(args, 'quantiles', '').strip())
_q_weight      = float(getattr(args, 'lambda_quantile', 0.0)) > 0.0
_gauss_on      = (int(getattr(args, 'use_gauss', 0)) == 1) and (float(getattr(args, 'lambda_nll', 0.0)) > 0.0)
_nb_on         = (int(getattr(args, 'use_nb_head', 0)) == 1) and (float(getattr(args, 'lambda_nll', 0.0)) > 0.0)
USE_TRAINER_PATH = bool(getattr(args, 'trainer_mode', False)) or (_has_quantiles and _q_weight) or _gauss_on or _nb_on

# Export MC run count early so temporal attention gating sees it during evaluation
import os as _os
_os.environ['BMTGNN_MC_RUNS'] = str(int(getattr(args, 'mc_runs', 50)))

use_cuda = torch.cuda.is_available() and args.device.startswith('cuda')
device = torch.device(args.device if use_cuda else 'cpu')
print(f"[Device] Using {device}")

# Print a compact resolved configuration banner (helps confirm metrics_json + CLI overrides)
try:
    print(
        "[Config] data=", getattr(args, 'data', None),
        "graph=", getattr(args, 'graph', None),
        "seq_in_len=", getattr(args, 'seq_in_len', None),
        "seq_out_len=", getattr(args, 'seq_out_len', None),
        "layers=", getattr(args, 'layers', None),
        "conv/res/skip/end=", (
            getattr(args, 'conv_channels', None),
            getattr(args, 'residual_channels', None),
            getattr(args, 'skip_channels', None),
            getattr(args, 'end_channels', None),
        ),
        "normalize=", getattr(args, 'normalize', None),
        "dual_channel=", getattr(args, 'dual_channel', None),
        "y_transform=", getattr(args, 'y_transform', None),
        "trainer_mode=", bool(getattr(args, 'trainer_mode', False)),
        "USE_TRAINER_PATH=", bool(globals().get('USE_TRAINER_PATH', False)),
        "search_trials=", getattr(args, 'search_trials', None),
        "epochs=", getattr(args, 'epochs', None),
        "eval_only=", bool(getattr(args, 'eval_only', False)),
        "chronological_split=", bool(getattr(args, 'chronological_split', False)),
        "use_chronological_split=", bool(getattr(args, 'use_chronological_split', False)),
        "mc_runs=", getattr(args, 'mc_runs', None),
        "use_gauss/lambda_nll=", (int(getattr(args, 'use_gauss', 0)), float(getattr(args, 'lambda_nll', 0.0))),
        sep='',
    )
except Exception:
    pass

# Start run logger if enabled
if getattr(args, "runlog", False):
    start_runlog(args)
    import torch
    devprops = {}
    if torch.cuda.is_available() and device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        devprops = {"name": p.name, "total_mem_gb": round(p.total_memory/1024**3, 2)}
    jlog("device_info", device=str(device), **devprops)

# clamp mc_runs to avoid accidental OOMs but allow higher user values
args.mc_runs = max(10, int(getattr(args, 'mc_runs', 50)))


# Removed duplicate set_random_seed - now imported from src.util

fixed_seed = 123

# Hyperparameter search option lists (module-level to avoid scope/analysis issues)
GCN_DEPTHS = [1, 2, 3]
LRS = [0.01, 0.001, 0.0005, 0.0008, 0.0001, 0.0003, 0.005]
CONVS = [4, 8, 16]
RESS = [16, 32, 64]
SKIPS = [64, 128, 256]
# Removed 1024 to reduce OOM risk under vectorized MC
ENDS = [256, 512]
if getattr(args, 'allow_wide_end', False) and 1024 not in ENDS:
    ENDS.append(1024)
LAYERS_CHOICES = [1, 2]
KS = [20, 30, 40, 50, 60, 70, 80, 90, 100]
DROPOUTS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
DILATION_EXS = [1, 2, 3]
NODE_DIMS = [20, 30, 40, 50, 60, 70, 80, 90, 100]
PROP_ALPHAS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8]
TANH_ALPHAS = [0.05, 0.1, 0.5, 1, 2, 3, 5, 7, 9]



def main(experiment): # pyright: ignore[reportGeneralTypeIssues]
    model: nn.Module | None = None
    # Set random seed for reproducibility (use --seed if provided)
    seed_to_use = fixed_seed if getattr(args, 'seed', None) is None else int(args.seed)
    set_random_seed(seed_to_use, getattr(args, 'cudnn_benchmark', False))

    # --- NEW: Configure Analysis Logger ---
    global ANALYSIS_LOGGER
    if args.analysis_log:
        ANALYSIS_LOGGER = AnalysisLogger(args.analysis_log)
        print(f"[AnalysisLogger] Logging detailed diagnostics to {args.analysis_log}")
    # ---

    # --- NEW: Early checkpoint hparams inference (before DataLoader construction) ---
    try:
        # Allow skipping automatic checkpoint-based hyperparameter inference
        # by setting environment variable BMTGNN_SKIP_CKPT_INFER=1
        if os.environ.get('BMTGNN_SKIP_CKPT_INFER', '0') == '1':
            print('[ckpt-early] skipping checkpoint inference due to BMTGNN_SKIP_CKPT_INFER=1')
        elif getattr(args, 'save', None) and os.path.exists(args.save):
            try:
                ck_early = torch.load(args.save, map_location='cpu')
                sd_early = ck_early.get('state_dict', {}) if isinstance(ck_early, dict) else {}
                if isinstance(sd_early, dict) and sd_early:
                    # lightweight inference of temporal kernel, layers and channels
                    seq_k = set(); layers_found = set()
                    conv_in = []
                    residual_out = []
                    skip_out = []
                    end_in = []
                    for k, v in sd_early.items():
                        if not hasattr(v, 'shape'):
                            continue
                        s = tuple(v.shape)
                        if k.startswith('filter_convs.') and k.endswith('.weight') and len(s) >= 4:
                            seq_k.add(s[-1])
                            try:
                                layers_found.add(int(k.split('.')[1]))
                            except Exception:
                                pass
                            conv_in.append(s[1])
                        if k.startswith('residual_convs.') and k.endswith('.weight') and len(s) >= 2:
                            residual_out.append(s[0])
                            try:
                                layers_found.add(int(k.split('.')[1]))
                            except Exception:
                                pass
                        if k.startswith('skip_convs.') and k.endswith('.weight') and len(s) >= 2:
                            skip_out.append(s[0])
                        if 'end_conv' in k and k.endswith('.weight') and len(s) >= 2:
                            end_in.append(s[1])
                    if seq_k:
                        args.seq_in_len = int(max(seq_k))
                    if layers_found:
                        args.layers = int(max(layers_found) + 1)
                    def _mode_or_none(lst):
                        try:
                            from statistics import mode
                            return int(mode(lst))
                        except Exception:
                            return int(max(set(lst), key=lst.count)) if lst else None
                    ci = _mode_or_none(conv_in)
                    if ci:
                        args.conv_channels = int(ci)
                    rc = _mode_or_none(residual_out)
                    if rc:
                        args.residual_channels = int(rc)
                    sc = _mode_or_none(skip_out)
                    if sc:
                        args.skip_channels = int(sc)
                    ei = _mode_or_none(end_in)
                    if ei:
                        args.end_channels = int(ei)
                    # Prefer exact checkpoint end_conv_1 output dim when present
                    try:
                        if 'end_conv_1.weight' in sd_early:
                            args.end_channels = int(sd_early['end_conv_1.weight'].shape[0])
                    except Exception:
                        pass
                    # detect quantile/nb head
                    if any(k.startswith('end_conv_q') or k.startswith('end_conv_nb') for k in sd_early.keys()):
                        setattr(args, 'use_nb_head', 1)
                    print('[ckpt-early] inferred seq_in_len=', getattr(args, 'seq_in_len', None),
                          'layers=', getattr(args, 'layers', None),
                          'conv=', getattr(args, 'conv_channels', None),
                          'res=', getattr(args, 'residual_channels', None),
                          'skip=', getattr(args, 'skip_channels', None),
                          'end=', getattr(args, 'end_channels', None),
                          'use_nb_head=', getattr(args, 'use_nb_head', None))
            except Exception as _e:
                print('[ckpt-early] failed to inspect checkpoint early:', _e)
    except Exception:
        pass
    # ---

    # v4_3ep preset: directly apply all settings from metrics_validation.json
    # MAE: 7.14, RMSE: 19.55, sMAPE: 4.98%, RSE: 0.0083, RAE: 0.0049
    # CLI arguments naturally override via argparse (we check against argparse defaults)
    
    # Only apply preset values if user didn't override via CLI
    # Check against argparse defaults to preserve CLI arguments
    if args.data == './data/sm_aggr_v4.csv': args.data = './data/sm_data.csv'
    args.graph = './data/graph_square.csv'  # Use square adjacency
    if args.out_dir == 'runs': args.out_dir = './outputs_v4_3ep'
    if args.save == 'model/Bayesian/model.pt': args.save = 'model/Bayesian/model.pt'
    if args.runlog == False: args.runlog = False
    if args.log_interval == 2000: args.log_interval = 2000
    if args.run_tag == '': args.run_tag = ''
    if args.val_plot_start == '': args.val_plot_start = ''
    if args.val_plot_end == '': args.val_plot_end = ''
    if args.optim == 'adamw': args.optim = 'adamw'
    if args.L1Loss == False: args.L1Loss = False
    if args.normalize == 2: args.normalize = 2
    
    # Model architecture
    if args.gcn_true == False: args.gcn_true = False
    if args.buildA_true == False: args.buildA_true = False
    if args.gcn_depth == 2: args.gcn_depth = 2
    if args.num_nodes == 95: args.num_nodes = 95
    if args.dropout == 0.3: args.dropout = 0.05
    if args.subgraph_size == 0: args.subgraph_size = 8
    if args.node_dim == 40: args.node_dim = 40
    if args.dilation_exponential == 2: args.dilation_exponential = 2
    if args.conv_channels == 16: args.conv_channels = 16
    if args.residual_channels == 16: args.residual_channels = 32
    if args.skip_channels == 32: args.skip_channels = 64
    if args.end_channels == 64: args.end_channels = 128
    if args.allow_wide_end == False: args.allow_wide_end = False
    if args.in_dim == 1: args.in_dim = 1
    if args.seq_in_len == 10: args.seq_in_len = 120
    if args.seq_out_len == 36: args.seq_out_len = 36
    if args.horizon == 1: args.horizon = 1
    if args.layers == 5: args.layers = 8
    if args.exclude_names == '': args.exclude_names = ''
    
    # Training hyperparameters
    if args.batch_size == 8: args.batch_size = 8
    if args.lr == 0.001: args.lr = 0.0004
    if args.weight_decay == 0.00001: args.weight_decay = 1e-05
    if args.clip == 10.0: args.clip = 1.5
    if args.grad_clip == None: args.grad_clip = None
    if args.propalpha == 0.05: args.propalpha = 0.05
    if args.tanhalpha == 3: args.tanhalpha = 3
    if args.step_size == 100: args.step_size = 100
    # epochs: don't override if user explicitly set it via CLI
    if args.epochs == 250: args.epochs = 0  # Only reset if still at argparse default
    if args.seed == None: args.seed = None
    if args.num_split == 1: args.num_split = 1
    if args.search_trials == 120: args.search_trials = 0
    if args.early_stop_patience == 150: args.early_stop_patience = 150
    
    # Training execution
    if args.amp == False: args.amp = False
    if args.no_dataparallel == False: args.no_dataparallel = False
    if args.cudnn_benchmark == False: args.cudnn_benchmark = False
    if args.mc_runs == 50: args.mc_runs = 10
    if args.vectorized_mc == False: args.vectorized_mc = False
    if args.no_vectorized_mc == False: args.no_vectorized_mc = False
    if args.mc_vec_max_elems == 120_000_000: args.mc_vec_max_elems = 120_000_000
    if args.grad_accum_steps == 1: args.grad_accum_steps = 1
    if args.fused_optim == False: args.fused_optim = False
    if args.trainer_mode == False: args.trainer_mode = True
    if args.use_cached_hp == True: args.use_cached_hp = False
    if args.eval_only == False: args.eval_only = True
    
    # Probabilistic features - Loss configuration - Data splitting
    # (Continuing conditional preset application)
    if args.use_gauss == 0: args.use_gauss = 1
    if args.lambda_nll == 0.0: args.lambda_nll = 0.1
    if args.use_nb_head == 0: args.use_nb_head = 0
    if args.use_zinb == 0: args.use_zinb = 0
    if args.quantiles == '': args.quantiles = ''
    if args.lambda_quantile == 0.0: args.lambda_quantile = 0.0
    if args.calibration == 'none': args.calibration = 'both'
    if args.y_transform == 'log1p': args.y_transform = 'none'
    if args.nonzero_weight == 4.0: args.nonzero_weight = 4.0
    if args.horizon_gamma == 1.5: args.horizon_gamma = 1.5
    if args.huber_delta == 1.0: args.huber_delta = 1.0
    if args.loss_alpha == 1.0: args.loss_alpha = 1.0
    if args.loss_beta == 0.4: args.loss_beta = 0.4
    if args.loss_gamma == 0.8: args.loss_gamma = 0.8
    if args.mae_weight == 0.2: args.mae_weight = 0.2
    if args.smooth_l1_only == False: args.smooth_l1_only = False
    if args.use_weighted_horizon_loss == False: args.use_weighted_horizon_loss = False
    if args.start_year == 2004: args.start_year = 2004
    if args.steps_per_year == 12: args.steps_per_year = 12
    if args.use_chronological_split == True: args.use_chronological_split = False
    if args.chronological_split == False: args.chronological_split = False
    if args.train_end_year == 2014: args.train_end_year = 2014
    if args.valid_end_year == 2021: args.valid_end_year = 2021
    if args.test_end_year == 2024: args.test_end_year = 2024
    if args.auto_disable_chrono == 1: args.auto_disable_chrono = 1
    if args.train_ratio == 0.60: args.train_ratio = 0.6
    if args.valid_ratio == 0.20: args.valid_ratio = 0.2
    if args.robust_metrics == False: args.robust_metrics = False
    if args.weight_nodes_in_loss == False: args.weight_nodes_in_loss = False
    if args.weight_nodes_in_metrics == False: args.weight_nodes_in_metrics = False
    if args.residual_head == False: args.residual_head = False
    if args.dual_channel == 'diff': args.dual_channel = 'pct'
    if args.pct_clip == 0.0: args.pct_clip = 0.0
    if args.temporal_attn == False: args.temporal_attn = True
    if args.temporal_transformer == 0: args.temporal_transformer = 1
    if args.tt_layers == 2: args.tt_layers = 3
    if args.attn_heads == 2: args.attn_heads = 4
    if args.attn_dim == 64: args.attn_dim = 128
    if args.attn_dropout == 0.10: args.attn_dropout = 0.08
    if args.attn_window == 0: args.attn_window = 0
    if args.attn_math_mode == False: args.attn_math_mode = False
    if args.attn_bn_chunk == 0: args.attn_bn_chunk = 0
    if args.attn_gate_threshold == 1500000: args.attn_gate_threshold = 1500000
    if args.graph_mix == 0.0: args.graph_mix = 0.6
    if args.dropedge_p == 0.0: args.dropedge_p = 0.0
    if args.graph_normalize == 'none': args.graph_normalize = 'none'
    if args.compile == 'auto': args.compile = 'off'
    if args.no_plots == False: args.no_plots = False
    if args.enable_test_plots == False: args.enable_test_plots = False
    if args.plot_norm_space == False: args.plot_norm_space = False
    if args.movement_debug == False: args.movement_debug = False
    if args.debug_rf == False: args.debug_rf = False
    if args.nan_debug == False: args.nan_debug = False
    if args.auto_window_adjust == False: args.auto_window_adjust = False
    if args.auto_window_min_train == 8: args.auto_window_min_train = 8
    if args.debug_layout == False: args.debug_layout = False
    if args.log_gpu_mem == False: args.log_gpu_mem = False
    if args.log_peak_mem == False: args.log_peak_mem = False
    if args.scheduler == 'cosine': args.scheduler = 'cosine'
    if args.sched_T0 == 10: args.sched_T0 = 10
    if args.sched_Tmult == 2: args.sched_Tmult = 2
    if args.onecycle_pct == 0.3: args.onecycle_pct = 0.3
    if args.ensemble == False: args.ensemble = False
    if args.ensemble_ckpts == '': args.ensemble_ckpts = ''
    if args.ensemble_mode == 'ivw': args.ensemble_mode = 'ivw'
    if args.conformal == False: args.conformal = False
    if args.conf_alpha == 0.05: args.conf_alpha = 0.05
    if args.conf_calibrate == False: args.conf_calibrate = False
    if args.ens_patchtst == False: args.ens_patchtst = False
    if args.ens_nhits == False: args.ens_nhits = False
    if args.ens_mlp == False: args.ens_mlp = False
    if args.ens_d_model == 128: args.ens_d_model = 128
    if args.ens_nhead == 8: args.ens_nhead = 8
    if args.ens_patch_len == 16: args.ens_patch_len = 16
    if args.ens_stride == 8: args.ens_stride = 8
    if args.ens_depth == 2: args.ens_depth = 2
    if args.ens_hidden == 256: args.ens_hidden = 256
    if args.ens_blocks == 3: args.ens_blocks = 3
    if args.ens_dropout == 0.1: args.ens_dropout = 0.1
    if args.ens_epochs == 0: args.ens_epochs = 0
    if args.ens_lr == 0.001: args.ens_lr = 0.001
    if args.auto_tune_dropout == False: args.auto_tune_dropout = False
    if args.auto_dropout_target == 0.35: args.auto_dropout_target = 0.35
    if args.auto_dropout_min == 0.02: args.auto_dropout_min = 0.02
    if args.auto_dropout_step == 0.10: args.auto_dropout_step = 0.1
    if args.smooth_plot == False: args.smooth_plot = False
    if args.smooth_alpha == 0.1: args.smooth_alpha = 0.1
    if args.amp_loss_weight == 0.3: args.amp_loss_weight = 0.3
    if args.persist_mc == '': args.persist_mc = ''
    if args.persist_mc_slides == '': args.persist_mc_slides = ''
    if args.preset == 'v4_3ep': args.preset = ''
    if args.ckpt_to_compare == '': args.ckpt_to_compare = ''
    if args.ckpt_compare_csv == '': args.ckpt_compare_csv = ''
    if args.analysis_log == None: args.analysis_log = None
    if args.strong_rmdpt == False: args.strong_rmdpt = False

    # Keep checkpoint output run-scoped for reproducibility (especially Optuna).
    _maybe_set_run_scoped_save_path(args)

    print('[Preset] applied v4_3ep defaults')

    # Post-preset: re-apply explicit CLI intent if needed
    if getattr(args, 'train', False):
        args.eval_only = False
        try:
            # If the JSON was eval-only it may have set use_cached_hp; training should not.
            args.use_cached_hp = False
        except Exception:
            pass
    if getattr(args, 'no_trainer_mode', False):
        args.trainer_mode = False

    # Recompute Trainer/search routing after presets (USE_TRAINER_PATH is used later).
    try:
        _has_quantiles = bool(getattr(args, 'quantiles', '').strip())
        _q_weight      = float(getattr(args, 'lambda_quantile', 0.0)) > 0.0
        _gauss_on      = (int(getattr(args, 'use_gauss', 0)) == 1) and (float(getattr(args, 'lambda_nll', 0.0)) > 0.0)
        _nb_on         = (int(getattr(args, 'use_nb_head', 0)) == 1) and (float(getattr(args, 'lambda_nll', 0.0)) > 0.0)
        globals()['USE_TRAINER_PATH'] = bool(getattr(args, 'trainer_mode', False)) or (_has_quantiles and _q_weight) or _gauss_on or _nb_on
    except Exception:
        pass

    # Keep env in sync for attention auto-gating
    try:
        os.environ['BMTGNN_MC_RUNS'] = str(int(getattr(args, 'mc_runs', 50)))
    except Exception:
        pass

    # Print a post-preset banner so logs reflect the effective config.
    try:
        print(
            "[ConfigResolved] data=", getattr(args, 'data', None),
            " graph=", getattr(args, 'graph', None),
            " seq_in_len=", getattr(args, 'seq_in_len', None),
            " seq_out_len=", getattr(args, 'seq_out_len', None),
            " layers=", getattr(args, 'layers', None),
            " trainer_mode=", bool(getattr(args, 'trainer_mode', False)),
            " USE_TRAINER_PATH=", bool(globals().get('USE_TRAINER_PATH', False)),
            " search_trials=", getattr(args, 'search_trials', None),
            " epochs=", getattr(args, 'epochs', None),
            " eval_only=", bool(getattr(args, 'eval_only', False)),
            sep='',
        )
    except Exception:
        pass

    # Stage user graph once (if provided) BEFORE DataLoaderS builds adjacency
    if getattr(args, 'graph', ''):
        user_graph = args.graph
        if os.path.exists(user_graph):
            try:
                os.makedirs('data', exist_ok=True)
                # Try to load as numeric square adjacency
                import numpy as np  # local import to ensure availability for static analyzers
                g = np.loadtxt(user_graph, delimiter=',')
                if g.ndim == 2 and g.shape[0] == g.shape[1]:
                    # preserve float weights
                    np.savetxt('data/graph_square.csv', g, delimiter=',', fmt='%.6g')
                    print('Copied numeric adjacency to data/graph_square.csv')
                else:
                    # treat as edge-list/headered CSV
                    import shutil
                    shutil.copy(user_graph, 'data/graph.csv')
                    print('Copied graph to data/graph.csv')
            except Exception:
                import shutil
                os.makedirs('data', exist_ok=True)
                shutil.copy(user_graph, 'data/graph.csv')
                print('Copied graph to data/graph.csv (fallback)')

    Data, use_chrono, steps_py, required_months = resolve_split_and_build_data(args, device)

    predefined_A = prepare_graph_and_subgraph(args, device, Data)

    # Log which adjacency source will be used by DataLoaderS
    try:
        if getattr(args, 'graph', ''):
            if os.path.exists('data/graph_square.csv'):
                print(f"[Graph] adjacency source=data/graph_square.csv (from {args.graph})")
            elif os.path.exists('data/graph.csv'):
                print(f"[Graph] adjacency source=data/graph.csv (from {args.graph})")
    except Exception:
        pass


    best_hp = []
    best_val = best_rse = best_rae = best_smape = 10000000
    best_corr = -10000000
    best_test_rse = 10000000
    best_test_corr = -10000000

    # Build Data once (DataLoaderS parameters are static w.r.t. HPs)
    # Validate data path and try sensible fallbacks before constructing DataLoaderS
    if not os.path.exists(args.data):
        # try project-local data directory with same basename
        local_candidate = os.path.join(os.getcwd(), 'data', os.path.basename(args.data))
        if os.path.exists(local_candidate):
            print(f"[Warning] specified data '{args.data}' not found; using '{local_candidate}' instead")
            args.data = local_candidate
        elif os.path.exists(os.path.join(os.getcwd(), 'data', 'sm_data.txt')):
            alt = os.path.join(os.getcwd(), 'data', 'sm_data.txt')
            print(f"[Warning] specified data '{args.data}' not found; falling back to '{alt}'")
            args.data = alt
        else:
            found = []
            try:
                found = os.listdir(os.path.join(os.getcwd(), 'data'))
            except Exception:
                found = []
            raise FileNotFoundError(f"Data file '{args.data}' not found. Place your dataset at that path or in ./data/. Files found in ./data/: {found}")

    # Data already built above as Data

    # If num_nodes still unset (0 or negative), adopt from loaded data
    if getattr(args, 'num_nodes', 0) <= 0:
        try:
            args.num_nodes = int(getattr(Data, 'm', 0))
            if args.num_nodes > 0:
                print(f"[AutoDetect] Set num_nodes from data: {args.num_nodes}")
        except Exception as _e:
            print(f"[AutoDetect] Warning: failed to set num_nodes from Data: {_e}")

    # Move entire (small) splits to GPU once to avoid per-batch host→device copies
    def _to_dev_split(tup):
        if tup is None or tup[0] is None or tup[1] is None:
            return tup
        return (tup[0].to(device, dtype=torch.float32, non_blocking=True),
                tup[1].to(device, dtype=torch.float32, non_blocking=True))
    if device.type == 'cuda':
        from typing import Any
        setattr(Data, 'train', _to_dev_split(Data.train) if Data.train is not None else None)
        setattr(Data, 'valid', _to_dev_split(Data.valid) if Data.valid is not None else None)
        setattr(Data, 'test', _to_dev_split(Data.test) if Data.test is not None else None)
        if getattr(Data, 'test_window', None) is not None:
            setattr(Data, 'test_window', torch.as_tensor(Data.test_window, dtype=torch.float32).to(device, non_blocking=True))

    # --- sanity: training windows must exist ---
    try:
        train_attr = getattr(Data, 'train', None)
        if train_attr is None or train_attr[0] is None:
            raise RuntimeError("No training windows: Data.train is missing or empty")
        B_train = int(train_attr[0].shape[0])
        jlog("window_guard", train_windows=B_train)
        if B_train < 1:
            if getattr(args, 'auto_window_adjust', False):
                print('[AutoWindow] No training windows; attempting automatic adjustment...')
                # Preserve original for logging
                orig_in, orig_out = int(args.seq_in_len), int(args.seq_out_len)
                target_min = max(1, int(getattr(args, 'auto_window_min_train', 4)))
                # Hard lower bounds to avoid degenerate setups
                min_in_allowed = 8
                min_out_allowed = 4
                adjusted = False
                # We'll try decreasing seq_in_len first (history length), then seq_out_len
                for _pass in range(2):
                    for _ in range(200):  # safety cap
                        # Rebuild DataLoaderS with current lengths
                        Data = DataLoaderS(args.data, float(getattr(args,'train_ratio',0.6)), float(getattr(args,'valid_ratio',0.2)), device, args.horizon, args.seq_in_len, args.normalize, args.seq_out_len,
                                           chronological=use_chrono,
                                           start_year=args.start_year, steps_per_year=args.steps_per_year,
                                           train_end_year=args.train_end_year, valid_end_year=args.valid_end_year, test_end_year=args.test_end_year,
                                           dual_channel=args.dual_channel,
                                           pct_clip=float(getattr(args, 'pct_clip', 0.0)),
                                           y_transform=getattr(args,'y_transform', None),
                                           exclude_names=(args.exclude_names if getattr(args, 'exclude_names', '') != '' else None))
                        train_attr = getattr(Data, 'train', None)
                        B_train = int(train_attr[0].shape[0]) if train_attr and train_attr[0] is not None else 0
                        if B_train >= target_min or (B_train >= 1 and _pass == 1):
                            adjusted = True
                            setattr(args, '_final_seq_in_len', int(args.seq_in_len))
                            setattr(args, '_final_seq_out_len', int(args.seq_out_len))
                            setattr(args, '_auto_window_adjusted', True)
                            print(f"[AutoWindow] Success: seq_in_len={args.seq_in_len}, seq_out_len={args.seq_out_len}, train_windows={B_train}")
                            break
                        # Adjust lengths
                        if _pass == 0 and args.seq_in_len > min_in_allowed:
                            # Decrease in_len in modest steps (e.g., 12 or 6) to keep monthly semantics
                            step = 12 if args.seq_in_len - 12 >= min_in_allowed else 6
                            args.seq_in_len = max(min_in_allowed, args.seq_in_len - step)
                        elif _pass == 1 and args.seq_out_len > min_out_allowed:
                            step = 12 if args.seq_out_len - 12 >= min_out_allowed else 4
                            args.seq_out_len = max(min_out_allowed, args.seq_out_len - step)
                        else:
                            break
                    if adjusted:
                        break
                if not adjusted:
                    raise RuntimeError(f"No training windows after auto-adjust attempts (final seq_in_len={args.seq_in_len}, seq_out_len={args.seq_out_len}). Consider widening --train_end_year.")
            else:
                raise RuntimeError(f"No training windows: decrease --seq_in_len/--seq_out_len or widen the train split.")
    except Exception as _e:
        print(f"[FATAL] {_e}")
        raise
    # ------------------------------------------
    # ------------------------------------------
    # --- RunLog: dataset & graph summary ---
    if getattr(args, "runlog", False):
        try:
            N = Data.m
            T_train = int(Data.train[0].shape[0]) if getattr(Data, 'train', None) and Data.train[0] is not None else 0
            T_valid = int(Data.valid[0].shape[0]) if getattr(Data, 'valid', None) and Data.valid[0] is not None else 0
            T_test  = int(Data.test[0].shape[0]) if getattr(Data, 'test', None) and Data.test[0] is not None else 0
            mode = "rolling" if getattr(Data, "rolling", False) else "global"
            dual = getattr(args, "dual_channel", "none")
            jlog("data_summary", nodes=N, t_train=T_train, t_valid=T_valid, t_test=T_test,
                 normalize_mode=mode, dual_channel=dual, seq_in=args.seq_in_len, seq_out=args.seq_out_len)

            # quick graph density (ignores diagonal)
            dens = None
            if getattr(args, "graph", None) and os.path.exists(args.graph):
                import numpy as np
                A = np.loadtxt(args.graph, delimiter=',')
                if A.shape[0] == A.shape[1] and A.shape[0] == N:
                    nnz_off = np.count_nonzero(A - np.diag(np.diag(A)))
                    dens = float(nnz_off) / float(max(1, N*(N-1)))
            jlog("graph_summary", path=args.graph, density=dens)
        except Exception as _e:
            jlog("warn_data_graph_summary", error=str(_e)[:160])
        inact = getattr(Data, "inactive_nodes", None)
        if inact is not None:
            jlog("inactive_nodes", count=len(inact))
    # Save per-node train scaler for reproducibility/debug
    try:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        mean_t = Data.mu.detach().cpu() if (hasattr(Data, 'mu') and Data.mu is not None and torch.is_tensor(Data.mu)) else None
        std_t = torch.clamp(Data.std.detach().cpu(), min=1e-6) if (hasattr(Data, 'std') and Data.std is not None and torch.is_tensor(Data.std)) else None
        torch.save({'mean': mean_t, 'std': std_t}, os.path.join(os.path.dirname(args.save), 'y_scaler.pt'))
    except Exception as _e:
        print(f"[scaler_save] warning: {_e}")

    # --- NEW: Archive checkpoint + scaler next to outputs for reproducibility ---
    # This prevents losing the exact checkpoint that produced a given metrics_validation.json.
    try:
        out_dir = getattr(args, 'out_dir', None)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            # stable tag: either run_tag or the out_dir basename
            run_tag = getattr(args, 'run_tag', '') or os.path.basename(os.path.normpath(out_dir))
            arch_dir = os.path.join(out_dir, f"_artifacts_{run_tag}")
            os.makedirs(arch_dir, exist_ok=True)
            # copy checkpoint
            if getattr(args, 'save', None) and os.path.exists(args.save):
                import shutil
                shutil.copy2(args.save, os.path.join(arch_dir, os.path.basename(args.save)))
            # copy scaler
            scaler_path = os.path.join(os.path.dirname(args.save), 'y_scaler.pt')
            if os.path.exists(scaler_path):
                import shutil
                shutil.copy2(scaler_path, os.path.join(arch_dir, 'y_scaler.pt'))
    except Exception as _e:
        print(f"[archive] warning: {_e}")
    # ---
    
    # Split summary logging (keep consistent with actual split mode)
    try:
        if bool(use_chrono):
            print(f"[SPLIT] start_year={args.start_year}, steps_per_year={args.steps_per_year}")
            print(
                f"[SPLIT] Train: {args.start_year}–{args.train_end_year} | "
                f"Valid: {args.train_end_year + 1}–{args.valid_end_year} | "
                f"Test: {args.valid_end_year + 1}–{args.test_end_year}"
            )
        else:
            # If chrono was requested but we auto-fell back, make it explicit.
            if bool(getattr(args, 'use_chronological_split', False)) or bool(getattr(args, 'chronological_split', False)):
                print("[SPLIT] Using ratio splits (chronological split auto-disabled due to insufficient windowable span)")
    except Exception:
        pass


    # Apply safer graph defaults for stability/generalization
    try:
        # prefer symmetric normalized adjacency to tame spectrum
        from src.util import sym_adj
        import numpy as np
        sym_path = 'data/graph_symnorm.csv'
        if os.path.exists(sym_path):
            try:
                A_symnorm = np.loadtxt(sym_path, delimiter=',')
                Data.adj = torch.from_numpy(A_symnorm.astype(np.float32)).to(device)
                print('[Graph] using cached symmetric-normalized adjacency (D^-1/2 A D^-1/2)')
            except Exception:
                pass
        if isinstance(Data.adj, torch.Tensor) and not os.path.exists(sym_path):
            A_np = Data.adj.detach().cpu().numpy()
            A_symnorm = sym_adj(A_np)
            Data.adj = torch.from_numpy(A_symnorm.astype(np.float32)).to(device)
            try:
                np.savetxt(sym_path, A_symnorm, delimiter=',', fmt='%.6f')
            except Exception:
                pass
            print('[Graph] using symmetric-normalized adjacency (D^-1/2 A D^-1/2)')
    except Exception:
        pass

    # conservative default subgraph_size if user didn't change it
    if getattr(args, 'subgraph_size', None) is None or int(getattr(args, 'subgraph_size', 20)) == 20:
        args.subgraph_size = 8
        print(f"[Graph] subgraph_size not specified or default; using conservative k={args.subgraph_size}")

    # Optional: skip random search if user wants to reuse cached best HPs / checkpoint
    if args.use_cached_hp and args.eval_only:
        print('[use_cached_hp + eval_only] Bypassing training and trials.')
        run_search = False
    elif USE_TRAINER_PATH:
        print('[trainer_mode/probabilistic] Engaging Trainer path and skipping random search.')
        run_search = False
    else:
        run_search = True

    # -------------------------------------------------------------------------
    # Trainer path (probabilistic / explicit switch). This performs a single,
    # deterministic training run with your current CLI hyperparameters and
    # wires pinball (quantiles) and/or Gaussian NLL, then falls through to the
    # standard evaluation + artifact writing code below.
    # -------------------------------------------------------------------------
    if not run_search and USE_TRAINER_PATH:
        run_trainer_path(
            args=args,
            Data=Data,
            device=device,
            q_list=q_list,
            has_quantiles=_has_quantiles,
            q_weight=_q_weight,
            use_trainer_path=USE_TRAINER_PATH,
            evaluate_fn=evaluate,
            evaluate_sliding_window_fn=evaluate_sliding_window,
            to_model_layout_fn=to_model_layout,
            jlog_fn=jlog,
        )

    if run_search:
        # random search loop
        trials = max(1, int(getattr(args,'search_trials',60)))
        for q in range(trials):
            # hps (sample from module-level options)
            gcn_depth = GCN_DEPTHS[randrange(len(GCN_DEPTHS))]
            lr = LRS[randrange(len(LRS))]
            conv = CONVS[randrange(len(CONVS))]
            res = RESS[randrange(len(RESS))]
            skip = SKIPS[randrange(len(SKIPS))]
            end = ENDS[randrange(len(ENDS))]
            # Safety caps to reduce OOM risk under large mc_runs or wide heads
            res = min(res, 256)
            skip = min(skip, 512)
            end = min(end, 512)
            layer = LAYERS_CHOICES[randrange(len(LAYERS_CHOICES))]
            k = KS[randrange(len(KS))]
            dropout = DROPOUTS[randrange(len(DROPOUTS))]
            dilation_ex = DILATION_EXS[randrange(len(DILATION_EXS))]
            node_dim = NODE_DIMS[randrange(len(NODE_DIMS))]
            prop_alpha = PROP_ALPHAS[randrange(len(PROP_ALPHAS))]
            tanh_alpha = TANH_ALPHAS[randrange(len(TANH_ALPHAS))]

            print('train X:',Data.train[0].shape)
            print('train Y:', Data.train[1].shape)
            print('valid X:',Data.valid[0].shape)
            print('valid Y:',Data.valid[1].shape)
            print('test X:',Data.test[0].shape)
            print('test Y:',Data.test[1].shape)
            print('test window:', Data.test_window.shape)

            tlen = (int(Data.train[0].shape[0]) if getattr(Data,'train',None) and Data.train[0] is not None else 0)
            vlen = (int(Data.valid[0].shape[0]) if getattr(Data,'valid',None) and Data.valid[0] is not None else 0)
            slen = (int(Data.test[0].shape[0]) if getattr(Data,'test',None) and Data.test[0] is not None else 0)
            print('length of training set=', tlen)
            print('length of validation set=', vlen)
            print('length of testing set=', slen)
            print('valid=',int((0.43 + 0.3) * Data.n))

            # If predefined adjacency exists, move to the same device
            if isinstance(Data.adj, torch.Tensor):
                Data.adj = Data.adj.to(device)

            try:
                in_dim_use = getattr(Data, 'in_dim', args.in_dim)
            except NameError:
                in_dim_use = args.in_dim
            model = gtnet(
                args.gcn_true, args.buildA_true, gcn_depth, int(Data.m),
                device, Data.adj, dropout=dropout, subgraph_size=k,
                node_dim=node_dim, dilation_exponential=dilation_ex,
                conv_channels=conv, residual_channels=res,
                skip_channels=skip, end_channels=end,
                seq_length=args.seq_in_len, in_dim=in_dim_use, out_dim=args.seq_out_len,
                layers=layer, propalpha=prop_alpha, tanhalpha=tanh_alpha, layer_norm_affline=False,
                temporal_attn=getattr(args,'temporal_attn', False), attn_dim=getattr(args,'attn_dim',64),
                attn_heads=getattr(args,'attn_heads',2), attn_dropout=getattr(args,'attn_dropout',0.1),
                attn_window=getattr(args,'attn_window',0), attn_math_mode=getattr(args,'attn_math_mode', False),
                attn_bn_chunk=int(getattr(args,'attn_bn_chunk',0)),
                attn_gate_threshold=int(getattr(args,'attn_gate_threshold',0)),
                temporal_transformer=bool(getattr(args,'temporal_transformer',0)),
                tt_layers=int(getattr(args,'tt_layers',2)),
                graph_mix=float(getattr(args,'graph_mix',0.0)),
                dropedge_p=float(getattr(args,'dropedge_p',0.0)),
                quantiles=q_list,
                nb_head=bool(int(getattr(args,'use_nb_head',0))==1),
                zinb=bool(int(getattr(args,'use_zinb',0))==1)
            )
            model.to(device)
            # Optional compile based on flag
            compile_mode = getattr(args,'compile','auto')
            if getattr(torch, 'compile', None) is not None and compile_mode != 'off':
                try:
                    from typing import cast
                    compiled2: nn.Module
                    if compile_mode == 'eager':
                        compiled2 = cast(nn.Module, torch.compile(model, backend='eager'))
                        print('[compile] backend=eager')
                    elif compile_mode == 'inductor':
                        compiled2 = cast(nn.Module, torch.compile(model))
                        print('[compile] backend=inductor')
                    else:  # auto
                        compiled2 = cast(nn.Module, torch.compile(model))
                        print('[compile] auto -> inductor')
                    model = compiled2
                except Exception as _e:
                    print(f'[compile] disabled (fallback): {_e}')
                else:
                    _compiled_model = model
                    class _SafeCompiled(nn.Module):
                        def __init__(self, inner: nn.Module):
                            super().__init__()
                            self.inner = inner
                            self._fallen_back = False
                        def forward(self, *a, **kw):  # type: ignore[override]
                            if self._fallen_back:
                                return self.inner(*a, **kw)
                            try:
                                return self.inner(*a, **kw)
                            except Exception as e:
                                msg = str(e).lower()
                                triggers = ['triton', 'inductor', 'driver', 'backend', 'codegen_kernel']
                                if any(t in msg for t in triggers):
                                    print('[compile] runtime failure; falling back to eager forward for remainder of run')
                                    self.inner = _compiled_model
                                    self._fallen_back = True
                                    return self.inner(*a, **kw)
                                raise
                    model = _SafeCompiled(_compiled_model)
            # Optional DataParallel
            use_dp = (not bool(getattr(args, 'no_dataparallel', False))) \
                     and (torch.cuda.device_count() > 1) and (device.type == 'cuda')
            # Avoid DataParallel with compiled models; DDP is preferred but DP is used here.
            # For simplicity, we disable DP if model is not strictly an nn.Module instance
            if use_dp:
                if not isinstance(model, nn.Module):
                    print("[DP] Skipping DataParallel because model is not a standard nn.Module instance.")
                    use_dp = False
                elif hasattr(model, "_orig_mod"):
                    # Compiled models have _orig_mod but may still cause typing issues
                    print("[DP] Skipping DataParallel for compiled model to avoid type issues.")
                    use_dp = False
                else:
                    print(f"[DP] Using DataParallel on {torch.cuda.device_count()} GPUs")
                    model = torch.nn.DataParallel(model)
            def _get_attr(m, name, default=None):
                return getattr(m.module if isinstance(m, torch.nn.DataParallel) else m, name, default)
            print(args)
            print('The receptive field size is', _get_attr(model, 'receptive_field'))
            core = model.module if isinstance(model, torch.nn.DataParallel) else model
            # Ensure core is a nn.Module before accessing parameters
            if isinstance(core, nn.Module):
                nParams = sum(p.numel() for p in core.parameters())
                print('Number of model parameters is', nParams, flush=True)
            else:
                print('Could not determine number of model parameters.')

            if args.L1Loss:
                criterion = nn.L1Loss(reduction='sum').to(device)
            else:
                criterion = nn.MSELoss(reduction='sum').to(device)
            evaluateL2 = nn.MSELoss(reduction='sum').to(device)
            evaluateL1 = nn.L1Loss(reduction='sum').to(device)
            if args.fused_optim:
                # Native fused AdamW (PyTorch 2.0+ with CUDA 11.6+)
                import torch.optim as _optim
                # Handle compiled models which may have _orig_mod attribute
                if isinstance(core, nn.Module) and hasattr(core, '_orig_mod') and isinstance(core._orig_mod, nn.Module):
                    params = core._orig_mod.parameters()
                elif isinstance(core, nn.Module):
                    params = core.parameters()
                else:
                    raise RuntimeError("Could not access parameters from model (check model type)")
                optim = _optim.AdamW(params, lr=lr,
                                     weight_decay=args.weight_decay, fused=True)
                print("[optim] Using fused AdamW")
            else:
                if isinstance(core, nn.Module):
                    params = core.parameters()
                elif 'model' in locals() and 'model' in globals() and model is not None and isinstance(model, nn.Module):
                    params = model.parameters()
                else:
                    raise RuntimeError("Could not access parameters from model (check model type)")
                optim = Optim(params, args.optim, lr, args.clip,
                              weight_decay=args.weight_decay)
            # NEW: epoch-wise LR scheduler (helps avoid late-epoch collapse/flat preds)
            scheduler = None
            try:
                import torch.optim as _optim
                from typing import cast
                # Unwrap custom Optim wrapper to a real torch optimizer for schedulers
                opt_for_sched = getattr(optim, 'optimizer', None)
                if opt_for_sched is None:
                    opt_for_sched = optim
                if not isinstance(opt_for_sched, _optim.Optimizer):
                    raise TypeError(f"Scheduler requires torch.optim.Optimizer, got {type(opt_for_sched)}")
                opt_for_sched = cast(_optim.Optimizer, opt_for_sched)

                if args.scheduler == 'cosine':
                    scheduler = _optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        opt_for_sched, T_0=args.sched_T0, T_mult=args.sched_Tmult)
                elif args.scheduler == 'onecycle':
                    # needs total steps = epochs * steps_per_epoch; approximate with len(train)/batch
                    steps_per_epoch = max(1, int(math.ceil(Data.train[0].shape[0] / args.batch_size)))
                    max_lr = lr
                    scheduler = _optim.lr_scheduler.OneCycleLR(
                        opt_for_sched, max_lr=max_lr, total_steps=max(steps_per_epoch*args.epochs, 10),
                        pct_start=args.onecycle_pct)
            except Exception as _e:
                print(f"[scheduler] disabled: {_e}")
            # Per-trial early stopping trackers
            best_val_loss_trial = float('inf')
            epochs_without_improvement = 0
            scaler = GradScaler(enabled=bool(args.amp) and device.type=='cuda')
            grad_scaler = scaler if getattr(scaler, 'is_enabled', lambda: True)() else None
            try:
                print('begin training')
                for epoch in range(1, args.epochs + 1):
                    print('Experiment:',(experiment+1))
                    print('Iter:',q)
                    print('epoch:',epoch)
                    print('hp=',[gcn_depth,lr,conv,res,skip,end, k, dropout, dilation_ex, node_dim, prop_alpha, tanh_alpha, layer, epoch])
                    print('best sum=',best_val)
                    print('best rrse=',best_rse)
                    print('best rrae=',best_rae)
                    print('best corr=',best_corr)
                    print('best smape=',best_smape)
                    print('best hps=',best_hp)
                    print('best test rse=',best_test_rse)
                    print('best test corr=',best_test_corr)
                    epoch_start_time = time.time()
                    train_loss = train(
                        Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size,
                        data_scaler=None,
                        alpha=float(args.loss_alpha), beta=float(args.loss_beta),
                        gamma=float(args.loss_gamma), mae_weight=float(args.mae_weight),
                        grad_scaler=grad_scaler,
                        scheduler=scheduler
                    )
                    if getattr(args, "runlog", False):
                        jlog("epoch_train", epoch=epoch, loss=train_loss)
                    val_loss, val_rae, val_corr, val_smape = (
                        evaluate(
                            Data, Data.valid[0], Data.valid[1], model if 'model' in locals() else None, evaluateL2, evaluateL1,
                            args.batch_size, False, mc_runs=args.mc_runs
                        )
                        if Data.valid[0] is not None and Data.valid[1] is not None
                        else (float('inf'), float('inf'), 0.0, float('inf'))
                    )
                    # Export mc_runs for attention auto-gating (mc_runs * batch * nodes threshold)
                    try:
                        os.environ['BMTGNN_MC_RUNS'] = str(int(args.mc_runs))
                    except Exception:
                        pass
                    if getattr(args, "runlog", False):
                        jlog("epoch_valid", epoch=epoch, rrse=val_loss, rae=val_rae, corr=val_corr, smape=val_smape)
                    print(
                        '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f} | valid smape  {:5.4f}'.format(
                            epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr, val_smape), flush=True)
                    # step scheduler once per epoch (or let OneCycle step per batch if integrated)
                    try:
                        if scheduler and args.scheduler == 'cosine':
                            scheduler.step(epoch)
                    except Exception:
                        pass
                    # GPU memory logging per-epoch
                    if (getattr(args, 'log_gpu_mem', False) or getattr(args, 'log_peak_mem', False)) and torch.cuda.is_available():
                        try:
                            torch.cuda.synchronize()
                            alloc = torch.cuda.memory_allocated() / 1024**2
                            reserved = torch.cuda.memory_reserved() / 1024**2
                            peak = torch.cuda.max_memory_allocated() / 1024**2
                            msg = f"[gpu-mem][epoch {epoch}] alloc={alloc:.1f}MB reserved={reserved:.1f}MB peak={peak:.1f}MB"
                            print(msg)
                            if getattr(args, 'runlog', False):
                                jlog('gpu_mem_epoch', epoch=epoch, alloc_mb=round(alloc,2), reserved_mb=round(reserved,2), peak_mb=round(peak,2))
                        except Exception as _mem_e:
                            print(f"[gpu-mem] logging failed: {_mem_e}")
                    # Per-trial early stopping bookkeeping
                    if val_loss < best_val_loss_trial:
                        best_val_loss_trial = val_loss
                        epochs_without_improvement = 0
                        # Global best model tracking (across all trials)
                        sum_loss = val_loss + val_rae - val_corr
                        improved_global = (not math.isnan(val_corr)) and val_loss < best_rse
                        if improved_global:
                            best_hp=[gcn_depth,lr,conv,res,skip,end, k, dropout, dilation_ex, node_dim, prop_alpha, tanh_alpha, layer, epoch]
                            save_dir = os.path.dirname(args.save)
                            if save_dir:
                                os.makedirs(save_dir, exist_ok=True)
                            if 'model' in locals() and model is not None:
                                core_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
                            else:
                                raise RuntimeError("Model is not defined.")
                            # If model is compiled, get original module from _orig_mod
                            if (isinstance(core_to_save, nn.Module) or isinstance(core_to_save, torch.nn.DataParallel)) and hasattr(core_to_save, '_orig_mod') and isinstance(core_to_save._orig_mod, nn.Module):
                                core_to_save = core_to_save._orig_mod
                            if not isinstance(core_to_save, nn.Module):
                                raise RuntimeError("Cannot save model: not an nn.Module instance.")
                            torch.save({'state_dict': core_to_save.state_dict(), 'hparams': best_hp}, args.save)
                            best_val = sum_loss
                            best_rse = val_loss
                            best_rae = val_rae
                            best_corr= val_corr
                            best_smape=val_smape
                            # Optionally produce Validation plots at improvement time
                            if (not getattr(args,'no_plots', False)) and getattr(args,'enable_test_plots', True):
                                if Data.valid[0] is not None and Data.valid[1] is not None:
                                    _ = evaluate(
                                        Data, Data.valid[0], Data.valid[1], model if 'model' in locals() else None, evaluateL2, evaluateL1,
                                        args.batch_size, True, mc_runs=args.mc_runs
                                    )

                            # Update env in case mc_runs changed mid-run
                            try:
                                os.environ['BMTGNN_MC_RUNS'] = str(int(args.mc_runs))
                            except Exception:
                                pass
                            test_acc, test_rae, test_corr, test_smape = (
                                evaluate_sliding_window(
                                    Data, Data.test_window, model, evaluateL2, evaluateL1,
                                    args.seq_in_len, (not getattr(args,'no_plots', False)) and getattr(args,'enable_test_plots', True),
                                    mc_runs=args.mc_runs
                                )
                                if Data.test_window is not None
                                else (float('inf'), float('inf'), 0.0, float('inf'))
                            )

                            # Print validation metrics (RSE, RAE, Corr, sMAPE) for quick inspection
                            try:
                                print(f"[Final Metrics] Test: RSE={test_acc:.6f} RAE={test_rae:.6f} Corr={test_corr:.6f} sMAPE={test_smape:.6f}")
                            except Exception:
                                pass
                            if getattr(args, "runlog", False):
                                jlog("epoch_test", epoch=epoch, rrse=test_acc, rae=test_rae, corr=test_corr, smape=test_smape)
                            print('********************************************************************************************************')
                            print("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}| test smape {:5.4f}".format(test_acc, test_rae, test_corr, test_smape), flush=True)
                            print('********************************************************************************************************')
                            best_test_rse = test_acc
                            best_test_corr = test_corr
                    else:
                        epochs_without_improvement += 1
                        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
                            print(f"[early_stop] Trial {q} stopping early at epoch {epoch} (patience={args.early_stop_patience}).")
                            break
            except KeyboardInterrupt:
                print('-' * 89)
                print('Exiting from training early')

        print('best val loss=',best_val)
        print('best hps=',best_hp)
        # save best hp to disk (ensure output dir exists)
        os.makedirs(os.path.join('model', 'Bayesian'), exist_ok=True)
        with open(os.path.join('model', 'Bayesian', 'hp.txt'), "w") as f:
            f.write(str(best_hp))

    # If using cached HPs and there was no search this run, attempt to load hp.txt (or rely on ckpt)
    if args.use_cached_hp and not best_hp:
        import ast
        hp_path = os.path.join('model','Bayesian','hp.txt')
        if os.path.exists(hp_path):
            try:
                best_hp = ast.literal_eval(open(hp_path).read())
                print('[use_cached_hp] Loaded hp.txt:', best_hp)
            except Exception as e:
                print(f'[use_cached_hp] Failed to parse hp.txt: {e}')
        else:
            print('[use_cached_hp] hp.txt not found; will rely on checkpoint hparams if present.')

    # Load the best saved model safely: load checkpoint to CPU first to
    # avoid large transient GPU allocations, then move model to the target device.
    ckpt = torch.load(args.save, map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        loaded_hp = ckpt.get('hparams', best_hp)
        # Treat empty/invalid hparams as missing and fall back to args/inference.
        if not isinstance(loaded_hp, (list, tuple)) or len(loaded_hp) == 0:
            loaded_hp = None

        # Fallback: checkpoint missing hparams (older ckpt) -> synthesize from current args
        if not loaded_hp:
            # If reproducing a metrics JSON run, you may want to keep CLI args authoritative.
            # Setting BMTGNN_SKIP_CKPT_INFER=1 disables shape-based inference (and avoids
            # overriding args.seq_in_len).
            if os.environ.get('BMTGNN_SKIP_CKPT_INFER', '0') == '1':
                loaded_hp = [
                    getattr(args,'gcn_depth', 2),
                    getattr(args,'lr', getattr(args,'learning_rate', 0.001)),
                    getattr(args,'conv_channels', 32),
                    getattr(args,'residual_channels', 32),
                    getattr(args,'skip_channels', 64),
                    getattr(args,'end_channels', 128),
                    getattr(args,'subgraph_size', 20),
                    getattr(args,'dropout', 0.3),
                    getattr(args,'dilation_exponential', 1),
                    getattr(args,'node_dim', 40),
                    getattr(args,'propalpha', 0.05),
                    getattr(args,'tanhalpha', 3),
                    getattr(args,'layers', 3),
                    -1
                ]
                print('[ckpt-load] checkpoint has no hparams; using CLI args because BMTGNN_SKIP_CKPT_INFER=1')
            else:
                # Try to infer common hyperparameters from the checkpoint state_dict shapes
                try:
                    state = ckpt.get('state_dict', {}) if isinstance(ckpt, dict) else {}
                    # defaults
                    conv_c = getattr(args,'conv_channels', 32)
                    res_c = getattr(args,'residual_channels', 32)
                    skip_c = getattr(args,'skip_channels', 64)
                    end_c = getattr(args,'end_channels', 128)
                    layers_c = getattr(args,'layers', 3)
                    seq_candidates = set()
                    # Inspect keys for typical naming conventions
                    for k,v in list(state.items()):
                        if not hasattr(v, 'shape'):
                            continue
                        s = tuple(v.shape)
                        # filter_convs.{layer}.tconv.{i}.weight -> (out, in, 1, t)
                        if k.startswith('filter_convs.') and k.endswith('.weight'):
                            parts = k.split('.')
                            try:
                                lidx = int(parts[1])
                                layers_c = max(layers_c, lidx+1)
                            except Exception:
                                pass
                            if len(s) >= 2:
                                conv_c = s[1]
                            if len(s) >= 4:
                                seq_candidates.add(s[3])
                        # gate_convs -> residual channels
                        if k.startswith('gate_convs.') and k.endswith('.weight') and len(s) >= 2:
                            res_c = s[1]
                            try:
                                lidx = int(k.split('.')[1]); layers_c = max(layers_c, lidx+1)
                            except Exception:
                                pass
                        # skip convs
                        if (k.startswith('skip_convs.') or k.startswith('skip0') or k.startswith('skipE')) and k.endswith('.weight'):
                            if len(s) >= 1:
                                skip_c = s[0]
                            if len(s) >= 4:
                                seq_candidates.add(s[3])
                        # end convs
                        if ('end_conv' in k or k.startswith('end_conv')) and k.endswith('.weight'):
                            if len(s) >= 1:
                                end_c = s[0]
                    # choose seq_in_len as max observed temporal kernel or fallback to args
                    seq_in_len_infer = max(seq_candidates) if seq_candidates else getattr(args,'seq_in_len', 120)
                    # assemble loaded_hp in same order used elsewhere
                    loaded_hp = [
                        getattr(args,'gcn_depth', 2),
                        getattr(args,'lr', getattr(args,'learning_rate', 0.001)),
                        int(conv_c),
                        int(res_c),
                        int(skip_c),
                        int(end_c),
                        getattr(args,'subgraph_size', 20),
                        getattr(args,'dropout', 0.3),
                        getattr(args,'dilation_exponential', 1),
                        getattr(args,'node_dim', 40),
                        getattr(args,'propalpha', 0.05),
                        getattr(args,'tanhalpha', 3),
                        int(layers_c),
                        -1
                    ]
                    # override args.seq_in_len so model is constructed to match checkpoint temporal dims
                    try:
                        args.seq_in_len = int(seq_in_len_infer)
                    except Exception:
                        pass
                    print('[info] Inferred hparams from checkpoint and set args.seq_in_len to', getattr(args,'seq_in_len', None))
                except Exception as _e:
                    # fallback to conservative defaults if inference fails
                    loaded_hp = [
                        getattr(args,'gcn_depth', 2),
                        getattr(args,'lr', getattr(args,'learning_rate', 0.001)),
                        getattr(args,'conv_channels', 32),
                        getattr(args,'residual_channels', 32),
                        getattr(args,'skip_channels', 64),
                        getattr(args,'end_channels', 128),
                        getattr(args,'subgraph_size', 20),
                        getattr(args,'dropout', 0.3),
                        getattr(args,'dilation_exponential', 1),
                        getattr(args,'node_dim', 40),
                        getattr(args,'propalpha', 0.05),
                        getattr(args,'tanhalpha', 3),
                        getattr(args,'layers', 3),
                        -1  # epoch placeholder
                    ]
                    print('[warning] Checkpoint has no hparams and inference failed; using args-derived fallback hyperparameters.', _e)

                # With --use_cached_hp we prefer checkpoint/hp.txt hyperparameters, but if they are missing
                # (hp.txt is [] in this repo) we still proceed using the derived hparams above.
                if args.use_cached_hp and (not loaded_hp or len(loaded_hp) < 13):
                    print('[use_cached_hp][warn] No valid hparams found in checkpoint/hp.txt; proceeding with derived args-based hparams.')
        # If hparams exist but are shorter than expected, pad conservatively
        if len(loaded_hp) < 13:
            print(f'[warning] Loaded hparams length {len(loaded_hp)} < 13; padding with current args.')
            pad_map = [
                getattr(args,'gcn_depth', 2),
                getattr(args,'lr', getattr(args,'learning_rate', 0.001)),
                getattr(args,'conv_channels', 32),
                getattr(args,'residual_channels', 32),
                getattr(args,'skip_channels', 64),
                getattr(args,'end_channels', 128),
                getattr(args,'subgraph_size', 20),
                getattr(args,'dropout', 0.3),
                getattr(args,'dilation_exponential', 1),
                getattr(args,'node_dim', 40),
                getattr(args,'propalpha', 0.05),
                getattr(args,'tanhalpha', 3),
                getattr(args,'layers', 3),
                -1
            ]
            for i in range(len(loaded_hp), 14):
                loaded_hp.append(pad_map[i])
        # Reconstruct model on CPU (pass a CPU device to the constructor)
        cpu_dev = torch.device('cpu')
        try:
            in_dim_use = getattr(Data, 'in_dim', args.in_dim)
        except NameError:
            in_dim_use = args.in_dim
        model = gtnet(
            args.gcn_true, args.buildA_true, loaded_hp[0], int(Data.m),
            cpu_dev, Data.adj, dropout=loaded_hp[7], subgraph_size=loaded_hp[6],
            node_dim=loaded_hp[9], dilation_exponential=loaded_hp[8],
            conv_channels=loaded_hp[2], residual_channels=loaded_hp[3],
            skip_channels=loaded_hp[4], end_channels=loaded_hp[5],
            seq_length=args.seq_in_len, in_dim=in_dim_use, out_dim=args.seq_out_len,
            layers=loaded_hp[12], propalpha=loaded_hp[10], tanhalpha=loaded_hp[11],
            layer_norm_affline=False,
            temporal_attn=getattr(args,'temporal_attn', False), attn_dim=getattr(args,'attn_dim',64),
            attn_heads=getattr(args,'attn_heads',2), attn_dropout=getattr(args,'attn_dropout',0.1),
            attn_window=getattr(args,'attn_window',0), attn_math_mode=getattr(args,'attn_math_mode', False),
            attn_bn_chunk=int(getattr(args,'attn_bn_chunk',0)),
            attn_gate_threshold=int(getattr(args,'attn_gate_threshold',0)),
            temporal_transformer=bool(getattr(args,'temporal_transformer',0)),
            tt_layers=int(getattr(args,'tt_layers',2)),
            graph_mix=float(getattr(args,'graph_mix',0.0)),
            dropedge_p=float(getattr(args,'dropedge_p',0.0)),
            quantiles=q_list,
            nb_head=bool(int(getattr(args,'use_nb_head',0))==1),
            zinb=bool(int(getattr(args,'use_zinb',0))==1)
        )
        # Prepare state_dict: handle possible DataParallel 'module.' prefixes
        state_dict = ckpt['state_dict']
        model_keys = set(model.state_dict().keys())
        if any(k.startswith('module.') for k in state_dict.keys()) and not any(k.startswith('module.') for k in model_keys):
            # strip leading 'module.' from checkpoint keys
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

        # Sanitize checkpoint by only keeping keys that both exist in the current
        # model and have matching shapes. This allows loading checkpoints saved
        # with different hyperparameters without raising a RuntimeError.
        model_state = model.state_dict()
        filtered_state = {}
        skipped = []
        for k in state_dict.keys():
            if k in model_state and model_state[k].shape == state_dict[k].shape:
                filtered_state[k] = state_dict[k]
            else:
                skipped.append(k)

        # Load filtered state dict non-strictly to allow missing keys
        res = model.load_state_dict(filtered_state, strict=False)
        try:
            missing = getattr(res, 'missing_keys', None)
            unexpected = getattr(res, 'unexpected_keys', None)
        except Exception:
            missing = None
            unexpected = None
        print(f"Loaded checkpoint with strict=False. missing_keys={missing}, unexpected_keys={unexpected}")
        loaded_strict = False

        # Detailed diagnostic comparing checkpoint keys vs model keys and shapes
        try:
            model_sd = model.state_dict()
            ck_keys = set(state_dict.keys())
            model_keys = set(model_sd.keys())
            missing_in_ckpt = sorted([k for k in model_keys if k not in ck_keys])
            unexpected_in_ckpt = sorted([k for k in ck_keys if k not in model_keys])
            shape_mismatches = []
            for k in sorted(ck_keys & model_keys):
                try:
                    ck_shape = tuple(state_dict[k].shape)
                    model_shape = tuple(model_sd[k].shape)
                except Exception:
                    continue
                if ck_shape != model_shape:
                    shape_mismatches.append((k, ck_shape, model_shape))

            if missing_in_ckpt or unexpected_in_ckpt or shape_mismatches:
                print('[Checkpoint diagnostic] summary:')
                print(f'  model params total: {len(model_keys)}, ckpt params: {len(ck_keys)}')
                print(f'  missing in ckpt: {len(missing_in_ckpt)}; unexpected in ckpt: {len(unexpected_in_ckpt)}; shape mismatches: {len(shape_mismatches)}')
                if missing_in_ckpt:
                    print('  sample missing (first 20):', missing_in_ckpt[:20])
                if unexpected_in_ckpt:
                    print('  sample unexpected (first 20):', unexpected_in_ckpt[:20])
                if shape_mismatches:
                    print('  sample shape mismatches (first 20):')
                    for item in shape_mismatches[:20]:
                        print('    ', item)
        except Exception as diag_e:
            print('[Checkpoint diagnostic] failed to compute detailed comparison:', diag_e)
    elif isinstance(ckpt, nn.Module):
        # legacy: checkpoint contains the full module; keep on CPU for now
        model = ckpt
    else:
        raise RuntimeError(f"Unsupported checkpoint format in {args.save}")

    # Free the checkpoint object and clear caches before moving model to GPU
    del ckpt
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    # Now move the model to the actual device (may be 'cpu' or 'cuda:0')
    model.to(device)

    # Reconstruct Data and evaluation losses to ensure they are in scope here
    Data = DataLoaderS(args.data, float(getattr(args,'train_ratio',0.6)), float(getattr(args,'valid_ratio',0.2)), device, args.horizon, args.seq_in_len, args.normalize, args.seq_out_len,
                       chronological=bool(getattr(args,'chronological_split', False)),
                       start_year=args.start_year, steps_per_year=args.steps_per_year,
                       train_end_year=args.train_end_year, valid_end_year=args.valid_end_year, test_end_year=args.test_end_year,
                       dual_channel=args.dual_channel,
                       pct_clip=float(getattr(args, 'pct_clip', 0.0)),
                       y_transform=getattr(args,'y_transform', None),
                       exclude_names=(args.exclude_names if getattr(args, 'exclude_names', '') != '' else None))
    # Load scaler and overwrite Data.mu/std just to be explicit and avoid drift
    try:
        s = torch.load(os.path.join(os.path.dirname(args.save), 'y_scaler.pt'), map_location=device)
        Data.mu = s['mean'].to(device)
        Data.std = s['std'].to(device).clamp_min(1e-6)
    except Exception as _e:
        print(f"[scaler_load] warning: {_e}")
    evaluateL2 = nn.MSELoss(reduction='sum').to(device)  # MSE
    evaluateL1 = nn.L1Loss(reduction='sum').to(device)  # MAE


    do_plots = not bool(getattr(args,'no_plots', False))
    # --- Calibration state dict ---
    calib_state = {}

    # Validation evaluation (with optional calibration)
    vtest_acc, vtest_rae, vtest_corr, vtest_smape = (
        evaluate(
            Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
            args.batch_size, do_plots, mc_runs=args.mc_runs
        )
        if Data.valid[0] is not None and Data.valid[1] is not None
        else (float('inf'), float('inf'), 0.0, float('inf'))
    )

    # Print validation metrics to console
    try:
        print(f"[Final Metrics] Validation: RSE={vtest_acc:.6f} RAE={vtest_rae:.6f} Corr={vtest_corr:.6f} sMAPE={vtest_smape:.6f}")
    except Exception:
        pass

    # Persist metrics even when --no_plots is set (Optuna depends on this file).
    try:
        _save_metrics(
            args,
            'Validation',
            {
                'RSE': float(vtest_acc),
                'RAE': float(vtest_rae),
                'Corr': float(vtest_corr),
                'sMAPE': float(vtest_smape),
            },
            extras={
                'calibration': getattr(args, 'calibration', 'none'),
            },
        )
    except Exception as _e:
        jlog("warn_save_metrics", split="Validation", error=str(_e)[:160])

    # If you have access to pred_full and true_full here, apply calibration as follows:
    # (If not, ensure this logic is in your evaluate function after pred_full/true_full are computed)
    # Example (pseudo):
    # if args.calibration in ('val', 'both'):
    #     a, b = _fit_linear_calibration(pred_full, true_full)
    #     pred_full = a * pred_full + b
    #     calib_state['lin_calib_params'] = (float(a), float(b))
    # else:
    #     calib_state.pop('lin_calib_params', None)

    # Test evaluation (with optional calibration application)
    test_acc, test_rae, test_corr, test_smape = (
        (os.environ.__setitem__('BMTGNN_MC_RUNS', str(int(args.mc_runs))) or True) and
        evaluate_sliding_window(
            Data, Data.test_window, model, evaluateL2, evaluateL1,
            args.seq_in_len, do_plots, mc_runs=args.mc_runs
        )
        if Data.test_window is not None
        else (float('inf'), float('inf'), 0.0, float('inf'))
    )
    # If you have access to pred_full here, apply test calibration as follows:
    # if args.calibration in ('test', 'both') and 'lin_calib_params' in calib_state:
    #     a, b = calib_state['lin_calib_params']
    #     pred_full = a * pred_full + b

    print('********************************************************************************************************')    
    print("final test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f} | test smape {:5.4f}".format(test_acc, test_rae, test_corr, test_smape))
    print('********************************************************************************************************')

    try:
        _save_metrics(
            args,
            'Testing',
            {
                'RSE': float(test_acc),
                'RAE': float(test_rae),
                'Corr': float(test_corr),
                'sMAPE': float(test_smape),
            },
            extras={
                'calibration': getattr(args, 'calibration', 'none'),
            },
        )
    except Exception as _e:
        jlog("warn_save_metrics", split="Testing", error=str(_e)[:160])
    # Optional: full-series MC variance + precise checkpoint IVW ensemble
    try:
        run_full_series_ensemble(model, Data, device, args)
    except Exception as _e:
        print(f"[full-series][warn] {_e}")
    return vtest_acc, vtest_rae, vtest_corr, vtest_smape, test_acc, test_rae, test_corr, test_smape

if __name__ == "__main__":
    # Handle quick debug/inspection flags before launching full experiments.
    if getattr(args, 'debug_rf', False) or getattr(args, 'ckpt_to_compare', ''):
        # Build transient model using CLI args (use args.num_nodes or default 105)
        # For transient debug model, prefer CLI arg for in_dim (Data may not exist yet)
        in_dim_use = args.in_dim
        tmp_model = gtnet(
            args.gcn_true, args.buildA_true, args.gcn_depth, int(getattr(args, 'num_nodes', 105)),
            device, None, dropout=args.dropout, subgraph_size=args.subgraph_size,
            node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
            conv_channels=args.conv_channels, residual_channels=args.residual_channels,
            skip_channels=args.skip_channels, end_channels=args.end_channels,
            seq_length=args.seq_in_len, in_dim=in_dim_use, out_dim=args.seq_out_len,
            layers=args.layers, propalpha=getattr(args, 'propalpha', 0.05),
            tanhalpha=getattr(args, 'tanhalpha', 3), layer_norm_affline=False,
            temporal_attn=getattr(args,'temporal_attn', False), attn_dim=getattr(args,'attn_dim',64),
            attn_heads=getattr(args,'attn_heads',2), attn_dropout=getattr(args,'attn_dropout',0.1),
            attn_window=getattr(args,'attn_window',0), attn_math_mode=getattr(args,'attn_math_mode', False),
            attn_bn_chunk=int(getattr(args,'attn_bn_chunk',0)),
            attn_gate_threshold=int(getattr(args,'attn_gate_threshold',0)),
            temporal_transformer=bool(getattr(args,'temporal_transformer',0)),
            tt_layers=int(getattr(args,'tt_layers',2)),
            graph_mix=float(getattr(args,'graph_mix',0.0)),
            dropedge_p=float(getattr(args,'dropedge_p',0.0)),
            quantiles=q_list,
            nb_head=bool(int(getattr(args,'use_nb_head',0))==1),
            zinb=bool(int(getattr(args,'use_zinb',0))==1)
        )
        if getattr(args, 'debug_rf', False):
            core = tmp_model
            print('Receptive field (model.receptive_field):', getattr(core, 'receptive_field', None))
            print('seq_length:', args.seq_in_len)
            # Some model variants may define filter_convs / skip_convs attributes as None; guard iteration.
            for i, conv in enumerate(getattr(core, 'filter_convs', []) or []):
                try:
                    sizes = [c.kernel_size for c in conv.tconv] if hasattr(conv, 'tconv') else 'n/a'
                except Exception:
                    sizes = 'n/a'
                print(f'filter_convs[{i}] kernel sizes: {sizes}')
            for i, conv in enumerate(getattr(core, 'skip_convs', []) or []):
                try:
                    print(f'skip_convs[{i}] kernel_size = {conv.kernel_size}')
                except Exception:
                    print(f'skip_convs[{i}] kernel_size = n/a')
            try:
                print('skip0.kernel_size =', core.skip0.kernel_size)
                print('skipE.kernel_size =', core.skipE.kernel_size)
            except Exception:
                pass
            print('\n-- debug_rf done --')
        if getattr(args, 'ckpt_to_compare', ''):
            out_csv = getattr(args, 'ckpt_compare_csv', '') or 'ckpt_compare.csv'
            dump_ckpt_vs_model_csv(getattr(args, 'ckpt_to_compare'), tmp_model, out_csv)
        sys.exit(0)

    vacc = []
    vrae = []
    vcorr = []
    vsmape=[]
    acc = []
    rae = []
    corr = []
    smape=[]
    for i in range(1):
        val_acc, val_rae, val_corr, val_smape, test_acc, test_rae, test_corr, test_smape = main(i)
        vacc.append(val_acc)
        vrae.append(val_rae)
        vcorr.append(val_corr)
        vsmape.append(val_smape)
        acc.append(test_acc)
        rae.append(test_rae)
        corr.append(test_corr)
        smape.append(test_smape)
        # Emit per-trial JSON for external parsers
        try:
            import json
            trial_res = {
                'trial': int(i),
                'valid_rse': float(val_acc),
                'valid_rae': float(val_rae),
                'test_rse': float(test_acc),
                'test_rae': float(test_rae),
                'seed': int(getattr(args, 'seed', -1)),
                'subgraph_k': int(getattr(args, 'subgraph_size', -1)),
            }
            print('RESULT_JSON:' + json.dumps(trial_res))
        except Exception:
            pass
        if getattr(args, "runlog", False):
            try:
                jlog("run_finished",
                     best_val_rrse=float(val_acc) if 'val_acc' in locals() else None,
                     best_test_rrse=float(test_acc) if 'test_acc' in locals() else None)
            except Exception:
                jlog("run_finished", best_val_rrse=None)
    print('\n\n')
    print('1 run average')
    print('\n\n')
    print("valid\trse\trae")
    print("mean\t{:5.4f}\t{:5.4f}".format(np.mean(vacc), np.mean(vrae)))
    print("std\t{:5.4f}\t{:5.4f}".format(np.std(vacc), np.std(vrae)))
    print('\n\n')
    print("test\trse\trae")
    print("mean\t{:5.4f}\t{:5.4f}".format(np.mean(acc), np.mean(rae)))
    print("std\t{:5.4f}\t{:5.4f}".format(np.std(acc), np.std(rae)))

    # Machine-parseable JSON summary for external validators
    try:
        res = {
            'valid_rse_mean': float(np.mean(vacc)),
            'valid_rse_std': float(np.std(vacc)),
            'valid_rae_mean': float(np.mean(vrae)),
            'test_rse_mean': float(np.mean(acc)),
            'test_rae_mean': float(np.mean(rae)),
            'seed': int(getattr(args, 'seed', -1)),
            'subgraph_k': int(getattr(args, 'subgraph_size', -1)),
        }
        import json
        print('RESULT_JSON:' + json.dumps(res))
    except Exception:
        pass


def run_experiment(config: dict) -> dict:
    """Programmatic single-run experiment. Returns a dict with validation and test metrics.

    config keys (optional): data, graph, device, epochs, batch_size, seed, subgraph_size,
    preset, lr, conv_channels, residual_channels, skip_channels, end_channels, gcn_depth
    """
    # merge config into args-like object
    c = argparse.Namespace(**{k: v for k, v in config.items()})
    # defaults from global args
    for name in vars(args):
        if not hasattr(c, name):
            setattr(c, name, getattr(args, name))

    # set deterministic seed
    seed_to_use = int(getattr(c, 'seed', fixed_seed))
    set_random_seed(seed_to_use, getattr(c, 'cudnn_benchmark', False))

    # Build Data
    Data = DataLoaderS(getattr(c, 'data', args.data), float(getattr(args,'train_ratio',0.6)), float(getattr(args,'valid_ratio',0.2)), getattr(c, 'device', args.device), getattr(c, 'horizon', args.horizon), getattr(c, 'seq_in_len', args.seq_in_len), getattr(c, 'normalize', args.normalize), getattr(c, 'seq_out_len', args.seq_out_len),
                       chronological=bool(getattr(args,'chronological_split', False)),
                       start_year=args.start_year, steps_per_year=args.steps_per_year,
                       train_end_year=args.train_end_year, valid_end_year=args.valid_end_year, test_end_year=args.test_end_year,
                       dual_channel=getattr(c, 'dual_channel', args.dual_channel),
                       pct_clip=float(getattr(c, 'pct_clip', getattr(args, 'pct_clip', 0.0))),
                       y_transform=getattr(c, 'y_transform', getattr(args, 'y_transform', None)),
                       exclude_names=(getattr(c, 'exclude_names', getattr(args, 'exclude_names', '')) if getattr(c, 'exclude_names', '') != '' else None))

    # normalize adjacency based on --graph_normalize
    try:
        if isinstance(Data.adj, torch.Tensor):
            A_np = Data.adj.detach().cpu().numpy().astype(np.float32)
            mode = getattr(args, 'graph_normalize', 'none')
            if mode == 'sym':
                from src.util import sym_adj
                A_np = sym_adj(A_np)
            elif mode == 'square':
                # simple A@A then row normalize
                A_sq = A_np @ A_np
                row_sum = A_sq.sum(axis=1, keepdims=True) + 1e-6
                A_np = A_sq / row_sum
            elif mode == 'row':
                row_sum = A_np.sum(axis=1, keepdims=True) + 1e-6
                A_np = A_np / row_sum
            Data.adj = torch.from_numpy(A_np).to(getattr(c, 'device', device))
    except Exception as _e:
        print(f"[graph_normalize] warning: {_e}")

    # build model with provided HPs
    hp = {
        'gcn_depth': int(getattr(c, 'gcn_depth', args.gcn_depth)),
        'subgraph_size': int(getattr(c, 'subgraph_size', args.subgraph_size)),
        'node_dim': int(getattr(c, 'node_dim', args.node_dim)),
        'dilation_exponential': int(getattr(c, 'dilation_exponential', args.dilation_exponential)),
        'conv_channels': int(getattr(c, 'conv_channels', args.conv_channels)),
        'residual_channels': int(getattr(c, 'residual_channels', args.residual_channels)),
        'skip_channels': int(getattr(c, 'skip_channels', args.skip_channels)),
        'end_channels': int(getattr(c, 'end_channels', args.end_channels)),
        'seq_length': int(getattr(c, 'seq_in_len', args.seq_in_len)),
        'out_dim': int(getattr(c, 'seq_out_len', args.seq_out_len)),
    }

    in_dim_use = getattr(Data, 'in_dim', getattr(c, 'in_dim', args.in_dim))
    # allow a simple smaller model override via env var BMTGNN_SIMPLE_MODEL=1
    layers_use = int(getattr(c, 'layers', args.layers))
    tt_layers_use = int(getattr(args,'tt_layers',2))
    attn_heads_use = int(getattr(args,'attn_heads',2))
    attn_dim_use = int(getattr(args,'attn_dim',64))
    dropout_use = float(getattr(c, 'dropout', args.dropout))
    if os.environ.get('BMTGNN_SIMPLE_MODEL', '0') == '1':
        print('[BMTGNN_SIMPLE_MODEL] enabling simple model override')
        layers_use = 2
        tt_layers_use = 1
        attn_heads_use = 1
        attn_dim_use = 32
        dropout_use = max(0.05, dropout_use)
        # shrink conv/residual/end channels if present in hp
        try:
            hp['conv_channels'] = min(64, int(hp.get('conv_channels', 64)))
            hp['residual_channels'] = min(64, int(hp.get('residual_channels', 64)))
            hp['end_channels'] = min(64, int(hp.get('end_channels', 64)))
        except Exception:
            pass
    model = gtnet(getattr(c, 'gcn_true', args.gcn_true), getattr(c, 'buildA_true', args.buildA_true), hp['gcn_depth'], int(Data.m), getattr(c, 'device', device), Data.adj, dropout=dropout_use, subgraph_size=hp['subgraph_size'], node_dim=hp['node_dim'], dilation_exponential=hp['dilation_exponential'], conv_channels=hp['conv_channels'], residual_channels=hp['residual_channels'], skip_channels=hp['skip_channels'], end_channels=hp['end_channels'], seq_length=hp['seq_in_len'], in_dim=in_dim_use, out_dim=hp['out_dim'], layers=layers_use, propalpha=float(int(getattr(c, 'propalpha', args.propalpha)) if isinstance(getattr(c, 'propalpha', args.propalpha), bool) else float(getattr(c, 'propalpha', args.propalpha))), tanhalpha=int(getattr(c, 'tanhalpha', args.tanhalpha)), layer_norm_affline=False, temporal_attn=getattr(args,'temporal_attn', False), attn_dim=attn_dim_use, attn_heads=attn_heads_use, attn_dropout=getattr(args,'attn_dropout',0.1), attn_window=getattr(args,'attn_window',0), attn_math_mode=getattr(args,'attn_math_mode', False), attn_bn_chunk=int(getattr(args,'attn_bn_chunk',0)), attn_gate_threshold=int(getattr(args,'attn_gate_threshold',0)), temporal_transformer=bool(getattr(args,'temporal_transformer',0)), tt_layers=tt_layers_use, graph_mix=float(getattr(args,'graph_mix',0.0)), dropedge_p=float(getattr(args,'dropedge_p',0.0)), quantiles=q_list, nb_head=bool(int(getattr(args,'use_nb_head',0))==1), zinb=bool(int(getattr(args,'use_zinb',0))==1))
    model.to(getattr(c, 'device', device))

    # optimizer
    optim = torch.optim.Adam(model.parameters(), lr=float(getattr(c, 'lr', args.lr)), weight_decay=float(getattr(c, 'weight_decay', args.weight_decay)))
    scaler = torch.cuda.amp.GradScaler() if (getattr(c, 'amp', False) and torch.cuda.is_available()) else None

    # training loop (simple, deterministic): early stopping on validation rrse
    best_val = float('inf')
    patience = int(getattr(c, 'early_stopping_patience', 3))
    wait = 0
    v_rse = float('inf')
    v_rae = 0.0
    # Unpack possibly list/tuple datasets for clarity
    trX, trY = Data.train[0], Data.train[1]
    vaX, vaY = Data.valid[0], Data.valid[1]
    teX, teY = Data.test[0], Data.test[1]
    # Defensive: ensure splits are tensors and not None (helps static analyzers complaining about None iteration)
    for name, tensor in [('trX', trX), ('trY', trY), ('vaX', vaX), ('vaY', vaY), ('teX', teX), ('teY', teY)]:
        if tensor is None:
            raise ValueError(f"Dataset tensor {name} is None; check DataLoaderS splitting logic.")
    for ep in range(int(getattr(c, 'epochs', 3))):
        train(Data, trX, trY, model, None, optim, int(getattr(c, 'batch_size', args.batch_size)), data_scaler=None)
        evaluateL2 = nn.MSELoss(reduction='sum').to(getattr(c, 'device', device))
        evaluateL1 = nn.L1Loss(reduction='sum').to(getattr(c, 'device', device))
        v_rse, v_rae, v_corr, v_smape = (
            evaluate(Data, vaX, vaY, model, evaluateL2, evaluateL1, int(getattr(c, 'batch_size', args.batch_size)), False, mc_runs=1)
            if vaX is not None and vaY is not None
            else (float('inf'), float('inf'), 0.0, float('inf'))
        )
        if v_rse < best_val - 1e-6:
            best_val = v_rse
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # final evaluation on test
    evaluateL2 = nn.MSELoss(reduction='sum').to(getattr(c, 'device', device))
    evaluateL1 = nn.L1Loss(reduction='sum').to(getattr(c, 'device', device))
    t_rse, t_rae, t_corr, t_smape = (
        evaluate(Data, teX, teY, model, evaluateL2, evaluateL1, int(getattr(c, 'batch_size', args.batch_size)), False, mc_runs=1)
        if teX is not None and teY is not None
        else (float('inf'), float('inf'), 0.0, float('inf'))
    )

    return {
        'valid_rse': best_val,
        'valid_rae': float(v_rae),
        'test_rse': float(t_rse),
        'test_rae': float(t_rae),
        'seed': int(seed_to_use),
        'subgraph_k': int(hp['subgraph_size']),
    }
