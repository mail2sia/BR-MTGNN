"""
Main runner: train and test (uses src/*)

This script supports two common workflows:
    - start a fresh training run from CLI flags
    - reproduce a prior run by loading embedded args from a metrics JSON via --metrics_json

For a strong default recipe targeting sparse/heavy-tailed series, use --strong_rmdpt.

OPTIMIZED DEFAULTS (v0.2.0):
    All default parameters are now configured for stable, high-quality forecasting based on
    the 7-fix optimized configuration (layers=5, seq_in_len=24, MAE-dominant loss, per-node
    normalization, pct dual-channel, temporal attention enabled). These defaults match the
    run_safe_train.sh recipe.

    DATA SPLITTING STRATEGY:
    - Historical data (2004-2025): Split via train_ratio/valid_ratio for model development
      * 60% train (~2004-2016), 20% val (~2017-2020), 20% test (~2021-2025)
    - Future forecasting (2026-2028): Use scripts/forecast.py with trained model

    Key optimizations:

    - Receptive field: layers=5, dilation=2 (RF: 187 steps)
    - Loss: MAE-dominant (alpha=0.5, beta=0.0, gamma=0.5, mae_weight=10.0)
    - Normalization: per-node z-score (normalize=3)
    - Input: scale-invariant pct channel (dual_channel='pct', pct_clip=3.0)
    - Sequence: 24-month input, 36-month output
    - Architecture: conv=96, residual=96, skip=192, end=384
    - Training: lr=0.0005, batch=8, clip=30.0, dropout=0.0
    - Temporal: attn enabled, 4 heads, dim=96, transformer=1, tt_layers=2
    - Graph: mix=0.5, dropedge=0.08
    - Scheduler: cosine (T0=15, Tmult=2), early_stop=80

Quick start:
    # Training from scratch with optimized defaults
    python scripts/grid_tuning.py --config configs/grid_ultra_hugging.json


CLI flags take precedence over metrics JSON and --strong_rmdpt defaults.
"""

import json
import math
import os
import shutil
import subprocess
import sys
import time
from random import randrange
from typing import cast

import numpy as np
import torch
import torch.nn as nn

# Ensure project root (parent of scripts/) is on sys.path so 'src' is importable
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.cleanup import cleanup_checkpoints_and_cache
from src.cli_args import build_parser
from src.losses import weighted_huber_horizon_loss
from src.net import gtnet
from src.trainer import Optim, run_trainer_path
from src.train_test_ckpt import (
    dump_ckpt_vs_model_csv,
    _infer_ckpt_num_nodes,
    _infer_scaler_num_nodes,
)
from src.train_test_defaults import (
    _apply_hugging_mode_defaults,
    _apply_strong_rmdpt_defaults,
    _flag_was_set,
    _get_cli_keys,
    _maybe_generate_aggressive_smoothed_data,
)
from src.train_test_eval import (
    evaluate_impl,
    evaluate_sliding_window_impl,
    generate_epoch_validation_test_plots_impl,
)
from src.train_test_train import train_impl
from src.util import (
    AnalysisLogger,
    DataLoaderS,
    jlog,
    prepare_graph_and_subgraph,
    resolve_split_and_build_data,
    set_random_seed,
    start_runlog,
    sym_adj,
    to_model_layout,
)

import gc  # noqa: E402
from pathlib import Path  # noqa: E402

from torch.amp.grad_scaler import GradScaler  # noqa: E402

_np = np

# Process title customization
try:
    import setproctitle
except ImportError:
    setproctitle = None

# Global logger instance, configured in main()
ANALYSIS_LOGGER = None
_CALIB_AB = None
_CONF_Z = 1.96  # global 95% z, may be recalibrated on validation if --conf_calibrate
_CONFORMAL_Q = None  # per-node conformal offset learned on Validation (original units)

# last seen Validation CI ratio (median over nodes) to drive auto-dropout
_LAST_VAL_CI_RATIO = None
# ---

EVAL_STATE = {
    "calib_ab": _CALIB_AB,
    "conf_z": _CONF_Z,
    "conformal_q": _CONFORMAL_Q,
    "last_val_ci_ratio": _LAST_VAL_CI_RATIO,
    "analysis_logger": ANALYSIS_LOGGER,
}


def _sync_eval_state() -> None:
    global _CALIB_AB
    global _CONF_Z
    global _CONFORMAL_Q
    global _LAST_VAL_CI_RATIO
    _CALIB_AB = EVAL_STATE.get("calib_ab")
    _CONF_Z = float(EVAL_STATE.get("conf_z", _CONF_Z))
    _CONFORMAL_Q = EVAL_STATE.get("conformal_q")
    _LAST_VAL_CI_RATIO = EVAL_STATE.get("last_val_ci_ratio")


def _print_gpu_pid_titles() -> None:
    if os.name == "nt":
        print("[gpu-titles] Not supported on Windows.")
        return
    if not shutil.which("nvidia-smi"):
        print("[gpu-titles] nvidia-smi not found in PATH.")
        return

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader",
            ],
            text=True,
        )
    except Exception as e:
        print(f"[gpu-titles] Failed to query nvidia-smi: {e}")
        return

    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if not lines:
        print("[gpu-titles] No GPU compute processes found.")
        return

    print("[gpu-titles] GPU PID -> process title (ps args):")
    for line in lines:
        parts = [p.strip() for p in line.split(",", 1)]
        if not parts or not parts[0].isdigit():
            continue
        pid = parts[0]
        try:
            ps_out = subprocess.check_output(["ps", "-o", "pid=,comm=,args=", "-p", pid], text=True).strip()
        except Exception as e:
            ps_out = f"pid={pid} (ps failed: {e})"
        print(f"  {ps_out}")


# consistent_name was only used to coerce to str(); inline with str() where needed


_BAYESIAN_DIR = Path("model") / "Bayesian"


def _sync_model_and_scaler_to_bayesian(args) -> None:
    try:
        _BAYESIAN_DIR.mkdir(parents=True, exist_ok=True)
        model_src = str(getattr(args, "save", "") or "")
        if model_src and os.path.exists(model_src):
            shutil.copy2(model_src, _BAYESIAN_DIR / "model.pt")
        if model_src:
            scaler_src = os.path.join(os.path.dirname(model_src), "y_scaler.pt")
            if os.path.exists(scaler_src):
                shutil.copy2(scaler_src, _BAYESIAN_DIR / "y_scaler.pt")
    except Exception:
        return


def _save_metrics(args, split_name: str, metrics: dict, extras: dict | None = None):
    run_id = os.environ.get("RUN_TAG") or (args.run_tag if getattr(args, "run_tag", "") else time.strftime("%Y%m%d-%H%M%S"))
    out_root = Path(getattr(args, "out_dir", "runs")) / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    payload = {"split": split_name, "metrics": metrics, "args": vars(args)}
    if extras:
        payload["extras"] = extras
    with open(out_root / f"metrics_{split_name.lower()}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    try:
        _BAYESIAN_DIR.mkdir(parents=True, exist_ok=True)
        with open(_BAYESIAN_DIR / f"metrics_{split_name.lower()}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


def _maybe_set_run_scoped_save_path(args) -> None:
    """If the user didn't explicitly set --save, place checkpoints under out_dir/<run_id>/.

    This prevents Optuna trials from overwriting a shared checkpoint file.
    """
    try:
        # Use _flag_was_set to detect explicit CLI flag for --save
        if _flag_was_set("save"):
            return

        run_id = os.environ.get("RUN_TAG") or getattr(args, "run_tag", "")
        if not run_id:
            return
        out_dir = getattr(args, "out_dir", None)
        if not out_dir:
            return
        run_dir = Path(out_dir) / str(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        args.save = str(run_dir / "model.pt")
    except Exception:
        return


# For testing the model on unseen data, a sliding window can be used when the output period of the model is smaller than the target period to be forecasted.
# The sliding window uses the output from previous step as input of the next step.
# In our case, the window will be slided if the total forecast period (e.g., 72 months) is longer than the model's output length (e.g., 36 months).
def evaluate_sliding_window(data, test_window, model, evaluateL2, evaluateL1, n_input, is_plot, mc_runs=None):
    result = evaluate_sliding_window_impl(
        data,
        test_window,
        model,
        evaluateL2,
        evaluateL1,
        n_input,
        is_plot,
        args=args,
        device=device,
        state=EVAL_STATE,
        mc_runs=mc_runs,
        save_metrics_fn=_save_metrics,
    )
    _sync_eval_state()
    return result


def evaluate(
    data,
    X,
    Y,
    model,
    evaluateL2,
    evaluateL1,
    batch_size,
    is_plot,
    mc_runs=None,
    kind="Validation",
):
    result = evaluate_impl(
        data,
        X,
        Y,
        model,
        evaluateL2,
        evaluateL1,
        batch_size,
        is_plot,
        args=args,
        device=device,
        state=EVAL_STATE,
        mc_runs=mc_runs,
        kind=kind,
        save_metrics_fn=_save_metrics,
    )
    _sync_eval_state()
    return result


def train(
    data,
    X,
    Y,
    model,
    criterion,
    optim,
    batch_size,
    data_scaler,
    alpha: float = 1.0,
    beta: float = 0.4,
    gamma: float = 0.8,
    clip: float = 10.0,
    mae_weight: float = 0.2,
    grad_scaler: "torch.amp.grad_scaler.GradScaler|None" = None,
    scheduler=None,
):
    return train_impl(
        data,
        X,
        Y,
        model,
        criterion,
        optim,
        batch_size,
        data_scaler,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        clip=clip,
        mae_weight=mae_weight,
        grad_scaler=grad_scaler,
        scheduler=scheduler,
        args=args,
        device=device,
    )


def _generate_epoch_validation_test_plots(Data, model, evaluateL2, evaluateL1) -> None:
    return generate_epoch_validation_test_plots_impl(
        Data,
        model,
        evaluateL2,
        evaluateL1,
        args=args,
        state=EVAL_STATE,
        evaluate_fn=evaluate,
        evaluate_sliding_window_fn=evaluate_sliding_window,
    )


parser = build_parser()


def _resolve_use_trainer_path(args) -> bool:
    try:
        has_quantiles = bool(getattr(args, "quantiles", "").strip())
        q_weight = float(getattr(args, "lambda_quantile", 0.0)) > 0.0
        gauss_on = (int(getattr(args, "use_gauss", 0)) == 1) and (float(getattr(args, "lambda_nll", 0.0)) > 0.0)
        nb_on = (int(getattr(args, "use_nb_head", 0)) == 1) and (float(getattr(args, "lambda_nll", 0.0)) > 0.0)
        return bool(getattr(args, "trainer_mode", False)) or (has_quantiles and q_weight) or gauss_on or nb_on
    except Exception:
        return bool(getattr(args, "trainer_mode", False))


def _export_mc_runs_env(args) -> None:
    try:
        os.environ["BMTGNN_MC_RUNS"] = str(int(getattr(args, "mc_runs", 50)))
    except Exception:
        pass


args, _ = parser.parse_known_args()
try:
    args._cli_keys = _get_cli_keys()
except Exception:
    args._cli_keys = []
try:
    args._metrics_keys = []
except Exception:
    args._metrics_keys = []

# Configure global plot CI behaviour from CLI (override module-level defaults)
try:
    _plot_pi_level = max(0.01, min(0.99, float(getattr(args, "plot_pi_level", 0.95))))
except Exception:
    _plot_pi_level = 0.95

try:
    _plot_ci_scale = max(0.01, float(getattr(args, "plot_ci_scale", 1.0)))
except Exception:
    _plot_ci_scale = 1.0

try:
    _plot_ci_cap_ratio = float(getattr(args, "plot_ci_cap_ratio", 1.0))
except Exception:
    _plot_ci_cap_ratio = 1.0

# set_plot_ci_config removed (src.plot_monitoring)


def _apply_args_from_metrics_json(args, path: str, label: str) -> None:
    try:
        with open(path, "r", encoding="utf-8") as _f:
            _metrics = json.load(_f)
        if "args" in _metrics and isinstance(_metrics["args"], dict):
            # Preserve explicit CLI overrides for any key the user passed.
            _cli_keys = set(getattr(args, "_cli_keys", []) or [])
            _metrics_keys = set(getattr(args, "_metrics_keys", []) or [])

            for _k, _v in _metrics["args"].items():
                if _v is None:
                    continue
                if isinstance(_v, str) and _v == "":
                    continue
                # don't override explicit CLI args
                if _k in _cli_keys:
                    continue
                # set attribute if parser knows about it
                if hasattr(args, _k):
                    try:
                        setattr(args, _k, _v)
                        _metrics_keys.add(_k)
                    except Exception:
                        pass

            try:
                args._metrics_keys = sorted(_metrics_keys)
            except Exception:
                pass

            print(f"[{label}] Applied args from {path} (preserved CLI overrides)")
    except Exception as _e:
        print(f"[{label}] Failed to apply metrics_json:", _e)


# If the user passed --metrics_json, apply the embedded args dict to override defaults.
if getattr(args, "metrics_json", None):
    _apply_args_from_metrics_json(args, args.metrics_json, "metrics_json")

# If requested, apply best-tune metrics for reproducing the target run.
if getattr(args, "use_best_tune", False) and getattr(args, "best_tune_path", ""):
    _apply_args_from_metrics_json(args, args.best_tune_path, "best_tune")

# ============================================================================
# AUTO-CLEANUP: Delete checkpoints and cache on pipeline start
# ============================================================================
if not getattr(args, "no_cleanup", False):
    try:
        cleanup_stats = cleanup_checkpoints_and_cache(dry_run=False, verbose=True)
        if cleanup_stats["total_failed"] > 0:
            print(f"[WARNING] {cleanup_stats['total_failed']} cleanup operations failed. Continuing anyway...")
    except Exception as e:
        print(f"[WARNING] Cleanup failed: {e}. Continuing without cleanup...")
else:
    print("[INFO] Skipping automatic cleanup (--no_cleanup flag set)")

# ============================================================================
# RATIO-BASED SPLITTING (chronological split options removed)
# ============================================================================

# If requested, set env var to skip checkpoint-shape-based inference
if getattr(args, "skip_ckpt_infer", False):
    os.environ["BMTGNN_SKIP_CKPT_INFER"] = "1"

# Force training mode even if metrics_json came from an eval-only run.
if getattr(args, "train", False):
    try:
        args.eval_only = False
    except Exception:
        pass
    try:
        args.use_cached_hp = False
    except Exception:
        pass

# Allow explicit disabling of trainer_mode when needed (so random search runs are performed)
if getattr(args, "no_trainer_mode", False):
    try:
        args.trainer_mode = False
        print("[cli] --no_trainer_mode: forcing trainer_mode=False so search_trials will run")
    except Exception:
        pass
if getattr(args, "strong_rmdpt", False):
    print("[strong_rmdpt] applying strong defaults...")
    _apply_strong_rmdpt_defaults(args)
    print(f"[final] early_stop_patience={args.early_stop_patience}")
if getattr(args, "hugging_mode", False):
    print("[hugging_mode] applying aggressive curve-hugging defaults...")
    _apply_hugging_mode_defaults(args)
    print(f"[hugging_mode] epochs={args.epochs} dropout={args.dropout} mc_runs={args.mc_runs}")

# Optional: generate aggressive-smoothed data before any auto-detection or split building
_maybe_generate_aggressive_smoothed_data(args)

# ============================================================================
# RATIO-BASED SPLITTING ONLY (Chronological options completely removed)
# ============================================================================
print("\n[SPLIT] Using RATIO-BASED splitting:")
print(f"  Train Ratio:      {getattr(args, 'train_ratio', 0.60):.2%}")
print(f"  Valid Ratio:      {getattr(args, 'valid_ratio', 0.20):.2%}")
print(f"  Test Ratio:       {1.0 - getattr(args, 'train_ratio', 0.60) - getattr(args, 'valid_ratio', 0.20):.2%}")
print()
# ---- Optional: force-disable hyperparameter search (debug only) ----
try:
    if os.environ.get("BMTGNN_FORCE_SEARCH_TRIALS_0", "0") == "1":
        if getattr(args, "search_trials", 0) and int(getattr(args, "search_trials", 0)) > 0:
            print("[warn] BMTGNN_FORCE_SEARCH_TRIALS_0=1; forcing search_trials=0")
            args.search_trials = 0
except Exception:
    pass
# Map grad_clip alias if provided
if getattr(args, "grad_clip", None) is not None:
    try:
        if args.grad_clip is not None:
            args.clip = float(args.grad_clip)
    except Exception:
        pass
    print("[Deprecation] --grad_clip is deprecated; use --clip instead.")

# --- Auto-detect number of nodes (variables) if not provided or <=0 ---
if getattr(args, "num_nodes", 0) <= 0:
    try:
        # local numpy import removed; use module-level `_np` alias
        _candidates = [args.data]
        if not os.path.exists(args.data):
            _candidates.append(os.path.join("data", os.path.basename(args.data)))
            _candidates.append("data/sm_data.csv")
            _candidates.append("data/sm_data.txt")
        for _p in _candidates:
            if _p and os.path.exists(_p):
                try:
                    _delim = "," if str(_p).lower().endswith(".csv") else "\t"
                    _row = _np.loadtxt(_p, delimiter=_delim, max_rows=1)
                    if _row.ndim == 1:
                        args.num_nodes = int(_row.shape[0])
                        print(f"[AutoDetect] Inferred num_nodes={args.num_nodes} from '{_p}'")
                        break
                except Exception:
                    pass
        if getattr(args, "num_nodes", 0) <= 0:
            print("[AutoDetect] Warning: could not infer num_nodes; will attempt after data load")
    except Exception as _e:
        print(f"[AutoDetect] Warning: num_nodes inference failed early: {_e}")

# Parse quantiles (if provided)
q_list = []
if getattr(args, "quantiles", ""):
    raw_q = str(getattr(args, "quantiles")).strip()
    if raw_q:
        try:
            parts = [p for p in raw_q.split(",") if p.strip()]
            q_list = [float(p) for p in parts]
            q_list = [q for q in q_list if 0.0 < q < 1.0]
        except Exception:
            q_list = []

# Harmonize vectorized MC flags (explicit --no_vectorized_mc overrides)
if getattr(args, "no_vectorized_mc", False):
    args.vectorized_mc = False

# Decide whether to use the Trainer path (probabilistic features ON or explicit switch)
USE_TRAINER_PATH = _resolve_use_trainer_path(args)

# Compute quantile flags for trainer path
_has_quantiles = bool(getattr(args, "quantiles", "").strip())
_q_weight = float(getattr(args, "lambda_quantile", 0.0)) > 0.0

# Export MC run count early so temporal attention gating sees it during evaluation
_export_mc_runs_env(args)

use_cuda = torch.cuda.is_available() and args.device.startswith("cuda")
device = torch.device(args.device if use_cuda else "cpu")
if use_cuda and device.type == "cuda":
    try:
        dev_count = torch.cuda.device_count()
        if device.index is not None and device.index >= dev_count:
            print(f"[Device] Requested {device}, but only {dev_count} CUDA device(s) visible. Falling back to cuda:0.")
            device = torch.device("cuda:0")
    except Exception as _e:
        print(f"[Device] CUDA device check failed: {_e}. Falling back to CPU.")
        device = torch.device("cpu")
# Device selected (only print if explicitly requested via debug flag)
if os.environ.get("BMTGNN_DEBUG_DEVICE", "0") == "1":
    print(f"[Device] Using {device}")

# Start run logger if enabled (silent unless debug enabled)
if getattr(args, "runlog", False):
    start_runlog(args)
    # use top-level `torch` import

    if os.environ.get("BMTGNN_DEBUG_DEVICE", "0") == "1":
        print("[RunLog] Logging to: model/Bayesian/logs/...")
    devprops = {}
    if torch.cuda.is_available() and device.type == "cuda":
        try:
            p = torch.cuda.get_device_properties(device)
            devprops = {
                "name": p.name,
                "total_mem_gb": round(p.total_memory / 1024**3, 2),
            }
        except Exception as _e:
            devprops = {"device_props_error": str(_e)[:160]}
    jlog("device_info", device=str(device), **devprops)

# clamp mc_runs to avoid accidental OOMs but allow higher user values
args.mc_runs = max(10, int(getattr(args, "mc_runs", 50)))


fixed_seed = 123

# Hyperparameter search option lists (module-level to avoid scope/analysis issues)
# Optimized for sparse/heavy-tailed RMD/PT forecasting (target RSE/RAE < 0.6)
GCN_DEPTHS = [2, 3]
LRS = [0.0003, 0.0005, 0.0007, 0.0008, 0.001]  # Conservative LRs for stability
CONVS = [16, 32]
RESS = [32, 64]
SKIPS = [64, 128]
# Removed 1024 to reduce OOM risk under vectorized MC
ENDS = [128, 256, 512]
if getattr(args, "allow_wide_end", False) and 1024 not in ENDS:
    ENDS.append(1024)
LAYERS_CHOICES = [5, 8]  # Deeper for better temporal patterns
KS = [40, 50, 60, 70, 80]
DROPOUTS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18]  # Lower dropout for curve hugging
DILATION_EXS = [2, 3]
NODE_DIMS = [40, 50, 60, 70, 80]
PROP_ALPHAS = [0.05, 0.1, 0.15, 0.2]
TANH_ALPHAS = [2, 3, 5]


def main(experiment):  # pyright: ignore[reportGeneralTypeIssues]
    model: nn.Module | None = None
    # Set random seed for reproducibility (use --seed if provided)
    seed_to_use = fixed_seed if getattr(args, "seed", None) is None else int(args.seed)
    set_random_seed(seed_to_use, getattr(args, "cudnn_benchmark", False))

    if getattr(args, "show_gpu_titles", False):
        _print_gpu_pid_titles()

    # --- NEW: Configure Analysis Logger ---
    global ANALYSIS_LOGGER
    if args.analysis_log:
        ANALYSIS_LOGGER = AnalysisLogger(args.analysis_log)
    EVAL_STATE["analysis_logger"] = ANALYSIS_LOGGER
    # ---

    # --- NEW: Early checkpoint hparams inference (before DataLoader construction) ---
    try:
        # Skip checkpoint inference if --fresh_start was requested
        if getattr(args, "fresh_start", False):
            pass  # Silent skip
        # Allow skipping automatic checkpoint-based hyperparameter inference
        # by setting environment variable BMTGNN_SKIP_CKPT_INFER=1
        elif os.environ.get("BMTGNN_SKIP_CKPT_INFER", "0") == "1":
            pass  # Silent skip
        elif getattr(args, "save", None) and os.path.exists(args.save):
            try:
                ck_early = torch.load(args.save, map_location="cpu")
                sd_early = ck_early.get("state_dict", {}) if isinstance(ck_early, dict) else {}
                if isinstance(sd_early, dict) and sd_early:
                    # lightweight inference of temporal kernel, layers and channels
                    seq_k = set()
                    layers_found = set()
                    filter_out = []
                    conv_in = []
                    residual_out = []
                    skip_out = []
                    end_in = []
                    for k, v in sd_early.items():
                        if not hasattr(v, "shape"):
                            continue
                        s = tuple(v.shape)
                        if k.startswith("filter_convs.") and k.endswith(".weight") and len(s) >= 4:
                            seq_k.add(s[-1])
                            try:
                                layers_found.add(int(k.split(".")[1]))
                            except Exception:
                                pass
                            # dilated_inception: per-branch out_channels = conv_channels / 4
                            filter_out.append(s[0])
                        if k.startswith("residual_convs.") and k.endswith(".weight") and len(s) >= 2:
                            residual_out.append(s[0])
                            conv_in.append(s[1])
                            try:
                                layers_found.add(int(k.split(".")[1]))
                            except Exception:
                                pass
                        if k.startswith("skip_convs.") and k.endswith(".weight") and len(s) >= 2:
                            skip_out.append(s[0])
                        if "end_conv" in k and k.endswith(".weight") and len(s) >= 2:
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
                    elif filter_out:
                        # fallback: infer conv_channels from filter conv branch width
                        args.conv_channels = int(_mode_or_none(filter_out) * 4)
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
                        if "end_conv_1.weight" in sd_early:
                            args.end_channels = int(sd_early["end_conv_1.weight"].shape[0])
                    except Exception:
                        pass
                    # detect quantile/nb head
                    if any(k.startswith("end_conv_q") or k.startswith("end_conv_nb") for k in sd_early.keys()):
                        setattr(args, "use_nb_head", 1)
            except Exception:
                pass  # Silent checkpoint inspection failure
    except Exception:
        pass
    # ---

    # Keep checkpoint output run-scoped for reproducibility (especially Optuna).
    _maybe_set_run_scoped_save_path(args)

    # Post-preset: re-apply explicit CLI intent if needed
    if getattr(args, "train", False):
        args.eval_only = False
        try:
            # If the JSON was eval-only it may have set use_cached_hp; training should not.
            args.use_cached_hp = False
        except Exception:
            pass
    if getattr(args, "no_trainer_mode", False):
        args.trainer_mode = False

    # Recompute Trainer/search routing after presets (USE_TRAINER_PATH is used later).
    globals()["USE_TRAINER_PATH"] = _resolve_use_trainer_path(args)

    # Keep env in sync for attention auto-gating
    _export_mc_runs_env(args)

    # Post-preset config resolved (only print if debug enabled)
    if os.environ.get("BMTGNN_DEBUG_CONFIG", "0") == "1":
        try:
            print(
                "[ConfigResolved]",
                f"data={getattr(args, 'data', None)}",
                f"graph={getattr(args, 'graph', None)}",
                f"seq_in_len={getattr(args, 'seq_in_len', None)}",
                f"seq_out_len={getattr(args, 'seq_out_len', None)}",
                f"layers={getattr(args, 'layers', None)}",
                f"trainer_mode={bool(getattr(args, 'trainer_mode', False))}",
                f"epochs={getattr(args, 'epochs', None)}",
            )
        except Exception:
            pass

    # Stage user graph once (if provided) BEFORE DataLoaderS builds adjacency
    if getattr(args, "graph", ""):
        user_graph = args.graph
        if os.path.exists(user_graph):
            try:
                os.makedirs("data", exist_ok=True)
                # Try to load as numeric square adjacency
                g = np.loadtxt(user_graph, delimiter=",")
                if g.ndim == 2 and g.shape[0] == g.shape[1]:
                    # preserve float weights
                    np.savetxt("data/graph_square.csv", g, delimiter=",", fmt="%.6g")
                    print("Copied numeric adjacency to data/graph_square.csv")
                else:
                    # treat as edge-list/headered CSV
                    shutil.copy(user_graph, "data/graph.csv")
                    print("Copied graph to data/graph.csv")
            except Exception:
                os.makedirs("data", exist_ok=True)
                shutil.copy(user_graph, "data/graph.csv")
                print("Copied graph to data/graph.csv (fallback)")

    Data, use_chrono, steps_py, required_months = resolve_split_and_build_data(args, device)

    # Align num_nodes with the data columns before graph loading.
    try:
        data_nodes = int(getattr(Data, "m", 0))
    except Exception:
        data_nodes = 0
    if data_nodes > 0:
        if getattr(args, "num_nodes", 0) <= 0:
            args.num_nodes = data_nodes
            print(f"[AutoDetect] Set num_nodes from data: {args.num_nodes}")
        elif int(getattr(args, "num_nodes", 0)) != data_nodes:
            print(f"[warn] num_nodes={args.num_nodes} disagrees with data columns={data_nodes}; using data value.")
            args.num_nodes = data_nodes

    # Guard against mismatched checkpoint/scaler from prior runs.
    if not getattr(args, "fresh_start", False):
        ckpt_nodes = None
        scaler_nodes = None
        if getattr(args, "save", None) and os.path.exists(args.save):
            ckpt_nodes = _infer_ckpt_num_nodes(args.save)
            scaler_path = os.path.join(os.path.dirname(args.save), "y_scaler.pt")
            if os.path.exists(scaler_path):
                scaler_nodes = _infer_scaler_num_nodes(scaler_path)
        mismatch = (ckpt_nodes is not None and data_nodes > 0 and ckpt_nodes != data_nodes) or (scaler_nodes is not None and data_nodes > 0 and scaler_nodes != data_nodes)
        if mismatch:
            print(f"[Mismatch] data_nodes={data_nodes} ckpt_nodes={ckpt_nodes} scaler_nodes={scaler_nodes}; " "forcing fresh_start to rebuild 95-node artifacts.")
            args.fresh_start = True
            if getattr(args, "eval_only", False):
                print("[Mismatch] Disabling eval_only due to node mismatch.")
                args.eval_only = False
            if getattr(args, "use_cached_hp", False):
                print("[Mismatch] Disabling use_cached_hp due to node mismatch.")
                args.use_cached_hp = False
            os.environ["BMTGNN_SKIP_CKPT_INFER"] = "1"

    predefined_A = prepare_graph_and_subgraph(args, device, Data)

    # Log which adjacency source will be used by DataLoaderS
    try:
        if getattr(args, "graph", ""):
            if os.path.exists("data/graph_square.csv"):
                print(f"[Graph] adjacency source=data/graph_square.csv (from {args.graph})")
            elif os.path.exists("data/graph.csv"):
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
        local_candidate = os.path.join(os.getcwd(), "data", os.path.basename(args.data))
        if os.path.exists(local_candidate):
            print(f"[Warning] specified data '{args.data}' not found; using '{local_candidate}' instead")
            args.data = local_candidate
        elif os.path.exists(os.path.join(os.getcwd(), "data", "sm_data.txt")):
            alt = os.path.join(os.getcwd(), "data", "sm_data.txt")
            print(f"[Warning] specified data '{args.data}' not found; falling back to '{alt}'")
            args.data = alt
        else:
            found = []
            try:
                found = os.listdir(os.path.join(os.getcwd(), "data"))
            except Exception:
                found = []
            raise FileNotFoundError(f"Data file '{args.data}' not found. Place your dataset at that path or in ./data/. Files found in ./data/: {found}")

    # Data already built above as Data

    # Move entire (small) splits to GPU once to avoid per-batch host->device copies
    def _to_dev_split(tup):
        if tup is None or tup[0] is None or tup[1] is None:
            return tup
        return (
            tup[0].to(device, dtype=torch.float32, non_blocking=True),
            tup[1].to(device, dtype=torch.float32, non_blocking=True),
        )

    if device.type == "cuda":
        setattr(Data, "train", _to_dev_split(Data.train) if Data.train is not None else None)
        setattr(Data, "valid", _to_dev_split(Data.valid) if Data.valid is not None else None)
        setattr(Data, "test", _to_dev_split(Data.test) if Data.test is not None else None)
        if getattr(Data, "test_window", None) is not None:
            setattr(
                Data,
                "test_window",
                torch.as_tensor(Data.test_window, dtype=torch.float32).to(device, non_blocking=True),
            )
        try:
            if getattr(Data, "train", None) is not None and Data.train[0] is not None:
                print(f"[Device] train X device: {Data.train[0].device}, train Y device: {Data.train[1].device}")
        except Exception as _e:
            print(f"[Device] train device check failed: {_e}")

    # --- Compute per-node training-derived weights (std-based) and apply name-based boosts ---
    node_weights_np = None
    try:
        train_split = getattr(Data, "train", None)
        if train_split is not None and train_split[1] is not None:
            Y_train = train_split[1]
            # Convert to numpy on CPU if tensor
            if hasattr(Y_train, "cpu") and hasattr(Y_train, "numpy"):
                Y_np = Y_train.cpu().numpy()
            else:
                # local numpy import removed; use module-level `_np` alias
                Y_np = _np.asarray(Y_train)
            if Y_np.ndim == 3:
                B_tr, H_tr, N_tr = Y_np.shape
                flat = Y_np.reshape(-1, N_tr)
                # local numpy import removed; use module-level `_np` alias
                node_std = _np.std(flat, axis=0)
                node_std = node_std + 1e-6
                node_w = node_std / (float(node_std.mean()) + 1e-12)

                # Apply name-based boosts: RMD, PT, other
                try:
                    col_names = Data.create_columns()
                    if col_names is None or len(col_names) != node_w.shape[0]:
                        col_names = None
                except Exception:
                    col_names = None

                rmd_boost = float(getattr(args, "rmd_loss_weight", 2.5))
                pt_boost = float(getattr(args, "pt_loss_weight", 1.0))
                other_boost = float(getattr(args, "other_loss_weight", 0.5))

                if col_names is not None:
                    factors = _np.ones_like(node_w, dtype=_np.float32)
                    for i, name in enumerate(col_names):
                        try:
                            if isinstance(name, str) and name.startswith("RMD_"):
                                factors[i] = rmd_boost
                            elif isinstance(name, str) and name.startswith("PT_"):
                                factors[i] = pt_boost
                            else:
                                factors[i] = other_boost
                        except Exception:
                            factors[i] = 1.0
                    node_w = node_w * factors
                    # Re-normalize to mean=1 to keep global scale stable, then clip
                    node_w = node_w / (float(node_w.mean()) + 1e-12)

                node_w = _np.clip(node_w, 0.5, 5.0).astype(_np.float32)
                node_weights_np = node_w
                print(f"[train] per-node weights computed: mean={node_w.mean():.3f} min={node_w.min():.3f} max={node_w.max():.3f} (rmd_boost={rmd_boost} pt_boost={pt_boost} other={other_boost})")
    except Exception as _e:
        print(f"[train] per-node weight computation failed: {_e}")
    setattr(Data, "node_weights", node_weights_np)

    # --- sanity: training windows must exist ---
    try:
        train_attr = getattr(Data, "train", None)
        if train_attr is None or train_attr[0] is None:
            raise RuntimeError("No training windows: Data.train is missing or empty")
        B_train = int(train_attr[0].shape[0])
        jlog("window_guard", train_windows=B_train)
        if B_train < 1:
            if getattr(args, "auto_window_adjust", False):
                print("[AutoWindow] No training windows; attempting automatic adjustment...")
                # Preserve original for logging
                orig_in, orig_out = int(args.seq_in_len), int(args.seq_out_len)
                target_min = max(1, int(getattr(args, "auto_window_min_train", 4)))
                # Hard lower bounds to avoid degenerate setups
                min_in_allowed = 8
                min_out_allowed = 4
                adjusted = False
                # We'll try decreasing seq_in_len first (history length), then seq_out_len
                for _pass in range(2):
                    for _ in range(200):  # safety cap
                        # Rebuild DataLoaderS with current lengths
                        Data = DataLoaderS(
                            args.data,
                            float(getattr(args, "train_ratio", 0.6)),
                            float(getattr(args, "valid_ratio", 0.2)),
                            device,
                            args.horizon,
                            args.seq_in_len,
                            args.normalize,
                            args.seq_out_len,
                            chronological=use_chrono,
                            start_year=args.start_year,
                            steps_per_year=args.steps_per_year,
                            train_end_year=args.train_end_year,
                            valid_end_year=args.valid_end_year,
                            test_end_year=args.test_end_year,
                            dual_channel=args.dual_channel,
                            pct_clip=float(getattr(args, "pct_clip", 0.0)),
                            y_transform=getattr(args, "y_transform", None),
                            trend_smooth=bool(getattr(args, "trend_smooth", False)),
                            trend_alpha=float(getattr(args, "trend_alpha", 0.25)),
                            trend_beta=float(getattr(args, "trend_beta", 0.05)),
                            resid_alpha=float(getattr(args, "resid_alpha", 0.20)),
                            has_header=bool(getattr(args, "has_header", False)),
                            drop_first_col=bool(getattr(args, "drop_first_col", False)),
                            exclude_names=(args.exclude_names if getattr(args, "exclude_names", "") != "" else None),
                        )
                        train_attr = getattr(Data, "train", None)
                        B_train = int(train_attr[0].shape[0]) if train_attr and train_attr[0] is not None else 0
                        if B_train >= target_min or (B_train >= 1 and _pass == 1):
                            adjusted = True
                            setattr(args, "_final_seq_in_len", int(args.seq_in_len))
                            setattr(args, "_final_seq_out_len", int(args.seq_out_len))
                            setattr(args, "_auto_window_adjusted", True)
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
                raise RuntimeError("No training windows: decrease --seq_in_len/--seq_out_len or widen the train split.")
    except Exception as _e:
        print(f"[FATAL] {_e}")
        raise
    # ------------------------------------------
    # ------------------------------------------
    # --- RunLog: dataset & graph summary ---
    if getattr(args, "runlog", False):
        try:
            N = Data.m
            T_train = int(Data.train[0].shape[0]) if getattr(Data, "train", None) and Data.train[0] is not None else 0
            T_valid = int(Data.valid[0].shape[0]) if getattr(Data, "valid", None) and Data.valid[0] is not None else 0
            T_test = int(Data.test[0].shape[0]) if getattr(Data, "test", None) and Data.test[0] is not None else 0
            mode = "rolling" if getattr(Data, "rolling", False) else "global"
            dual = getattr(args, "dual_channel", "none")
            jlog(
                "data_summary",
                nodes=N,
                t_train=T_train,
                t_valid=T_valid,
                t_test=T_test,
                normalize_mode=mode,
                dual_channel=dual,
                seq_in=args.seq_in_len,
                seq_out=args.seq_out_len,
            )

            # quick graph density (ignores diagonal)
            dens = None
            if getattr(args, "graph", None) and os.path.exists(args.graph):
                A = np.loadtxt(args.graph, delimiter=",")
                if A.shape[0] == A.shape[1] and A.shape[0] == N:
                    nnz_off = np.count_nonzero(A - np.diag(np.diag(A)))
                    dens = float(nnz_off) / float(max(1, N * (N - 1)))
            jlog("graph_summary", path=args.graph, density=dens)
        except Exception as _e:
            jlog("warn_data_graph_summary", error=str(_e)[:160])
        inact = getattr(Data, "inactive_nodes", None)
        if inact is not None:
            jlog("inactive_nodes", count=len(inact))
    # Save per-node train scaler for reproducibility/debug
    try:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        mean_t = Data.mu.detach().cpu() if (hasattr(Data, "mu") and Data.mu is not None and torch.is_tensor(Data.mu)) else None
        std_t = torch.clamp(Data.std.detach().cpu(), min=1e-6) if (hasattr(Data, "std") and Data.std is not None and torch.is_tensor(Data.std)) else None
        torch.save(
            {"mean": mean_t, "std": std_t},
            os.path.join(os.path.dirname(args.save), "y_scaler.pt"),
        )
        _sync_model_and_scaler_to_bayesian(args)
    except Exception as _e:
        print(f"[scaler_save] warning: {_e}")

    # --- Artifact archiving disabled to prevent false runs ---
    # Checkpoint and scaler archiving is skipped to avoid creating unnecessary artifact directories
    # ---

    # Split summary logging (keep consistent with actual split mode)
    try:
        if bool(use_chrono):
            print(f"[SPLIT] start_year={args.start_year}, steps_per_year={args.steps_per_year}")
            print(f"[SPLIT] Train: {args.start_year}-{args.train_end_year} | " f"Valid: {args.train_end_year + 1}-{args.valid_end_year} | " f"Test: {args.valid_end_year + 1}-{args.test_end_year}")
        else:
            # If chrono was requested but we auto-fell back, make it explicit.
            if bool(getattr(args, "use_chronological_split", False)) or bool(getattr(args, "chronological_split", False)):
                print("[SPLIT] Using ratio splits (chronological split auto-disabled due to insufficient windowable span)")
    except Exception:
        pass

    # Apply safer graph defaults for stability/generalization
    try:
        # prefer symmetric normalized adjacency to tame spectrum
        sym_path = "data/graph_symnorm.csv"
        if os.path.exists(sym_path):
            try:
                A_symnorm = np.loadtxt(sym_path, delimiter=",")
                Data.adj = torch.from_numpy(A_symnorm.astype(np.float32)).to(device)
                print("[Graph] using cached symmetric-normalized adjacency (D^-1/2 A D^-1/2)")
            except Exception:
                pass
        if isinstance(Data.adj, torch.Tensor) and not os.path.exists(sym_path):
            A_np = Data.adj.detach().cpu().numpy()
            A_symnorm = sym_adj(A_np)
            Data.adj = torch.from_numpy(A_symnorm.astype(np.float32)).to(device)
            try:
                np.savetxt(sym_path, A_symnorm, delimiter=",", fmt="%.6f")
            except Exception:
                pass
            print("[Graph] using symmetric-normalized adjacency (D^-1/2 A D^-1/2)")
    except Exception:
        pass

    # conservative default subgraph_size if user didn't change it
    if getattr(args, "subgraph_size", None) is None or int(getattr(args, "subgraph_size", 20)) == 20:
        args.subgraph_size = 8
        print(f"[Graph] subgraph_size not specified or default; using conservative k={args.subgraph_size}")

    # Optional: skip random search if user wants to reuse cached best HPs / checkpoint
    if args.use_cached_hp and args.eval_only:
        run_search = False
    elif USE_TRAINER_PATH:
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
        model = run_trainer_path(
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
        if model is not None:
            setattr(args, "_skip_ckpt_reload", True)

    if run_search:
        # random search loop
        trials = max(1, int(getattr(args, "search_trials", 60)))

        # Initial GPU cleanup before starting hyperparameter search
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("[cleanup] Initial GPU memory cleared before search")

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

            # Data shape calculations (verbose debug info removed)
            tlen = int(Data.train[0].shape[0]) if getattr(Data, "train", None) and Data.train[0] is not None else 0
            vlen = int(Data.valid[0].shape[0]) if getattr(Data, "valid", None) and Data.valid[0] is not None else 0
            slen = int(Data.test[0].shape[0]) if getattr(Data, "test", None) and Data.test[0] is not None else 0

            # If predefined adjacency exists, move to the same device
            if isinstance(Data.adj, torch.Tensor):
                Data.adj = Data.adj.to(device)

            try:
                in_dim_use = getattr(Data, "in_dim", args.in_dim)
            except NameError:
                in_dim_use = args.in_dim
            model = gtnet(
                args.gcn_true,
                args.buildA_true,
                gcn_depth,
                int(Data.m),
                device,
                Data.adj,
                dropout=dropout,
                subgraph_size=k,
                node_dim=node_dim,
                dilation_exponential=dilation_ex,
                conv_channels=conv,
                residual_channels=res,
                skip_channels=skip,
                end_channels=end,
                seq_length=args.seq_in_len,
                in_dim=in_dim_use,
                out_dim=args.seq_out_len,
                layers=layer,
                propalpha=prop_alpha,
                tanhalpha=tanh_alpha,
                layer_norm_affline=False,
                temporal_attn=getattr(args, "temporal_attn", False),
                attn_dim=getattr(args, "attn_dim", 64),
                attn_heads=getattr(args, "attn_heads", 2),
                attn_dropout=getattr(args, "attn_dropout", 0.1),
                attn_window=getattr(args, "attn_window", 0),
                attn_math_mode=getattr(args, "attn_math_mode", False),
                attn_bn_chunk=int(getattr(args, "attn_bn_chunk", 0)),
                attn_gate_threshold=int(getattr(args, "attn_gate_threshold", 0)),
                temporal_transformer=bool(getattr(args, "temporal_transformer", 0)),
                tt_layers=int(getattr(args, "tt_layers", 2)),
                graph_mix=float(getattr(args, "graph_mix", 0.0)),
                dropedge_p=float(getattr(args, "dropedge_p", 0.0)),
                quantiles=q_list,
                nb_head=bool(int(getattr(args, "use_nb_head", 0)) == 1),
                zinb=bool(int(getattr(args, "use_zinb", 0)) == 1),
                gauss_head=bool(int(getattr(args, "use_gauss", 0)) == 1),
            )
            model.to(device)
            # Optional compile based on flag (simplified safe attempt)
            compile_mode = getattr(args, "compile", "auto")
            if getattr(torch, "compile", None) is not None and compile_mode != "off":
                try:
                    if compile_mode == "eager":
                        model = cast(nn.Module, torch.compile(model, backend="eager"))
                        print("[compile] backend=eager")
                    elif compile_mode == "inductor":
                        model = cast(nn.Module, torch.compile(model))
                        print("[compile] backend=inductor")
                    else:  # auto
                        model = cast(nn.Module, torch.compile(model))
                        print("[compile] auto -> inductor")
                except Exception as _e:
                    print(f"[compile] disabled (fallback): {_e}")

            def _get_attr(m, name, default=None):
                return getattr(m, name, default)

            print(args)
            print("The receptive field size is", _get_attr(model, "receptive_field"))
            core = model
            # Ensure core is a nn.Module before accessing parameters
            if isinstance(core, nn.Module):
                nParams = sum(p.numel() for p in core.parameters())
                print("Number of model parameters is", nParams, flush=True)
            else:
                print("Could not determine number of model parameters.")

            if args.L1Loss:
                criterion = nn.L1Loss(reduction="mean").to(device)
            else:
                criterion = nn.MSELoss(reduction="mean").to(device)
            evaluateL2 = nn.MSELoss(reduction="sum").to(device)
            evaluateL1 = nn.L1Loss(reduction="sum").to(device)
            if args.fused_optim:
                # Native fused AdamW (PyTorch 2.0+ with CUDA 11.6+)
                import torch.optim as _optim

                # Handle compiled models which may have _orig_mod attribute
                if isinstance(core, nn.Module) and hasattr(core, "_orig_mod") and isinstance(core._orig_mod, nn.Module):
                    params = core._orig_mod.parameters()
                elif isinstance(core, nn.Module):
                    params = core.parameters()
                else:
                    raise RuntimeError("Could not access parameters from model (check model type)")
                optim = _optim.AdamW(params, lr=lr, weight_decay=args.weight_decay, fused=True)
                print("[optim] Using fused AdamW")
            else:
                if isinstance(core, nn.Module):
                    params = core.parameters()
                elif "model" in locals() and "model" in globals() and model is not None and isinstance(model, nn.Module):
                    params = model.parameters()
                else:
                    raise RuntimeError("Could not access parameters from model (check model type)")
                optim = Optim(params, args.optim, lr, args.clip, weight_decay=args.weight_decay)
            # NEW: epoch-wise LR scheduler (helps avoid late-epoch collapse/flat preds)
            scheduler = None
            try:
                import torch.optim as _optim

                # Unwrap custom Optim wrapper to a real torch optimizer for schedulers
                opt_for_sched = getattr(optim, "optimizer", None)
                if opt_for_sched is None:
                    opt_for_sched = optim
                if not isinstance(opt_for_sched, _optim.Optimizer):
                    raise TypeError(f"Scheduler requires torch.optim.Optimizer, got {type(opt_for_sched)}")
                opt_for_sched = cast(_optim.Optimizer, opt_for_sched)

                if args.scheduler == "cosine":
                    scheduler = _optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_for_sched, T_0=args.sched_T0, T_mult=args.sched_Tmult)
                elif args.scheduler == "onecycle":
                    # needs total steps = epochs * steps_per_epoch; approximate with len(train)/batch
                    steps_per_epoch = max(1, int(math.ceil(Data.train[0].shape[0] / args.batch_size)))
                    max_lr = lr
                    scheduler = _optim.lr_scheduler.OneCycleLR(
                        opt_for_sched,
                        max_lr=max_lr,
                        total_steps=max(steps_per_epoch * args.epochs, 10),
                        pct_start=args.onecycle_pct,
                    )
            except Exception as _e:
                print(f"[scheduler] disabled: {_e}")
            # Per-trial early stopping trackers
            best_val_loss_trial = float("inf")
            epochs_without_improvement = 0
            ensemble_ckpts = int(getattr(args, "ensemble_ckpts", 0))
            ensemble_every = max(1, int(getattr(args, "ensemble_every", 20)))
            ensemble_paths = []
            scaler = GradScaler(enabled=bool(args.amp) and device.type == "cuda")
            grad_scaler = scaler if getattr(scaler, "is_enabled", lambda: True)() else None

            # Outer try-catch for OOM errors (catch and skip trial gracefully)
            try:
                print("begin training")
                for epoch in range(1, args.epochs + 1):
                    print("Experiment:", (experiment + 1))
                    print("Iter:", q)
                    print("epoch:", epoch)
                    print(
                        "hp=",
                        [
                            gcn_depth,
                            lr,
                            conv,
                            res,
                            skip,
                            end,
                            k,
                            dropout,
                            dilation_ex,
                            node_dim,
                            prop_alpha,
                            tanh_alpha,
                            layer,
                            epoch,
                        ],
                    )
                    print("best sum=", best_val)
                    print("best rrse=", best_rse)
                    print("best rrae=", best_rae)
                    print("best corr=", best_corr)
                    print("best smape=", best_smape)
                    print("best hps=", best_hp)
                    print("best test rse=", best_test_rse)
                    print("best test corr=", best_test_corr)
                    epoch_start_time = time.time()
                    train_loss = train(
                        Data,
                        Data.train[0],
                        Data.train[1],
                        model,
                        criterion,
                        optim,
                        args.batch_size,
                        data_scaler=None,
                        alpha=float(args.loss_alpha),
                        beta=float(args.loss_beta),
                        gamma=float(args.loss_gamma),
                        mae_weight=float(args.mae_weight),
                        grad_scaler=grad_scaler,
                        scheduler=scheduler,
                    )
                    if getattr(args, "runlog", False):
                        jlog("epoch_train", epoch=epoch, loss=train_loss)
                    val_loss, val_rae, val_corr, val_smape = (
                        evaluate(
                            Data,
                            Data.valid[0],
                            Data.valid[1],
                            model if "model" in locals() else None,
                            evaluateL2,
                            evaluateL1,
                            args.batch_size,
                            False,
                            mc_runs=args.mc_runs,
                        )
                        if Data.valid[0] is not None and Data.valid[1] is not None
                        else (float("inf"), float("inf"), 0.0, float("inf"))
                    )
                    # Export mc_runs for attention auto-gating (mc_runs * batch * nodes threshold)
                    _export_mc_runs_env(args)
                    # Store current epoch for plot tracking
                    args._current_epoch = epoch

                    if getattr(args, "runlog", False):
                        jlog(
                            "epoch_valid",
                            epoch=epoch,
                            rrse=val_loss,
                            rae=val_rae,
                            corr=val_corr,
                            smape=val_smape,
                        )
                    print(
                        "| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f} | valid smape  {:5.4f}".format(
                            epoch,
                            (time.time() - epoch_start_time),
                            train_loss,
                            val_loss,
                            val_rae,
                            val_corr,
                            val_smape,
                        ),
                        flush=True,
                    )
                    # step scheduler once per epoch (or let OneCycle step per batch if integrated)
                    try:
                        if scheduler and args.scheduler == "cosine":
                            scheduler.step(epoch)
                    except Exception:
                        pass
                    # GPU memory logging per-epoch
                    if (getattr(args, "log_gpu_mem", False) or getattr(args, "log_peak_mem", False)) and torch.cuda.is_available():
                        try:
                            torch.cuda.synchronize()
                            alloc = torch.cuda.memory_allocated() / 1024**2
                            reserved = torch.cuda.memory_reserved() / 1024**2
                            peak = torch.cuda.max_memory_allocated() / 1024**2
                            msg = f"[gpu-mem][epoch {epoch}] alloc={alloc:.1f}MB reserved={reserved:.1f}MB peak={peak:.1f}MB"
                            print(msg)
                            if getattr(args, "runlog", False):
                                jlog(
                                    "gpu_mem_epoch",
                                    epoch=epoch,
                                    alloc_mb=round(alloc, 2),
                                    reserved_mb=round(reserved, 2),
                                    peak_mb=round(peak, 2),
                                )
                        except Exception as _mem_e:
                            print(f"[gpu-mem] logging failed: {_mem_e}")
                    # Per-trial early stopping bookkeeping
                    if val_loss < best_val_loss_trial:
                        best_val_loss_trial = val_loss
                        epochs_without_improvement = 0
                        if "model" in locals() and model is not None:
                            core_to_save = model
                        else:
                            raise RuntimeError("Model is not defined.")
                        if isinstance(core_to_save, nn.Module) and hasattr(core_to_save, "_orig_mod") and isinstance(core_to_save._orig_mod, nn.Module):
                            core_to_save = core_to_save._orig_mod
                        if not isinstance(core_to_save, nn.Module):
                            raise RuntimeError("Cannot save model: not an nn.Module instance.")
                        # Global best model tracking (across all trials)
                        sum_loss = val_loss + val_rae - val_corr
                        improved_global = (not math.isnan(val_corr)) and val_loss < best_rse
                        if improved_global:
                            best_hp = [
                                gcn_depth,
                                lr,
                                conv,
                                res,
                                skip,
                                end,
                                k,
                                dropout,
                                dilation_ex,
                                node_dim,
                                prop_alpha,
                                tanh_alpha,
                                layer,
                                epoch,
                            ]
                            save_dir = os.path.dirname(args.save)
                            if save_dir:
                                os.makedirs(save_dir, exist_ok=True)
                            # Only persist checkpoints every 20 epochs to reduce IO
                            # (still update best_* metrics each time). Also always
                            # save on the final epoch to ensure a final checkpoint.
                            total_epochs = getattr(args, "epochs", 1)
                            should_save = (epoch % 20 == 0) or (epoch >= total_epochs)
                            if should_save:
                                torch.save(
                                    {
                                        "state_dict": core_to_save.state_dict(),
                                        "hparams": best_hp,
                                    },
                                    args.save,
                                )
                                _sync_model_and_scaler_to_bayesian(args)
                            else:
                                print(f"[save-skip] improved_global at epoch={epoch} but skipping save until epoch multiple of 20 (or final epoch {total_epochs})")
                            best_val = sum_loss
                            best_rse = val_loss
                            best_rae = val_rae
                            best_corr = val_corr
                            best_smape = val_smape

                            # Update env in case mc_runs changed mid-run
                            _export_mc_runs_env(args)
                            test_eval_mode = str(getattr(args, "test_eval_mode", "sliding")).lower()
                            test_batch = None
                            if test_eval_mode in ("batch", "both"):
                                test_batch = evaluate(
                                    Data,
                                    Data.test[0],
                                    Data.test[1],
                                    model,
                                    evaluateL2,
                                    evaluateL1,
                                    args.batch_size,
                                    False,
                                    mc_runs=args.mc_runs,
                                )
                            test_sliding = None
                            if test_eval_mode in ("sliding", "both"):
                                test_sliding = (
                                    evaluate_sliding_window(
                                        Data,
                                        Data.test_window,
                                        model,
                                        evaluateL2,
                                        evaluateL1,
                                        args.seq_in_len,
                                        False,
                                        mc_runs=args.mc_runs,
                                    )
                                    if Data.test_window is not None
                                    else (float("inf"), float("inf"), 0.0, float("inf"))
                                )
                            if test_eval_mode == "batch" and test_batch is not None:
                                test_acc, test_rae, test_corr, test_smape = test_batch
                            elif test_eval_mode == "both" and test_sliding is not None:
                                test_acc, test_rae, test_corr, test_smape = test_sliding
                            else:
                                test_acc, test_rae, test_corr, test_smape = test_sliding or (float("inf"), float("inf"), 0.0, float("inf"))

                            # Print validation metrics (RSE, RAE, Corr, sMAPE) for quick inspection
                            try:
                                print(f"[Final Metrics] Test: RSE={test_acc:.6f} RAE={test_rae:.6f} Corr={test_corr:.6f} sMAPE={test_smape:.6f}")
                            except Exception:
                                pass
                            if getattr(args, "runlog", False):
                                jlog(
                                    "epoch_test",
                                    epoch=epoch,
                                    rrse=test_acc,
                                    rae=test_rae,
                                    corr=test_corr,
                                    smape=test_smape,
                                )
                            print("********************************************************************************************************")
                            print(
                                "test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}| test smape {:5.4f}".format(test_acc, test_rae, test_corr, test_smape),
                                flush=True,
                            )
                            print("********************************************************************************************************")
                            best_test_rse = test_acc
                            best_test_corr = test_corr
                        if ensemble_ckpts > 0:
                            total_epochs = getattr(args, "epochs", 1)
                            if (epoch % ensemble_every == 0) or (epoch >= total_epochs):
                                save_dir = os.path.dirname(getattr(args, "save", "")) or "model/Bayesian"
                                os.makedirs(save_dir, exist_ok=True)
                                ens_path = os.path.join(save_dir, f"model_ens_e{int(epoch):04d}.pt")
                                torch.save(
                                    {
                                        "state_dict": core_to_save.state_dict(),
                                        "hparams": best_hp,
                                    },
                                    ens_path,
                                )
                                ensemble_paths.append(ens_path)
                                while len(ensemble_paths) > ensemble_ckpts:
                                    old_path = ensemble_paths.pop(0)
                                    try:
                                        if os.path.exists(old_path):
                                            os.remove(old_path)
                                    except Exception as _e:
                                        print(f"[ensemble] warning: failed to remove {old_path}: {_e}")
                    else:
                        epochs_without_improvement += 1
                        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
                            print(f"[early_stop] Trial {q} stopping early at epoch {epoch} (patience={args.early_stop_patience}).")
                            break
                # Generate validation + testing plots once per trial (after training completes)
                plot_nodes = int(getattr(args, "plot_top_k", 95) or 95)
                print(f"[Plot] Validation: plotting {plot_nodes} nodes")
                print(f"[Plot] Test: plotting {plot_nodes} nodes")
                _generate_epoch_validation_test_plots(Data, model, evaluateL2, evaluateL1)
            except KeyboardInterrupt:
                print("-" * 89)
                print("Exiting from training early")
            except torch.cuda.OutOfMemoryError as oom_err:
                print(f"[OOM] Trial {q} failed due to CUDA OOM: {oom_err}")
                print(f"[OOM] Skipping trial {q} and moving to next configuration...")
                # Clear GPU memory before next trial
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

            # GPU memory cleanup after each trial to prevent OOM in subsequent trials
            try:
                del model
                del optim
                if scheduler is not None:
                    del scheduler
                if grad_scaler is not None:
                    del grad_scaler
                del criterion, evaluateL1, evaluateL2
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    if os.environ.get("BMTGNN_DEBUG_CONFIG", "0") == "1":
                        freed = torch.cuda.memory_allocated() / 1024**2
                        print(f"[cleanup] Trial {q} completed. CUDA memory allocated: {freed:.1f}MB")
            except Exception as cleanup_err:
                print(f"[cleanup] Warning: cleanup after trial {q} failed: {cleanup_err}")

        print("best val loss=", best_val)
        print("best hps=", best_hp)
        # save best hp to disk (ensure output dir exists)
        os.makedirs(os.path.join("model", "Bayesian"), exist_ok=True)
        with open(os.path.join("model", "Bayesian", "hp.txt"), "w") as f:
            f.write(str(best_hp))

    # If using cached HPs and there was no search this run, attempt to load hp.txt (or rely on ckpt)
    if args.use_cached_hp and not best_hp:
        import ast

        hp_path = os.path.join("model", "Bayesian", "hp.txt")
        if os.path.exists(hp_path):
            try:
                best_hp = ast.literal_eval(open(hp_path).read())
                if os.environ.get("BMTGNN_DEBUG_CONFIG", "0") == "1":
                    print("[use_cached_hp] Loaded hp.txt:", best_hp)
            except Exception as e:
                if os.environ.get("BMTGNN_DEBUG_CONFIG", "0") == "1":
                    print(f"[use_cached_hp] Failed to parse hp.txt: {e}")
        else:
            if os.environ.get("BMTGNN_DEBUG_CONFIG", "0") == "1":
                print("[use_cached_hp] hp.txt not found; will rely on checkpoint hparams if present.")

    # Load the best saved model safely: load checkpoint to CPU first to
    # avoid large transient GPU allocations, then move model to the target device.
    # Skip checkpoint loading if --fresh_start was requested

    # Cleanup any residual GPU memory before loading final model
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if os.environ.get("BMTGNN_DEBUG_CONFIG", "0") == "1":
            alloc = torch.cuda.memory_allocated() / 1024**2
            print(f"[cleanup] Before final model load: {alloc:.1f}MB allocated")

    if getattr(args, "_skip_ckpt_reload", False) and isinstance(model, nn.Module):
        # Trainer path already produced a trained model; avoid reloading a mismatched checkpoint.
        pass
    elif getattr(args, "fresh_start", False):
        if os.environ.get("BMTGNN_DEBUG_CONFIG", "0") == "1":
            print("[SKIP] --fresh_start flag detected; skipping checkpoint load.")
        print("[INFO] Starting fresh training with command-line hyperparameters.")
        # Determine correct in_dim based on dual_channel setting
        if getattr(args, "dual_channel", "none") != "none":
            in_dim_actual = 2  # dual-channel input (original + diff/pct)
        else:
            in_dim_actual = int(getattr(args, "in_dim", 1))
        # Create a fresh model with current args (no checkpoint loading)
        model = gtnet(
            gcn_true=args.gcn_true,
            buildA_true=args.buildA_true,
            gcn_depth=args.gcn_depth,
            num_nodes=args.num_nodes,
            device=device,
            predefined_A=predefined_A,
            dropout=args.dropout,
            subgraph_size=args.subgraph_size,
            node_dim=args.node_dim,
            dilation_exponential=args.dilation_exponential,
            conv_channels=args.conv_channels,
            residual_channels=args.residual_channels,
            skip_channels=args.skip_channels,
            end_channels=args.end_channels,
            seq_length=args.seq_in_len,
            in_dim=in_dim_actual,
            out_dim=args.seq_out_len,
            layers=args.layers,
            propalpha=args.propalpha,
            tanhalpha=args.tanhalpha,
            layer_norm_affline=True,
            # Pass optional features
            temporal_attn=bool(getattr(args, "temporal_attn", False)),
            attn_heads=int(getattr(args, "attn_heads", 2)),
            attn_dim=int(getattr(args, "attn_dim", 64)),
            attn_dropout=float(getattr(args, "attn_dropout", 0.10)),
            attn_window=int(getattr(args, "attn_window", 0)),
            attn_math_mode=bool(getattr(args, "attn_math_mode", False)),
            attn_bn_chunk=int(getattr(args, "attn_bn_chunk", 0)),
            temporal_transformer=bool(getattr(args, "temporal_transformer", False)),
            tt_layers=int(getattr(args, "tt_layers", 2)),
            graph_mix=float(getattr(args, "graph_mix", 0.0)),
            dropedge_p=float(getattr(args, "dropedge_p", 0.0)),
            gauss_head=bool(int(getattr(args, "use_gauss", 0)) == 1),
        ).to(device)
        print(f"[FRESH] Created new model: seq_in_len={args.seq_in_len}, layers={args.layers}, in_dim={in_dim_actual}, dual_channel={args.dual_channel}")
    else:
        ckpt = torch.load(args.save, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            loaded_hp = ckpt.get("hparams", best_hp)
            # Treat empty/invalid hparams as missing and fall back to args/inference.
            if not isinstance(loaded_hp, (list, tuple)) or len(loaded_hp) == 0:
                loaded_hp = None

            state = ckpt.get("state_dict", {}) if isinstance(ckpt, dict) else {}
            # If cached hparams disagree with checkpoint shapes, prefer the checkpoint.
            try:
                if loaded_hp and isinstance(state, dict) and len(loaded_hp) >= 6:
                    end_c_ckpt = None
                    skip_c_ckpt = None
                    for k, v in state.items():
                        if not hasattr(v, "shape"):
                            continue
                        if k.endswith("end_conv_1.weight"):
                            end_c_ckpt = int(v.shape[0])
                        elif k.endswith("skip0.weight"):
                            skip_c_ckpt = int(v.shape[0])
                    mismatch = False
                    if end_c_ckpt is not None and int(loaded_hp[5]) != end_c_ckpt:
                        mismatch = True
                    if skip_c_ckpt is not None and int(loaded_hp[4]) != skip_c_ckpt:
                        mismatch = True
                    if mismatch:
                        print("[ckpt-load][warn] Cached hparams mismatch checkpoint shapes; inferring from checkpoint instead.")
                        loaded_hp = None
            except Exception:
                pass

            # Fallback: checkpoint missing hparams (older ckpt) -> synthesize from current args
            if not loaded_hp:
                # If reproducing a metrics JSON run, you may want to keep CLI args authoritative.
                # Setting BMTGNN_SKIP_CKPT_INFER=1 disables shape-based inference (and avoids
                # overriding args.seq_in_len).
                if os.environ.get("BMTGNN_SKIP_CKPT_INFER", "0") == "1":
                    loaded_hp = [
                        getattr(args, "gcn_depth", 2),
                        getattr(args, "lr", getattr(args, "learning_rate", 0.001)),
                        getattr(args, "conv_channels", 32),
                        getattr(args, "residual_channels", 32),
                        getattr(args, "skip_channels", 64),
                        getattr(args, "end_channels", 128),
                        getattr(args, "subgraph_size", 20),
                        getattr(args, "dropout", 0.3),
                        getattr(args, "dilation_exponential", 1),
                        getattr(args, "node_dim", 40),
                        getattr(args, "propalpha", 0.05),
                        getattr(args, "tanhalpha", 3),
                        getattr(args, "layers", 3),
                        -1,
                    ]
                    print("[ckpt-load] checkpoint has no hparams; using CLI args because BMTGNN_SKIP_CKPT_INFER=1")
                else:
                    # Try to infer common hyperparameters from the checkpoint state_dict shapes
                    try:
                        state = ckpt.get("state_dict", {}) if isinstance(ckpt, dict) else {}
                        # defaults
                        conv_c = getattr(args, "conv_channels", 32)
                        res_c = getattr(args, "residual_channels", 32)
                        skip_c = getattr(args, "skip_channels", 64)
                        end_c = getattr(args, "end_channels", 128)
                        layers_c = getattr(args, "layers", 3)
                        seq_candidates = set()
                        # Inspect keys for typical naming conventions
                        for k, v in list(state.items()):
                            if not hasattr(v, "shape"):
                                continue
                            s = tuple(v.shape)
                            # filter_convs.{layer}.tconv.{i}.weight -> (out, in, 1, t)
                            if k.startswith("filter_convs.") and k.endswith(".weight"):
                                parts = k.split(".")
                                try:
                                    lidx = int(parts[1])
                                    layers_c = max(layers_c, lidx + 1)
                                except Exception:
                                    pass
                                if len(s) >= 2:
                                    conv_c = s[1]
                                if len(s) >= 4:
                                    seq_candidates.add(s[3])
                            # gate_convs -> residual channels
                            if k.startswith("gate_convs.") and k.endswith(".weight") and len(s) >= 2:
                                res_c = s[1]
                                try:
                                    lidx = int(k.split(".")[1])
                                    layers_c = max(layers_c, lidx + 1)
                                except Exception:
                                    pass
                            # skip convs
                            if (k.startswith("skip_convs.") or k.startswith("skip0") or k.startswith("skipE")) and k.endswith(".weight"):
                                if len(s) >= 1:
                                    skip_c = s[0]
                                if len(s) >= 4:
                                    seq_candidates.add(s[3])
                            # end convs
                            if ("end_conv" in k or k.startswith("end_conv")) and k.endswith(".weight"):
                                if len(s) >= 1:
                                    end_c = s[0]
                        # CRITICAL FIX: Do NOT infer seq_in_len from kernel sizes - use args!
                        # seq_candidates contains kernel sizes (e.g., 7), NOT sequence lengths (e.g., 48)
                        # Inferring seq_in_len from kernels causes catastrophic shape mismatches
                        seq_in_len_infer = getattr(args, "seq_in_len", 120)  # ALWAYS use command-line arg
                        # assemble loaded_hp in same order used elsewhere
                        loaded_hp = [
                            getattr(args, "gcn_depth", 2),
                            getattr(args, "lr", getattr(args, "learning_rate", 0.001)),
                            int(conv_c),
                            int(res_c),
                            int(skip_c),
                            int(end_c),
                            getattr(args, "subgraph_size", 20),
                            getattr(args, "dropout", 0.3),
                            getattr(args, "dilation_exponential", 1),
                            getattr(args, "node_dim", 40),
                            getattr(args, "propalpha", 0.05),
                            getattr(args, "tanhalpha", 3),
                            int(layers_c),
                            -1,
                        ]
                        # CRITICAL FIX: Do NOT override args.seq_in_len - keep command-line value
                        # The old code was incorrectly setting seq_in_len to kernel size (7) instead of sequence length (48)
                        # This caused shape mismatches and NaN errors in evaluation
                        # args.seq_in_len should NEVER be overridden from checkpoint inference
                        print(
                            "[info] Using seq_in_len from command-line args:",
                            getattr(args, "seq_in_len", None),
                        )
                    except Exception as _e:
                        # fallback to conservative defaults if inference fails
                        loaded_hp = [
                            getattr(args, "gcn_depth", 2),
                            getattr(args, "lr", getattr(args, "learning_rate", 0.001)),
                            getattr(args, "conv_channels", 32),
                            getattr(args, "residual_channels", 32),
                            getattr(args, "skip_channels", 64),
                            getattr(args, "end_channels", 128),
                            getattr(args, "subgraph_size", 20),
                            getattr(args, "dropout", 0.3),
                            getattr(args, "dilation_exponential", 1),
                            getattr(args, "node_dim", 40),
                            getattr(args, "propalpha", 0.05),
                            getattr(args, "tanhalpha", 3),
                            getattr(args, "layers", 3),
                            -1,  # epoch placeholder
                        ]
                        print(
                            "[warning] Checkpoint has no hparams and inference failed; using args-derived fallback hyperparameters.",
                            _e,
                        )

                    # With --use_cached_hp we prefer checkpoint/hp.txt hyperparameters, but if they are missing
                    # (hp.txt is [] in this repo) we still proceed using the derived hparams above.
                    if args.use_cached_hp and (not loaded_hp or len(loaded_hp) < 13):
                        if os.environ.get("BMTGNN_DEBUG_CONFIG", "0") == "1":
                            print("[use_cached_hp][warn] No valid hparams found in checkpoint/hp.txt; proceeding with derived args-based hparams.")
            # If hparams exist but are shorter than expected, pad conservatively
            if len(loaded_hp) < 13:
                print(f"[warning] Loaded hparams length {len(loaded_hp)} < 13; padding with current args.")
                pad_map = [
                    getattr(args, "gcn_depth", 2),
                    getattr(args, "lr", getattr(args, "learning_rate", 0.001)),
                    getattr(args, "conv_channels", 32),
                    getattr(args, "residual_channels", 32),
                    getattr(args, "skip_channels", 64),
                    getattr(args, "end_channels", 128),
                    getattr(args, "subgraph_size", 20),
                    getattr(args, "dropout", 0.3),
                    getattr(args, "dilation_exponential", 1),
                    getattr(args, "node_dim", 40),
                    getattr(args, "propalpha", 0.05),
                    getattr(args, "tanhalpha", 3),
                    getattr(args, "layers", 3),
                    -1,
                ]
                for i in range(len(loaded_hp), 14):
                    loaded_hp.append(pad_map[i])
            # Reconstruct model on CPU (pass a CPU device to the constructor)
            cpu_dev = torch.device("cpu")
            try:
                in_dim_use = getattr(Data, "in_dim", args.in_dim)
            except NameError:
                in_dim_use = args.in_dim
            model = gtnet(
                args.gcn_true,
                args.buildA_true,
                loaded_hp[0],
                int(Data.m),
                cpu_dev,
                Data.adj,
                dropout=loaded_hp[7],
                subgraph_size=loaded_hp[6],
                node_dim=loaded_hp[9],
                dilation_exponential=loaded_hp[8],
                conv_channels=loaded_hp[2],
                residual_channels=loaded_hp[3],
                skip_channels=loaded_hp[4],
                end_channels=loaded_hp[5],
                seq_length=args.seq_in_len,
                in_dim=in_dim_use,
                out_dim=args.seq_out_len,
                layers=loaded_hp[12],
                propalpha=loaded_hp[10],
                tanhalpha=loaded_hp[11],
                layer_norm_affline=False,
                temporal_attn=getattr(args, "temporal_attn", False),
                attn_dim=getattr(args, "attn_dim", 64),
                attn_heads=getattr(args, "attn_heads", 2),
                attn_dropout=getattr(args, "attn_dropout", 0.1),
                attn_window=getattr(args, "attn_window", 0),
                attn_math_mode=getattr(args, "attn_math_mode", False),
                attn_bn_chunk=int(getattr(args, "attn_bn_chunk", 0)),
                attn_gate_threshold=int(getattr(args, "attn_gate_threshold", 0)),
                temporal_transformer=bool(getattr(args, "temporal_transformer", 0)),
                tt_layers=int(getattr(args, "tt_layers", 2)),
                graph_mix=float(getattr(args, "graph_mix", 0.0)),
                dropedge_p=float(getattr(args, "dropedge_p", 0.0)),
                quantiles=q_list,
                nb_head=bool(int(getattr(args, "use_nb_head", 0)) == 1),
                zinb=bool(int(getattr(args, "use_zinb", 0)) == 1),
                gauss_head=bool(int(getattr(args, "use_gauss", 0)) == 1),
            )
            state_dict = ckpt["state_dict"]

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
                missing = getattr(res, "missing_keys", None)
                unexpected = getattr(res, "unexpected_keys", None)
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
                    print("[Checkpoint diagnostic] summary:")
                    print(f"  model params total: {len(model_keys)}, ckpt params: {len(ck_keys)}")
                    print(f"  missing in ckpt: {len(missing_in_ckpt)}; unexpected in ckpt: {len(unexpected_in_ckpt)}; shape mismatches: {len(shape_mismatches)}")
                    if missing_in_ckpt:
                        print("  sample missing (first 20):", missing_in_ckpt[:20])
                    if unexpected_in_ckpt:
                        print("  sample unexpected (first 20):", unexpected_in_ckpt[:20])
                    if shape_mismatches:
                        print("  sample shape mismatches (first 20):")
                        for item in shape_mismatches[:20]:
                            print("    ", item)
            except Exception as diag_e:
                print(
                    "[Checkpoint diagnostic] failed to compute detailed comparison:",
                    diag_e,
                )
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

    # Load scaler and overwrite Data.mu/std just to be explicit and avoid drift
    try:
        s = torch.load(
            os.path.join(os.path.dirname(args.save), "y_scaler.pt"),
            weights_only=True,
            map_location=device,
        )
        Data.mu = s["mean"].to(device)
        Data.std = s["std"].to(device).clamp_min(1e-6)
    except Exception as _e:
        print(f"[scaler_load] warning: {_e}")
    evaluateL2 = nn.MSELoss(reduction="sum").to(device)  # MSE
    evaluateL1 = nn.L1Loss(reduction="sum").to(device)  # MAE

    do_plots = not bool(getattr(args, "no_plots", False))
    # Validation evaluation (with optional calibration)
    vtest_acc, vtest_rae, vtest_corr, vtest_smape = (
        evaluate(
            Data,
            Data.valid[0],
            Data.valid[1],
            model,
            evaluateL2,
            evaluateL1,
            args.batch_size,
            do_plots,
            mc_runs=args.mc_runs,
        )
        if Data.valid[0] is not None and Data.valid[1] is not None
        else (float("inf"), float("inf"), 0.0, float("inf"))
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
            "Validation",
            {
                "RSE": float(vtest_acc),
                "RAE": float(vtest_rae),
                "Corr": float(vtest_corr),
                "sMAPE": float(vtest_smape),
            },
            extras={
                "calibration": getattr(args, "calibration", "none"),
            },
        )
    except Exception as _e:
        jlog("warn_save_metrics", split="Validation", error=str(_e)[:160])

    # Test evaluation (with optional calibration application)
    _export_mc_runs_env(args)
    test_eval_mode = str(getattr(args, "test_eval_mode", "sliding")).lower()
    test_batch = None
    if test_eval_mode in ("batch", "both"):
        test_batch = evaluate(
            Data,
            Data.test[0],
            Data.test[1],
            model,
            evaluateL2,
            evaluateL1,
            args.batch_size,
            do_plots,
            mc_runs=args.mc_runs,
        )
    test_sliding = None
    if test_eval_mode in ("sliding", "both"):
        test_sliding = (
            evaluate_sliding_window(
                Data,
                Data.test_window,
                model,
                evaluateL2,
                evaluateL1,
                args.seq_in_len,
                do_plots,
                mc_runs=args.mc_runs,
            )
            if Data.test_window is not None
            else (float("inf"), float("inf"), 0.0, float("inf"))
        )
    if test_eval_mode == "batch" and test_batch is not None:
        test_acc, test_rae, test_corr, test_smape = test_batch
    elif test_eval_mode == "both" and test_sliding is not None:
        test_acc, test_rae, test_corr, test_smape = test_sliding
    else:
        test_acc, test_rae, test_corr, test_smape = test_sliding or (float("inf"), float("inf"), 0.0, float("inf"))
    print("********************************************************************************************************")
    print("final test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f} | test smape {:5.4f}".format(test_acc, test_rae, test_corr, test_smape))
    print("********************************************************************************************************")

    try:
        _save_metrics(
            args,
            "Testing",
            {
                "RSE": float(test_acc),
                "RAE": float(test_rae),
                "Corr": float(test_corr),
                "sMAPE": float(test_smape),
            },
            extras={
                "calibration": getattr(args, "calibration", "none"),
                "test_eval_mode": test_eval_mode,
                "test_batch_metrics": (
                    {"RSE": float(test_batch[0]), "RAE": float(test_batch[1]), "Corr": float(test_batch[2]), "sMAPE": float(test_batch[3])}
                    if test_batch is not None
                    else None
                ),
                "test_sliding_metrics": (
                    {"RSE": float(test_sliding[0]), "RAE": float(test_sliding[1]), "Corr": float(test_sliding[2]), "sMAPE": float(test_sliding[3])}
                    if test_sliding is not None
                    else None
                ),
            },
        )
        _sync_model_and_scaler_to_bayesian(args)
    except Exception as _e:
        jlog("warn_save_metrics", split="Testing", error=str(_e)[:160])
    # Save calibrated CONF_Z (if available) so forecast.py can reuse it
    try:
        ckpt_dir = Path(os.path.dirname(getattr(args, "save", "")) or "model/Bayesian")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        confz_path = ckpt_dir / "conf_z.json"
        with open(confz_path, "w") as _f:
            json.dump({"CONF_Z": float(_CONF_Z)}, _f)
        print(f"[conf] Saved calibrated CONF_Z={_CONF_Z:.3f} to {confz_path}")
    except Exception as _e:
        print(f"[conf] WARNING: could not save CONF_Z ({_e})")

    return (
        vtest_acc,
        vtest_rae,
        vtest_corr,
        vtest_smape,
        test_acc,
        test_rae,
        test_corr,
        test_smape,
    )


if __name__ == "__main__":
    # Handle quick debug/inspection flags before launching full experiments.
    if getattr(args, "debug_rf", False) or getattr(args, "ckpt_to_compare", ""):
        # Build transient model using CLI args (use args.num_nodes or default 105)
        # For transient debug model, prefer CLI arg for in_dim (Data may not exist yet)
        in_dim_use = args.in_dim
        tmp_model = gtnet(
            args.gcn_true,
            args.buildA_true,
            args.gcn_depth,
            int(getattr(args, "num_nodes", 105)),
            device,
            None,
            dropout=args.dropout,
            subgraph_size=args.subgraph_size,
            node_dim=args.node_dim,
            dilation_exponential=args.dilation_exponential,
            conv_channels=args.conv_channels,
            residual_channels=args.residual_channels,
            skip_channels=args.skip_channels,
            end_channels=args.end_channels,
            seq_length=args.seq_in_len,
            in_dim=in_dim_use,
            out_dim=args.seq_out_len,
            layers=args.layers,
            propalpha=getattr(args, "propalpha", 0.05),
            tanhalpha=getattr(args, "tanhalpha", 3),
            layer_norm_affline=False,
            temporal_attn=getattr(args, "temporal_attn", False),
            attn_dim=getattr(args, "attn_dim", 64),
            attn_heads=getattr(args, "attn_heads", 2),
            attn_dropout=getattr(args, "attn_dropout", 0.1),
            attn_window=getattr(args, "attn_window", 0),
            attn_math_mode=getattr(args, "attn_math_mode", False),
            attn_bn_chunk=int(getattr(args, "attn_bn_chunk", 0)),
            attn_gate_threshold=int(getattr(args, "attn_gate_threshold", 0)),
            temporal_transformer=bool(getattr(args, "temporal_transformer", 0)),
            tt_layers=int(getattr(args, "tt_layers", 2)),
            graph_mix=float(getattr(args, "graph_mix", 0.0)),
            dropedge_p=float(getattr(args, "dropedge_p", 0.0)),
            quantiles=q_list,
            nb_head=bool(int(getattr(args, "use_nb_head", 0)) == 1),
            zinb=bool(int(getattr(args, "use_zinb", 0)) == 1),
            gauss_head=bool(int(getattr(args, "use_gauss", 0)) == 1),
        )
        if getattr(args, "debug_rf", False):
            core = tmp_model
            print(
                "Receptive field (model.receptive_field):",
                getattr(core, "receptive_field", None),
            )
            print("seq_length:", args.seq_in_len)
            # Some model variants may define filter_convs / skip_convs attributes as None; guard iteration.
            for i, conv in enumerate(getattr(core, "filter_convs", []) or []):
                try:
                    sizes = [c.kernel_size for c in conv.tconv] if hasattr(conv, "tconv") else "n/a"
                except Exception:
                    sizes = "n/a"
                print(f"filter_convs[{i}] kernel sizes: {sizes}")
            for i, conv in enumerate(getattr(core, "skip_convs", []) or []):
                try:
                    print(f"skip_convs[{i}] kernel_size = {conv.kernel_size}")
                except Exception:
                    print(f"skip_convs[{i}] kernel_size = n/a")
            try:
                print("skip0.kernel_size =", core.skip0.kernel_size)
                print("skipE.kernel_size =", core.skipE.kernel_size)
            except Exception:
                pass
            print("\n-- debug_rf done --")
        if getattr(args, "ckpt_to_compare", ""):
            out_csv = getattr(args, "ckpt_compare_csv", "") or "ckpt_compare.csv"
            dump_ckpt_vs_model_csv(getattr(args, "ckpt_to_compare"), tmp_model, out_csv)
        sys.exit(0)

    vacc = []
    vrae = []
    vcorr = []
    vsmape = []
    acc = []
    rae = []
    corr = []
    smape = []
    for i in range(1):
        (
            val_acc,
            val_rae,
            val_corr,
            val_smape,
            test_acc,
            test_rae,
            test_corr,
            test_smape,
        ) = main(i)
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
            trial_res = {
                "trial": int(i),
                "valid_rse": float(val_acc),
                "valid_rae": float(val_rae),
                "test_rse": float(test_acc),
                "test_rae": float(test_rae),
                "seed": int(getattr(args, "seed", -1)),
                "subgraph_k": int(getattr(args, "subgraph_size", -1)),
            }
            print("RESULT_JSON:" + json.dumps(trial_res))
        except Exception:
            pass
        if getattr(args, "runlog", False):
            try:
                jlog(
                    "run_finished",
                    best_val_rrse=float(val_acc) if "val_acc" in locals() else None,
                    best_test_rrse=float(test_acc) if "test_acc" in locals() else None,
                )
            except Exception:
                jlog("run_finished", best_val_rrse=None)
    print("\n\n")
    print("1 run average")
    print("\n\n")
    print("valid\trse\trae")
    print("mean\t{:5.4f}\t{:5.4f}".format(np.mean(vacc), np.mean(vrae)))
    print("std\t{:5.4f}\t{:5.4f}".format(np.std(vacc), np.std(vrae)))
    print("\n\n")
    print("test\trse\trae")
    print("mean\t{:5.4f}\t{:5.4f}".format(np.mean(acc), np.mean(rae)))
    print("std\t{:5.4f}\t{:5.4f}".format(np.std(acc), np.std(rae)))

    # Machine-parseable JSON summary for external validators
    try:
        res = {
            "valid_rse_mean": float(np.mean(vacc)),
            "valid_rse_std": float(np.std(vacc)),
            "valid_rae_mean": float(np.mean(vrae)),
            "test_rse_mean": float(np.mean(acc)),
            "test_rae_mean": float(np.mean(rae)),
            "seed": int(getattr(args, "seed", -1)),
            "subgraph_k": int(getattr(args, "subgraph_size", -1)),
        }
        print("RESULT_JSON:" + json.dumps(res))
    except Exception:
        pass
