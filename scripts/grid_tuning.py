#!/usr/bin/env python3
"""
Grid tuning wrapper to sweep specified hyperparameters.

Example:
  python scripts/grid_tuning.py --config configs/grid_tune_combo.json

Unknown args are not accepted; provide overrides in the config file.
"""
import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _strip_flag(args: List[str], name: str) -> List[str]:
    key = f"--{name}"
    out: List[str] = []
    skip_next = False
    for i, a in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if a == key:
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                skip_next = True
            continue
        if a.startswith(key + "="):
            continue
        out.append(a)
    return out


def _has_flag(args: List[str], name: str) -> bool:
    key = f"--{name}"
    for a in args:
        if a == key or a.startswith(key + "="):
            return True
    return False


def _load_validation_metrics(run_dir: str) -> Tuple[float, float, float]:
    path = os.path.join(run_dir, "metrics_validation.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    metrics = data.get("metrics", {})
    extras = data.get("extras", {})
    rse = float(metrics.get("RSE", float("nan")))
    rae = float(metrics.get("RAE", float("nan")))
    ci_ratio = float(extras.get("val_ci_ratio", float("nan")))
    return rse, rae, ci_ratio


def _iter_grid(grid: Dict[str, Iterable[object]]) -> Iterable[Dict[str, object]]:
    keys = list(grid.keys())
    values = [list(grid[k]) for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def _apply_grid_args(base_args: List[str], combo: Dict[str, object]) -> List[str]:
    args = list(base_args)
    for name, value in combo.items():
        args = _strip_flag(args, name)
        if isinstance(value, bool):
            if value:
                args.append(f"--{name}")
            continue
        args.extend([f"--{name}", str(value)])
    return args


def _copy_all_plots(run_dir: str, dashboard_dir: str, run_tag: str, rank: int):
    """Copy all plot images from a run to dashboard for visual monitoring."""
    plots_dir = os.path.join(run_dir, "plots")
    if not os.path.exists(plots_dir):
        return
    
    rank_dir = os.path.join(dashboard_dir, f"rank{rank:02d}_{run_tag}")
    os.makedirs(rank_dir, exist_ok=True)
    
    # Copy all generated plot images
    for fname in os.listdir(plots_dir):
        if fname.endswith((".png", ".pdf")):
            src = os.path.join(plots_dir, fname)
            dst = os.path.join(rank_dir, fname)
            try:
                shutil.copy2(src, dst)
            except Exception:
                pass


def _update_dashboard(top_runs: List[Tuple[float, str, Dict, float, float, float]], 
                      dashboard_dir: str, total_trials: int):
    """Create dashboard summary showing top runs"""
    summary_path = os.path.join(dashboard_dir, "TOP_RUNS_SUMMARY.txt")
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("HYPERPARAMETER SEARCH - TOP RUNS DASHBOARD\n")
        f.write(f"Total trials completed: {total_trials}\n")
        f.write(f"Top {len(top_runs)} runs tracked\n")
        f.write("=" * 80 + "\n\n")
        
        for i, (score, run_tag, params, rse, rae, ci_ratio) in enumerate(top_runs, 1):
            f.write(f"RANK {i}: {run_tag}\n")
            f.write(f"  Score: {score:.6f}  RSE: {rse:.6f}  RAE: {rae:.6f}  CI: {ci_ratio:.4f}\n")
            f.write(f"  Params: {', '.join(f'{k}={v}' for k, v in params.items())}\n")
            f.write(f"  Plots: rank{i:02d}_{run_tag}/\n")
            f.write("\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("To stop the search when satisfied with plots:\n")
        f.write(f"  Create file: {os.path.join(dashboard_dir, 'STOP_SEARCH')}\n")
        f.write("=" * 80 + "\n")
    
    return


def _save_best_hp(best_params: Dict[str, object], root: str) -> str:
    """Persist best grid params to model/Bayesian/hp.txt."""
    hp_dir = os.path.join(root, "model", "Bayesian")
    os.makedirs(hp_dir, exist_ok=True)
    hp_path = os.path.join(hp_dir, "hp.txt")
    with open(hp_path, "w", encoding="utf-8") as f:
        f.write(str(best_params))
    return hp_path


# ---------------------------------------------------------------------------
# Best model syncing
# Copies the best model and scaler from a run directory into model/Bayesian
def _copy_best_model(run_dir: str, root: str) -> None:
    """Copy model.pt and y_scaler.pt from run_dir to model/Bayesian."""
    try:
        src_model = os.path.join(run_dir, 'model.pt')
        if not os.path.exists(src_model):
            metrics_path = os.path.join(run_dir, 'metrics_validation.json')
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, 'r', encoding='utf-8') as f:
                        payload = json.load(f)
                    save_arg = str(payload.get('args', {}).get('save', '')).strip()
                    if save_arg:
                        cand = save_arg if os.path.isabs(save_arg) else os.path.join(root, save_arg)
                        if os.path.isdir(cand):
                            cand = os.path.join(cand, 'model.pt')
                        if os.path.exists(cand):
                            src_model = cand
                except Exception:
                    pass
        dst_dir = os.path.join(root, 'model', 'Bayesian')
        os.makedirs(dst_dir, exist_ok=True)
        if os.path.exists(src_model):
            shutil.copy2(src_model, os.path.join(dst_dir, 'model.pt'))
        scaler_src = os.path.join(run_dir, 'y_scaler.pt')
        if not os.path.exists(scaler_src):
            model_dir = os.path.dirname(src_model)
            cand_scaler = os.path.join(model_dir, 'y_scaler.pt')
            if os.path.exists(cand_scaler):
                scaler_src = cand_scaler
        if os.path.exists(scaler_src):
            shutil.copy2(scaler_src, os.path.join(dst_dir, 'y_scaler.pt'))
    except Exception:
        pass


def main() -> int:
    p = argparse.ArgumentParser(description="Grid sweep for train_test.py")
    p.add_argument("--config", type=str, required=True, help="Path to grid config JSON.")
    p.add_argument("--max-trials", type=int, default=0, help="Limit trials (0 = no limit).")
    p.add_argument("--timeout", type=int, default=0, help="Per-trial timeout in seconds (0 disables).")
    p.add_argument("--top-n", type=int, default=25, help="Track top N runs for dashboard (default: 25).")
    p.add_argument("--dry-run", action="store_true")

    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Optional runtime caps from config (used when CLI values are not provided)
    cfg_max_trials = int(cfg.get("max_trials", 0) or 0)
    cfg_timeout = int(cfg.get("timeout", 0) or 0)
    cfg_top_n = int(cfg.get("top_n", 0) or 0)
    if int(args.max_trials) <= 0 and cfg_max_trials > 0:
        args.max_trials = cfg_max_trials
    if int(args.timeout) <= 0 and cfg_timeout > 0:
        args.timeout = cfg_timeout
    if int(args.top_n) <= 0 and cfg_top_n > 0:
        args.top_n = cfg_top_n

    # Convert base_args from dict to list with proper -- prefixes
    base_args_cfg = cfg.get("base_args", {})
    if isinstance(base_args_cfg, dict):
        base_args = []
        for key, value in base_args_cfg.items():
            if isinstance(value, bool):
                if value:
                    base_args.append(f"--{key}")
            else:
                base_args.extend([f"--{key}", str(value)])
    else:
        base_args = list(base_args_cfg)
    
    # Support both "grid" and "params" keys for hyperparameter grid
    grid = cfg.get("grid", cfg.get("params", {}))
    targets = cfg.get("targets", {})
    score = cfg.get("score", {})
    force_fresh_start = bool(cfg.get("force_fresh_start", False))

    target_rse = float(targets.get("rse", 0.6))
    target_rae = float(targets.get("rae", 0.6))
    w_rse = float(score.get("w_rse", 0.5))
    w_rae = float(score.get("w_rae", 0.5))
    w_ci = float(score.get("w_ci", 0.15))

    run_tag_prefix = cfg.get("run_tag_prefix", "grid")
    runs_dir = cfg.get("runs_dir", "runs")

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_test = os.path.join(root, "scripts", "train_test.py")
    py = sys.executable

    # Create dashboard directory
    dashboard_dir = os.path.join(root, f"dashboard_{run_tag_prefix}")
    os.makedirs(dashboard_dir, exist_ok=True)
    
    # Track top N runs: (score, run_tag, params, rse, rae, ci_ratio)
    top_runs = []

    base_args = _strip_flag(base_args, "run_tag")

    trial = 0
    best_score = float("inf")
    best_run = None
    best_params: Dict[str, object] = {}

    for combo in _iter_grid(grid):
        trial += 1
        
        # Check for manual stop signal
        stop_file = os.path.join(dashboard_dir, "STOP_SEARCH")
        if os.path.exists(stop_file):
            print(f"\n[grid] STOP signal detected: {stop_file}")
            print("[grid] User requested early termination.")
            break
        
        if args.max_trials and trial > int(args.max_trials):
            print("[grid] Reached max trials; stopping.")
            return 1

        run_tag = f"{run_tag_prefix}_{time.strftime('%Y%m%d-%H%M%S')}_{trial:03d}"
        run_dir = os.path.join(root, runs_dir, run_tag)

        cmd_args = _apply_grid_args(base_args, combo)
        if force_fresh_start and not _has_flag(cmd_args, "fresh_start"):
            cmd_args.append("--fresh_start")
        cmd = [py, train_test, *cmd_args, "--run_tag", run_tag]

        print(f"\n{'='*80}")
        print(f"[grid] Trial {trial}")
        print("[grid] params:", ", ".join(f"{k}={v}" for k, v in combo.items()))
        print("[grid] run_tag:", run_tag)
        if args.dry_run:
            print("[grid] dry-run cmd:", " ".join(cmd))
            continue

        timeout = None if int(args.timeout) <= 0 else int(args.timeout)
        # Set PYTHONPATH to include project root for src module imports
        env = os.environ.copy()
        env['PYTHONPATH'] = root + os.pathsep + env.get('PYTHONPATH', '')
        res = subprocess.run(cmd, cwd=root, env=env, check=False, timeout=timeout)
        if res.returncode != 0:
            print(f"[grid] train_test failed (code={res.returncode}).")
            continue

        try:
            rse, rae, ci_ratio = _load_validation_metrics(run_dir)
        except Exception as exc:
            print(f"[grid] metrics load failed: {exc}")
            continue

        ci_pen = 0.0 if not (ci_ratio == ci_ratio) else ci_ratio
        score_val = (w_rse * rse) + (w_rae * rae) + (w_ci * ci_pen)

        # Update top runs leaderboard
        top_runs.append((score_val, run_tag, dict(combo), rse, rae, ci_ratio))
        top_runs.sort(key=lambda x: x[0])  # Sort by score (lower is better)
        top_runs = top_runs[:args.top_n]  # Keep only top N
        
        # Update dashboard with current top runs
        _update_dashboard(top_runs, dashboard_dir, trial)
        
        # Copy plots from top runs to dashboard
        for rank, (s, tag, params, r, a, c) in enumerate(top_runs, 1):
            run_path = os.path.join(root, runs_dir, tag)
            _copy_all_plots(run_path, dashboard_dir, tag, rank)

        if score_val < best_score:
            best_score = score_val
            best_run = run_tag
            best_params = dict(combo)
            # copy checkpoint & scaler immediately
            _copy_best_model(run_dir, root)

        print(
            f"[grid] Validation: RSE={rse:.6f} RAE={rae:.6f} "
            f"CIratio={ci_ratio:.4f} score={score_val:.6f}"
        )
        
        # Show current rank
        current_rank = next((i for i, (s, t, _, _, _, _) in enumerate(top_runs, 1) if t == run_tag), None)
        if current_rank:
            print(f"[grid] ⭐ This run ranks #{current_rank} of {len(top_runs)} tracked")
        
        if rse <= target_rse and rae <= target_rae:
            print("[grid] Target achieved. Stopping.")
            print(f"[grid] Best score so far: {best_score:.6f} run={best_run}")
            _update_dashboard(top_runs, dashboard_dir, trial)
            if best_params:
                hp_path = _save_best_hp(best_params, root)
                print(f"[grid] Saved best hp to: {hp_path}")
            return 0

    print("\n" + "="*80)
    print("[grid] Sweep complete.")
    if best_run is not None:
        print(f"[grid] Best score: {best_score:.6f} run={best_run}")
        print(f"[grid] Dashboard: {dashboard_dir}")
        if best_params:
            hp_path = _save_best_hp(best_params, root)
            print(f"[grid] Saved best hp to: {hp_path}")
        # ensure final best checkpoint is synced
        run_dir = os.path.join(root, runs_dir, best_run)
        _copy_best_model(run_dir, root)
        print(f"[grid] Copied best model and scaler from {best_run} to model/Bayesian")
    return 0 if best_run else 2


if __name__ == "__main__":
    raise SystemExit(main())
