#!/usr/bin/env python3
"""
Targeted tuning wrapper to auto-sweep aggressive smoothing strength
until Validation RSE/RAE drop below a target.

Example:
  python scripts/targeted_tuning.py --data data/sm_data_g.csv --device cuda:0 \
    --train --trainer_mode --has_header --drop_first_col

Any unknown args are forwarded to scripts/train_test.py.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from typing import List, Tuple


def _strip_flag(args: List[str], name: str) -> List[str]:
    key = f"--{name}"
    out = []
    skip_next = False
    for i, a in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if a == key:
            # consume optional value if provided as separate token
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                skip_next = True
            continue
        if a.startswith(key + "="):
            continue
        out.append(a)
    return out


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


def _ensure_flag(args: List[str], flag: str) -> List[str]:
    if flag in args:
        return args
    return args + [flag]


def main() -> int:
    p = argparse.ArgumentParser(description="Targeted smoothing sweep for Validation RSE/RAE")
    p.add_argument("--alpha-list", type=str, default="0.01,0.02,0.03,0.05")
    p.add_argument("--beta-list", type=str, default="0.01,0.02,0.03,0.05")
    p.add_argument("--gauss-list", type=str, default="0.8,1.0,1.2,1.5")
    p.add_argument("--ma-window-list", type=str, default="3,5,7")
    p.add_argument("--target-rse", type=float, default=0.6)
    p.add_argument("--target-rae", type=float, default=0.6)
    p.add_argument("--score-w-rse", type=float, default=0.5,
                   help="Weight for RSE in composite score.")
    p.add_argument("--score-w-rae", type=float, default=0.5,
                   help="Weight for RAE in composite score.")
    p.add_argument("--score-w-ci", type=float, default=0.15,
                   help="Penalty weight for val_ci_ratio (tighter plots -> lower score).")
    p.add_argument("--max-trials", type=int, default=32)
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--run-tag-prefix", type=str, default="tune")
    p.add_argument("--timeout", type=int, default=0, help="Per-trial timeout in seconds (0 disables)")
    p.add_argument("--dry-run", action="store_true")

    known, unknown = p.parse_known_args()

    # Base args forwarded to train_test.py
    base_args = unknown
    base_args = _strip_flag(base_args, "run_tag")
    base_args = _strip_flag(base_args, "aggressive_smooth")
    base_args = _strip_flag(base_args, "aggr_alpha")
    base_args = _strip_flag(base_args, "aggr_beta")
    base_args = _strip_flag(base_args, "aggr_gauss_sigma")
    base_args = _strip_flag(base_args, "aggr_ma_window")
    base_args = _strip_flag(base_args, "aggr_out_prefix")
    base_args = _strip_flag(base_args, "aggr_force")

    # Ensure aggressive smoothing is on
    base_args = _ensure_flag(base_args, "--aggressive_smooth")

    # Parse sweep lists
    def _parse_list(s: str) -> List[float]:
        out = []
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            out.append(float(part))
        return out

    alphas = _parse_list(known.alpha_list)
    betas = _parse_list(known.beta_list)
    gauss = _parse_list(known.gauss_list)
    ma_windows = [int(x) for x in _parse_list(known.ma_window_list)]

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_test = os.path.join(root, "scripts", "train_test.py")
    py = sys.executable

    trial = 0
    best_score = float("inf")
    best_run = None
    for a in alphas:
        for b in betas:
            for g in gauss:
                for m in ma_windows:
                    trial += 1
                    if trial > int(known.max_trials):
                        print("[tune] Reached max trials; stopping.")
                        return 1

                    run_tag = f"{known.run_tag_prefix}_{time.strftime('%Y%m%d-%H%M%S')}_{trial:02d}"
                    run_dir = os.path.join(root, known.runs_dir, run_tag)

                    cmd = [
                        py,
                        train_test,
                        *base_args,
                        "--aggressive_smooth",
                        "--aggr_alpha", str(a),
                        "--aggr_beta", str(b),
                        "--aggr_gauss_sigma", str(g),
                        "--aggr_ma_window", str(m),
                        "--aggr_out_prefix", os.path.join(known.runs_dir, run_tag, "sm_aggr"),
                        "--aggr_force",
                        "--run_tag", run_tag,
                    ]

                    print("[tune] Trial", trial)
                    print("[tune] params:", f"alpha={a}", f"beta={b}", f"gauss={g}", f"ma={m}")
                    print("[tune] run_tag:", run_tag)
                    if known.dry_run:
                        print("[tune] dry-run cmd:", " ".join(cmd))
                        continue

                    timeout = None if int(known.timeout) <= 0 else int(known.timeout)
                    res = subprocess.run(cmd, cwd=root, check=False, timeout=timeout)
                    if res.returncode != 0:
                        print(f"[tune] train_test failed (code={res.returncode}).")
                        continue

                    try:
                        rse, rae, ci_ratio = _load_validation_metrics(run_dir)
                    except Exception as e:
                        print(f"[tune] metrics load failed: {e}")
                        continue

                    w_rse = float(known.score_w_rse)
                    w_rae = float(known.score_w_rae)
                    w_ci = float(known.score_w_ci)
                    ci_pen = 0.0 if not (ci_ratio == ci_ratio) else ci_ratio
                    score = (w_rse * rse) + (w_rae * rae) + (w_ci * ci_pen)

                    if score < best_score:
                        best_score = score
                        best_run = run_tag

                    print(
                        f"[tune] Validation: RSE={rse:.6f} RAE={rae:.6f} "
                        f"CIratio={ci_ratio:.4f} score={score:.6f}"
                    )
                    if rse <= known.target_rse and rae <= known.target_rae:
                        print("[tune] Target achieved. Stopping.")
                        print(f"[tune] Best score so far: {best_score:.6f} run={best_run}")
                        return 0

    print("[tune] Sweep complete without reaching target.")
    if best_run is not None:
        print(f"[tune] Best score: {best_score:.6f} run={best_run}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
