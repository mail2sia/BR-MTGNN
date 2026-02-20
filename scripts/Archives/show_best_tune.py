#!/usr/bin/env python3
"""
Print the best tuning run and its smoothing parameters from runs/*/metrics_validation.json.

Example:
  python scripts/show_best_tune.py --runs-dir runs
"""
import argparse
import json
import os
from typing import Any, Dict, Tuple


def _load_metrics(path: str) -> Tuple[float, float, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    metrics = data.get("metrics", {})
    args = data.get("args", {})
    rse = float(metrics.get("RSE", float("nan")))
    rae = float(metrics.get("RAE", float("nan")))
    return rse, rae, args


def main() -> int:
    p = argparse.ArgumentParser(description="Show best tuning params from runs")
    p.add_argument("--runs-dir", type=str, default="runs")
    args = p.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    runs_dir = os.path.join(root, args.runs_dir)
    if not os.path.isdir(runs_dir):
        print(f"[best_tune] runs dir not found: {runs_dir}")
        return 1

    best = None
    for name in os.listdir(runs_dir):
        run_dir = os.path.join(runs_dir, name)
        if not os.path.isdir(run_dir):
            continue
        mpath = os.path.join(run_dir, "metrics_validation.json")
        if not os.path.isfile(mpath):
            continue
        try:
            rse, rae, run_args = _load_metrics(mpath)
        except Exception:
            continue
        score = rse + rae
        if best is None or score < best[0]:
            best = (score, rse, rae, run_dir, run_args)

    if best is None:
        print("[best_tune] No metrics_validation.json files found.")
        return 2

    _, rse, rae, run_dir, run_args = best
    print("[best_tune] run_dir:", run_dir)
    print(f"[best_tune] Validation: RSE={rse:.6f} RAE={rae:.6f}")

    keys = ["aggr_alpha", "aggr_beta", "aggr_gauss_sigma", "aggr_ma_window"]
    print("[best_tune] smoothing params:")
    for k in keys:
        if k in run_args:
            print(f"  {k} = {run_args[k]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
