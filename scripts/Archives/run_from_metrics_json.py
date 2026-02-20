import argparse
import json
import os
import subprocess
import sys
from typing import Any


# Some argparse options in train_test.py use `action='store_true'` (flags), while a few
# others are boolean-like but expect an explicit value (e.g., "--use_chronological_split 1").
STORE_TRUE_KEYS = {
    "runlog",
    "L1Loss",
    "gcn_true",
    "buildA_true",
    "amp",
    "no_dataparallel",
    "cudnn_benchmark",
    "vectorized_mc",
    "no_vectorized_mc",
    "trainer_mode",
    "debug_rf",
    "use_cached_hp",
    "eval_only",
    "no_plots",
    "enable_test_plots",
    "smooth_l1_only",
    "use_weighted_horizon_loss",
    "plot_norm_space",
    "movement_debug",
    "chronological_split",
    "robust_metrics",
    "residual_head",
    "ensemble",
    "conformal",
    "ens_patchtst",
    "ens_nhits",
    "ens_mlp",
    "auto_tune_dropout",
    "smooth_plot",
    "conf_calibrate",
    "temporal_attn",
    "attn_math_mode",
    "nan_debug",
    "auto_window_adjust",
    "debug_layout",
    "log_gpu_mem",
    "log_peak_mem",
    "strong_rmdpt",
    "weight_nodes_in_loss",
    "weight_nodes_in_metrics",
    "allow_wide_end",
    "fused_optim",
}

BOOL_VALUE_KEYS = {
    "use_chronological_split",
}


def _is_nullish(v: Any) -> bool:
    return v is None


def _stringy(v: Any) -> bool:
    return isinstance(v, str)


def _should_skip(k: str, v: Any) -> bool:
    # These are not CLI args for train_test.py, or are handled separately.
    if k in {"device", "save", "out_dir"}:
        return True
    # Empty strings usually mean "not set".
    if _stringy(v) and v == "":
        return True
    if _is_nullish(v):
        return True
    return False


def main() -> int:
    p = argparse.ArgumentParser(description="Run scripts/train_test.py using args from a metrics_*.json file")
    p.add_argument("metrics_json", help="Path to metrics_validation.json (or metrics_test.json) containing an args dict")
    p.add_argument("--device", default=None, help="Override device (e.g. cpu or cuda:0)")
    p.add_argument("--save", default=None, help="Override checkpoint path")
    p.add_argument("--out_dir", default=None, help="Override output directory")
    p.add_argument("--skip_ckpt_infer", action="store_true", help="Set BMTGNN_SKIP_CKPT_INFER=1")
    p.add_argument("--dry_run", action="store_true", help="Print command only")
    args = p.parse_args()

    with open(args.metrics_json, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    if "args" not in metrics or not isinstance(metrics["args"], dict):
        raise SystemExit("metrics_json must contain a top-level 'args' object")

    run_args: dict[str, Any] = dict(metrics["args"])

    # Apply overrides
    if args.device is not None:
        run_args["device"] = args.device
    if args.save is not None:
        run_args["save"] = args.save
    if args.out_dir is not None:
        run_args["out_dir"] = args.out_dir

    cmd: list[str] = [sys.executable, "-u", os.path.join("scripts", "train_test.py")]

    # First handle common required args explicitly
    if "save" in run_args and run_args["save"]:
        cmd += ["--save", str(run_args["save"])]
    if "out_dir" in run_args and run_args["out_dir"]:
        cmd += ["--out_dir", str(run_args["out_dir"])]
    if "device" in run_args and run_args["device"]:
        cmd += ["--device", str(run_args["device"])]

    # Now translate the rest
    for k, v in run_args.items():
        if _should_skip(k, v):
            continue
        flag = f"--{k}"
        if isinstance(v, bool):
            if k in BOOL_VALUE_KEYS:
                cmd += [flag, "1" if v else "0"]
            elif k in STORE_TRUE_KEYS:
                if v:
                    cmd.append(flag)
            else:
                # Default behavior for unknown boolean-like args: pass as 1/0
                cmd += [flag, "1" if v else "0"]
        else:
            cmd += [flag, str(v)]

    env = os.environ.copy()
    if args.skip_ckpt_infer:
        env["BMTGNN_SKIP_CKPT_INFER"] = "1"

    print("Running:")
    print(" ".join(cmd))

    if args.dry_run:
        return 0

    proc = subprocess.run(cmd, env=env)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
