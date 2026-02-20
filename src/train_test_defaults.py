import os
import sys

import numpy as np


def _flag_was_set(name: str) -> bool:
    """
    True if user explicitly set --<name> or --<name>=... in CLI.
    Keeps strong mode from overriding your manual choices.
    """
    key = f"--{name}"
    for a in sys.argv[1:]:
        if a == key or a.startswith(key + "="):
            return True
    return False


def _apply_defaults_if_unset(args, defaults: dict) -> None:
    """Set multiple args from a dict only if the CLI did not explicitly set them."""
    for k, v in defaults.items():
        if not _flag_was_set(k):
            try:
                setattr(args, k, v)
            except Exception:
                pass


def _apply_curve_hugging_loss_defaults(
    args,
    *,
    nonzero_weight: float,
    horizon_gamma: float,
    huber_delta: float,
) -> None:
    _apply_defaults_if_unset(
        args,
        {
            "use_weighted_horizon_loss": True,
            "nonzero_weight": nonzero_weight,
            "horizon_gamma": horizon_gamma,
            "huber_delta": huber_delta,
            "loss_beta": 0.0,
            "loss_gamma": 0.3,
            "mae_weight": 1.0,
            "movement_loss_weight": 0.6,
        },
    )


def _apply_low_regularization_hugging_defaults(args) -> None:
    _apply_defaults_if_unset(
        args,
        {
            "dropout": 0.0,
            "dropedge_p": 0.0,
            "smooth_plot": False,
            "plot_trend_smooth": False,
        },
    )


def _apply_strong_rmdpt_defaults(args):
    _apply_defaults_if_unset(
        args,
        {
            "normalize": 4,
            "dual_channel": "pct",
            "pct_clip": 3.0,
            "y_transform": "log1p",
        },
    )

    _apply_curve_hugging_loss_defaults(
        args,
        nonzero_weight=8.0,
        horizon_gamma=2.5,
        huber_delta=1.5,
    )

    if not _flag_was_set("graph_mix"):
        args.graph_mix = max(float(getattr(args, "graph_mix", 0.0)), 0.6)
    if not _flag_was_set("dropedge_p"):
        args.dropedge_p = max(float(getattr(args, "dropedge_p", 0.0)), 0.10)
    _apply_defaults_if_unset(
        args,
        {
            "residual_head": True,
            "graph_normalize": "sym",
            "weight_nodes_in_loss": True,
        },
    )

    _apply_defaults_if_unset(
        args,
        {
            "temporal_attn": True,
            "temporal_transformer": 1,
            "tt_layers": 3,
            "attn_heads": 4,
            "attn_dim": 128,
            "attn_dropout": 0.08,
        },
    )

    _apply_defaults_if_unset(
        args,
        {
            "lr": 4e-4,
            "weight_decay": 1e-5,
            "dropout": 0.0,
            "clip": 1.5,
            "early_stop_patience": 100,
        },
    )

    _apply_defaults_if_unset(
        args,
        {
            "layers": 5,
            "seq_in_len": 24,
            "seq_out_len": 36,
            "conv_channels": 96,
            "residual_channels": 96,
            "skip_channels": 192,
            "end_channels": 384,
            "batch_size": 8,
            "epochs": 300,
        },
    )

    _apply_defaults_if_unset(
        args,
        {
            "robust_metrics": True,
            "weight_nodes_in_metrics": True,
            "smooth_plot": False,
            "plot_trend_smooth": False,
        },
    )


def _apply_hugging_mode_defaults(args):
    _apply_defaults_if_unset(
        args,
        {
            "normalize": 3,
            "dual_channel": "pct",
            "pct_clip": 3.0,
            "y_transform": "log1p",
            "aggressive_smooth": True,
            "aggr_alpha": 0.01,
            "aggr_beta": 0.01,
            "aggr_gauss_sigma": 1.2,
            "aggr_ma_window": 5,
        },
    )

    _apply_curve_hugging_loss_defaults(args, nonzero_weight=6.0, horizon_gamma=2.0, huber_delta=2.0)
    _apply_defaults_if_unset(args, {"loss_alpha": 1.5})

    _apply_defaults_if_unset(
        args,
        {
            "residual_head": True,
            "graph_mix": 0.6,
            "graph_normalize": "sym",
            "weight_nodes_in_loss": True,
            "weight_nodes_in_metrics": True,
        },
    )

    _apply_defaults_if_unset(
        args,
        {
            "temporal_attn": True,
            "temporal_transformer": 1,
            "tt_layers": 2,
            "attn_heads": 4,
            "attn_dim": 96,
            "attn_dropout": 0.0,
        },
    )

    _apply_low_regularization_hugging_defaults(args)
    _apply_defaults_if_unset(args, {"epochs": 400, "early_stop_patience": 120, "mc_runs": 50})

    _apply_defaults_if_unset(
        args,
        {
            "layers": 5,
            "seq_in_len": 24,
            "seq_out_len": 36,
            "conv_channels": 96,
            "residual_channels": 96,
            "skip_channels": 192,
            "end_channels": 384,
            "batch_size": 8,
        },
    )


def _maybe_generate_aggressive_smoothed_data(args) -> None:
    if not getattr(args, "aggressive_smooth", False):
        return
    in_path = getattr(args, "data", "")
    if not in_path:
        print("[aggressive_smooth] No input data path provided; skipping.")
        return

    out_prefix = getattr(args, "aggr_out_prefix", "data/sm_aggr")
    out_csv = out_prefix + ".csv"
    out_txt = out_prefix + ".txt"
    if (not getattr(args, "aggr_force", False)) and os.path.exists(out_csv):
        args.data = out_csv
        if not _flag_was_set("has_header"):
            args.has_header = False
        if not _flag_was_set("drop_first_col"):
            args.drop_first_col = False
        print(f"[aggressive_smooth] Using cached smoothed data: {out_csv}")
        return

    try:
        import pandas as pd
        from scipy.ndimage import gaussian_filter1d
    except Exception as _e:
        print(f"[aggressive_smooth] Missing dependency: {_e}. Install pandas + scipy or disable --aggressive_smooth.")
        return

    def _read_matrix(path: str) -> np.ndarray:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".txt":
            arr = np.loadtxt(path, delimiter="\t")
            if arr.ndim == 1:
                arr = arr[:, None]
            return arr.astype(float)
        if ext == ".csv":
            df = pd.read_csv(path)
            if df.shape[1] > 1:
                col0 = str(df.columns[0]).lower()
                if col0 in ("date", "month", "time"):
                    df = df.iloc[:, 1:]
                elif getattr(args, "drop_first_col", False):
                    df = df.iloc[:, 1:]
            return df.to_numpy(dtype=float)
        raise ValueError("Unsupported input extension")

    def _double_exp(series: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        T = len(series)
        if T == 0:
            return np.array([], dtype=float)
        level = series[0]
        trend = series[1] - series[0] if T > 1 else 0.0
        out = np.empty(T, dtype=float)
        out[0] = series[0]
        for t in range(1, T):
            last_level = level
            level = alpha * series[t] + (1 - alpha) * (level + trend)
            trend = beta * (level - last_level) + (1 - beta) * trend
            out[t] = level + trend
        return out

    def _moving_avg(a: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return a
        return np.convolve(a, np.ones(window) / window, mode="same")

    def _agg_smooth_matrix(
        X: np.ndarray,
        alpha: float,
        beta: float,
        gauss_sigma: float,
        ma_window: int,
    ) -> np.ndarray:
        T, N = X.shape
        S = np.empty_like(X, dtype=float)
        for j in range(N):
            col = X[:, j]
            s = _double_exp(col, alpha, beta)
            if gauss_sigma > 0:
                s = gaussian_filter1d(s, sigma=gauss_sigma, mode="reflect")
            if ma_window > 1:
                s = _moving_avg(s, ma_window)
            S[:, j] = s
        return S

    X = _read_matrix(in_path)
    S = _agg_smooth_matrix(
        X,
        alpha=float(getattr(args, "aggr_alpha", 0.01)),
        beta=float(getattr(args, "aggr_beta", 0.01)),
        gauss_sigma=float(getattr(args, "aggr_gauss_sigma", 1.2)),
        ma_window=int(getattr(args, "aggr_ma_window", 5)),
    )

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        for row in S:
            f.write(",".join([f"{x:.6f}" for x in row]) + "\n")
    np.savetxt(out_txt, S, fmt="%.6f", delimiter="\t")
    args.data = out_csv
    if not _flag_was_set("has_header"):
        args.has_header = False
    if not _flag_was_set("drop_first_col"):
        args.drop_first_col = False
    print(f"[aggressive_smooth] Wrote {out_csv} and {out_txt} ({S.shape[0]}x{S.shape[1]})")


def _get_cli_keys() -> list[str]:
    keys = []
    for _tok in sys.argv[1:]:
        if _tok.startswith("--"):
            name = _tok[2:].split("=")[0].replace("-", "_")
            if name:
                keys.append(name)
    return sorted(set(keys))
