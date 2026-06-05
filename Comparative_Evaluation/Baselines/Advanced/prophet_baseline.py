"""
Prophet baseline for BR-MTGNN comparative evaluation.

Fits one Prophet model per time series (univariate, per-node).
Uses the same 70/20/10 chronological train/val/test split as BR-MTGNN.

Train  : first 60% of rows  (matches common.py load_data n_train)
Val    : next  20% of rows
Test   : remaining 20% of rows

Metrics reported (identical definitions to common.py compute_metrics):
  RAE  - Relative Absolute Error
  RSE  - Relative Squared Error
  Corr - Mean Pearson correlation across nodes
  Coverage - fraction of test actuals within Prophet's yhat_lower/yhat_upper

Prophet settings:
  - yearly_seasonality=True  (monthly data spans multiple years)
  - weekly_seasonality=False (monthly granularity; no sub-week signal)
  - daily_seasonality=False
  - changepoint_prior_scale=0.05  (moderate flexibility)
  - interval_width=0.95  (for PICP / Coverage)

Output (stdout, parsed by run_all_baselines.py extract_metrics):
  final test rse <rse> | test rae <rae> | test corr <corr> | test smape <smape>
  Coverage <coverage>
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data helpers (mirror common.py logic exactly)
# ---------------------------------------------------------------------------

def load_csv_values(data_path: Path) -> tuple[np.ndarray, int]:
    """Load numeric columns, return (array [T, N], num_nodes)."""
    df = pd.read_csv(str(data_path))
    # Drop any date/timestamp column
    date_like = [c for c in df.columns if str(c).strip().lower() in
                 {"date", "month", "month-year", "time", "timestamp", "ds"}
                 or str(c).strip().lower().startswith("date")]
    df = df.drop(columns=date_like, errors="ignore")
    arr = df.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float64)
    return arr, arr.shape[1]


def chronological_split(arr: np.ndarray, train_frac: float = 0.6, val_frac: float = 0.2):
    """Return (train, val, test) arrays with strict chronological ordering."""
    T = arr.shape[0]
    n_train = int(T * train_frac)
    n_val = int(T * val_frac)
    train = arr[:n_train]
    val = arr[n_train: n_train + n_val]
    test = arr[n_train + n_val:]
    return train, val, test


# ---------------------------------------------------------------------------
# Metrics (identical formulas to common.py)
# ---------------------------------------------------------------------------

def compute_metrics(pred: np.ndarray, target: np.ndarray):
    """
    pred, target: shape [T, N] (time × nodes).
    Returns dict with RSE, RAE, Corr, sMAPE.
    """
    diff = pred - target
    mean_tgt = target.mean()

    rse_denom = math.sqrt(float(np.sum((target - mean_tgt) ** 2))) + 1e-8
    rae_denom = float(np.sum(np.abs(target - mean_tgt))) + 1e-8

    rse = float(math.sqrt(float(np.sum(diff ** 2)))) / rse_denom
    rae = float(np.sum(np.abs(diff))) / rae_denom

    # Per-node Pearson correlation, then mean over finite values
    corrs = []
    for n in range(pred.shape[1]):
        p = pred[:, n]
        t = target[:, n]
        p_c = p - p.mean()
        t_c = t - t.mean()
        denom = (np.std(p) * np.std(t)) + 1e-8
        c = float(np.mean(p_c * t_c)) / denom
        if math.isfinite(c):
            corrs.append(max(0.0, c))
    corr = float(np.mean(corrs)) if corrs else 0.0

    smape = float(np.mean(
        2.0 * np.abs(diff) / (np.abs(pred) + np.abs(target) + 1e-8)
    ))

    return {"RSE": rse, "RAE": rae, "Corr": corr, "sMAPE": smape}


# ---------------------------------------------------------------------------
# Prophet fitting
# ---------------------------------------------------------------------------

def fit_prophet_series(
    train_vals: np.ndarray,
    horizon: int,
    changepoint_prior_scale: float,
    interval_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit Prophet on a single univariate series and forecast `horizon` steps.

    Returns:
        yhat       [horizon]  point forecasts
        yhat_lower [horizon]  lower PI bound
        yhat_upper [horizon]  upper PI bound
    """
    from prophet import Prophet  # imported here so the module loads even if prophet missing

    T = len(train_vals)
    # Monthly dates starting at an arbitrary origin (values only; Prophet ignores abs dates for forecasting)
    ds = pd.date_range("2004-01-01", periods=T, freq="MS")
    df_fit = pd.DataFrame({"ds": ds, "y": train_vals})

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        interval_width=interval_width,
        uncertainty_samples=200,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(df_fit)

    future = m.make_future_dataframe(periods=horizon, freq="MS")
    forecast = m.predict(future)
    tail = forecast.tail(horizon)

    yhat = tail["yhat"].to_numpy()
    yhat_lower = tail["yhat_lower"].to_numpy()
    yhat_upper = tail["yhat_upper"].to_numpy()
    return yhat, yhat_lower, yhat_upper


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Prophet univariate baseline")
    p.add_argument("--data", type=str, default="../../data/sm_data.csv")
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--seq_out_len", type=int, default=36,
                   help="Forecast horizon (must match test window length or be clipped)")
    p.add_argument("--changepoint_prior_scale", type=float, default=0.05)
    p.add_argument("--interval_width", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    # Accept (and ignore) common args passed by run_all_baselines.py
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seq_in_len", type=int, default=10)
    p.add_argument("--num_nodes", type=int, default=190)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--lr_schedule", type=str, default="none")
    p.add_argument("--lr_patience_sched", type=int, default=10)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--lr_decay_factor", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--corr_lambda", type=float, default=0.2)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    data_path = Path(args.data)
    arr, num_nodes = load_csv_values(data_path)
    T = arr.shape[0]

    train_arr, val_arr, test_arr = chronological_split(arr, args.train_frac, args.val_frac)

    # Prophet is fitted on train only; we forecast over the full val+test horizon
    # then evaluate on the test portion.
    val_len = len(val_arr)
    test_len = len(test_arr)
    total_horizon = val_len + test_len  # forecast from end of train

    horizon = min(args.seq_out_len, test_len)  # evaluation horizon

    print(f"Dataset : {T} timesteps × {num_nodes} nodes")
    print(f"Train   : {len(train_arr)} | Val: {val_len} | Test: {test_len}")
    print(f"Horizon : {horizon} steps")
    print(f"Fitting Prophet on {num_nodes} series...")

    # Scale by train max (same as common.py) to keep Prophet numerics stable
    scale = np.abs(train_arr).max(axis=0)
    scale[scale == 0] = 1.0

    train_norm = train_arr / scale
    test_norm = test_arr / scale

    preds = np.zeros((test_len, num_nodes), dtype=np.float64)
    lowers = np.zeros((test_len, num_nodes), dtype=np.float64)
    uppers = np.zeros((test_len, num_nodes), dtype=np.float64)

    failed = 0
    for n in range(num_nodes):
        if (n + 1) % 20 == 0 or n == num_nodes - 1:
            print(f"  series {n+1}/{num_nodes}")
        try:
            yhat, yhat_lo, yhat_hi = fit_prophet_series(
                train_norm[:, n],
                horizon=total_horizon,
                changepoint_prior_scale=args.changepoint_prior_scale,
                interval_width=args.interval_width,
            )
            # Take the test-window slice (last `test_len` steps of the forecast)
            preds[:, n] = yhat[val_len:val_len + test_len]
            lowers[:, n] = yhat_lo[val_len:val_len + test_len]
            uppers[:, n] = yhat_hi[val_len:val_len + test_len]
        except Exception as exc:
            # Degenerate series (all-zero, constant): use last training value
            failed += 1
            preds[:, n] = train_norm[-1, n]
            lowers[:, n] = train_norm[-1, n]
            uppers[:, n] = train_norm[-1, n]

    if failed:
        print(f"  {failed} series fell back to persistence (constant/degenerate).")

    # Rescale back to original units for metric computation (matches common.py evaluate)
    scale_row = scale.reshape(1, -1)
    pred_orig = preds * scale_row
    test_orig = test_norm * scale_row

    # Clip test evaluation to `horizon` steps (in case test_len > seq_out_len)
    pred_eval = pred_orig[:horizon]
    test_eval = test_orig[:horizon]

    metrics = compute_metrics(pred_eval, test_eval)

    # Coverage: fraction of actuals within [lower, upper] (in normalised space)
    lower_orig = lowers[:horizon] * scale_row
    upper_orig = uppers[:horizon] * scale_row
    coverage = float(np.mean((test_eval >= lower_orig) & (test_eval <= upper_orig)))

    # Print in the format expected by run_all_baselines.py extract_metrics
    print(
        f"final test rse {metrics['RSE']:.4f} | test rae {metrics['RAE']:.4f} | "
        f"test corr {metrics['Corr']:.4f} | test smape {metrics['sMAPE']:.4f}"
    )
    print(f"Coverage {coverage:.4f}")

    # Also print the canonical table format (matches other baselines)
    print("test\trse\trae\tcorr\ts-mape")
    print(
        f"mean\t{metrics['RSE']:.4f}\t{metrics['RAE']:.4f}\t"
        f"{metrics['Corr']:.4f}\t{metrics['sMAPE']:.4f}"
    )

    print(f"\nProphet results summary:")
    print(f"  RAE      = {metrics['RAE']:.4f}")
    print(f"  RSE      = {metrics['RSE']:.4f}")
    print(f"  Corr     = {metrics['Corr']:.4f}")
    print(f"  sMAPE    = {metrics['sMAPE']:.4f}")
    print(f"  Coverage = {coverage:.4f}  (target 0.95)")


if __name__ == "__main__":
    main()
