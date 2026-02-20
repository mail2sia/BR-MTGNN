#!/usr/bin/env python3
"""
Run baseline/ablation forecasts on a holdout split and report metrics.

Baselines:
- persistence: predict last value
- mean: predict mean of input window
- drift: predict last + h * (last - prev)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def safe_corr(pred: np.ndarray, true: np.ndarray, eps: float = 1e-12) -> float:
    p = pred.reshape(-1).astype(float)
    t = true.reshape(-1).astype(float)
    if p.size == 0 or t.size == 0:
        return float('nan')
    pm = p.mean()
    tm = t.mean()
    ps = p - pm
    ts = t - tm
    denom = np.sqrt((ps * ps).mean()) * np.sqrt((ts * ts).mean())
    if denom <= eps:
        return float('nan')
    return float((ps * ts).mean() / denom)


def smape(pred: np.ndarray, true: np.ndarray, eps: float = 1e-6) -> float:
    p = pred.astype(float)
    t = true.astype(float)
    denom = np.maximum(eps, np.abs(p) + np.abs(t))
    return float(np.mean(2.0 * np.abs(p - t) / denom) * 100.0)


def rse_rae(pred: np.ndarray, true: np.ndarray, eps: float = 1e-8) -> tuple[float, float]:
    p = pred.astype(float)
    t = true.astype(float)
    diff = p - t
    numer_rse = np.sum(diff * diff)
    denom_rse = np.sum((t - t.mean()) ** 2) + eps
    rse = float(np.sqrt(numer_rse / denom_rse))

    numer_rae = np.sum(np.abs(diff))
    denom_rae = np.sum(np.abs(t - t.mean())) + eps
    rae = float(numer_rae / denom_rae)
    return rse, rae


def load_series(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    if 'date' in df.columns:
        series = df.drop(columns=['date'])
    else:
        series = df
    series = series.apply(pd.to_numeric, errors='coerce')
    series = series.dropna(axis=0, how='any')
    return series, list(series.columns)


def build_windows(data: np.ndarray, seq_len: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    total = len(data)
    for i in range(seq_len, total - horizon + 1):
        xs.append(data[i - seq_len:i])
        ys.append(data[i:i + horizon])
    if not xs:
        return np.empty((0, seq_len, data.shape[1])), np.empty((0, horizon, data.shape[1]))
    return np.stack(xs, axis=0), np.stack(ys, axis=0)


def split_indices(total: int, train_ratio: float, valid_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_train = int(total * train_ratio)
    n_valid = int(total * valid_ratio)
    n_test = total - n_train - n_valid
    idx = np.arange(total)
    return idx[:n_train], idx[n_train:n_train + n_valid], idx[n_train + n_valid:n_train + n_valid + n_test]


def main() -> None:
    parser = argparse.ArgumentParser(description='Baseline/ablation evaluation for time series forecasts')
    parser.add_argument('--data', type=str, default='data/sm_data_g.csv')
    parser.add_argument('--seq-len', type=int, default=120)
    parser.add_argument('--horizon', type=int, default=6)
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--valid-ratio', type=float, default=0.2)
    parser.add_argument('--split-mode', choices=['ratio', 'holdout-index'], default='ratio')
    parser.add_argument('--holdout-steps', type=int, default=36)
    parser.add_argument('--output', type=str, default='model/Bayesian/forecast/uncertainty/baseline_ablation_metrics.csv')
    args = parser.parse_args()

    series, cols = load_series(Path(args.data))
    data = series.to_numpy(dtype=float)

    x_all, y_all = build_windows(data, seq_len=int(args.seq_len), horizon=int(args.horizon))
    if len(x_all) == 0:
        raise RuntimeError('Not enough data for requested seq_len/horizon')

    if args.split_mode == 'ratio':
        train_idx, valid_idx, test_idx = split_indices(len(x_all), float(args.train_ratio), float(args.valid_ratio))
    else:
        holdout = int(args.holdout_steps)
        test_idx = np.arange(max(0, len(x_all) - holdout), len(x_all))
        train_idx = np.arange(0, max(0, len(x_all) - holdout))
        valid_idx = np.array([], dtype=int)

    split_sets = {
        'train': train_idx,
        'valid': valid_idx,
        'test': test_idx,
    }

    def predict_baseline(x: np.ndarray, mode: str) -> np.ndarray:
        last = x[:, -1, :]
        if mode == 'persistence':
            base = last[:, None, :]
            return np.repeat(base, int(args.horizon), axis=1)
        if mode == 'mean':
            mean = x.mean(axis=1)
            base = mean[:, None, :]
            return np.repeat(base, int(args.horizon), axis=1)
        if mode == 'drift':
            prev = x[:, -2, :]
            slope = last - prev
            steps = np.arange(1, int(args.horizon) + 1, dtype=float)[None, :, None]
            return last[:, None, :] + slope[:, None, :] * steps
        raise ValueError(f'Unknown baseline mode: {mode}')

    rows = []
    for split_name, idx in split_sets.items():
        if len(idx) == 0:
            continue
        x = x_all[idx]
        y = y_all[idx]
        for mode in ['persistence', 'mean', 'drift']:
            pred = predict_baseline(x, mode)
            pred_flat = pred.reshape(-1, pred.shape[-1])
            true_flat = y.reshape(-1, y.shape[-1])

            rse, rae = rse_rae(pred_flat, true_flat)
            corr = safe_corr(pred_flat, true_flat)
            sm = smape(pred_flat, true_flat)

            rows.append({
                'baseline': mode,
                'split': split_name,
                'samples': int(len(pred_flat)),
                'RSE': rse,
                'RAE': rae,
                'Corr': corr,
                'sMAPE': sm,
                'seq_len': int(args.seq_len),
                'horizon': int(args.horizon),
                'data': str(args.data).replace('\\', '/'),
                'split_mode': args.split_mode,
                'holdout_steps': int(args.holdout_steps),
            })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[baseline] Saved metrics: {out_path}")


if __name__ == '__main__':
    main()
