#!/usr/bin/env python3
"""
Aggressive smoothing helper for B-MTGNN.
Usage:
  python scripts/agg_smoothing.py --input data/data.csv --out-prefix data/sm_aggr
This will write:
  data/sm_aggr.csv, data/sm_aggr.txt, data/dual_input_sm_aggr.npy (if --save-dual)
"""
import argparse, os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def read_matrix(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        arr = np.loadtxt(path, delimiter="\t")
        if arr.ndim == 1:
            arr = arr[:, None]
        return arr.astype(float)
    elif ext == ".csv":
        df = pd.read_csv(path)
        if df.shape[1] > 1 and str(df.columns[0]).lower() in ("date","month","time"):
            df = df.iloc[:, 1:]
        return df.to_numpy(dtype=float)
    else:
        raise ValueError("Unsupported input extension")


def write_outputs(sm: np.ndarray, out_csv: str, out_txt: str) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        for row in sm:
            f.write(",".join([f"{x:.6f}" for x in row]) + "\n")
    np.savetxt(out_txt, sm, fmt="%.6f", delimiter="\t")


def double_exp(series: np.ndarray, alpha: float, beta: float) -> np.ndarray:
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


def moving_avg(a: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 1:
        return a
    ret = np.convolve(a, np.ones(window)/window, mode='same')
    return ret


def agg_smooth_matrix(X: np.ndarray, alpha: float=0.01, beta: float=0.01, gauss_sigma: float=1.0, ma_window:int=3) -> np.ndarray:
    T, N = X.shape
    S = np.empty_like(X, dtype=float)
    for j in range(N):
        col = X[:, j]
        # 1) double exponential with tiny alpha/beta (very smooth)
        s = double_exp(col, alpha, beta)
        # 2) gaussian filter along time
        if gauss_sigma > 0:
            s = gaussian_filter1d(s, sigma=gauss_sigma, mode='reflect')
        # 3) moving average to remove small wiggles
        if ma_window > 1:
            s = moving_avg(s, ma_window)
        S[:, j] = s
    return S


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--out-prefix', default='data/sm_aggr')
    p.add_argument('--alpha', type=float, default=0.01)
    p.add_argument('--beta', type=float, default=0.01)
    p.add_argument('--gauss_sigma', type=float, default=1.0)
    p.add_argument('--ma_window', type=int, default=3)
    p.add_argument('--save_dual', action='store_true')
    args = p.parse_args()

    X = read_matrix(args.input)
    T,N = X.shape
    print(f"[agg_smoothing] Loaded {args.input}: shape {T} x {N}")
    S = agg_smooth_matrix(X, alpha=args.alpha, beta=args.beta, gauss_sigma=args.gauss_sigma, ma_window=args.ma_window)

    out_csv = args.out_prefix + '.csv'
    out_txt = args.out_prefix + '.txt'
    write_outputs(S, out_csv, out_txt)
    print(f"[agg_smoothing] Wrote {out_csv} and {out_txt} ({S.shape[0]}x{S.shape[1]})")

    if args.save_dual:
        dual = np.stack([X.astype(np.float32), S.astype(np.float32)], axis=-1)
        dual_path = args.out_prefix + '_dual.npy'
        np.save(dual_path, dual)
        print(f"[agg_smoothing] Wrote {dual_path} shape={dual.shape}")

if __name__ == '__main__':
    main()
