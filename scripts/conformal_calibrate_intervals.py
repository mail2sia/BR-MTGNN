#!/usr/bin/env python3
"""
Split-conformal interval calibration with per-horizon (or grouped) qhat.

Uses absolute residual exceedance: s = max(|y-forecast| - ci_95, 0).
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ConformalResult:
    group: str
    samples_calib: int
    qhat: float
    used_fallback: bool


def assign_group(df: pd.DataFrame, group_by: str) -> pd.Series:
    if group_by == 'global':
        return pd.Series(['ALL'] * len(df), index=df.index)
    if group_by == 'horizon':
        return df['horizon'].map(lambda h: f'H{int(h)}')
    if group_by == 'category':
        return df['category'].astype(str)
    return df.apply(lambda r: f"H{int(r['horizon'])}|{r['category']}", axis=1)


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['forecast'] = pd.to_numeric(out['forecast'], errors='coerce')
    out['ci_95'] = pd.to_numeric(out['ci_95'], errors='coerce')
    out['actual'] = pd.to_numeric(out['actual'], errors='coerce')
    out = out.dropna(subset=['forecast', 'ci_95', 'actual'])
    out = out[out['ci_95'] > 0]

    if 'horizon' in out.columns:
        out['horizon'] = pd.to_numeric(out['horizon'], errors='coerce').fillna(1).round().astype(int)
    else:
        out['horizon'] = 1

    if 'category' in out.columns:
        out['category'] = out['category'].fillna('Other').astype(str)
    elif 'node' in out.columns:
        out['category'] = out['node'].astype(str).map(lambda n: 'PT' if str(n).startswith('PT_') else 'RMD' if str(n).startswith('RMD_') else 'Other')
    else:
        out['category'] = 'Other'

    return out


def compute_qhat(residuals: pd.Series, alpha: float) -> float:
    if residuals.empty:
        return 0.0
    q = float(np.quantile(residuals.to_numpy(dtype=float), 1.0 - alpha))
    if not np.isfinite(q):
        return 0.0
    return float(max(0.0, q))


def main() -> None:
    parser = argparse.ArgumentParser(description='Split-conformal calibration by horizon/category')
    parser.add_argument('--input', type=str, default='model/Bayesian/forecast/calibration_interval_samples.csv')
    parser.add_argument('--output', type=str, default='model/Bayesian/forecast/calibration_interval_samples_conformal_horizon.csv')
    parser.add_argument('--summary', type=str, default='model/Bayesian/forecast/uncertainty/calibration_conformal_summary.csv')
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--calib-frac', type=float, default=0.7)
    parser.add_argument('--group-by', choices=['global', 'horizon', 'category', 'horizon_category'], default='horizon')
    parser.add_argument('--min-group-size', type=int, default=50)
    parser.add_argument('--fallback', choices=['global', 'none'], default='global')
    parser.add_argument('--split-col', type=str, default='window_index')
    parser.add_argument('--output-all', action='store_true')
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(f'Input file not found: {inp}')

    df = pd.read_csv(inp)
    df = ensure_columns(df)
    if df.empty:
        raise RuntimeError('No valid rows for conformal calibration')

    split_col = args.split_col if args.split_col in df.columns else None
    if split_col:
        ordered = df.sort_values(split_col)
    else:
        ordered = df.sort_values('t_index') if 't_index' in df.columns else df.copy()

    n = len(ordered)
    n_calib = max(1, int(n * float(args.calib_frac)))
    calib = ordered.iloc[:n_calib].copy()
    eval_df = ordered.iloc[n_calib:].copy()

    if calib.empty or eval_df.empty:
        raise RuntimeError('Calibration/eval split failed; adjust --calib-frac or input size')

    calib['group'] = assign_group(calib, args.group_by)
    eval_df['group'] = assign_group(eval_df, args.group_by)

    calib['residual'] = (calib['actual'] - calib['forecast']).abs() - calib['ci_95']
    calib['residual'] = calib['residual'].clip(lower=0.0)

    global_qhat = compute_qhat(calib['residual'], alpha=float(args.alpha))

    group_qhat: dict[str, float] = {}
    summary_rows: list[dict[str, object]] = []
    min_group_size = int(max(1, args.min_group_size))

    for group_name, g in calib.groupby('group', dropna=False):
        group_name = str(group_name)
        used_fallback = False
        if len(g) < min_group_size:
            if args.fallback == 'global':
                qhat = global_qhat
                used_fallback = True
            else:
                qhat = 0.0
                used_fallback = True
        else:
            qhat = compute_qhat(g['residual'], alpha=float(args.alpha))
        group_qhat[group_name] = float(qhat)
        summary_rows.append({
            'group': group_name,
            'samples_calib': int(len(g)),
            'qhat': float(qhat),
            'used_fallback': bool(used_fallback),
            'group_by': args.group_by,
            'alpha': float(args.alpha),
            'calib_frac': float(args.calib_frac),
        })

    def apply_qhat(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out['conformal_qhat'] = out['group'].map(group_qhat).fillna(global_qhat)
        out['ci_95'] = out['ci_95'] + out['conformal_qhat']
        out['lower'] = out['forecast'] - out['ci_95']
        out['upper'] = out['forecast'] + out['ci_95']
        return out

    eval_out = apply_qhat(eval_df)
    eval_out['split'] = 'eval'

    if args.output_all:
        calib_out = calib.copy()
        calib_out['conformal_qhat'] = calib_out['group'].map(group_qhat).fillna(global_qhat)
        calib_out['split'] = 'calib'
        out_df = pd.concat([calib_out, eval_out], ignore_index=True)
    else:
        out_df = eval_out

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"[conformal] Saved: {out_path}")
    print(f"[conformal] Summary: {summary_path}")


if __name__ == '__main__':
    main()
