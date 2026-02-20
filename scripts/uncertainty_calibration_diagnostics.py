#!/usr/bin/env python3
"""
Generate calibration curves, PIT histograms, and per-horizon interval coverage tables.

Uses forecast mean and ci_95 to approximate predictive distributions.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


def load_interval_frame(path: Path, max_horizon: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {'forecast', 'ci_95', 'actual'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

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

    if max_horizon is not None:
        out = out[out['horizon'] <= int(max_horizon)]

    return out


def interval_metrics(frame: pd.DataFrame, target: float = 0.95) -> tuple[int, float, float, float]:
    lower = frame['forecast'] - frame['ci_95']
    upper = frame['forecast'] + frame['ci_95']
    inside = (frame['actual'] >= lower) & (frame['actual'] <= upper)
    picp = float(inside.mean()) if len(frame) else float('nan')
    ace = float(picp - target) if np.isfinite(picp) else float('nan')
    mpiw = float((upper - lower).mean()) if len(frame) else float('nan')
    return int(len(frame)), picp, ace, mpiw


def safe_int(value: object, default: int = 0) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return int(default)
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return int(default)
    return int(default)


def build_horizon_table(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    n_all, picp_all, ace_all, mpiw_all = interval_metrics(frame)
    rows.append({
        'set': label,
        'horizon': 'ALL',
        'samples': n_all,
        'picp_95': picp_all,
        'ace_95': ace_all,
        'mpiw_95': mpiw_all,
    })

    for horizon, group in frame.groupby('horizon'):
        n_h, picp_h, ace_h, mpiw_h = interval_metrics(group)
        rows.append({
            'set': label,
            'horizon': safe_int(horizon),
            'samples': n_h,
            'picp_95': picp_h,
            'ace_95': ace_h,
            'mpiw_95': mpiw_h,
        })

    return pd.DataFrame(rows)


def calibration_curve(frame: pd.DataFrame, levels: list[float], label: str) -> pd.DataFrame:
    sigma = frame['ci_95'] / 1.96
    sigma = sigma.replace([np.inf, -np.inf], np.nan)
    valid = frame.copy()
    valid['sigma'] = sigma
    valid = valid[valid['sigma'] > 0]
    if valid.empty:
        return pd.DataFrame(columns=['set', 'level', 'picp'])

    err = (valid['actual'] - valid['forecast']).abs().to_numpy(dtype=float)
    sigma_vals = valid['sigma'].to_numpy(dtype=float)

    rows = []
    for level in levels:
        level = float(level)
        z = float(norm.ppf(0.5 + level / 2.0))
        within = err <= (z * sigma_vals)
        picp = float(np.mean(within)) if len(within) else float('nan')
        rows.append({'set': label, 'level': level, 'picp': picp})

    return pd.DataFrame(rows)


def pit_values(frame: pd.DataFrame) -> np.ndarray:
    sigma = frame['ci_95'] / 1.96
    sigma = sigma.replace([np.inf, -np.inf], np.nan)
    valid = frame.copy()
    valid['sigma'] = sigma
    valid = valid[valid['sigma'] > 0]
    if valid.empty:
        return np.array([], dtype=float)

    z = (valid['actual'] - valid['forecast']) / valid['sigma']
    pit = norm.cdf(z.to_numpy(dtype=float))
    pit = pit[np.isfinite(pit)]
    return pit


def plot_calibration_curve(curve_df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(curve_df['level'], curve_df['picp'], marker='o', color='#2c7fb8', label='Empirical')
    ax.plot([0, 1], [0, 1], linestyle='--', color='#444444', label='Ideal')
    ax.set_xlabel('Nominal Coverage')
    ax.set_ylabel('Empirical Coverage')
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def plot_pit_histogram(pit: np.ndarray, out_path: Path, title: str, bins: int) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(pit, bins=bins, range=(0, 1), color='#74a9cf', edgecolor='white', alpha=0.85)
    ax.axhline(len(pit) / bins if bins else 0, color='#444444', linestyle='--', linewidth=1)
    ax.set_xlabel('PIT Value')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate calibration curves, PIT, and coverage tables')
    parser.add_argument('--input-raw', type=str, default='model/Bayesian/forecast/calibration_interval_samples.csv')
    parser.add_argument('--input-recal', type=str, default='model/Bayesian/forecast/calibration_interval_samples_recalibrated.csv')
    parser.add_argument('--output-dir', type=str, default='model/Bayesian/forecast/uncertainty')
    parser.add_argument('--max-horizon', type=int, default=0)
    parser.add_argument('--pit-bins', type=int, default=20)
    parser.add_argument('--levels', type=str, default='0.5,0.6,0.7,0.8,0.9,0.95')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_horizon = int(args.max_horizon) if int(args.max_horizon) > 0 else None

    levels = []
    for part in str(args.levels).split(','):
        part = part.strip()
        if not part:
            continue
        try:
            levels.append(float(part))
        except ValueError:
            continue
    if not levels:
        levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    outputs = []
    curve_rows = []
    pit_rows = []
    coverage_rows = []

    for label, path_str in [('raw', args.input_raw), ('recalibrated', args.input_recal)]:
        path = Path(path_str)
        if not path.exists():
            continue

        frame = load_interval_frame(path, max_horizon=max_horizon)
        outputs.append(label)

        coverage_rows.append(build_horizon_table(frame, label=label))

        curve_df = calibration_curve(frame, levels=levels, label=label)
        if len(curve_df):
            curve_rows.append(curve_df)
            plot_calibration_curve(
                curve_df,
                output_dir / f'calibration_curve_{label}',
                title=f'Calibration Curve ({label})',
            )

        pit = pit_values(frame)
        if pit.size:
            plot_pit_histogram(
                pit,
                output_dir / f'calibration_pit_hist_{label}',
                title=f'PIT Histogram ({label})',
                bins=int(max(5, args.pit_bins)),
            )
            hist_counts, bin_edges = np.histogram(pit, bins=int(max(5, args.pit_bins)), range=(0, 1))
            for idx in range(len(hist_counts)):
                pit_rows.append({
                    'set': label,
                    'bin_left': float(bin_edges[idx]),
                    'bin_right': float(bin_edges[idx + 1]),
                    'count': int(hist_counts[idx]),
                    'density': float(hist_counts[idx] / max(1, pit.size)),
                })

    if coverage_rows:
        coverage_df = pd.concat(coverage_rows, ignore_index=True)
        coverage_df.to_csv(output_dir / 'calibration_interval_coverage_by_horizon.csv', index=False)

    if curve_rows:
        curve_df = pd.concat(curve_rows, ignore_index=True)
        curve_df.to_csv(output_dir / 'calibration_curve.csv', index=False)

    if pit_rows:
        pd.DataFrame(pit_rows).to_csv(output_dir / 'calibration_pit_hist.csv', index=False)

    if outputs:
        print(f"[diag] Generated calibration diagnostics for: {', '.join(outputs)}")
    else:
        print('[diag] No input files found for calibration diagnostics')


if __name__ == '__main__':
    main()
