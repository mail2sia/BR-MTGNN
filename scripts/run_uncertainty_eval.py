#!/usr/bin/env python3
"""
Reproducible uncertainty evaluation runner with seed control and basic CI checks.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


_ROOT = Path(__file__).parent.parent


def run_cmd(cmd: list[str], label: str) -> None:
    print(f"[eval] {label}")
    subprocess.run(cmd, cwd=str(_ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run uncertainty eval pipeline with reproducibility')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mc-runs', type=int, default=200)
    parser.add_argument('--max-horizon', type=int, default=6)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--group-by', type=str, default='horizon_category')
    parser.add_argument('--cap-candidates', type=str, default='4,6,8,10,12,15,20')
    parser.add_argument('--max-mpiw-multiplier', type=float, default=12.0)
    parser.add_argument('--skip-backtest', action='store_true')
    parser.add_argument('--skip-recalibration', action='store_true')
    parser.add_argument('--skip-plot', action='store_true')
    parser.add_argument('--skip-calibration-diagnostics', action='store_true')
    parser.add_argument('--ci-mode', action='store_true')
    parser.add_argument('--ci-min-picp', type=float, default=0.15)
    args = parser.parse_args()

    backtest_out = _ROOT / 'model' / 'Bayesian' / 'forecast' / 'calibration_interval_samples.csv'

    if not args.skip_backtest:
        run_cmd(
            [
                sys.executable,
                str(_ROOT / 'scripts' / 'generate_interval_backtest.py'),
                '--mc_runs', str(int(args.mc_runs)),
                '--max_horizon', str(int(args.max_horizon)),
                '--device', str(args.device),
                '--seed', str(int(args.seed)),
                '--output', str(backtest_out),
            ],
            label='Generate interval backtest samples',
        )

    if not args.skip_recalibration:
        run_cmd(
            [
                sys.executable,
                str(_ROOT / 'scripts' / 'recalibrate_intervals.py'),
                '--auto-tune-cap',
                '--cap-candidates', str(args.cap_candidates),
                '--max-mpiw-multiplier', str(float(args.max_mpiw_multiplier)),
                '--group-by', str(args.group_by),
            ],
            label='Auto-tuned recalibration',
        )

    if not args.skip_plot:
        run_cmd(
            [sys.executable, str(_ROOT / 'scripts' / 'plot_uncertainty_insights.py')],
            label='Plot uncertainty insights',
        )

    if not args.skip_calibration_diagnostics:
        run_cmd(
            [sys.executable, str(_ROOT / 'scripts' / 'uncertainty_calibration_diagnostics.py')],
            label='Calibration curve + PIT diagnostics',
        )

    if args.ci_mode:
        diag_path = _ROOT / 'model' / 'Bayesian' / 'forecast' / 'uncertainty' / 'calibration_interval_diagnostics.csv'
        if not diag_path.exists():
            raise RuntimeError(f"Missing diagnostics CSV for CI checks: {diag_path}")

        diag = pd.read_csv(diag_path)
        match = diag[diag['source_file'].astype(str).str.contains('calibration_interval_samples_recalibrated.csv')]
        if match.empty:
            raise RuntimeError('No recalibrated diagnostics row found for CI checks')

        picp = float(match.iloc[0]['picp'])
        if picp < float(args.ci_min_picp):
            raise SystemExit(f"CI check failed: PICP {picp:.4f} < {float(args.ci_min_picp):.4f}")

        print(f"[eval] CI check passed: PICP {picp:.4f} >= {float(args.ci_min_picp):.4f}")


if __name__ == '__main__':
    main()
