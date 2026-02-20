#!/usr/bin/env python3
"""
Recalibrate interval widths using capped quantile scaling.

Supports grouped recalibration by horizon, category, or both.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class RecalibrationRunResult:
    max_scale: float
    global_raw_scale: float
    global_scale: float
    global_capped: bool
    group_scales: dict[str, float]
    group_meta: dict[str, dict[str, object]]
    out: pd.DataFrame
    picp_after: float
    ace_after: float
    mpiw_after: float


def infer_category(node_name: object) -> str:
    if not isinstance(node_name, str) or not node_name:
        return 'Other'
    token = node_name.strip()
    if token.startswith('PT_'):
        return 'PT'
    if token.startswith('RMD_'):
        return 'RMD'
    return 'Other'


def safe_scale_from_ratio(ratio: pd.Series, target: float, min_scale: float, max_scale: float, shrinkage: float) -> tuple[float, float, bool]:
    raw = float(np.quantile(ratio.to_numpy(dtype=float), target))
    if not np.isfinite(raw):
        raw = 1.0
    shrunk = 1.0 + (raw - 1.0) * (1.0 - shrinkage)
    clipped = float(np.clip(shrunk, min_scale, max_scale))
    was_capped = bool(clipped != shrunk)
    return raw, clipped, was_capped


def diagnostics(frame: pd.DataFrame, target: float) -> tuple[int, float, float, float]:
    lower = frame['forecast'] - frame['ci_95']
    upper = frame['forecast'] + frame['ci_95']
    inside = (frame['actual'] >= lower) & (frame['actual'] <= upper)
    picp = float(inside.mean())
    ace = float(picp - target)
    mpiw = float((upper - lower).mean())
    return int(len(frame)), picp, ace, mpiw


def ensure_group_columns(df: pd.DataFrame, horizon_col: str, category_col: str | None) -> pd.DataFrame:
    out = df.copy()

    if horizon_col in out.columns:
        out['recal_horizon'] = pd.to_numeric(out[horizon_col], errors='coerce').fillna(1).round().astype(int)
    elif 'step_ahead' in out.columns:
        out['recal_horizon'] = pd.to_numeric(out['step_ahead'], errors='coerce').fillna(1).round().astype(int)
    else:
        out['recal_horizon'] = 1

    if category_col and category_col in out.columns:
        out['recal_category'] = out[category_col].fillna('Other').astype(str)
    elif 'category' in out.columns:
        out['recal_category'] = out['category'].fillna('Other').astype(str)
    elif 'node' in out.columns:
        out['recal_category'] = out['node'].map(infer_category)
    else:
        out['recal_category'] = 'Other'

    return out


def assign_recal_group(df: pd.DataFrame, group_by: str) -> pd.Series:
    if group_by == 'global':
        return pd.Series(['ALL'] * len(df), index=df.index)
    if group_by == 'horizon':
        return df['recal_horizon'].map(lambda h: f'H{int(h)}')
    if group_by == 'category':
        return df['recal_category'].astype(str)
    return df.apply(lambda r: f"H{int(r['recal_horizon'])}|{r['recal_category']}", axis=1)


def parse_cap_candidates(text: str, min_scale: float) -> list[float]:
    parts = [p.strip() for p in str(text).split(',') if p.strip()]
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except ValueError:
            continue
    vals = [v for v in vals if np.isfinite(v) and v >= min_scale]
    if not vals:
        vals = [min_scale]
    return sorted(set(vals))


def main():
    parser = argparse.ArgumentParser(description='Recalibrate forecast intervals by scale factor')
    parser.add_argument('--input', type=str, default='model/Bayesian/forecast/calibration_interval_samples.csv')
    parser.add_argument('--output', type=str, default='model/Bayesian/forecast/calibration_interval_samples_recalibrated.csv')
    parser.add_argument('--summary', type=str, default='model/Bayesian/forecast/uncertainty/calibration_recalibration_summary.csv')
    parser.add_argument('--target', type=float, default=0.95)
    parser.add_argument('--min-ci', type=float, default=1e-9)
    parser.add_argument('--group-by', type=str, default='horizon_category', choices=['global', 'horizon', 'category', 'horizon_category'])
    parser.add_argument('--horizon-col', type=str, default='horizon')
    parser.add_argument('--category-col', type=str, default='')
    parser.add_argument('--min-group-size', type=int, default=25)
    parser.add_argument('--fallback', type=str, default='global', choices=['global', 'none'])
    parser.add_argument('--min-scale', type=float, default=1.0)
    parser.add_argument('--max-scale', type=float, default=12.0)
    parser.add_argument('--shrinkage', type=float, default=0.2)
    parser.add_argument('--auto-tune-cap', action='store_true')
    parser.add_argument('--cap-candidates', type=str, default='5,8,10,12,15,20')
    parser.add_argument('--max-mpiw-multiplier', type=float, default=12.0)
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(f'Input file not found: {inp}')

    df = pd.read_csv(inp)
    required = {'forecast', 'ci_95', 'actual'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')

    d = ensure_group_columns(df, horizon_col=str(args.horizon_col), category_col=(str(args.category_col).strip() or None))
    d['forecast'] = pd.to_numeric(d['forecast'], errors='coerce')
    d['ci_95'] = pd.to_numeric(d['ci_95'], errors='coerce')
    d['actual'] = pd.to_numeric(d['actual'], errors='coerce')
    d = d.dropna(subset=['forecast', 'ci_95', 'actual'])
    d = d[d['ci_95'] > 0]

    if d.empty:
        raise RuntimeError('No valid rows for recalibration')

    target = float(args.target)
    target = max(0.5, min(0.999, target))
    min_scale = float(max(0.01, args.min_scale))
    max_scale_requested = float(max(min_scale, args.max_scale))
    shrinkage = float(np.clip(args.shrinkage, 0.0, 0.95))
    min_group_size = int(max(1, args.min_group_size))

    d['recal_group'] = assign_recal_group(d, args.group_by)
    d['ratio'] = (d['actual'] - d['forecast']).abs() / d['ci_95']

    out_template = ensure_group_columns(df, horizon_col=str(args.horizon_col), category_col=(str(args.category_col).strip() or None))
    out_template['recal_group'] = assign_recal_group(out_template, args.group_by)

    base_eval = d[['recal_group', 'forecast', 'ci_95', 'actual']].copy()
    n_before, picp_before, ace_before, mpiw_before = diagnostics(base_eval[['forecast', 'ci_95', 'actual']], target=target)

    def run_with_cap(active_max_scale: float) -> RecalibrationRunResult:
        global_raw_scale, global_scale, global_capped = safe_scale_from_ratio(
            d['ratio'],
            target=target,
            min_scale=min_scale,
            max_scale=active_max_scale,
            shrinkage=shrinkage,
        )

        group_scales: dict[str, float] = {}
        group_meta: dict[str, dict[str, object]] = {}

        for group_name, g in d.groupby('recal_group', dropna=False):
            group_name = str(group_name)
            used_fallback = False
            if len(g) < min_group_size:
                if args.fallback == 'global':
                    raw_s = global_raw_scale
                    final_s = global_scale
                    capped_s = global_capped
                    used_fallback = True
                else:
                    raw_s = 1.0
                    final_s = 1.0
                    capped_s = False
                    used_fallback = True
            else:
                raw_s, final_s, capped_s = safe_scale_from_ratio(
                    g['ratio'],
                    target=target,
                    min_scale=min_scale,
                    max_scale=active_max_scale,
                    shrinkage=shrinkage,
                )

            group_scales[group_name] = final_s
            group_meta[group_name] = {
                'raw_scale': raw_s,
                'scale_factor': final_s,
                'was_capped': bool(capped_s),
                'used_fallback': bool(used_fallback),
                'samples': int(len(g)),
            }

        out_i = out_template.copy()
        out_i['recal_scale'] = out_i['recal_group'].map(group_scales).fillna(global_scale)
        out_i['ci_95'] = pd.to_numeric(out_i['ci_95'], errors='coerce') * out_i['recal_scale']
        out_i['ci_95'] = out_i['ci_95'].clip(lower=float(args.min_ci))
        out_i['lower'] = pd.to_numeric(out_i['forecast'], errors='coerce') - out_i['ci_95']
        out_i['upper'] = pd.to_numeric(out_i['forecast'], errors='coerce') + out_i['ci_95']

        out_valid_i = out_i[['forecast', 'ci_95', 'actual']].apply(pd.to_numeric, errors='coerce').dropna()
        _, picp_after_i, ace_after_i, mpiw_after_i = diagnostics(out_valid_i, target=target)

        return RecalibrationRunResult(
            max_scale=float(active_max_scale),
            global_raw_scale=float(global_raw_scale),
            global_scale=float(global_scale),
            global_capped=bool(global_capped),
            group_scales=group_scales,
            group_meta=group_meta,
            out=out_i,
            picp_after=float(picp_after_i),
            ace_after=float(ace_after_i),
            mpiw_after=float(mpiw_after_i),
        )

    selected_max_scale = max_scale_requested
    tuning_rows: list[dict[str, object]] = []

    if args.auto_tune_cap:
        cap_candidates = parse_cap_candidates(args.cap_candidates, min_scale=min_scale)
        if max_scale_requested not in cap_candidates:
            cap_candidates.append(max_scale_requested)
            cap_candidates = sorted(set(cap_candidates))

        mpiw_limit = float(max(1.0, args.max_mpiw_multiplier) * mpiw_before)
        results = [run_with_cap(cap) for cap in cap_candidates]
        for res in results:
            feasible = bool(res.mpiw_after <= mpiw_limit)
            tuning_rows.append({
                'candidate_max_scale': float(res.max_scale),
                'picp_after': float(res.picp_after),
                'ace_after': float(res.ace_after),
                'mpiw_after': float(res.mpiw_after),
                'is_feasible': feasible,
                'mpiw_limit': mpiw_limit,
            })

        feasible_results = [r for r in results if r.mpiw_after <= mpiw_limit]
        if feasible_results:
            best = max(feasible_results, key=lambda r: (r.picp_after, -r.mpiw_after, -r.max_scale))
        else:
            best = max(results, key=lambda r: (r.picp_after - 0.05 * (r.mpiw_after / max(mpiw_limit, 1e-12)), -r.mpiw_after))

        selected_max_scale = float(best.max_scale)
        run_result = best
    else:
        run_result = run_with_cap(selected_max_scale)

    global_raw_scale = float(run_result.global_raw_scale)
    global_scale = float(run_result.global_scale)
    global_capped = bool(run_result.global_capped)
    group_meta = run_result.group_meta
    out = run_result.out
    picp_after = float(run_result.picp_after)
    ace_after = float(run_result.ace_after)
    mpiw_after = float(run_result.mpiw_after)
    max_scale = float(selected_max_scale)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    summary_rows = [{
        'level': 'global',
        'group': 'ALL',
        'input_file': str(inp).replace('\\\\', '/'),
        'output_file': str(out_path).replace('\\\\', '/'),
        'samples': int(n_before),
        'target_coverage': target,
        'group_by': args.group_by,
        'min_group_size': min_group_size,
        'min_scale': min_scale,
        'max_scale': max_scale,
        'max_scale_requested': max_scale_requested,
        'shrinkage': shrinkage,
        'auto_tune_cap': bool(args.auto_tune_cap),
        'max_mpiw_multiplier': float(args.max_mpiw_multiplier),
        'raw_scale': global_raw_scale,
        'scale_factor': global_scale,
        'was_capped': bool(global_capped),
        'used_fallback': False,
        'picp_before': picp_before,
        'ace_before': ace_before,
        'mpiw_before': mpiw_before,
        'picp_after': picp_after,
        'ace_after': ace_after,
        'mpiw_after': mpiw_after,
    }]

    out_eval = out.copy()
    out_eval['forecast'] = pd.to_numeric(out_eval['forecast'], errors='coerce')
    out_eval['ci_95'] = pd.to_numeric(out_eval['ci_95'], errors='coerce')
    out_eval['actual'] = pd.to_numeric(out_eval['actual'], errors='coerce')
    out_eval = out_eval.dropna(subset=['forecast', 'ci_95', 'actual'])
    out_eval = out_eval[out_eval['ci_95'] > 0]

    base_eval = d[['recal_group', 'forecast', 'ci_95', 'actual']].copy()
    for group_name, g_before in base_eval.groupby('recal_group', dropna=False):
        g_after = out_eval[out_eval['recal_group'].astype(str) == str(group_name)]
        if g_after.empty:
            continue
        g_n_before, g_picp_before, g_ace_before, g_mpiw_before = diagnostics(g_before, target=target)
        g_n_after, g_picp_after, g_ace_after, g_mpiw_after = diagnostics(g_after[['forecast', 'ci_95', 'actual']], target=target)
        meta = group_meta.get(str(group_name), {})
        raw_scale_val = meta.get('raw_scale', np.nan)
        scale_factor_val = meta.get('scale_factor', np.nan)
        if not isinstance(raw_scale_val, (int, float, np.floating)):
            raw_scale_val = np.nan
        if not isinstance(scale_factor_val, (int, float, np.floating)):
            scale_factor_val = np.nan
        summary_rows.append({
            'level': 'group',
            'group': str(group_name),
            'input_file': str(inp).replace('\\\\', '/'),
            'output_file': str(out_path).replace('\\\\', '/'),
            'samples': int(min(g_n_before, g_n_after)),
            'target_coverage': target,
            'group_by': args.group_by,
            'min_group_size': min_group_size,
            'min_scale': min_scale,
            'max_scale': max_scale,
            'max_scale_requested': max_scale_requested,
            'shrinkage': shrinkage,
            'auto_tune_cap': bool(args.auto_tune_cap),
            'max_mpiw_multiplier': float(args.max_mpiw_multiplier),
            'raw_scale': float(raw_scale_val),
            'scale_factor': float(scale_factor_val),
            'was_capped': bool(meta.get('was_capped', False)),
            'used_fallback': bool(meta.get('used_fallback', False)),
            'picp_before': g_picp_before,
            'ace_before': g_ace_before,
            'mpiw_before': g_mpiw_before,
            'picp_after': g_picp_after,
            'ace_after': g_ace_after,
            'mpiw_after': g_mpiw_after,
        })

    summary = pd.DataFrame(summary_rows)

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    if args.auto_tune_cap and tuning_rows:
        tuning_path = summary_path.with_name(summary_path.stem + '_tuning.csv')
        pd.DataFrame(tuning_rows).to_csv(tuning_path, index=False)
        print(f'[recal] Saved cap tuning summary: {tuning_path}')

    print(f'[recal] Saved recalibrated samples: {out_path}')
    print(f'[recal] group_by={args.group_by}, max_scale_selected={max_scale:.4f}, global_scale={global_scale:.4f}, capped={global_capped}')
    print(f'After results of interval recalibration: picp={picp_after:.4f}, mpiw={mpiw_after:.4f}')
    print(f'[recal] Saved summary: {summary_path}')


if __name__ == '__main__':
    main()
