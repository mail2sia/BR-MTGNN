"""
=============================================================================
PIPELINE STEP 1 of 5 — smoothing.py
=============================================================================
Purpose : Validate raw PT data via Gemini API, then apply double exponential
          smoothing (Holt's method) to produce the model-ready time series.

Run AFTER: data/data.csv is available (raw input — no prior step required)
Run BEFORE: make_sparse_graph.py  (Step 2)

HOW TO RUN
----------
    cd /home/sahsan03/B-MTGNN/BR_MTGNN_main

    python smoothing.py \\
        --input_csv  data/data.csv \\
        --output_csv data/sm_data.csv \\
        --alpha 0.10 --beta 0.05 --mode des \\
        --validation_report model/Bayesian/validation_report.json

Key options:
    --alpha         Level smoothing factor (0–1, default 0.10)
    --beta          Trend smoothing factor (0–1, default 0.05)
    --mode          des = Holt double exponential | copy = clean only
    --no_gemini     Skip Gemini API (no corrections applied)
    --skip_validation  Bypass all validation entirely
    --no_cache      Force fresh Gemini API call even if cache exists

Outputs:
    data/sm_data.csv                       Smoothed, validated time series
    model/Bayesian/validation_report.json  Gemini audit log
    data/validation_cache.json             Cache (reused by all later steps)
=============================================================================
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def double_exponential_smoothing(series: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    series = np.asarray(series, dtype=float)
    if len(series) == 0:
        return series
    if len(series) == 1:
        return series.copy()
    result = [series[0]]
    level = series[0]
    trend = series[1] - series[0]
    for n in range(1, len(series)):
        value = series[n]
        last_level = level
        level = alpha * value + (1.0 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1.0 - beta) * trend
        result.append(level + trend)
    return np.asarray(result, dtype=float)


def _default_input_csv() -> str:
    """Use validated CSV if it exists, otherwise fall back to raw data."""
    validated = Path("data/data_validated.csv")
    return str(validated) if validated.exists() else "data/data.csv"


def parse_args():
    p = argparse.ArgumentParser(description="Create model-ready data/sm_data.csv with preserved headers and Date column")
    p.add_argument("--input_csv", type=str, default=None,
                   help="Input CSV (default: data/data_validated.csv if it exists, else data/data.csv)")
    p.add_argument("--output_csv", type=str, default="data/sm_data.csv")
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--clip_nonnegative", action="store_true", default=True)
    p.add_argument("--no_clip_nonnegative", dest="clip_nonnegative", action="store_false")
    p.add_argument("--mode", type=str, default="des", choices=["des", "copy"], help="des = causal double exponential smoothing; copy = only clean and save")
    # Gemini validation options
    p.add_argument("--gemini_api_key", type=str, default="",
                   help="Gemini API key for historical trend validation (overrides GEMINI_API_KEY env var)")
    p.add_argument("--no_gemini", action="store_true",
                   help="Skip Gemini API validation; no anachronism corrections applied (Gemini is sole mechanism)")
    p.add_argument("--skip_validation", action="store_true",
                   help="Skip all Gemini validation (not recommended for production runs)")
    p.add_argument("--validation_report", type=str, default="",
                   help="Path to write the JSON validation report (optional)")
    p.add_argument("--cache_path", type=str, default="",
                   help="Path to validation cache JSON (default: data/validation_cache.json)")
    p.add_argument("--no_cache", action="store_true",
                   help="Disable cache; always call the Gemini API")
    return p.parse_args()


def main():
    args = parse_args()
    input_csv = args.input_csv or _default_input_csv()
    print(f"[smoothing.py] Input: {input_csv}")
    df = pd.read_csv(input_csv)
    date_col = None
    for c in df.columns:
        if str(c).strip().lower() in {"date", "month", "time", "timestamp", "ds",
                                       "month-year", "month_year", "year-month"}:
            date_col = c
            break

    # ------------------------------------------------------------------
    # Gemini validation: zero-out anachronistic PT values before smoothing
    # so the model never trains on historically impossible signal.
    # ------------------------------------------------------------------
    if not args.skip_validation:
        try:
            from gemini_validator import validate_and_correct
            api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY", "") or None
            from pathlib import Path as _Path
            _cache_path = None if args.no_cache else (_Path(args.cache_path) if args.cache_path else None)
            df, val_report = validate_and_correct(
                df,
                api_key=api_key,
                use_gemini=not args.no_gemini,
                verbose=True,
                cache_path=_cache_path,
            )
            if args.validation_report:
                import json
                Path(args.validation_report).parent.mkdir(parents=True, exist_ok=True)
                Path(args.validation_report).write_text(json.dumps(val_report, indent=2))
                print(f"[smoothing.py] Validation report saved to {args.validation_report}")
        except ImportError:
            print("[smoothing.py] WARNING: gemini_validator.py not found — skipping validation.")
    else:
        print("[smoothing.py] Gemini validation skipped (--skip_validation).")

    if date_col is not None:
        date_values = df[date_col].astype(str).tolist()
        output_date_col = str(date_col)
    else:
        output_date_col = "Month-Year"
        date_values = pd.date_range(start="2004-01-01", periods=len(df), freq="MS").strftime("%b-%y").tolist()
        print("[smoothing.py] WARNING: No date column found; generated Month-Year from Jan-04.")

    numeric_cols = [c for c in df.columns if c != date_col]
    columns = {}
    for c in numeric_cols:
        values = pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if args.clip_nonnegative:
            values = np.clip(values, 0.0, None)
        if args.mode == "des":
            values = double_exponential_smoothing(values, args.alpha, args.beta)
            if args.clip_nonnegative:
                values = np.clip(values, 0.0, None)
        columns[c] = values
    out = pd.DataFrame(columns)
    out.insert(0, output_date_col, date_values)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Saved {args.output_csv} with shape {out.shape}")


if __name__ == "__main__":
    main()
