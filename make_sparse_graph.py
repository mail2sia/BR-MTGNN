from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_corr(x: np.ndarray, y: np.ndarray, min_obs: int = 12) -> float:
    """Finite-sample Pearson correlation with safe fallbacks."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < int(min_obs):
        return 0.0
    x = x[ok]
    y = y[ok]
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    corr = float(np.corrcoef(x, y)[0, 1])
    return corr if np.isfinite(corr) else 0.0


def _winsorize(x: np.ndarray, q: float = 0.02) -> np.ndarray:
    """Clip extreme changes so one launch spike cannot dominate a correlation."""
    x = np.asarray(x, dtype=float).copy()
    ok = np.isfinite(x)
    if ok.sum() < 4 or q <= 0:
        return x
    lo, hi = np.nanquantile(x[ok], [q, 1.0 - q])
    if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
        x[ok] = np.clip(x[ok], lo, hi)
    return x


def _residualize(y: np.ndarray, design: np.ndarray, min_obs: int = 12) -> np.ndarray:
    """Remove shared trend/global-attention effects through least-squares residuals."""
    y = np.asarray(y, dtype=float)
    design = np.asarray(design, dtype=float)
    ok = np.isfinite(y) & np.all(np.isfinite(design), axis=1)
    if ok.sum() < max(min_obs, design.shape[1] + 2):
        return y - np.nanmean(y)
    beta, *_ = np.linalg.lstsq(design[ok], y[ok], rcond=None)
    out = np.full_like(y, np.nan, dtype=float)
    out[ok] = y[ok] - design[ok] @ beta
    return out


def _lagged_best_corr(x: np.ndarray, y: np.ndarray, max_lag: int, min_obs: int = 12) -> tuple[float, int]:
    """Best positive lagged correlation.

    lag=0 compares RMD(t) with PT(t). lag>0 compares RMD(t) with PT(t+lag),
    which lets technology response follow the disease signal.
    """
    best = 0.0
    best_lag = 0
    for lag in range(max(0, int(max_lag)) + 1):
        if lag == 0:
            xs, ys = x, y
        else:
            xs, ys = x[:-lag], y[lag:]
        corr = _safe_corr(xs, ys, min_obs=min_obs)
        if corr > best:
            best = corr
            best_lag = lag
    return best, best_lag


def _segment_support(r: np.ndarray, p: np.ndarray, max_lag: int, blocks: int, min_obs: int, threshold: float) -> float:
    """Fraction of time segments showing at least weak pair-specific co-movement."""
    n = len(r)
    if blocks <= 1 or n < max(blocks * min_obs, 24):
        return 1.0
    supports = []
    for idx in np.array_split(np.arange(n), blocks):
        if len(idx) < min_obs:
            continue
        corr, _ = _lagged_best_corr(r[idx], p[idx], min(max_lag, max(0, len(idx) - min_obs)), min_obs=min_obs)
        supports.append(corr >= threshold)
    if not supports:
        return 1.0
    return float(np.mean(supports))


def _series_score(
    rmd: np.ndarray,
    pt: np.ndarray,
    max_lag: int,
    level_design: np.ndarray | None = None,
    diff_design: np.ndarray | None = None,
    min_obs: int = 12,
    winsor_q: float = 0.02,
    stability_blocks: int = 3,
    stability_threshold: float = 0.10,
) -> dict[str, float | int]:
    """Robust RMD->PT edge score.

    The original score used correlation on log-levels only. That over-rewards technologies
    with a broad post-launch/global-attention surge. This score keeps the raw level
    correlation as a diagnostic, but ranks edges mostly by:
      1) correlation after removing shared time/global-attention effects, and
      2) correlation of winsorized month-to-month changes.
    """
    r = np.log1p(np.clip(np.asarray(rmd, dtype=float), 0.0, None))
    p = np.log1p(np.clip(np.asarray(pt, dtype=float), 0.0, None))

    raw_corr, raw_lag = _lagged_best_corr(r, p, max_lag=max_lag, min_obs=min_obs)

    if level_design is not None:
        r_level = _residualize(r, level_design, min_obs=min_obs)
        p_level = _residualize(p, level_design, min_obs=min_obs)
    else:
        r_level = r - np.nanmean(r)
        p_level = p - np.nanmean(p)
    resid_corr, resid_lag = _lagged_best_corr(r_level, p_level, max_lag=max_lag, min_obs=min_obs)

    r_diff = _winsorize(np.diff(r), q=winsor_q)
    p_diff = _winsorize(np.diff(p), q=winsor_q)
    if diff_design is not None and len(diff_design) == len(r_diff):
        r_diff = _residualize(r_diff, diff_design, min_obs=min_obs)
        p_diff = _residualize(p_diff, diff_design, min_obs=min_obs)
    diff_corr, diff_lag = _lagged_best_corr(r_diff, p_diff, max_lag=max_lag, min_obs=min_obs)

    # Segment support prevents a single late regime from fully defining the edge while
    # still allowing newer technologies to remain in the graph when their signal is specific.
    support = _segment_support(r_level, p_level, max_lag, stability_blocks, min_obs, stability_threshold)
    support_multiplier = 0.50 + 0.50 * support

    base = (0.10 * raw_corr + 0.45 * resid_corr + 0.45 * diff_corr) * support_multiplier
    base = float(max(0.0, min(base, 1.0)))

    active_overlap = int(np.sum((np.asarray(rmd, dtype=float) > 0) & (np.asarray(pt, dtype=float) > 0)))
    return {
        "base_score": base,
        "raw_corr": float(raw_corr),
        "raw_lag": int(raw_lag),
        "resid_corr": float(resid_corr),
        "resid_lag": int(resid_lag),
        "diff_corr": float(diff_corr),
        "diff_lag": int(diff_lag),
        "segment_support": float(support),
        "active_overlap": active_overlap,
    }


def _design_matrices(recent: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Create controls for broad trend and global attention shared by many series."""
    values = recent[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    log_values = np.log1p(np.clip(values, 0.0, None))
    global_attention = np.nanmean(log_values, axis=1)
    n = len(recent)
    t = np.linspace(-1.0, 1.0, n)

    level_design = np.column_stack([
        np.ones(n),
        t,
        t ** 2,
        global_attention,
    ])

    dg = np.diff(global_attention)
    dt = t[1:]
    diff_design = np.column_stack([
        np.ones(n - 1),
        dt,
        dg,
    ])
    return level_design, diff_design


def main() -> None:
    ap = argparse.ArgumentParser(description="Create sparse undirected graph from RMD–PT historical co-movement; scored by RMD→PT correlation.")
    ap.add_argument("--data_csv", default="data/sm_data.csv")
    ap.add_argument("--out_csv", default="data/graph_sparse.csv")
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--window", type=int, default=96, help="Recent months used for correlation ranking")
    ap.add_argument("--max_lag", type=int, default=6, help="Allow PT to lag RMD by up to this many months")
    ap.add_argument("--min_corr", type=float, default=0.05, help="Minimum robust score; top_k still backfills if needed")
    ap.add_argument("--min_obs", type=int, default=12, help="Minimum paired observations for a correlation")
    ap.add_argument("--winsor_q", type=float, default=0.02, help="Tail fraction clipped in month-to-month changes")
    ap.add_argument("--stability_blocks", type=int, default=3, help="Number of time blocks for segment-support penalty")
    ap.add_argument("--stability_threshold", type=float, default=0.10, help="Weak segment-level correlation counted as support")
    ap.add_argument("--activity_weight", type=float, default=0.02, help="Small bounded activity bonus for non-inactive PTs")
    ap.add_argument("--activity_bonus_cap", type=float, default=0.10, help="Maximum activity bonus above 1.0")
    ap.add_argument("--hub_threshold", type=float, default=0.35, help="Base score threshold used to identify global PT hubs")
    ap.add_argument("--hub_penalty", type=float, default=0.50, help="Penalty strength for PTs that score high against many RMDs")
    ap.add_argument("--train_end_date", type=str, default="",
                    help="Last month of training period (YYYY-MM). When set, graph is built "
                         "only from rows up to this date, preventing evaluation-period leakage.")
    ap.add_argument("--dates_csv", type=str, default="",
                    help="CSV with a date column used to apply --train_end_date when data_csv "
                         "(e.g. sm_data.csv) has no date column. Defaults to "
                         "data/data_validated.csv if it exists, else data/data.csv.")
    args = ap.parse_args()

    df = pd.read_csv(args.data_csv)
    rmd_cols = [c for c in df.columns if str(c).startswith("RMD_") and str(c).endswith("_NoM")]
    pt_cols = [c for c in df.columns if str(c).startswith("PT_") and str(c).endswith("_NoM")]
    if not rmd_cols or not pt_cols:
        raise ValueError("Expected RMD_*_NoM and PT_*_NoM columns in data_csv.")

    if args.train_end_date:
        # Restrict to training period only — prevents graph leakage from val/test data.
        # Look for a date column in data_csv first, then fall back to dates_csv.
        date_col = next((c for c in df.columns if str(c).strip().lower() in
                         {"date", "month", "time", "month-year", "month_year", "year-month"}), None)
        if date_col:
            date_series = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
        else:
            # sm_data.csv has no date column — load dates from the raw/validated CSV
            from pathlib import Path as _Path
            dates_csv = args.dates_csv or (
                "data/data_validated.csv" if _Path("data/data_validated.csv").exists()
                else "data/data.csv"
            )
            dates_df = pd.read_csv(dates_csv)
            date_col = next((c for c in dates_df.columns if str(c).strip().lower() in
                             {"date", "month", "time", "month-year", "month_year", "year-month"}), None)
            if date_col is None:
                print("[make_sparse_graph] WARNING: --train_end_date set but no date column found — using full dataset.")
                date_series = None
            else:
                date_series = pd.to_datetime(dates_df[date_col], errors="coerce", format="mixed")
                if len(date_series) != len(df):
                    print(f"[make_sparse_graph] WARNING: dates_csv row count ({len(date_series)}) "
                          f"!= data_csv row count ({len(df)}) — using full dataset.")
                    date_series = None

        if date_series is not None:
            cutoff = pd.Timestamp(args.train_end_date)
            mask = date_series.values <= cutoff.to_datetime64()
            df = df[mask].copy()
            print(f"[make_sparse_graph] train_end_date={args.train_end_date}: "
                  f"using {len(df)} rows (up to {cutoff.strftime('%Y-%m')}) for graph construction.")

    recent = df.tail(max(args.window, 24)).copy()
    all_signal_cols = rmd_cols + pt_cols
    level_design, diff_design = _design_matrices(recent, all_signal_cols)

    pair_rows = []
    for rmd in rmd_cols:
        r = pd.to_numeric(recent[rmd], errors="coerce").fillna(0.0).to_numpy(float)
        for pt in pt_cols:
            p = pd.to_numeric(recent[pt], errors="coerce").fillna(0.0).to_numpy(float)
            metrics = _series_score(
                r,
                p,
                max_lag=args.max_lag,
                level_design=level_design,
                diff_design=diff_design,
                min_obs=args.min_obs,
                winsor_q=args.winsor_q,
                stability_blocks=args.stability_blocks,
                stability_threshold=args.stability_threshold,
            )
            activity = math.log1p(float(np.nanmean(np.clip(p, 0.0, None))))
            activity_multiplier = 1.0 + min(float(args.activity_bonus_cap), float(args.activity_weight) * activity)
            pair_rows.append({
                "RMD": rmd,
                "PT": pt,
                "activity_multiplier": activity_multiplier,
                **metrics,
            })

    scores_df = pd.DataFrame(pair_rows)

    # Generic/global technologies are not excluded. They are down-weighted when they
    # correlate strongly with many RMDs, so only pair-specific signal survives top-k ranking.
    pt_hubness = scores_df.groupby("PT")["base_score"].transform(lambda s: float((s >= args.hub_threshold).mean()))
    pt_median = scores_df.groupby("PT")["base_score"].transform("median")
    scores_df["pt_hubness"] = pt_hubness
    scores_df["pt_median_base_score"] = pt_median
    scores_df["specificity_score"] = np.clip(scores_df["base_score"] - pt_median, 0.0, None)
    scores_df["hub_multiplier"] = np.clip(1.0 - args.hub_penalty * pt_hubness, 0.05, 1.0)
    scores_df["score"] = scores_df["base_score"] * scores_df["activity_multiplier"] * scores_df["hub_multiplier"]

    rows = []
    for rmd in rmd_cols:
        sub = scores_df[scores_df["RMD"] == rmd].sort_values(
            ["score", "specificity_score", "base_score", "raw_corr"], ascending=False
        )
        chosen = sub.loc[sub["base_score"] >= args.min_corr, "PT"].head(args.top_k).tolist()
        if len(chosen) < args.top_k:
            for pt in sub["PT"].tolist():
                if pt not in chosen:
                    chosen.append(pt)
                if len(chosen) >= args.top_k:
                    break
        rows.append([rmd] + chosen)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, header=False, index=False)
    scores_df.to_csv(out.with_suffix(".scores.csv"), index=False)
    print(f"Saved sparse graph: {out}")
    print(f"RMD rows: {len(rows)} | PT per RMD: {args.top_k} | total undirected edges: {len(rows) * args.top_k}")
    print(f"Scores saved: {out.with_suffix('.scores.csv')}")
    print("Score columns include raw_corr, resid_corr, diff_corr, segment_support, pt_hubness, and final score.")
    print("Note: adjacency matrix is symmetrized during model training (undirected graph).")


if __name__ == "__main__":
    main()
