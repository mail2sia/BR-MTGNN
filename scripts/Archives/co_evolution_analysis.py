#!/usr/bin/env python3
"""
Co-evolution analysis for RMD/PT series.

Outputs:
- learned adjacency "attention" heatmap + edge list
- node influence scores (in/out/total weight)
- leading/lagging correlations + optional Granger p-values
- early-warning gap metrics based on recent trend divergence
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from statsmodels.tsa.stattools import grangercausalitytests
    _HAS_GRANGER = True
except Exception:
    grangercausalitytests = None
    _HAS_GRANGER = False

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.forecast import load_trained_model
from src.util import DataLoaderS


def _load_nodes(nodes_path: str) -> List[str]:
    df = pd.read_csv(nodes_path)
    if "token" in df.columns:
        return df["token"].tolist()
    return df.iloc[:, 0].tolist()


def _load_rmd_pt_map(path: str) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",") if p.strip()]
            if not parts:
                continue
            mapping[parts[0]] = parts[1:]
    return mapping


def _load_series(data_path: str, has_header: bool = True, drop_first_col: bool = True) -> pd.DataFrame:
    df = pd.read_csv(data_path, header=0 if has_header else None)
    if drop_first_col and df.shape[1] > 1:
        df = df.iloc[:, 1:]
    return df


def _append_forecast(base_df: pd.DataFrame, forecast_path: str) -> pd.DataFrame:
    forecast_df = pd.read_csv(forecast_path)
    if "Date" in forecast_df.columns:
        forecast_df = forecast_df.set_index("Date")
    elif "t" in forecast_df.columns:
        forecast_df = forecast_df.set_index("t")
    else:
        forecast_df = forecast_df.set_index(forecast_df.columns[0])
    forecast_df = forecast_df.reset_index(drop=True)
    forecast_df = forecast_df[base_df.columns.intersection(forecast_df.columns)]
    return pd.concat([base_df, forecast_df], axis=0, ignore_index=True)


def _trend_slope(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    x = np.arange(values.size, dtype=float)
    slope = np.polyfit(x, values, 1)[0]
    return float(slope)


def _lag_corr(x: np.ndarray, y: np.ndarray, max_lag: int) -> Tuple[int, float]:
    best_lag = 0
    best_corr = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            xs = x[-lag:]
            ys = y[: len(xs)]
        elif lag > 0:
            xs = x[: -lag]
            ys = y[lag:]
        else:
            xs = x
            ys = y
        if xs.size < 3:
            continue
        corr = np.corrcoef(xs, ys)[0, 1]
        if np.isnan(corr):
            continue
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    return best_lag, float(best_corr) if np.isfinite(best_corr) else float("nan")


def _granger_pvalue(x: np.ndarray, y: np.ndarray, max_lag: int) -> float:
    if not _HAS_GRANGER or grangercausalitytests is None or x.size < (max_lag + 3):
        return float("nan")
    data = np.column_stack([y, x])
    try:
        results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        pvals = [results[l][0]["ssr_ftest"][1] for l in results]
        return float(np.nanmin(pvals)) if pvals else float("nan")
    except Exception:
        return float("nan")


def _save_attention_outputs(adj: np.ndarray, nodes: List[str], out_dir: Path, top_k: int = 200) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    adj = np.asarray(adj, dtype=float)
    np.fill_diagonal(adj, 0.0)

    # Save edge list
    edges = []
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            w = adj[i, j]
            if w > 0:
                edges.append((nodes[i], nodes[j], float(w)))
    edges.sort(key=lambda x: x[2], reverse=True)
    edge_df = pd.DataFrame(edges, columns=["src", "dst", "weight"])
    edge_df.head(top_k).to_csv(out_dir / "attention_edges_topk.csv", index=False)
    edge_df.to_csv(out_dir / "attention_edges_all.csv", index=False)

    # Node influence scores
    out_strength = adj.sum(axis=1)
    in_strength = adj.sum(axis=0)
    total_strength = out_strength + in_strength
    infl_df = pd.DataFrame({
        "node": nodes,
        "out_strength": out_strength,
        "in_strength": in_strength,
        "total_strength": total_strength,
    }).sort_values("total_strength", ascending=False)
    infl_df.to_csv(out_dir / "node_influence_strength.csv", index=False)

    # Heatmap for attention map
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(adj, cmap="viridis", ax=ax, cbar_kws={"label": "Learned edge weight"})
    ax.set_title("Learned Graph Attention (Adjacency)")
    ax.set_xlabel("Target node")
    ax.set_ylabel("Source node")
    plt.tight_layout()
    fig.savefig(out_dir / "attention_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "attention_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def _compute_pair_metrics(df: pd.DataFrame, rmd_pt: Dict[str, List[str]], max_lag: int, trend_window: int) -> pd.DataFrame:
    rows = []
    for rmd, pts in rmd_pt.items():
        if rmd not in df.columns:
            continue
        rmd_series = df[rmd].to_numpy(dtype=float)
        if rmd_series.size < 5:
            continue
        rmd_recent = rmd_series[-trend_window:] if rmd_series.size >= trend_window else rmd_series
        rmd_slope = _trend_slope(rmd_recent)
        for pt in pts:
            if pt not in df.columns:
                continue
            pt_series = df[pt].to_numpy(dtype=float)
            if pt_series.size < 5:
                continue
            pt_recent = pt_series[-trend_window:] if pt_series.size >= trend_window else pt_series
            pt_slope = _trend_slope(pt_recent)

            lag, corr = _lag_corr(rmd_series, pt_series, max_lag=max_lag)
            pval = _granger_pvalue(rmd_series, pt_series, max_lag=max_lag)

            gap_score = rmd_slope - pt_slope
            warning = int(rmd_slope > 0 and pt_slope <= 0)

            rows.append({
                "rmd": rmd,
                "pt": pt,
                "rmd_slope": rmd_slope,
                "pt_slope": pt_slope,
                "gap_score": gap_score,
                "warning_flag": warning,
                "lag_best": lag,
                "lag_corr": corr,
                "granger_p": pval,
            })
    return pd.DataFrame(rows)


def _compute_network_dynamics(df: pd.DataFrame, rmd_pt: Dict[str, List[str]], window: int = 24, step: int = 3) -> pd.DataFrame:
    rows = []
    n = df.shape[0]
    for end in range(window, n + 1, step):
        start = end - window
        slice_df = df.iloc[start:end]
        edge_vals = []
        for rmd, pts in rmd_pt.items():
            if rmd not in slice_df.columns:
                continue
            rmd_series = slice_df[rmd].to_numpy(dtype=float)
            for pt in pts:
                if pt not in slice_df.columns:
                    continue
                pt_series = slice_df[pt].to_numpy(dtype=float)
                if rmd_series.size < 3 or pt_series.size < 3:
                    continue
                corr = np.corrcoef(rmd_series, pt_series)[0, 1]
                if not np.isnan(corr):
                    edge_vals.append(corr)
        avg_corr = float(np.nanmean(edge_vals)) if edge_vals else float("nan")
        rows.append({
            "window_start": start,
            "window_end": end,
            "avg_rmd_pt_corr": avg_corr,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Co-evolution analysis for RMD/PT series")
    parser.add_argument("--data", type=str, default="data/sm_data_g.csv")
    parser.add_argument("--nodes", type=str, default="data/nodes.csv")
    parser.add_argument("--rmd-pt-map", type=str, default="data/Archives/RMD_PT_map.csv")
    parser.add_argument("--graph", type=str, default="data/graph_topk_k12.csv")
    parser.add_argument("--checkpoint", type=str, default="model/Bayesian/model.pt")
    parser.add_argument("--forecast", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="model/Bayesian/forecast/co_evolution")
    parser.add_argument("--trend-window", type=int, default=24)
    parser.add_argument("--max-lag", type=int, default=12)
    parser.add_argument("--net-window", type=int, default=24)
    parser.add_argument("--net-step", type=int, default=3)
    parser.add_argument("--topk-edges", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data series
    df = _load_series(args.data, has_header=True, drop_first_col=True)
    if args.forecast:
        df = _append_forecast(df, args.forecast)

    # Load mappings
    nodes = _load_nodes(args.nodes)
    rmd_pt = _load_rmd_pt_map(args.rmd_pt_map)

    # Learned adjacency (attention map)
    adj = None
    if os.path.exists(args.checkpoint):
        model, _, _ = load_trained_model(args.checkpoint, device=args.device, args=None)
        if getattr(model, "gc", None) is not None:
            idx = model.idx
            try:
                adj = model.gc.fullA(idx).detach().cpu().numpy()
            except Exception:
                try:
                    adj = model.gc(idx).detach().cpu().numpy()
                except Exception:
                    adj = None

    if adj is None and os.path.exists(args.graph):
        adj = np.loadtxt(args.graph, delimiter=",")

    if adj is not None:
        size = min(len(nodes), adj.shape[0], adj.shape[1])
        _save_attention_outputs(adj[:size, :size], nodes[:size], out_dir, top_k=args.topk_edges)

    # Pair metrics: leading/lagging + early-warning gaps
    pair_df = _compute_pair_metrics(df, rmd_pt, max_lag=args.max_lag, trend_window=args.trend_window)
    pair_df.to_csv(out_dir / "rmd_pt_lead_lag_metrics.csv", index=False)

    warning_df = pair_df.sort_values("gap_score", ascending=False)
    warning_df.to_csv(out_dir / "early_warning_gaps.csv", index=False)

    # Network dynamics (rolling correlation)
    net_df = _compute_network_dynamics(df, rmd_pt, window=args.net_window, step=args.net_step)
    net_df.to_csv(out_dir / "network_dynamics.csv", index=False)

    print("[co-evolution] Outputs saved to", out_dir)
    print("[co-evolution] Pair metrics:", out_dir / "rmd_pt_lead_lag_metrics.csv")
    print("[co-evolution] Early-warning gaps:", out_dir / "early_warning_gaps.csv")
    print("[co-evolution] Network dynamics:", out_dir / "network_dynamics.csv")
    if adj is not None:
        print("[co-evolution] Attention heatmap:", out_dir / "attention_heatmap.png")


if __name__ == "__main__":
    main()
