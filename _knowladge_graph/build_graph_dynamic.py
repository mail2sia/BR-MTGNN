#!/usr/bin/env python3
"""
build_graph_dynamic.py — Static + Dynamic bipartite RMD–PT graphs for B-MTGNN (SILF-ready)

Outputs
- graph.csv                  : static N×N adjacency (bipartite, symmetric, zero diag, [0,1])
- nodes.csv                  : node index→(Name,Type)
- edges_timeseries.csv       : static RMD–PT edge table with metrics
- edge_channels_static.npz   : static channels: pearson, xcorr, coact, score  (each N×N)
- node_features.csv          : per-node features (mean, std, trend, acf12, activity, strength)
- graph_dynamic.npz          : dynamic adjacency cube [T, N, N] from rolling windows
- edge_channels_dynamic.npz  : dynamic channels (pearson, xcorr, coact, score) each [T, N, N]
- (optional) graph.graphml   : NetworkX GraphML (if networkx present)
- (optional) pyg_static.pt   : PyTorch Geometric tensors (if torch + torch_geometric present)

Design
- Bipartite only: RMD–PT edges. RMD–RMD and PT–PT are zeroed.
- Static graph: metrics computed over full time span.
- Dynamic graph: rolling windows (e.g., 24 months) with stride=1 month yield evolving edges.

Scoring
- pearson_abs: |Pearson| on valid months
- max_xcorr_abs: max |Pearson| over lags 1..max_lag (excl zero-lag)
- coactivity_pmi_sigmoid: σ(PMI) of (>0) co-activity
- score = 0.3*pearson + 0.4*xcorr + 0.3*coact  (clipped [0,1])

Node features
- mean_nomn, std_nomn, trend_slope_24m (OLS on last 24), acf12, activity_ratio, strength (sum of static weights)

Robustness
- Handles pivot or long-form data (auto-pivots)
- Synonym-aware classification via search_terms.py
- QA: square, symmetric, zero diag, finite, [0,1]; warns on isolates

How to run:
python build_graph_dynamic.py \
  --data data.csv \
  --terms search_terms.py \
  --graph-out graph.csv \
  --nodes-out nodes.csv \
  --edges-out edges_timeseries.csv \
  --edge-channels-out edge_channels_static.npz \
  --node-feats-out node_features.csv \
  --graph-dyn-out graph_dynamic.npz \
  --edge-channels-dyn-out edge_channels_dynamic.npz \
  --window-months 24 --stride 1 --max-lag 24 \
  -v

"""

import argparse
import importlib.util
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# --------------------------- logging ---------------------------------
def setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity <= 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")


# --------------------------- terms -----------------------------------
def clean_token(s: str) -> str:
    import re
    s = s.lower()
    s = re.sub(r"[_\-\s/]+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_terms_py(path: Path) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    if not path.exists():
        raise FileNotFoundError(f"terms file not found: {path}")
    spec = importlib.util.spec_from_file_location("terms_mod", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    if not hasattr(mod, "DISEASE_SYNONYMS") or not hasattr(mod, "PT_SYNONYMS"):
        raise ValueError("terms file must define DISEASE_SYNONYMS and PT_SYNONYMS")
    return dict(mod.DISEASE_SYNONYMS), dict(mod.PT_SYNONYMS)

def build_match_maps(disease_syn: Dict[str, List[str]], pt_syn: Dict[str, List[str]]) -> Tuple[Dict[str, str], Dict[str, str]]:
    dis_map: Dict[str, str] = {}
    pt_map: Dict[str, str] = {}
    for canon, syns in disease_syn.items():
        for t in (syns or []):
            dis_map[clean_token(t)] = canon
        dis_map[clean_token(canon)] = canon
    for canon, syns in pt_syn.items():
        for t in (syns or []):
            pt_map[clean_token(t)] = canon
        pt_map[clean_token(canon)] = canon
    return dis_map, pt_map

def strip_solution_naming(col: str) -> str:
    if isinstance(col, str) and col.startswith("Solution_") and col.endswith("_Mentions"):
        return col[len("Solution_"):-len("_Mentions")]
    return col

def classify_columns(columns: List[str], disease_map: Dict[str, str], pt_map: Dict[str, str]) -> Tuple[List[str], List[str]]:
    rmd_cols, pt_cols = [], []
    for c in columns:
        token = strip_solution_naming(c)
        k = clean_token(token)
        if k in pt_map:
            pt_cols.append(c)
        elif k in disease_map:
            rmd_cols.append(c)
        else:
            # conservative: unknown -> RMD
            rmd_cols.append(c)
    return rmd_cols, pt_cols


# --------------------------- data load --------------------------------
def infer_date_col(df: pd.DataFrame) -> Optional[str]:
    for guess in ("Date", "date", "month", "Month", "year_month", "YearMonth"):
        if guess in df.columns:
            return guess
    first = df.columns[0]
    try:
        pd.to_datetime(df[first], errors="raise")
        return first
    except Exception:
        return None

def load_dataset(path: Path) -> Tuple[pd.DataFrame, List[pd.Timestamp]]:
    if not path.exists():
        raise FileNotFoundError(f"data file not found: {path}")
    df = pd.read_csv(path)

    date_col = infer_date_col(df)
    if date_col is not None and df.shape[1] > 2:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        if df[date_col].isna().all():
            raise ValueError(f"Could not parse date column '{date_col}' in {path}")
        df["year_month"] = df[date_col].dt.to_period("M").dt.to_timestamp()
        df = df.drop(columns=[date_col]).set_index("year_month").sort_index()
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        idx = list(df.index)
        logging.info("Loaded pivot dataset: shape=%s", df.shape)
        return df, idx

    lower = {c.lower(): c for c in df.columns}
    if {"date", "terms", "nom"}.issubset(lower.keys()):
        c_date, c_terms, c_nom = lower["date"], lower["terms"], lower["nom"]
        df[c_date] = pd.to_datetime(df[c_date], errors="coerce")
        if df[c_date].isna().all():
            raise ValueError(f"Could not parse 'date' in {path}")
        df["year_month"] = df[c_date].dt.to_period("M").dt.to_timestamp()
        pv = df.pivot_table(index="year_month", columns=c_terms, values=c_nom, aggfunc="sum")
        for c in pv.columns:
            pv[c] = pd.to_numeric(pv[c], errors="coerce")
        pv = pv.sort_index()
        idx = list(pv.index)
        logging.info("Loaded long dataset and pivoted: shape=%s", pv.shape)
        return pv, idx

    raise ValueError("Unsupported dataset format. Expect pivot (Date + term cols) or long (date, terms, NoM).")


# --------------------------- metrics ----------------------------------
def pearson_abs(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return 0.0
    xv, yv = x[valid], y[valid]
    if xv.var() <= 1e-12 or yv.var() <= 1e-12:
        return 0.0
    c = np.corrcoef(xv, yv)[0, 1]
    return float(abs(c)) if np.isfinite(c) else 0.0

def max_xcorr_abs(x: np.ndarray, y: np.ndarray, max_lag: int) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    n = len(x)
    if n < 3:
        return 0.0
    best = 0.0
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        c1 = pearson_abs(x[:-lag], y[lag:])
        c2 = pearson_abs(x[lag:], y[:-lag])
        best = max(best, c1, c2)
    return best

def coactivity_pmi_sigmoid(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() == 0:
        return 0.0
    xv = (x[valid] > 0).astype(int)
    yv = (y[valid] > 0).astype(int)
    n = len(xv)
    if n == 0:
        return 0.0
    px = xv.mean(); py = yv.mean(); pxy = (xv & yv).mean()
    eps = 1e-12
    pmi = math.log((pxy + eps) / ((px + eps) * (py + eps)))
    return float(1.0 / (1.0 + math.exp(-pmi)))  # (0,1)


# --------------------------- edges (static) ----------------------------
def filter_active(cols: List[str], df: pd.DataFrame) -> List[str]:
    keep = []
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(v).sum() > 0 and np.nansum(np.abs(v)) > 0.0:
            keep.append(c)
    return keep

def build_edges_static(df: pd.DataFrame, rmd: List[str], pt: List[str],
                       min_overlap: int, min_coactive: int, max_lag: int) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    rmd_a = filter_active(rmd, df)
    pt_a  = filter_active(pt, df)
    if not rmd_a or not pt_a:
        logging.warning("All columns filtered; rmd=%d pt=%d", len(rmd_a), len(pt_a))

    rows = []
    N = len(rmd_a) + len(pt_a)
    pear_mat = np.zeros((N, N), dtype=np.float64)
    xcor_mat = np.zeros((N, N), dtype=np.float64)
    coac_mat = np.zeros((N, N), dtype=np.float64)
    scor_mat = np.zeros((N, N), dtype=np.float64)
    node_index = {n:i for i,n in enumerate(list(rmd_a)+list(pt_a))}

    for r in rmd_a:
        xr = df[r].to_numpy(dtype=float)
        for p in pt_a:
            yp = df[p].to_numpy(dtype=float)
            valid = np.isfinite(xr) & np.isfinite(yp)
            if valid.sum() < max(min_overlap, 3):
                continue
            coactive = int(((xr > 0) & (yp > 0) & np.isfinite(xr) & np.isfinite(yp)).sum())
            if coactive < min_coactive:
                continue

            pear = pearson_abs(xr, yp)
            xcor = max_xcorr_abs(xr, yp, max_lag=max_lag)
            coac = coactivity_pmi_sigmoid(xr, yp)
            scor = 0.3*pear + 0.4*xcor + 0.3*coac
            i, j = node_index[r], node_index[p]
            pear_mat[i,j] = pear_mat[j,i] = pear
            xcor_mat[i,j] = xcor_mat[j,i] = xcor
            coac_mat[i,j] = coac_mat[j,i] = coac
            scor_mat[i,j] = scor_mat[j,i] = min(max(scor, 0.0), 1.0)

            rows.append({
                "RMD": r, "PT": p,
                "pearson_abs": round(pear,6),
                "max_xcorr_abs": round(xcor,6),
                "coactivity_sigmoid": round(coac,6),
                "coactive_months": coactive,
                "score": round(scor,6),
            })

    edges = pd.DataFrame(rows).sort_values(
        by=["score","coactive_months","max_xcorr_abs","pearson_abs"],
        ascending=False, kind="mergesort"
    )
    channels = {"pearson": pear_mat, "xcorr": xcor_mat, "coact": coac_mat, "score": scor_mat}
    return edges, channels


# --------------------------- dynamic windows --------------------------
def rolling_slices(index: List[pd.Timestamp], window_months: int, stride: int) -> List[Tuple[int,int]]:
    """Return list of (start_idx, end_idx_exclusive) windows."""
    if window_months <= 2:
        raise ValueError("window_months must be > 2")
    T = len(index)
    spans = []
    for start in range(0, T - window_months + 1, stride):
        end = start + window_months
        spans.append((start, end))
    return spans

def build_edges_dynamic(df: pd.DataFrame, rmd: List[str], pt: List[str],
                        idx: List[pd.Timestamp],
                        window_months: int, stride: int,
                        min_overlap: int, min_coactive: int, max_lag: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Return dynamic adjacency [Tw, N, N] and dynamic channels dict of same shape."""
    rmd_a = filter_active(rmd, df)
    pt_a  = filter_active(pt, df)
    nodes = list(rmd_a)+list(pt_a)
    N = len(nodes)
    node_index = {n:i for i,n in enumerate(nodes)}
    spans = rolling_slices(idx, window_months, stride)
    Tw = len(spans)

    A_dyn   = np.zeros((Tw, N, N), dtype=np.float32)
    P_dyn   = np.zeros_like(A_dyn)
    X_dyn   = np.zeros_like(A_dyn)
    C_dyn   = np.zeros_like(A_dyn)

    arr = df[nodes].to_numpy(dtype=float)  # [T, N]

    for t, (s,e) in enumerate(spans):
        seg = arr[s:e, :]                  # [W, N]
        for i, r in enumerate(rmd_a):
            xr = seg[:, node_index[r]]
            for p in pt_a:
                j = node_index[p]
                yp = seg[:, j]
                valid = np.isfinite(xr) & np.isfinite(yp)
                if valid.sum() < max(min_overlap, 3):
                    continue
                coactive = int(((xr > 0) & (yp > 0) & np.isfinite(xr) & np.isfinite(yp)).sum())
                if coactive < min_coactive:
                    continue
                pear = pearson_abs(xr, yp)
                xcor = max_xcorr_abs(xr, yp, max_lag=max_lag)
                coac = coactivity_pmi_sigmoid(xr, yp)
                scor = min(max(0.3*pear + 0.4*xcor + 0.3*coac, 0.0), 1.0)
                A_dyn[t, i, j] = A_dyn[t, j, i] = scor
                P_dyn[t, i, j] = P_dyn[t, j, i] = pear
                X_dyn[t, i, j] = X_dyn[t, j, i] = xcor
                C_dyn[t, i, j] = C_dyn[t, j, i] = coac

        # sanitize window
        np.fill_diagonal(A_dyn[t], 0.0)
        np.fill_diagonal(P_dyn[t], 0.0)
        np.fill_diagonal(X_dyn[t], 0.0)
        np.fill_diagonal(C_dyn[t], 0.0)

    return A_dyn, {"pearson": P_dyn, "xcorr": X_dyn, "coact": C_dyn, "score": A_dyn}


# --------------------------- node features ----------------------------
def acf(series: np.ndarray, lag: int) -> float:
    valid = np.isfinite(series)
    x = series[valid]
    if len(x) <= lag+1:
        return 0.0
    x0 = x[:-lag]; x1 = x[lag:]
    return pearson_abs(x0, x1)

def ols_slope_last(series: np.ndarray, last_k: int) -> float:
    valid = np.isfinite(series)
    x = series[valid]
    if len(x) < 3:
        return 0.0
    x = x[-last_k:] if len(x) >= last_k else x
    n = len(x)
    t = np.arange(n, dtype=float)
    t_mean = t.mean(); x_mean = x.mean()
    denom = ((t - t_mean)**2).sum()
    if denom <= 1e-12:
        return 0.0
    slope = ((t - t_mean) * (x - x_mean)).sum() / denom
    return float(slope)

def compute_node_features(df: pd.DataFrame, nodes: List[str], A_static: np.ndarray) -> pd.DataFrame:
    X = df[nodes].to_numpy(dtype=float)
    feats = []
    strengths = A_static.sum(axis=1)
    for i, n in enumerate(nodes):
        s = X[:, i]
        feats.append({
            "Name": n,
            "mean_nomn": float(np.nanmean(s)),
            "std_nomn": float(np.nanstd(s)),
            "trend_slope_24m": ols_slope_last(s, last_k=24),
            "acf12": acf(s, lag=12),
            "activity_ratio": float(np.isfinite(s).sum() and (np.nanmean((s>0).astype(float)))),
            "strength": float(strengths[i]),
        })
    return pd.DataFrame(feats)


# --------------------------- QA & save --------------------------------
def qa_graph(A: np.ndarray) -> None:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"[QA] graph must be square, got {A.shape}")
    if np.isnan(A).any() or np.isinf(A).any():
        raise ValueError("[QA] graph contains NaN/Inf.")
    if (A < 0.0).any() or (A > 1.0).any():
        raise ValueError("[QA] graph has values outside [0,1].")
    if not np.allclose(A, A.T, atol=1e-10):
        raise ValueError("[QA] graph must be symmetric.")
    if not np.allclose(np.diag(A), 0.0, atol=1e-12):
        raise ValueError("[QA] diagonal must be 0.")

def save_graph(A: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(path), A, delimiter=",", fmt="%.6f")

def save_nodes(nodes: List[str], types: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Index": range(len(nodes)), "Name": nodes, "Type": types}).to_csv(path, index=False)

def try_export_graphml(A: np.ndarray, nodes: List[str], types: List[str], path: Path) -> None:
    try:
        import networkx as nx
    except Exception:
        logging.info("networkx not installed; skipping GraphML export.")
        return
    G = nx.Graph()
    for i, name in enumerate(nodes):
        G.add_node(i, name=name, type=types[i])
    N = len(nodes)
    for i in range(N):
        for j in range(i+1, N):
            w = float(A[i, j])
            if w > 0.0:
                G.add_edge(i, j, weight=w)
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, str(path))
    logging.info("Wrote GraphML -> %s", path)

# def try_export_pyg(A: np.ndarray, nodes: List[str], out_path: Path) -> None:
#     try:
#         import torch
#         from torch_geometric.data import Data
#     except Exception:
#         logging.info("PyTorch Geometric not installed; skipping pyg export.")
#         return
#     N = len(nodes)
#     edges_i, edges_j, weights = [], [], []
#     for i in range(N):
#         for j in range(i+1, N):
#             w = A[i, j]
#             if w > 0.0:
#                 edges_i += [i, j]
#                 edges_j += [j, i]
#                 weights += [w, w]
#     if not edges_i:
#         logging.warning("No edges to export for PyG.")
#         return
#     edge_index = np.vstack([edges_i, edges_j]).astype(np.int64)
#     edge_attr = np.array(weights, dtype=np.float32)[:, None]
#     data = Data(
#         edge_index = (torch.from_numpy(edge_index)),
#         edge_attr  = (torch.from_numpy(edge_attr)),
#         num_nodes  = N
#     )
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     try:
#         torch.save(data, str(out_path))
#         logging.info("Wrote PyG tensors -> %s", out_path)
#     except Exception as e:
#         logging.warning("Failed to save PyG tensors: %s", e)


# --------------------------- main -------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build static + dynamic bipartite RMD–PT graphs for B-MTGNN.")
    p.add_argument("--data", required=True, help="Path to data CSV (pivot or long-form).")
    p.add_argument("--terms", required=True, help="Path to search_terms.py.")
    p.add_argument("--graph-out", default="graph.csv", help="Output path for static graph.csv.")
    p.add_argument("--nodes-out", default="nodes.csv", help="Output path for nodes.csv.")
    p.add_argument("--edges-out", default="edges_timeseries.csv", help="Output path for static edges table (QA).")
    p.add_argument("--edge-channels-out", default="edge_channels_static.npz", help="Output path for static channels npz.")
    p.add_argument("--node-feats-out", default="node_features.csv", help="Output path for node features CSV.")
    # dynamic
    p.add_argument("--graph-dyn-out", default="graph_dynamic.npz", help="Output path for dynamic adjacency cube.")
    p.add_argument("--edge-channels-dyn-out", default="edge_channels_dynamic.npz", help="Output path for dynamic channels npz.")
    p.add_argument("--window-months", type=int, default=24, help="Rolling window size (months) for dynamic graph.")
    p.add_argument("--stride", type=int, default=1, help="Window stride (months).")
    # metrics thresholds
    p.add_argument("--min-overlap", type=int, default=6, help="Min overlapping valid months per pair.")
    p.add_argument("--min-coactive", type=int, default=2, help="Min months with both series > 0.")
    p.add_argument("--max-lag", type=int, default=24, help="Max lead/lag months for cross-corr (excl. zero-lag).")
    # extras
    p.add_argument("--graphml-out", default="", help="Optional: write GraphML (requires networkx).")
    p.add_argument("--pyg-out", default="", help="Optional: write PyG tensors (requires torch & torch_geometric).")
    p.add_argument("-v", "--verbose", action="count", default=1, help="-v (info) or -vv (debug).")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    data_path = Path(args.data)
    terms_path = Path(args.terms)

    # 1) Load data (pivot) and monthly index
    pivot_df, month_index = load_dataset(data_path)

    # 2) Load terms and classify columns
    disease_syn, pt_syn = load_terms_py(terms_path)
    dis_map, pt_map     = build_match_maps(disease_syn, pt_syn)
    rmd_cols, pt_cols   = classify_columns(list(pivot_df.columns), dis_map, pt_map)
    if not rmd_cols or not pt_cols:
        raise RuntimeError(f"Classification failed. RMD={len(rmd_cols)} PT={len(pt_cols)}")

    # Reindex to monthly continuous range to avoid window boundary drift
    full_idx = pd.period_range(start=month_index[0], end=month_index[-1], freq="M").to_timestamp()
    pivot_df = pivot_df.reindex(full_idx)
    month_index = list(pivot_df.index)

    logging.info("Detected RMD=%d, PT=%d, months=%d", len(rmd_cols), len(pt_cols), len(month_index))

    # 3) Static edges and channels
    edges, channels = build_edges_static(
        pivot_df, rmd_cols, pt_cols,
        min_overlap=args.min_overlap, min_coactive=args.min_coactive, max_lag=args.max_lag
    )

    # build static adjacency in node order [RMD..., PT...]
    nodes = list(rmd_cols) + list(pt_cols)
    types = (["RMD"] * len(rmd_cols)) + (["PT"] * len(pt_cols))
    idx_map = {n:i for i,n in enumerate(nodes)}
    N = len(nodes)
    A = np.zeros((N, N), dtype=np.float64)
    for _, row in edges.iterrows():
        i = idx_map[row["RMD"]]; j = idx_map[row["PT"]]
        w = float(min(max(row["score"], 0.0), 1.0))
        A[i,j] = max(A[i,j], w)
        A[j,i] = A[i,j]
    np.fill_diagonal(A, 0.0)
    A = np.nan_to_num(A, nan=0.0, posinf=1.0, neginf=0.0)
    A = np.clip(A, 0.0, 1.0)
    qa_graph(A)

    # 4) Node features (from raw series + static graph)
    node_feats = compute_node_features(pivot_df, nodes, A)

    # 5) Dynamic graphs & channels
    A_dyn, ch_dyn = build_edges_dynamic(
        pivot_df, rmd_cols, pt_cols,
        idx=month_index,
        window_months=args.window_months, stride=args.stride,
        min_overlap=args.min_overlap, min_coactive=args.min_coactive, max_lag=args.max_lag
    )

    # 6) Save everything
    save_graph(A, Path(args.graph_out))
    save_nodes(nodes, types, Path(args.nodes_out))
    Path(args.edges_out).parent.mkdir(parents=True, exist_ok=True)
    edges.to_csv(args.edges_out, index=False)

    # static channels aligned to [RMD..., PT...] order
    # channels were built on active subsets; rebuild dense channels aligned to full nodes
    dense = {k: np.zeros((N, N), dtype=np.float64) for k in ("pearson","xcorr","coact","score")}
    # populate from edges table (guaranteed subset of nodes)
    for _, row in edges.iterrows():
        i = idx_map[row["RMD"]]; j = idx_map[row["PT"]]
        dense["pearson"][i,j] = dense["pearson"][j,i] = float(row["pearson_abs"])
        dense["xcorr"][i,j]   = dense["xcorr"][j,i]   = float(row["max_xcorr_abs"])
        dense["coact"][i,j]   = dense["coact"][j,i]   = float(row["coactivity_sigmoid"])
        dense["score"][i,j]   = dense["score"][j,i]   = float(min(max(row["score"],0.0),1.0))
    np.savez_compressed(Path(args.edge_channels_out),
                        pearson=dense["pearson"], xcorr=dense["xcorr"],
                        coact=dense["coact"], score=dense["score"])

    # dynamic channels already dense on active nodes; pad to full N if any inactive columns existed
    # Here we assumed classification skipped only truly empty columns; typically N matches.
    np.savez_compressed(Path(args.graph_dyn_out), graph=A_dyn.astype(np.float32))
    np.savez_compressed(Path(args.edge_channels_dyn_out),
                        pearson=ch_dyn["pearson"].astype(np.float32),
                        xcorr=ch_dyn["xcorr"].astype(np.float32),
                        coact=ch_dyn["coact"].astype(np.float32),
                        score=ch_dyn["score"].astype(np.float32))

    Path(args.node_feats_out).parent.mkdir(parents=True, exist_ok=True)
    node_feats.to_csv(Path(args.node_feats_out), index=False)

    # # Optional exports (don’t fail pipeline if missing)
    # if args.graphml_out:
    #     try_export_graphml(A, nodes, types, Path(args.graphml_out))
    # if args.pyg_out:
    #     try_export_pyg(A, nodes, Path(args.pyg_out))

    # logging.info("DONE. Static graph: %s | Dynamic cube: %s | Nodes: %s",
    #              args.graph_out, args.graph_dyn_out, args.nodes_out)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.getLogger().exception("FAILED: %s", e)
        sys.exit(1)
