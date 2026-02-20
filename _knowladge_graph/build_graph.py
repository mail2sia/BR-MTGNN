#!/usr/bin/env python3
"""
build_graph.py — Create a bipartite RMD–PT adjacency matrix (graph.csv) for B-MTGNN.

Features
- Accepts pivot (wide) or long-form input.
- Classifies columns using DISEASE_SYNONYMS and PT_SYNONYMS from search_terms.py.
- Builds a square, symmetric, zero-diagonal adjacency with values in [0,1].
- Only RMD–PT edges (bipartite); RMD–RMD and PT–PT are zero.
- Exposes robust CLI for paths and scoring options.
- Emits nodes.csv (index→name,type) and optional edges_timeseries.csv for QA.

Scoring
- pearson_pos: |Pearson| on aligned valid months (no NaN).
- max_xcorr: max |Pearson| over lags 1..max_lag (lead/lag), excluding zero-lag.
- coactivity_pmi: PMI of monthly (>0) co-activity, mapped by logistic σ(PMI) to (0,1).
- score = 0.3*pearson_pos + 0.4*max_xcorr + 0.3*coactivity_pmi

QA
- Enforces square, symmetric, zero diagonal, finite, [0,1].
- Warns on isolated nodes (row-sum=0).
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
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


# --------------------------- term maps --------------------------------

def clean_token(s: str) -> str:
    """Normalize a token for matching: lowercase, collapse separators, strip punctuation."""
    import re
    s = s.lower()
    s = re.sub(r"[_\-\s/]+", " ", s)      # normalize separators
    s = re.sub(r"[^\w\s]", "", s)         # drop punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_terms_py(path: Path) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Load DISEASE_SYNONYMS and PT_SYNONYMS from search_terms.py."""
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


def build_match_maps(
    disease_syn: Dict[str, List[str]],
    pt_syn: Dict[str, List[str]]
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Build norm_token -> canonical maps for diseases and PTs using canonical + synonyms.
    Canonical overwrites synonyms on collision.
    """
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
    """
    Handle legacy "Solution_*_Mentions" style; returns a normalized middle token if matched,
    otherwise returns the original column.
    """
    if not isinstance(col, str):
        return col
    if col.startswith("Solution_") and col.endswith("_Mentions"):
        return col[len("Solution_"):-len("_Mentions")]
    return col


def classify_columns(
    columns: List[str],
    disease_map: Dict[str, str],
    pt_map: Dict[str, str],
    date_col_name: Optional[str]
) -> Tuple[List[str], List[str]]:
    """
    Classify pivot columns into RMD vs PT using term maps and legacy naming rules.
    Unknown columns: default to RMD (conservative).
    """
    rmd_cols, pt_cols = [], []
    for c in columns:
        if date_col_name and c == date_col_name:
            continue
        token = strip_solution_naming(c)
        k = clean_token(token)

        if k in pt_map:
            pt_cols.append(c)
        elif k in disease_map:
            rmd_cols.append(c)
        else:
            # Legacy rule: Solution_*_Mentions => PT
            if isinstance(c, str) and c.startswith("Solution_") and c.endswith("_Mentions"):
                pt_cols.append(c)
            else:
                rmd_cols.append(c)  # conservative default
    return rmd_cols, pt_cols


# --------------------------- load data --------------------------------

def infer_date_col(df: pd.DataFrame) -> Optional[str]:
    """Try to find a date column."""
    for guess in ("Date", "date", "month", "Month", "year_month", "YearMonth"):
        if guess in df.columns:
            return guess
    # if first column looks like date, allow it
    first = df.columns[0]
    try:
        pd.to_datetime(df[first], errors="raise")
        return first
    except Exception:
        return None


def load_dataset(path: Path) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Load a dataset as pivot (wide) monthly frame: index=datetime (monthly), numeric columns=terms.
    Returns (df, date_col_name if present else None).
    Accepts:
      - Pivot CSV with a date-like column + numeric columns for terms
      - Long CSV with columns (date, terms, NoM) => auto-pivoted
    """
    if not path.exists():
        raise FileNotFoundError(f"data file not found: {path}")

    df = pd.read_csv(path)
    # pivot-like?
    date_col = infer_date_col(df)
    if date_col is not None and df.shape[1] > 2:
        # parse date col to month start
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        if df[date_col].isna().all():
            raise ValueError(f"Could not parse date column '{date_col}' in {path}")
        df["year_month"] = df[date_col].dt.to_period("M").dt.to_timestamp()
        df = df.drop(columns=[date_col]).set_index("year_month").sort_index()
        # numeric only; keep NaN (mask later)
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        logging.info("Loaded pivot dataset with shape=%s", df.shape)
        return df, date_col

    # long-like?
    lower_cols = {c.lower(): c for c in df.columns}
    if {"date", "terms", "nom"}.issubset(lower_cols.keys()):
        c_date, c_terms, c_nom = lower_cols["date"], lower_cols["terms"], lower_cols["nom"]
        df[c_date] = pd.to_datetime(df[c_date], errors="coerce")
        if df[c_date].isna().all():
            raise ValueError(f"Could not parse 'date' in {path}")
        df["year_month"] = df[c_date].dt.to_period("M").dt.to_timestamp()
        pt = df.pivot_table(index="year_month", columns=c_terms, values=c_nom, aggfunc="sum")
        # numeric; keep NaN (mask later)
        for c in pt.columns:
            pt[c] = pd.to_numeric(pt[c], errors="coerce")
        pt = pt.sort_index()
        logging.info("Loaded long dataset and pivoted to shape=%s", pt.shape)
        return pt, None

    raise ValueError(
        f"Unsupported dataset format for {path}. Expected pivot (Date + term columns) or long (date, terms, NoM)."
    )


# --------------------------- metrics ----------------------------------

def _pearson_abs(x: np.ndarray, y: np.ndarray) -> float:
    """Absolute Pearson on aligned valid samples; returns 0 for degenerate cases."""
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return 0.0
    xv = x[valid]
    yv = y[valid]
    vx = xv.var()
    vy = yv.var()
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    c = np.corrcoef(xv, yv)[0, 1]
    if not np.isfinite(c):
        return 0.0
    return abs(float(c))


def _max_xcorr_abs(x: np.ndarray, y: np.ndarray, max_lag: int = 24) -> float:
    """Max absolute Pearson over lags 1..max_lag (lead/lag), excluding zero-lag."""
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    n = len(x)
    if n < 3:
        return 0.0
    best = 0.0
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        c1 = _pearson_abs(x[:-lag], y[lag:])  # x leads
        c2 = _pearson_abs(x[lag:], y[:-lag])  # y leads
        best = max(best, c1, c2)
    return best


def _coactivity_pmi_sigmoid(x: np.ndarray, y: np.ndarray) -> float:
    """
    PMI-like co-activity: months where x>0 and y>0 relative to marginals.
    Uses logistic σ(PMI) to map to (0,1). Negative PMI maps to <0.5 naturally.
    """
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() == 0:
        return 0.0
    xv = (x[valid] > 0).astype(int)
    yv = (y[valid] > 0).astype(int)
    n = len(xv)
    if n == 0:
        return 0.0
    px = xv.mean()
    py = yv.mean()
    pxy = (xv & yv).mean()
    eps = 1e-12
    # PMI in nats; base doesn't matter since passed through logistic
    pmi = math.log((pxy + eps) / ((px + eps) * (py + eps)))
    # logistic to (0,1)
    return float(1.0 / (1.0 + math.exp(-pmi)))


# --------------------------- graph build -------------------------------

def build_edges(
    pivot_df: pd.DataFrame,
    rmd_cols: List[str],
    pt_cols: List[str],
    min_overlap: int,
    min_coactive: int,
    max_lag: int
) -> pd.DataFrame:
    """Compute RMD–PT edge metrics and composite score; returns sorted edges DataFrame."""
    rows = []

    # Filter out columns that are entirely zero or all-NaN
    def _is_all_zero_or_nan(col: pd.Series) -> bool:
        v = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)
        if np.isfinite(v).sum() == 0:
            return True
        return np.nansum(np.abs(v)) == 0.0

    act_rmd = [c for c in rmd_cols if not _is_all_zero_or_nan(pivot_df[c])]
    act_pt = [c for c in pt_cols if not _is_all_zero_or_nan(pivot_df[c])]
    dropped_r = sorted(set(rmd_cols) - set(act_rmd))
    dropped_p = sorted(set(pt_cols) - set(act_pt))
    if dropped_r or dropped_p:
        logging.warning("Dropped %d RMD and %d PT all-zero/NaN columns from scoring.",
                        len(dropped_r), len(dropped_p))

    for r in act_rmd:
        xr = pivot_df[r].to_numpy(dtype=float)
        for p in act_pt:
            yp = pivot_df[p].to_numpy(dtype=float)

            valid = np.isfinite(xr) & np.isfinite(yp)
            if valid.sum() < max(min_overlap, 3):
                continue

            coactive_months = int(((xr > 0) & (yp > 0) & np.isfinite(xr) & np.isfinite(yp)).sum())
            if coactive_months < min_coactive:
                continue

            pear = _pearson_abs(xr, yp)
            xcorr = _max_xcorr_abs(xr, yp, max_lag=max_lag)
            coact = _coactivity_pmi_sigmoid(xr, yp)
            score = 0.3 * pear + 0.4 * xcorr + 0.3 * coact

            rows.append({
                "RMD": r,
                "PT": p,
                "pearson_pos": round(pear, 6),
                "max_xcorr": round(xcorr, 6),
                "coactivity_pmi": round(coact, 6),
                "coactive_months": coactive_months,
                "score": round(score, 6),
            })

    if not rows:
        logging.warning("No edges were generated; check your data and thresholds.")
        return pd.DataFrame(columns=["RMD", "PT", "pearson_pos", "max_xcorr", "coactivity_pmi", "coactive_months", "score"])

    edges = pd.DataFrame(rows).sort_values(
        by=["score", "coactive_months", "max_xcorr", "pearson_pos"],
        ascending=False,
        kind="mergesort"
    )
    return edges


def edges_to_adjacency(
    edges: pd.DataFrame,
    rmd_cols: List[str],
    pt_cols: List[str],
    norm: str = "none"
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Build symmetric N×N adjacency from edges. Node order: [RMD..., PT...].
    norm: none | row  (row-normalization)
    """
    nodes = list(rmd_cols) + list(pt_cols)
    types = (["RMD"] * len(rmd_cols)) + (["PT"] * len(pt_cols))
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    A = np.zeros((N, N), dtype=np.float64)

    for _, row in edges.iterrows():
        u, v, w = row["RMD"], row["PT"], float(row["score"])
        w = float(np.clip(w, 0.0, 1.0))
        i, j = idx[u], idx[v]
        A[i, j] = max(A[i, j], w)  # keep strongest if duplicates
        A[j, i] = A[i, j]

    # Zero diagonal and sanitize
    np.fill_diagonal(A, 0.0)
    A = np.nan_to_num(A, nan=0.0, posinf=1.0, neginf=0.0)
    A = np.clip(A, 0.0, 1.0)

    if norm == "row":
        rs = A.sum(axis=1, keepdims=True)
        rs[rs == 0.0] = 1.0
        A = A / rs
    elif norm != "none":
        raise ValueError("--norm must be one of: none,row")

    qa_validate_graph(A)
    return A, nodes, types


# --------------------------- QA ---------------------------------------

def qa_validate_graph(A: np.ndarray) -> None:
    """Hard QA checks for N×N adjacency."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"[QA] graph.csv must be square, got {A.shape}")
    if np.isnan(A).any() or np.isinf(A).any():
        raise ValueError("[QA] graph.csv contains NaN/Inf.")
    if (A < 0.0).any() or (A > 1.0).any():
        raise ValueError("[QA] graph.csv has values outside [0,1].")
    if not np.allclose(A, A.T, atol=1e-10):
        raise ValueError("[QA] graph.csv must be symmetric.")
    if not np.allclose(np.diag(A), 0.0, atol=1e-12):
        raise ValueError("[QA] graph.csv diagonal must be 0.")
    iso = np.where(A.sum(axis=1) == 0.0)[0]
    if len(iso) > 0:
        logging.warning("%d isolated node(s) in graph (row-sum=0).", len(iso))


def save_graph(A: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(path), A, delimiter=",", fmt="%.6f")


def save_nodes(nodes: List[str], types: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Index": range(len(nodes)), "Name": nodes, "Type": types}).to_csv(path, index=False)


# --------------------------- main -------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build bipartite RMD–PT graph.csv from dataset.")
    p.add_argument("--data", type=str, required=True, help="Path to data CSV (pivot or long-form).")
    p.add_argument("--terms", type=str, required=True, help="Path to search_terms.py (with DISEASE_SYNONYMS, PT_SYNONYMS).")
    p.add_argument("--graph-out", type=str, default="graph.csv", help="Output path for graph.csv (numeric only, no header).")
    p.add_argument("--nodes-out", type=str, default="nodes.csv", help="Output path for nodes.csv (Index,Name,Type).")
    p.add_argument("--edges-out", type=str, default="", help="Optional: write edges_timeseries.csv for QA.")
    p.add_argument("--min-overlap", type=int, default=6, help="Min overlapping valid months per pair.")
    p.add_argument("--min-coactive", type=int, default=2, help="Min months with both series > 0.")
    p.add_argument("--max-lag", type=int, default=24, help="Max lead/lag months for cross-correlation (excludes zero-lag).")
    p.add_argument("--norm", choices=["none", "row"], default="none", help="Adjacency normalization (default: none).")
    p.add_argument("-v", "--verbose", action="count", default=1, help="Increase verbosity: -v (info), -vv (debug).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    data_path = Path(args.data)
    terms_path = Path(args.terms)
    graph_out = Path(args.graph_out)
    nodes_out = Path(args.nodes_out)
    edges_out = Path(args.edges_out) if args.edges_out else None

    # 1) Load data
    pivot_df, date_col = load_dataset(data_path)

    # 2) Load terms and build classification maps
    disease_syn, pt_syn = load_terms_py(terms_path)
    dis_map, pt_map = build_match_maps(disease_syn, pt_syn)

    # 3) Classify columns
    rmd_cols, pt_cols = classify_columns(list(pivot_df.columns), dis_map, pt_map, date_col_name=None)
    if not rmd_cols or not pt_cols:
        raise RuntimeError(f"Classification failed: RMD={len(rmd_cols)} PT={len(pt_cols)}. "
                           f"Check your columns and {terms_path}.")

    # Preserve original column order: RMDs then PTs (stable)
    logging.info("Detected %d RMD columns and %d PT columns.", len(rmd_cols), len(pt_cols))

    # 4) Build edges
    edges = build_edges(
        pivot_df,
        rmd_cols=rmd_cols,
        pt_cols=pt_cols,
        min_overlap=args.min_overlap,
        min_coactive=args.min_coactive,
        max_lag=args.max_lag
    )

    # Optional: write edges for QA
    if edges_out:
        edges_out.parent.mkdir(parents=True, exist_ok=True)
        edges.to_csv(edges_out, index=False)
        logging.info("Wrote %d edges -> %s", len(edges), edges_out)

    # 5) Build adjacency
    A, nodes, types = edges_to_adjacency(edges, rmd_cols, pt_cols, norm=args.norm)

    # 6) Save outputs
    save_graph(A, graph_out)
    save_nodes(nodes, types, nodes_out)

    logging.info("Wrote graph.csv %s and nodes.csv %s", graph_out, nodes_out)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.getLogger().exception("FAILED: %s", e)
        sys.exit(1)
