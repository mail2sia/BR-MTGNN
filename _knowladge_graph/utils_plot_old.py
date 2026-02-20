#!/usr/bin/env python3
"""
utils_plot.py — Plot RMD–PT knowledge graph from an edge list.

Input CSV (edges_timeseries.csv) columns expected:
  RMD, PT, score
Optional columns (if present, ignored for plotting but ok to include):
  pearson_pos, max_xcorr, coactivity_pmi, coactive_months

Usage:
  python scripts/utils_plot.py \
    --edges outputs/edges_timeseries.csv \
    --out   outputs/graphs/rmd_pt_graph.png \
    --score-threshold 0.45 \
    --topk-per-rmd 5 \
    --seed 42 \
    --figsize 18 12

Dependencies:
  pandas, numpy, networkx, matplotlib
"""

import argparse
from pathlib import Path
import math
from typing import Sequence, cast, Tuple

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Plot an RMD–PT knowledge graph from edges_timeseries.csv")
    p.add_argument("--edges", required=True, help="Path to edges_timeseries.csv (RMD,PT,score,...)")
    p.add_argument("--out", required=True, help="Output figure path (png/pdf/svg)")
    p.add_argument("--score-threshold", type=float, default=0.0,
                   help="Keep only edges with score >= threshold")
    p.add_argument("--topk-per-rmd", type=int, default=0,
                   help="If >0, for each RMD keep only top-K PT edges by score (after threshold)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for layout")
    p.add_argument("--figsize", nargs=2, type=float, default=[16, 10],
                   help="Figure size, e.g. --figsize 16 10")
    p.add_argument("--rmd-color", default="#1f77b4", help="Hex/color name for RMD nodes (default blue)")
    p.add_argument("--pt-color", default="#2ca02c", help="Hex/color name for PT nodes (default green)")
    p.add_argument("--edge-alpha", type=float, default=0.6, help="Edge alpha transparency")
    p.add_argument("--edge-width-min", type=float, default=0.5, help="Minimum edge width")
    p.add_argument("--edge-width-max", type=float, default=6.0, help="Maximum edge width")
    p.add_argument("--node-size-min", type=int, default=250, help="Minimum node size")
    p.add_argument("--node-size-max", type=int, default=1600, help="Maximum node size")
    p.add_argument("--label-max", type=int, default=80,
                   help="Max number of labels to draw (to avoid clutter). Set 0 to draw none.")
    p.add_argument("--title", default="RMD–PT Knowledge Graph", help="Plot title")
    return p.parse_args()


def load_edges(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"RMD", "PT", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    # Clean types
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    # Drop obvious bad rows
    df = df[(df["RMD"].astype(str) != "") & (df["PT"].astype(str) != "")]
    return df


def filter_edges(df: pd.DataFrame, score_threshold: float, topk_per_rmd: int) -> pd.DataFrame:
    out = df[df["score"] >= score_threshold].copy()
    if topk_per_rmd and topk_per_rmd > 0:
        out = (out.sort_values(["RMD", "score"], ascending=[True, False])
                  .groupby("RMD", as_index=False)
                  .head(topk_per_rmd))
    return out.sort_values("score", ascending=False).reset_index(drop=True)


def build_graph(df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    # Add nodes with bipartite partition attribute
    rmds = df["RMD"].unique().tolist()
    pts  = df["PT"].unique().tolist()

    G.add_nodes_from(rmds, bipartite="RMD", kind="RMD")
    G.add_nodes_from(pts,  bipartite="PT",  kind="PT")

    # Add edges with score
    for _, row in df.iterrows():
        G.add_edge(row["RMD"], row["PT"], score=float(row["score"]))
    return G


def scale_series(values, vmin, vmax, out_min, out_max):
    """Min-max scale a list/array of values to [out_min, out_max]. Handles constant lists."""
    vals = np.asarray(values, dtype=float)
    if len(vals) == 0:
        return np.array([])
    if math.isclose(vmax, vmin):
        return np.full_like(vals, (out_min + out_max) / 2.0, dtype=float)
    scaled = (vals - vmin) / (vmax - vmin)
    return out_min + scaled * (out_max - out_min)


def compute_layout(G: nx.Graph, seed: int):
    # Use a spring layout with deterministic seed for reproducibility
    return nx.spring_layout(G, seed=seed)


def draw_graph(G: nx.Graph, pos, args):
    # Partition nodes
    rmd_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "RMD"]
    pt_nodes  = [n for n, d in G.nodes(data=True) if d.get("kind") == "PT"]

    # Node sizes: degree-based scaling
    degrees = dict(list(G.degree()))  # type: ignore
    if not G.nodes:
        rmd_sizes, pt_sizes = [], []
    else:
        deg_vals = np.array([degrees.get(n, 0) for n in G.nodes()], dtype=float)
        vmin = float(deg_vals.min()) if deg_vals.size else 0.0
        vmax = float(deg_vals.max()) if deg_vals.size else 1.0
        nsizes_all = scale_series(deg_vals, vmin, vmax, args.node_size_min, args.node_size_max)
        nsizes = {n: nsizes_all[i] for i, n in enumerate(G.nodes())}
        rmd_sizes = [int(nsizes[n]) for n in rmd_nodes]
        pt_sizes  = [int(nsizes[n]) for n in pt_nodes]


    # Edge widths: score-based scaling
    scores = [float(d.get("score", 0.0)) for _, _, d in G.edges(data=True)]
    wmin = min(scores) if scores else 0.0
    wmax = max(scores) if scores else 1.0
    wscaled = scale_series(scores, wmin, wmax, args.edge_width_min, args.edge_width_max)
    wscaled_list = wscaled.tolist() if isinstance(wscaled, np.ndarray) else [float(wscaled)]

    # Robust figsize (avoid shadowed built-ins)
    try:
        figsize: Tuple[float, float] = (float(args.figsize[0]), float(args.figsize[1]))
    except Exception:
        figsize = (16.0, 10.0)

    fig = plt.figure(figsize=figsize, dpi=150)
    ax = plt.gca()
    ax.set_axis_off()
    plt.title(args.title, pad=10)

    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        width=wscaled_list,  # type: ignore[arg-type]
        alpha=args.edge_alpha
    )

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=rmd_nodes,
                           node_color=args.rmd_color,
                           node_size=rmd_sizes,  # type: ignore[arg-type]  # type: ignore
                           linewidths=0.5, edgecolors="white")
    nx.draw_networkx_nodes(G, pos, nodelist=pt_nodes,
                           node_color=args.pt_color,
                           node_size=pt_sizes,  # type: ignore[arg-type]  # type: ignore
                           linewidths=0.5, edgecolors="white")

    # Labels: limit to avoid clutter
    if args.label_max != 0 and G.nodes:
        # Select top-N highest-degree nodes for labeling
        nodes_sorted = sorted(G.nodes(), key=lambda n: degrees.get(n, 0), reverse=True)
        to_label = nodes_sorted[:args.label_max]
        labels = {n: n for n in to_label}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    # Legend (simple color boxes)
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=args.rmd_color, edgecolor='white', label='RMD'),
        Patch(facecolor=args.pt_color,  edgecolor='white', label='PT'),
    ]
    ax.legend(handles=legend_elems, loc="lower left", frameon=False)

    return fig


def main():
    args = parse_args()
    edges_path = Path(args.edges)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_edges(str(edges_path))
    df_f = filter_edges(df, score_threshold=args.score_threshold, topk_per_rmd=args.topk_per_rmd)

    if df_f.empty:
        print("No edges to plot after filtering. Adjust --score-threshold or --topk-per-rmd.")
        return

    G = build_graph(df_f)
    pos = compute_layout(G, seed=args.seed)
    fig = draw_graph(G, pos, args)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved graph -> {out_path}")


if __name__ == "__main__":
    main()
