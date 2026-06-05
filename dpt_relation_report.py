#!/usr/bin/env python3
"""
Generate DPT (Disorders ↔ Pertinent Technologies) relationship figures and tables.

This script reads a graph CSV of RMD-PT links and produces:
 - Table_DPT_Top_Evidence_Links.csv / .md
 - Figure_DPT_Evidence_Heatmap.png/.pdf
 - Figure_DPT_Reduced_Bipartite_Network.png/.pdf
 - Figure_DPT_Ego_Networks.png/.pdf
 - dpt_relation_manifest.json
 - dpt_relation_summary.md

The script uses real repository files only and will fail clearly if required inputs
are missing or cannot be parsed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RMD_ALIASES = ["rmd_name", "RMD", "disorder", "source", "source_name", "burden_id"]
PT_ALIASES = ["pt_name", "PT", "technology", "target", "target_name", "readiness_id"]
WEIGHT_ALIASES = ["weight", "evidence_weight", "link_weight", "score", "final_score", "edge_weight"]


def find_column(columns, candidates):
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def clean_display(name: str) -> str:
    if pd.isna(name):
        return ""
    s = str(name)
    for token in ["RMD_", "PT_"]:
        s = s.replace(token, "")
    for token in ["_NoM", "_NoP"]:
        s = s.replace(token, "")
    s = s.replace("_", " ")
    s = " ".join(s.split())
    s = s.strip()
    # Keep original casing mostly, but collapse excessive uppercase
    if s.isupper():
        s = s.title()
    return s


def short_abbr(name: str, maxlen: int = 6) -> str:
    words = clean_display(name).split()
    if not words:
        return ""
    if len(words) == 1:
        return words[0][:maxlen].upper()
    abbr = "".join(w[0].upper() for w in words)
    if len(abbr) <= maxlen:
        return abbr
    return (words[0][: max(1, maxlen - 1)] + words[-1][0]).upper()


def detect_columns(df: pd.DataFrame):
    cols = list(df.columns)
    rmd_col = find_column(cols, RMD_ALIASES)
    pt_col = find_column(cols, PT_ALIASES)
    weight_col = find_column(cols, WEIGHT_ALIASES)
    return rmd_col, pt_col, weight_col


def load_graph_csv(path: Path) -> tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing graph CSV: {path}")
    # Try standard CSV load first
    df = pd.read_csv(path)
    rmd_col, pt_col, weight_col = detect_columns(df)
    warnings = []
    # If no explicit RMD/PT columns found, attempt adjacency-list parsing (rows: RMD, PT1, PT2, ...)
    if rmd_col is None or pt_col is None:
        # attempt to parse as adjacency list by reading raw lines
        rows = []
        with open(path, "r") as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split(",") if p.strip()]
                if not parts:
                    continue
                rmd_raw = parts[0]
                for pt in parts[1:]:
                    rows.append({"rmd_raw": rmd_raw, "pt_raw": pt, "weight": 1.0})
        if not rows:
            raise ValueError(f"Could not find RMD/PT columns in {path}. Tried aliases: {RMD_ALIASES} / {PT_ALIASES}")
        df2 = pd.DataFrame(rows)
        df2["rmd_display"] = df2["rmd_raw"].apply(clean_display)
        df2["pt_display"] = df2["pt_raw"].apply(clean_display)
        agg = df2.groupby(["rmd_raw", "pt_raw", "rmd_display", "pt_display"], dropna=False, as_index=False)["weight"].sum()
        return agg, warnings
    if weight_col is None:
        warnings.append("No weight column found; using weight=1.0 for all edges.")
        df["weight"] = 1.0
        weight_col = "weight"
    else:
        df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")
        df[weight_col] = df[weight_col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # rename canonical columns
    df = df.rename(columns={rmd_col: "rmd_raw", pt_col: "pt_raw", weight_col: "weight"})

    # Keep raw IDs and create display names
    df["rmd_display"] = df["rmd_raw"].apply(clean_display)
    df["pt_display"] = df["pt_raw"].apply(clean_display)

    # Aggregate duplicate edges by summing weights
    agg = df.groupby(["rmd_raw", "pt_raw", "rmd_display", "pt_display"], dropna=False, as_index=False)["weight"].sum()
    return agg, warnings


def build_heatmap_matrix(edges: pd.DataFrame, top_n_rmd: int, top_n_pt: int):
    rmd_totals = edges.groupby("rmd_display", as_index=False)["weight"].sum().rename(columns={"weight": "rmd_total"})
    pt_totals = edges.groupby("pt_display", as_index=False)["weight"].sum().rename(columns={"weight": "pt_total"})

    top_rmds = rmd_totals.sort_values("rmd_total", ascending=False).head(top_n_rmd)["rmd_display"].tolist()
    top_pts = pt_totals.sort_values("pt_total", ascending=False).head(top_n_pt)["pt_display"].tolist()

    sub = edges[edges["rmd_display"].isin(top_rmds) & edges["pt_display"].isin(top_pts)].copy()
    matrix = sub.pivot_table(index="rmd_display", columns="pt_display", values="weight", aggfunc="sum", fill_value=0.0)

    # Ensure full ordering
    matrix = matrix.reindex(index=top_rmds, columns=top_pts, fill_value=0.0)
    rmd_order = top_rmds
    pt_order = top_pts
    return matrix, rmd_totals, pt_totals, rmd_order, pt_order


def plot_heatmap(matrix: pd.DataFrame, out_png: Path, out_pdf: Path):
    # size adaptive
    nrows, ncols = matrix.shape
    figsize = (max(9, ncols * 0.36 + 3), max(8, nrows * 0.22 + 3))
    fig, ax = plt.subplots(figsize=figsize)
    # make room for rotated x-labels and long y-labels
    fig.subplots_adjust(bottom=0.32, left=0.28, right=0.95, top=0.90)
    im = ax.imshow(matrix.values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(ncols))
    ax.set_yticks(np.arange(nrows))
    ax.set_xticklabels(matrix.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(matrix.index, fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Evidence Link Weight")
    ax.set_title("RMD–PT Evidence Link Matrix\nTop RMDs and PTs by total weighted evidence strength", fontsize=12, fontweight="bold")
    # constrained_layout used to manage margins for labels
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def build_reduced_bipartite(edges: pd.DataFrame, top_network_rmd: int, top_k_links_per_rmd: int):
    rmd_tot = edges.groupby("rmd_raw", as_index=False)["weight"].sum()
    top_rmd_raw = rmd_tot.sort_values("weight", ascending=False).head(top_network_rmd)["rmd_raw"].tolist()

    reduced_edges = []
    for r in top_rmd_raw:
        sub = edges[edges["rmd_raw"] == r].copy()
        sub = sub.sort_values("weight", ascending=False).head(top_k_links_per_rmd)
        reduced_edges.append(sub)
    if reduced_edges:
        reduced = pd.concat(reduced_edges, ignore_index=True)
    else:
        reduced = pd.DataFrame(columns=edges.columns)
    # also compute node totals for the reduced set
    return reduced


def plot_reduced_bipartite(reduced: pd.DataFrame, out_png: Path, out_pdf: Path, pt_scores_path: Path | None = None):
    if reduced.empty:
        # create empty figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No edges to plot", ha="center", va="center")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
        return

    # compute weighted degrees for nodes
    rmd_group = reduced.groupby(["rmd_raw", "rmd_display"], as_index=False)["weight"].sum().rename(columns={"weight": "rmd_weight"})
    pt_group = reduced.groupby(["pt_raw", "pt_display"], as_index=False)["weight"].sum().rename(columns={"weight": "pt_weight"})

    # orders
    rmd_group = rmd_group.sort_values("rmd_weight", ascending=False).reset_index(drop=True)
    pt_group = pt_group.sort_values("pt_weight", ascending=False).reset_index(drop=True)

    rmds = rmd_group["rmd_raw"].tolist()
    pts = pt_group["pt_raw"].tolist()

    rmd_pos = {r: i for i, r in enumerate(rmds)}
    pt_pos = {p: i for i, p in enumerate(pts)}

    # y positions normalized
    rmd_y = {r: 1.0 - i / max(1, len(rmds) - 1) if len(rmds) > 1 else 0.5 for i, r in enumerate(rmds)}
    pt_y = {p: 1.0 - i / max(1, len(pts) - 1) if len(pts) > 1 else 0.5 for i, p in enumerate(pts)}

    fig, ax = plt.subplots(figsize=(12, max(6, len(rmds) * 0.35)))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92)

    # draw edges
    maxw = reduced["weight"].max()
    for _, row in reduced.iterrows():
        r = row["rmd_raw"]
        p = row["pt_raw"]
        w = float(row["weight"])
        y1 = rmd_y.get(r, 0.5)
        y2 = pt_y.get(p, 0.5)
        lw = 0.6 + 4.0 * (w / maxw) if maxw > 0 else 0.6
        ax.plot([0.02, 0.98], [y1, y2], color="#666666", linewidth=lw, alpha=0.38, zorder=1)

    # draw nodes
    # RMD nodes
    rmd_weights = rmd_group.set_index("rmd_raw")["rmd_weight"].to_dict()
    max_r = max(rmd_weights.values()) if rmd_weights else 1.0
    for r in rmds:
        y = rmd_y[r]
        size = 200 + 1800 * (rmd_weights.get(r, 0.0) / max_r)
        ax.scatter(0.02, y, s=size, color="#b2182b", edgecolor="black", zorder=3)
        label = short_abbr(r)
        ax.text(0.06, y, label, va="center", fontsize=9, fontweight="bold")

    pt_weights = pt_group.set_index("pt_raw")["pt_weight"].to_dict()
    max_p = max(pt_weights.values()) if pt_weights else 1.0
    for p in pts:
        y = pt_y[p]
        size = 160 + 1400 * (pt_weights.get(p, 0.0) / max_p)
        ax.scatter(0.98, y, s=size, color="#2166ac", edgecolor="black", zorder=3)
        label = short_abbr(p)
        ax.text(0.92, y, label, va="center", ha="right", fontsize=9, fontweight="bold")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_axis_off()
    ax.set_title("Filtered RMD–PT Bipartite Evidence Network\nOnly strongest links are shown to avoid visual clutter", fontsize=12, fontweight="bold")
    # constrained_layout handles margins
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_ego_networks(edges: pd.DataFrame, ego_rmds: list[str], out_png: Path, out_pdf: Path):
    # Create 2x2 panels
    panels = min(4, len(ego_rmds))
    cols = 2
    rows = math.ceil(panels / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))
    fig.subplots_adjust(top=0.92)
    axes = np.array(axes).reshape(-1)

    for i in range(panels):
        ax = axes[i]
        r = ego_rmds[i]
        sub = edges[edges["rmd_raw"] == r].sort_values("weight", ascending=False).head(6)
        center_x = 0.18
        right_x = 0.82
        # center node
        ax.scatter(center_x, 0.5, s=900, color="#b2182b", edgecolor="black", zorder=3)
        ax.text(center_x - 0.02, 0.5, short_abbr(r), fontsize=10, fontweight="bold", ha="right", va="center")

        # target PTs
        n = len(sub)
        if n == 0:
            ax.text(0.5, 0.5, "No links", ha="center")
            continue
        ys = list(np.linspace(0.9, 0.1, n))
        maxw = sub["weight"].max()
        for (idx, row), y in zip(sub.iterrows(), ys):
            w = float(row["weight"])
            lw = 0.8 + 4.0 * (w / maxw) if maxw > 0 else 0.8
            ax.plot([center_x + 0.02, right_x - 0.02], [0.5, y], color="#666666", linewidth=lw, alpha=0.6)
            ax.scatter(right_x, y, s=300, color="#2166ac", edgecolor="black")
            ax.text(right_x + 0.02, y, short_abbr(row["pt_raw"]) + " — " + clean_display(row["pt_raw"]), va="center", fontsize=9)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        ax.set_title(clean_display(r), fontsize=10, fontweight="bold")

    # hide unused axes
    for j in range(panels, len(axes)):
        axes[j].set_axis_off()

    fig.suptitle("Selected RMD Ego Networks: Strongest Linked PTs", fontsize=13, fontweight="bold")
    # constrained_layout manages axis decorations
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def write_tables(edges: pd.DataFrame, rmd_totals: pd.DataFrame, pt_totals: pd.DataFrame, out_csv: Path, out_md: Path, top_n: int = 50):
    # Merge totals
    rmd_tot_map = rmd_totals.set_index("rmd_display")["rmd_total"].to_dict()
    pt_tot_map = pt_totals.set_index("pt_display")["pt_total"].to_dict()

    edges2 = edges.copy()
    edges2["rmd_total"] = edges2["rmd_display"].map(rmd_tot_map).fillna(0.0)
    edges2["pt_total"] = edges2["pt_display"].map(pt_tot_map).fillna(0.0)
    edges2["link_share_within_rmd"] = edges2.apply(lambda r: (r["weight"] / r["rmd_total"]) if r["rmd_total"] > 0 else 0.0, axis=1)
    edges2["link_share_within_pt"] = edges2.apply(lambda r: (r["weight"] / r["pt_total"]) if r["pt_total"] > 0 else 0.0, axis=1)

    df_out = edges2.sort_values("weight", ascending=False).reset_index(drop=True)
    df_out.insert(0, "rank", np.arange(1, len(df_out) + 1))

    cols = ["rank", "rmd_display", "pt_display", "weight", "rmd_total", "pt_total", "link_share_within_rmd", "link_share_within_pt"]
    df_out[cols].head(top_n).to_csv(out_csv, index=False)

    # write markdown
    with open(out_md, "w") as f:
        f.write("# Table: Top DPT Evidence Links\n\n")
        f.write(df_out[cols].head(top_n).to_markdown(index=False))
        f.write("\n")


def build_manifest(out_dir: Path, inputs: dict, stats: dict, warnings: list[str]):
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "stats": stats,
        "warnings": warnings,
    }
    with open(out_dir / "dpt_relation_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def build_summary(out_dir: Path, top_links: pd.DataFrame):
    md = []
    md.append("# DPT Relation Report Summary\n")
    md.append("## Why the full node-link graph was replaced\n")
    md.append("The original full node-link visualization produced a dense ‘hairball’ with many overlapping edges and labels, reducing interpretability. To emphasize the strongest evidence links we provide a compact heatmap, a reduced bipartite network showing only the top edges per RMD, and focused ego-network panels for example RMDs.\n")

    md.append("## How to read the heatmap\n")
    md.append("Rows are RMDs and columns are PTs. Cell color shows summed evidence weight for the RMD–PT pair. Rows and columns are ordered by total evidence strength.\n")

    md.append("## How to read the filtered bipartite graph\n")
    md.append("Left nodes are selected RMDs, right nodes are PTs connecting to those RMDs. Edge width is proportional to evidence weight. Only the top links per RMD are shown to avoid clutter.\n")

    md.append("## How to read the ego-network panels\n")
    md.append("Each panel centers one RMD and displays its strongest PT links (up to six). Edge thickness reflects evidence weight.\n")

    md.append("## Top 10 strongest RMD–PT evidence links\n")
    md.append(top_links.head(10)[["rmd_display", "pt_display", "weight"]].to_markdown(index=False))
    md.append("\n")
    md.append("## Caution\n")
    md.append("The DPT graph represents weighted evidence links between RMDs and PTs. It is not proof of clinical efficacy.\n")

    with open(out_dir / "dpt_relation_summary.md", "w") as f:
        f.write("\n".join(md))


def parse_args():
    p = argparse.ArgumentParser(description="Build DPT relation report from graph CSV")
    p.add_argument("--graph_csv", type=str, default="data/graph_sparse_2025cut.csv")
    p.add_argument("--brgi_csv", type=str, default="")
    p.add_argument("--priority_csv", type=str, default="")
    p.add_argument("--pt_scores", type=str, default="model/Bayesian/forecast/hype_cycle/rmd_pt_hype_cycle_scores.csv")
    p.add_argument("--out_dir", type=str, default="model/Retrained_TDB_combined_10in_36out_delta010/forecast/dpt_relation_report")
    p.add_argument("--top_n_rmd", type=int, default=25)
    p.add_argument("--top_n_pt", type=int, default=25)
    p.add_argument("--top_network_rmd", type=int, default=12)
    p.add_argument("--top_k_links_per_rmd", type=int, default=4)
    p.add_argument("--ego_top_k", type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    inputs = {"graph_csv": args.graph_csv, "brgi_csv": args.brgi_csv, "priority_csv": args.priority_csv}
    warnings = []

    edges, load_warnings = load_graph_csv(Path(args.graph_csv))
    warnings.extend(load_warnings)

    # compute totals
    rmd_totals = edges.groupby("rmd_display", as_index=False)["weight"].sum().rename(columns={"weight": "rmd_total"})
    pt_totals = edges.groupby("pt_display", as_index=False)["weight"].sum().rename(columns={"weight": "pt_total"})

    # Heatmap
    matrix, rmd_totals_full, pt_totals_full, rmd_order, pt_order = build_heatmap_matrix(edges, args.top_n_rmd, args.top_n_pt)
    heat_png = out / "Figure_DPT_Evidence_Heatmap.png"
    heat_pdf = out / "Figure_DPT_Evidence_Heatmap.pdf"
    plot_heatmap(matrix, heat_png, heat_pdf)

    # Reduced bipartite network
    reduced = build_reduced_bipartite(edges, args.top_network_rmd, args.top_k_links_per_rmd)
    bip_png = out / "Figure_DPT_Reduced_Bipartite_Network.png"
    bip_pdf = out / "Figure_DPT_Reduced_Bipartite_Network.pdf"
    plot_reduced_bipartite(reduced, bip_png, bip_pdf, Path(args.pt_scores) if args.pt_scores else None)

    # Ego networks selection
    ego_rmds = []
    # try brgi_csv or priority_csv to find notable RMDs
    brgi_path = Path(args.brgi_csv) if args.brgi_csv else None
    prio_path = Path(args.priority_csv) if args.priority_csv else None
    if brgi_path and brgi_path.exists():
        try:
            brgi_df = pd.read_csv(brgi_path)
            # attempt to find an RMD identifier column
            rcol = find_column(brgi_df.columns, RMD_ALIASES)
            score_col = find_column(brgi_df.columns, ["brgi_gap", "brgi_score", "gap", "final_priority_score", "score"]) or find_column(brgi_df.columns, [c for c in brgi_df.columns])
            if rcol and score_col:
                tmp = brgi_df[[rcol, score_col]].dropna()
                tmp = tmp.rename(columns={rcol: "rmd", score_col: "score"})
                tmp = tmp.groupby("rmd", as_index=False)["score"].max().sort_values("score", ascending=False)
                ego_rmds = tmp.head(4)["rmd"].tolist()
        except Exception:
            pass

    if not ego_rmds and prio_path and prio_path.exists():
        try:
            p_df = pd.read_csv(prio_path)
            rcol = find_column(p_df.columns, RMD_ALIASES)
            score_col = find_column(p_df.columns, ["final_priority_score", "final_priority", "brgi_score", "score"]) or find_column(p_df.columns, [c for c in p_df.columns])
            if rcol and score_col:
                tmp = p_df[[rcol, score_col]].dropna()
                tmp = tmp.rename(columns={rcol: "rmd", score_col: "score"})
                tmp = tmp.groupby("rmd", as_index=False)["score"].max().sort_values("score", ascending=False)
                ego_rmds = tmp.head(4)["rmd"].tolist()
        except Exception:
            pass

    if not ego_rmds:
        # fallback top by evidence
        ego_rmds = rmd_totals.sort_values("rmd_total", ascending=False).head(4)["rmd_display"].tolist()

    # map ego_rmds from display back to raw where possible
    ego_raw = []
    disp_to_raw = edges.drop_duplicates(["rmd_raw", "rmd_display"]).set_index("rmd_display")["rmd_raw"].to_dict()
    for e in ego_rmds:
        ego_raw.append(disp_to_raw.get(e, e))

    ego_png = out / "Figure_DPT_Ego_Networks.png"
    ego_pdf = out / "Figure_DPT_Ego_Networks.pdf"
    plot_ego_networks(edges, ego_raw, ego_png, ego_pdf)

    # Top links table
    table_csv = out / "Table_DPT_Top_Evidence_Links.csv"
    table_md = out / "Table_DPT_Top_Evidence_Links.md"
    write_tables(edges, rmd_totals, pt_totals, table_csv, table_md, top_n=50)

    # Build manifest and summary
    stats = {
        "raw_edges": int(len(edges)),
        "rmd_count": int(edges["rmd_raw"].nunique()),
        "pt_count": int(edges["pt_raw"].nunique()),
        "edges_in_reduced_network": int(len(reduced)),
        "top_n_rmd": int(args.top_n_rmd),
        "top_n_pt": int(args.top_n_pt),
        "top_network_rmd": int(args.top_network_rmd),
        "top_k_links_per_rmd": int(args.top_k_links_per_rmd),
    }

    inputs_record = {k: str(v) for k, v in inputs.items()}
    # outputs list
    outputs = {
        "Table_DPT_Top_Evidence_Links.csv": str(table_csv),
        "Table_DPT_Top_Evidence_Links.md": str(table_md),
        "Figure_DPT_Evidence_Heatmap.png": str(heat_png),
        "Figure_DPT_Evidence_Heatmap.pdf": str(heat_pdf),
        "Figure_DPT_Reduced_Bipartite_Network.png": str(bip_png),
        "Figure_DPT_Reduced_Bipartite_Network.pdf": str(bip_pdf),
        "Figure_DPT_Ego_Networks.png": str(ego_png),
        "Figure_DPT_Ego_Networks.pdf": str(ego_pdf),
        "dpt_relation_manifest.json": str(out / "dpt_relation_manifest.json"),
        "dpt_relation_summary.md": str(out / "dpt_relation_summary.md"),
    }

    build_manifest(out, inputs_record, stats, warnings)

    # top links for summary
    top_links = edges.sort_values("weight", ascending=False).reset_index(drop=True)
    build_summary(out, top_links)

    print("DPT relation report generated successfully.")


if __name__ == "__main__":
    main()
