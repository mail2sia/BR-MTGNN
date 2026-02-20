#!/usr/bin/env python3
"""Create summary tables and composite figure for RQ5 co-evolution outputs."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd


def _shorten(name: str, max_len: int = 28) -> str:
    if len(name) <= max_len:
        return name
    return name[: max_len - 1] + "…"


def _table_from_df(ax, df: pd.DataFrame, title: str) -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=11, pad=8)
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize co-evolution outputs")
    parser.add_argument("--input-dir", type=str, default="model/Bayesian/forecast/co_evolution")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    topk = int(args.topk)

    gaps_path = in_dir / "early_warning_gaps.csv"
    infl_path = in_dir / "node_influence_strength.csv"
    net_path = in_dir / "network_dynamics.csv"
    heatmap_path = in_dir / "attention_heatmap.png"

    gaps_df = pd.read_csv(gaps_path)
    infl_df = pd.read_csv(infl_path)
    net_df = pd.read_csv(net_path)

    gaps_top = gaps_df.sort_values("gap_score", ascending=False).head(topk).copy()
    gaps_top["rmd"] = gaps_top["rmd"].map(lambda x: _shorten(str(x)))
    gaps_top["pt"] = gaps_top["pt"].map(lambda x: _shorten(str(x)))
    gaps_top = gaps_top[["rmd", "pt", "gap_score", "lag_best", "lag_corr", "granger_p"]]

    infl_top = infl_df.sort_values("total_strength", ascending=False).head(topk).copy()
    infl_top["node"] = infl_top["node"].map(lambda x: _shorten(str(x)))
    infl_top = infl_top[["node", "out_strength", "in_strength", "total_strength"]]

    gaps_top.to_csv(in_dir / "early_warning_gaps_top10.csv", index=False)
    infl_top.to_csv(in_dir / "node_influence_top10.csv", index=False)

    # Standalone network dynamics figure
    fig_net, ax_net = plt.subplots(figsize=(8, 4.5))
    ax_net.plot(net_df["window_end"], net_df["avg_rmd_pt_corr"], color="#2c7fb8", linewidth=2)
    ax_net.set_title("Network Dynamics (Avg RMD-PT Corr)")
    ax_net.set_xlabel("Window end index")
    ax_net.set_ylabel("Avg corr")
    ax_net.grid(alpha=0.3)
    fig_net.tight_layout()
    fig_net.savefig(in_dir / "network_dynamics.png", dpi=300, bbox_inches="tight")
    fig_net.savefig(in_dir / "network_dynamics.pdf", bbox_inches="tight")
    plt.close(fig_net)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], width_ratios=[1.1, 1.0])

    # Attention heatmap panel
    ax0 = fig.add_subplot(gs[0, 0])
    if heatmap_path.exists():
        img = mpimg.imread(heatmap_path)
        ax0.imshow(img)
        ax0.axis("off")
        ax0.set_title("Learned Attention Map", fontsize=12, pad=6)
    else:
        ax0.axis("off")
        ax0.set_title("Attention map not found", fontsize=12, pad=6)

    # Network dynamics panel
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(net_df["window_end"], net_df["avg_rmd_pt_corr"], color="#2c7fb8", linewidth=2)
    ax1.set_title("Network Dynamics (Avg RMD-PT Corr)")
    ax1.set_xlabel("Window end index")
    ax1.set_ylabel("Avg corr")
    ax1.grid(alpha=0.3)

    # Top-10 gaps table
    ax2 = fig.add_subplot(gs[1, 0])
    _table_from_df(ax2, gaps_top, "Top-10 Early-Warning Gaps")

    # Top-10 influence table
    ax3 = fig.add_subplot(gs[1, 1])
    _table_from_df(ax3, infl_top, "Top-10 Node Influence")

    fig.tight_layout()
    out_png = in_dir / "co_evolution_summary.png"
    out_pdf = in_dir / "co_evolution_summary.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("[co-evolution] Summary tables saved:", in_dir / "early_warning_gaps_top10.csv")
    print("[co-evolution] Summary tables saved:", in_dir / "node_influence_top10.csv")
    print("[co-evolution] Composite figure:", out_png)
    print("[co-evolution] Network dynamics figure:", in_dir / "network_dynamics.png")


if __name__ == "__main__":
    main()
