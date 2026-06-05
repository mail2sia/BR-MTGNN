#!/usr/bin/env python3
"""
Generate publication-style BRGI+CISN priority plots:
  Panel a: BRGI-vs-Spillover scatter with marginal KDEs
  Panel b: Cleveland dot plot of top-ranked candidates
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import pandas as pd
import seaborn as sns

DEFAULT_PRIORITY_CSV = "model/Bayesian/forecast/brgi_cisn_decision_map/brgi_cisn_priority_matrix.csv"
DEFAULT_OUT_DIR = "model/Bayesian/forecast/brgi_cisn_decision_map"

PALETTE: Dict[str, str] = {
    "Strategic Priority": "#D55E00",
    "Targeted Priority": "#0072B2",
    "Platform Opportunity": "#2E7D32",
    "Watch": "#999999",
}

CANON_COLS = {
    "rank": ["rank", "Rank"],
    "rmd_name": ["rmd_name", "RMD", "RMD Name"],
    "pt_name": ["pt_name", "PT", "PT Name"],
    "brgi_norm": ["brgi_norm", "BRGI Norm"],
    "spillover_norm": ["cisn_spillover_norm", "spillover_value_norm", "Spillover Norm"],
    "final_priority": ["final_priority_score", "Final Priority"],
    "label": ["priority_label", "Label"],
}


def set_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["axes.titlesize"] = 8
    plt.rcParams["xtick.labelsize"] = 7
    plt.rcParams["ytick.labelsize"] = 7
    plt.rcParams["legend.fontsize"] = 6
    plt.rcParams["legend.title_fontsize"] = 7
    plt.rcParams["axes.linewidth"] = 0.5
    plt.rcParams["xtick.major.width"] = 0.5
    plt.rcParams["ytick.major.width"] = 0.5
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build BRGI+CISN publication-style panel figures.")
    p.add_argument("--priority_csv", default=DEFAULT_PRIORITY_CSV,
                   help="Priority matrix input (CSV or pipe-delimited markdown/text table).")
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR,
                   help="Output directory for rendered figures.")
    p.add_argument("--top_k", type=int, default=20,
                   help="Number of top candidates in Panel b.")
    p.add_argument("--x_threshold", type=float, default=0.5,
                   help="Vertical threshold line in Panel a (BRGI norm).")
    p.add_argument("--y_threshold", type=float, default=0.2,
                   help="Horizontal threshold line in Panel a (spillover norm).")
    p.add_argument("--plot_3d_panel_a", action="store_true",
                   help="Also render a 3D version of Panel a with z = final priority score.")
    p.add_argument("--elev", type=float, default=26.0,
                   help="3D view elevation angle in degrees (Panel a 3D).")
    p.add_argument("--azim", type=float, default=132.0,
                   help="3D view azimuth angle in degrees (Panel a 3D).")
    return p.parse_args()


def _read_pipe_table(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8").splitlines()
    table_lines = [ln.strip() for ln in lines if "|" in ln and ln.strip()]
    table_lines = [ln for ln in table_lines if not re.match(r"^\|?[-\s|:]+\|?$", ln)]
    if not table_lines:
        raise ValueError(f"No parseable pipe table lines found in: {path}")
    rows = [[c.strip() for c in ln.strip("|").split("|")] for ln in table_lines]
    header = rows[0]
    body = [r for r in rows[1:] if len(r) == len(header)]
    return pd.DataFrame(body, columns=header)


def _pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_priority_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Priority matrix not found: {path}")
    raw = _read_pipe_table(path) if path.suffix.lower() in {".txt", ".md"} else pd.read_csv(path)

    rename_map: Dict[str, str] = {}
    for canon, variants in CANON_COLS.items():
        src = _pick_col(raw, variants)
        if src:
            rename_map[src] = canon
    df = raw.rename(columns=rename_map).copy()

    required = ["brgi_norm", "spillover_norm", "final_priority", "label", "pt_name", "rmd_name"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns {missing}. Found columns: {list(raw.columns)}")

    if "rank" not in df.columns:
        df["rank"] = range(1, len(df) + 1)

    for c in ["rank", "brgi_norm", "spillover_norm", "final_priority"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["brgi_norm", "spillover_norm", "final_priority", "label"]).copy()

    for c in ["pt_name", "rmd_name", "label"]:
        df[c] = df[c].astype(str).str.strip()
    return df


def clean_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"^(RMD_|PT_)", "", name)
    return name.replace("_", " ")


def _kde_safe(data: pd.DataFrame, ax: plt.Axes, axis: str) -> None:
    for label, sub in data.groupby("label"):
        if label not in PALETTE:
            continue
        s = sub[axis].dropna()
        if len(s) < 3 or s.nunique() < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if axis == "brgi_norm":
                sns.kdeplot(x=s, fill=True, alpha=0.18, linewidth=0.8,
                            color=PALETTE[label], ax=ax, common_norm=False)
            else:
                sns.kdeplot(y=s, fill=True, alpha=0.18, linewidth=0.8,
                            color=PALETTE[label], ax=ax, common_norm=False)


def plot_panel_a(df: pd.DataFrame, out_dir: Path, x_threshold: float, y_threshold: float) -> None:
    fig = plt.figure(figsize=(7.08, 5.5))
    fig.patch.set_facecolor("#FFFFFF")
    gs = gridspec.GridSpec(4, 4, hspace=0.15, wspace=0.45)

    ax_main = fig.add_subplot(gs[1:4, 0:3])
    ax_histx = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    ax_histy = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
    ax_legend = fig.add_subplot(gs[0, 3])

    for ax in (ax_main, ax_histx, ax_histy):
        ax.set_facecolor("#FFFFFF")
    ax_legend.set_facecolor("#FFFFFF")
    ax_legend.axis("off")

    sns.scatterplot(data=df, x="brgi_norm", y="spillover_norm", hue="label",
                    palette=PALETTE, s=30, alpha=0.7, edgecolor="none", ax=ax_main,
                    legend=False)

    ax_main.axvline(x=x_threshold, color="#666666", linestyle=":", linewidth=0.8)
    ax_main.axhline(y=y_threshold, color="#666666", linestyle=":", linewidth=0.8)
    ax_main.set_xlabel("Need Urgency (BRGI Norm)")
    ax_main.set_ylabel("Cross-Disorder Value (Spillover Norm)")
    ax_main.set_xlim(-0.05, 1.05)
    ax_main.set_ylim(-0.03, 0.60)
    sns.despine(ax=ax_main, top=True, right=True)

    # Priority regions
    x_min, x_max = ax_main.get_xlim()
    y_min, y_max = ax_main.get_ylim()
    x_th, y_th = x_threshold, y_threshold

    ax_main.add_patch(mpatches.Rectangle((x_min, y_min), x_th - x_min, y_th - y_min,
                                         facecolor=PALETTE["Watch"], alpha=0.07, zorder=0))
    ax_main.add_patch(mpatches.Rectangle((x_th, y_min), x_max - x_th, y_th - y_min,
                                         facecolor=PALETTE["Targeted Priority"], alpha=0.07, zorder=0))
    ax_main.add_patch(mpatches.Rectangle((x_min, y_th), x_th - x_min, y_max - y_th,
                                         facecolor=PALETTE["Platform Opportunity"], alpha=0.07, zorder=0))
    ax_main.add_patch(mpatches.Rectangle((x_th, y_th), x_max - x_th, y_max - y_th,
                                         facecolor=PALETTE["Strategic Priority"], alpha=0.07, zorder=0))

    ax_main.text((x_min + x_th) / 2, (y_min + y_th) / 2, "Watch",
                 ha="center", va="center", fontsize=7, color="#666666")
    ax_main.text((x_th + x_max) / 2, (y_min + y_th) / 2, "Targeted Priority",
                 ha="center", va="center", fontsize=7, color="#0B4F7A")
    ax_main.text((x_min + x_th) / 2, (y_th + y_max) / 2, "Platform Opportunity",
                 ha="center", va="center", fontsize=7, color="#7A3F64")
    ax_main.text((x_th + x_max) / 2, (y_th + y_max) / 2, "Strategic Priority",
                 ha="center", va="center", fontsize=7, color="#8A3B00")

    legend_order = ["Strategic Priority", "Targeted Priority", "Platform Opportunity", "Watch"]
    handles = [mpatches.Patch(color=PALETTE[k], label=k) for k in legend_order if k in set(df["label"].unique())]
    if handles:
        legend = ax_legend.legend(handles=handles, title="Priority Tier", loc="upper left",
                                  frameon=False, borderpad=0.2, handlelength=1.2)
        if legend and legend.get_title():
            legend.get_title().set_weight("bold")

    _kde_safe(df, ax_histx, axis="brgi_norm")
    ax_histx.axis("off")
    _kde_safe(df, ax_histy, axis="spillover_norm")
    ax_histy.axis("off")

    out_pdf = out_dir / "Fig_Panel_A_BRGI_CISN_Scatter_Marginals.pdf"
    out_png = out_dir / "Fig_Panel_A_BRGI_CISN_Scatter_Marginals.png"
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_panel_b(df: pd.DataFrame, out_dir: Path, top_k: int) -> None:
    top = df.sort_values("final_priority", ascending=False).head(top_k).copy()
    top["pt_clean"] = top["pt_name"].apply(clean_name)
    top["rmd_clean"] = top["rmd_name"].apply(clean_name)
    top["label_clean"] = top["pt_clean"] + "\n(" + top["rmd_clean"] + ")"
    top = top.sort_values("final_priority", ascending=True)

    fig, ax = plt.subplots(figsize=(8.6, max(3.8, 0.34 * top_k)))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    x_floor = min(0.85, float(top["final_priority"].min()) - 0.01)
    x_cap = max(1.02, float(top["final_priority"].max()) + 0.005)
    ax.hlines(y=top["label_clean"], xmin=x_floor, xmax=top["final_priority"],
              color="#B8B8B8", linewidth=1.0, alpha=0.9)

    sns.scatterplot(data=top, x="final_priority", y="label_clean", hue="label",
                    palette=PALETTE, s=55, edgecolor="none", alpha=0.95, ax=ax)

    ax.set_xlabel("Final Combined Priority Score", labelpad=4)
    ax.set_ylabel("")
    ax.set_xlim(x_floor, x_cap)
    ax.xaxis.grid(True, linestyle=":", alpha=0.45, color="#BDBDBD")
    ax.yaxis.grid(False)
    sns.despine(ax=ax, top=True, right=True)

    legend = ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, title="Priority Tier")
    if legend and legend.get_title():
        legend.get_title().set_weight("bold")

    plt.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))

    out_pdf = out_dir / "Fig_Panel_B_TopPriority_ClevelandDot.pdf"
    out_png = out_dir / "Fig_Panel_B_TopPriority_ClevelandDot.png"
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_panel_a_3d(
    df: pd.DataFrame,
    out_dir: Path,
    x_threshold: float,
    y_threshold: float,
    elev: float,
    azim: float,
) -> None:
    fig = plt.figure(figsize=(7.2, 5.5))
    fig.patch.set_facecolor("#FFFFFF")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#FFFFFF")

    for label, sub in df.groupby("label"):
        if label not in PALETTE:
            continue
        ax.scatter(
            sub["brgi_norm"],
            sub["spillover_norm"],
            sub["final_priority"],
            s=26,
            alpha=0.78,
            c=PALETTE[label],
            edgecolors="none",
            label=label,
            depthshade=True,
        )

    z_min = float(df["final_priority"].min())
    z_max = float(df["final_priority"].max())
    z_pad = max(0.01, (z_max - z_min) * 0.04)
    z_low = z_min - z_pad
    z_high = z_max + z_pad

    # Project threshold guides on the floor plane for consistent interpretation with 2D panel.
    ax.plot([x_threshold, x_threshold], [-0.03, 0.60], [z_low, z_low],
            color="#666666", linestyle=":", linewidth=0.8, alpha=0.9)
    ax.plot([-0.05, 1.05], [y_threshold, y_threshold], [z_low, z_low],
            color="#666666", linestyle=":", linewidth=0.8, alpha=0.9)

    ax.set_xlabel("Need Urgency (BRGI Norm)", labelpad=6)
    ax.set_ylabel("Cross-Disorder Value (Spillover Norm)", labelpad=6)
    ax.set_zlabel("Final Priority Score", labelpad=8)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.03, 0.60)
    ax.set_zlim(z_low, z_high)
    ax.view_init(elev=elev, azim=azim)

    # Light pane styling to avoid clutter in print.
    ax.xaxis.pane.set_alpha(0.04)
    ax.yaxis.pane.set_alpha(0.04)
    ax.zaxis.pane.set_alpha(0.02)
    ax.grid(True, alpha=0.22, linestyle=":")

    legend = ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.98), title="Priority Tier")
    if legend and legend.get_title():
        legend.get_title().set_weight("bold")

    fig.tight_layout()
    out_pdf = out_dir / "Fig_Panel_A_BRGI_CISN_Scatter_3D.pdf"
    out_png = out_dir / "Fig_Panel_A_BRGI_CISN_Scatter_3D.png"
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    args = parse_args()
    set_style()

    priority_path = Path(args.priority_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_priority_matrix(priority_path)
    plot_panel_a(df, out_dir, x_threshold=args.x_threshold, y_threshold=args.y_threshold)
    if args.plot_3d_panel_a:
        plot_panel_a_3d(
            df,
            out_dir,
            x_threshold=args.x_threshold,
            y_threshold=args.y_threshold,
            elev=args.elev,
            azim=args.azim,
        )
    plot_panel_b(df, out_dir, top_k=args.top_k)

    print("Saved:")
    print(f"  - {out_dir / 'Fig_Panel_A_BRGI_CISN_Scatter_Marginals.pdf'}")
    print(f"  - {out_dir / 'Fig_Panel_A_BRGI_CISN_Scatter_Marginals.png'}")
    if args.plot_3d_panel_a:
        print(f"  - {out_dir / 'Fig_Panel_A_BRGI_CISN_Scatter_3D.pdf'}")
        print(f"  - {out_dir / 'Fig_Panel_A_BRGI_CISN_Scatter_3D.png'}")
    print(f"  - {out_dir / 'Fig_Panel_B_TopPriority_ClevelandDot.pdf'}")
    print(f"  - {out_dir / 'Fig_Panel_B_TopPriority_ClevelandDot.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
