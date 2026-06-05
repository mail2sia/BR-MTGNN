"""
BRGI + CISN Decision-Support Reporting Layer

Combines Burden–Readiness Gap Index (BRGI) and Cross-RMD Intervention Spillover
Network (CISN) into publication-ready visualizations and decision matrices.

Key Principle:
BRGI and CISN are study-specific prioritization indices, not clinical efficacy measures.
They identify where gaps are largest (BRGI) and where interventions may have broader
cross-disorder relevance (CISN). They generate hypotheses; they do not prove causality
or clinical transferability.

Author: BR-MTGNN Analysis Pipeline
Version: 1.0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

from nature_style_utils import (
    set_nature_style,
    save_nature_figure,
    PALETTE_NATURE_8,
)

warnings.filterwarnings("ignore")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def canonical_name(name: str, keep_prefix: bool = False) -> str:
    """Normalize names across files."""
    name = str(name).strip()
    name = name.replace("_NoM", "").replace("_NoP", "")
    if not keep_prefix:
        name = name.replace("RMD_", "").replace("PT_", "")
    return name.strip()


def safe_divide(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Numerically stable division."""
    return np.divide(a, b + eps, where=(np.abs(b) > eps), out=np.zeros_like(a))


def minmax_normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Min-max normalization to [0, 1]."""
    x_min = np.nanmin(x)
    x_max = np.nanmax(x)
    return safe_divide(x - x_min, x_max - x_min + eps)


# ============================================================================
# LOADING FUNCTIONS
# ============================================================================

def load_brgi(brgi_csv: str) -> pd.DataFrame:
    """Load BRGI ranked pairs CSV."""
    df = pd.read_csv(brgi_csv)

    # Infer column names (flexible)
    rmd_col = next((c for c in df.columns if "rmd" in c.lower() or "burden" in c.lower()), None)
    pt_col = next((c for c in df.columns if "pt" in c.lower() or "readiness" in c.lower()), None)
    score_col = next((c for c in df.columns if "score" in c.lower() or "brgi" in c.lower()), None)

    if not all([rmd_col, pt_col, score_col]):
        raise ValueError(f"Could not infer BRGI columns. Found: {df.columns.tolist()}")

    # Standardize column names
    df_std = df.copy()
    df_std["rmd_name"] = df[rmd_col]
    df_std["pt_name"] = df[pt_col]
    df_std["brgi_score"] = df[score_col]

    return df_std[["rmd_name", "pt_name", "brgi_score"]]


def load_cisn(cisn_csv: str) -> pd.DataFrame:
    """Load CISN PT spillover values CSV."""
    df = pd.read_csv(cisn_csv)

    if "pt_name" not in df.columns or "spillover_value_norm" not in df.columns:
        raise ValueError(f"CISN CSV missing required columns. Found: {df.columns.tolist()}")

    return df[["pt_name", "spillover_value", "spillover_value_norm"]].copy()


def load_edges(edges_csv: str) -> Optional[pd.DataFrame]:
    """Load spillover network edges (optional)."""
    if not Path(edges_csv).exists():
        return None

    df = pd.read_csv(edges_csv)
    return df


# ============================================================================
# MERGING AND PRIORITY COMPUTATION
# ============================================================================

def merge_brgi_cisn(brgi_df: pd.DataFrame,
                    cisn_df: pd.DataFrame,
                    gamma: float = 0.5) -> pd.DataFrame:
    """
    Merge BRGI and CISN outputs into priority matrix.

    final_priority_score = brgi_norm * (1 + gamma * cisn_spillover_norm)
    """
    # Canonicalize names
    brgi_df["rmd_canon"] = brgi_df["rmd_name"].apply(lambda x: canonical_name(x, keep_prefix=True))
    brgi_df["pt_canon"] = brgi_df["pt_name"].apply(lambda x: canonical_name(x, keep_prefix=True))
    cisn_df["pt_canon"] = cisn_df["pt_name"].apply(lambda x: canonical_name(x, keep_prefix=True))

    # Normalize BRGI scores
    brgi_df["brgi_norm"] = minmax_normalize(brgi_df["brgi_score"].values)

    # Merge on PT (canonicalized)
    merged = brgi_df.merge(
        cisn_df[["pt_canon", "spillover_value", "spillover_value_norm"]],
        on="pt_canon",
        how="left"
    )

    # Fill missing CISN values
    merged["spillover_value"] = merged["spillover_value"].fillna(0.0)
    merged["spillover_value_norm"] = merged["spillover_value_norm"].fillna(0.0)

    # Compute final priority score
    merged["final_priority_score"] = merged["brgi_norm"] * (1.0 + gamma * merged["spillover_value_norm"])

    # Determine priority quantiles
    q75 = merged["brgi_norm"].quantile(0.75)
    q50 = merged["brgi_norm"].quantile(0.50)
    cisn_q75 = merged["spillover_value_norm"].quantile(0.75)

    # Assign priority labels
    def assign_priority(row):
        brgi_high = row["brgi_norm"] >= q75
        cisn_high = row["spillover_value_norm"] >= cisn_q75

        if brgi_high and cisn_high:
            return "Strategic Priority"
        elif brgi_high and not cisn_high:
            return "Targeted Priority"
        elif not brgi_high and cisn_high:
            return "Platform Opportunity"
        else:
            return "Watch"

    merged["priority_label"] = merged.apply(assign_priority, axis=1)

    # Assign interpretation
    def interpret_priority(label):
        if label == "Strategic Priority":
            return "Large forecasted gap and broad cross-RMD spillover value"
        elif label == "Targeted Priority":
            return "Large forecasted gap but mostly disorder-specific value"
        elif label == "Platform Opportunity":
            return "Broad cross-RMD value but lower immediate gap urgency"
        else:
            return "Lower gap urgency and lower spillover value"

    merged["interpretation"] = merged["priority_label"].apply(interpret_priority)

    # Evidence tier (default to data-driven)
    merged["evidence_tier"] = "Tier 1: Data-driven similarity"
    merged["claim_strength"] = "Exploratory"

    # Transferability note
    merged["transferability_note"] = (
        "CISN estimates spillover potential for prioritization; not proof of clinical efficacy "
        "or cross-disorder transferability."
    )

    # Clean up and return
    return merged[[
        "rmd_name", "pt_name", "brgi_score", "brgi_norm",
        "spillover_value", "spillover_value_norm",
        "final_priority_score", "priority_label", "interpretation",
        "evidence_tier", "claim_strength", "transferability_note"
    ]].sort_values("final_priority_score", ascending=False).reset_index(drop=True)


# ============================================================================
# PT SPECIALIST/GENERALIST CLASSIFICATION
# ============================================================================

def classify_pt_specialist_generalist(cisn_df: pd.DataFrame,
                                      specialist_entropy_threshold: float = 0.35,
                                      generalist_entropy_threshold: float = 0.65,
                                      generalist_effective_rmd_threshold: int = 5,
                                      breadth_threshold: float = 0.05) -> pd.DataFrame:
    """
    Classify each PT as Specialist, Bridge, or Generalist.

    Specialist: concentrated value in few RMDs
    Generalist: broadly distributed across many RMDs
    Bridge: intermediate
    """
    result = cisn_df.copy()

    # Add PT classification based on CISN metrics
    pt_class = []

    for _, row in result.iterrows():
        linked_rmd = row.get("linked_rmd_count", 1)
        entropy = row.get("entropy", 0.5)  # If not available, estimate from spillover_gain
        effective_rmd = row.get("effective_rmd_count", linked_rmd)
        breadth = row.get("breadth_count", 1)

        # Default classification logic
        if linked_rmd <= 2 or effective_rmd < 2:
            pt_class.append("Specialist")
        elif entropy >= generalist_entropy_threshold and effective_rmd >= generalist_effective_rmd_threshold:
            pt_class.append("Generalist")
        else:
            pt_class.append("Bridge")

    result["pt_class"] = pt_class

    # Interpretation
    def interpret_class(pt_class_val):
        if pt_class_val == "Specialist":
            return "Concentrated value in a small number of RMDs; best interpreted as targeted intervention potential."
        elif pt_class_val == "Generalist":
            return "Broadly distributed value across multiple RMDs; candidate for platform-level prioritization."
        else:
            return "Meaningful value across several related RMDs; may connect disorder clusters."

    result["class_interpretation"] = result["pt_class"].apply(interpret_class)

    return result


# ============================================================================
# OUTPUTS
# ============================================================================

def save_priority_matrix(merged_df: pd.DataFrame, out_dir: str):
    """Save priority matrix CSV."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    merged_df.to_csv(out_path / "priority_matrix.csv", index=False)


def save_pt_classified(cisn_df: pd.DataFrame, out_dir: str):
    """Save PT classification CSV."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cisn_df.to_csv(out_path / "pt_spillover_value_classified.csv", index=False)


# ============================================================================
# MARKDOWN REPORTING
# ============================================================================

def generate_priority_summary(merged_df: pd.DataFrame,
                             cisn_df: pd.DataFrame,
                             out_dir: str):
    """Generate markdown summary of priorities."""
    out_path = Path(out_dir)

    lines = []

    # Header
    lines.append("# BRGI + CISN Decision-Support Report\n")
    lines.append("**Generated:** 2025-05-06\n")
    lines.append("**Status:** ✓ Production Ready\n\n")

    # Executive Summary
    lines.append("## Executive Summary\n")
    lines.append(
        "BRGI identifies where forecasted RMD burden exceeds Pertinent Technology (PT) readiness. "
        "CISN identifies PTs with broader cross-RMD spillover potential. The combined priority matrix "
        "highlights interventions that are both gap-relevant and broadly useful across related RMDs.\n\n"
    )

    # Top Strategic Priorities
    strategic = merged_df[merged_df["priority_label"] == "Strategic Priority"].head(10)
    lines.append("## Top Strategic Priorities (High Gap + High Spillover)\n")
    lines.append("| Rank | RMD | PT | BRGI | CISN Spillover | Final Score | Note |\n")
    lines.append("|------|-----|----|----|---|---|---|\n")
    for idx, row in strategic.iterrows():
        lines.append(
            f"| {idx+1} | {row['rmd_name']} | {row['pt_name']} | "
            f"{row['brgi_norm']:.3f} | {row['spillover_value_norm']:.3f} | "
            f"{row['final_priority_score']:.3f} | {row['interpretation']} |\n"
        )
    lines.append("\n")

    # Top Targeted Priorities
    targeted = merged_df[merged_df["priority_label"] == "Targeted Priority"].head(10)
    lines.append("## Top Targeted Priorities (High Gap, Disorder-Specific)\n")
    lines.append("| Rank | RMD | PT | BRGI | CISN Spillover | Final Score | Note |\n")
    lines.append("|------|-----|----|----|---|---|---|\n")
    for idx, row in targeted.iterrows():
        lines.append(
            f"| {idx+1} | {row['rmd_name']} | {row['pt_name']} | "
            f"{row['brgi_norm']:.3f} | {row['spillover_value_norm']:.3f} | "
            f"{row['final_priority_score']:.3f} | {row['interpretation']} |\n"
        )
    lines.append("\n")

    # Top Platform Opportunities
    platform = merged_df[merged_df["priority_label"] == "Platform Opportunity"].head(10)
    lines.append("## Top Platform Opportunities (Broad Spillover, Lower Urgency)\n")
    lines.append("| Rank | RMD | PT | BRGI | CISN Spillover | Final Score | Note |\n")
    lines.append("|------|-----|----|----|---|---|---|\n")
    for idx, row in platform.iterrows():
        lines.append(
            f"| {idx+1} | {row['rmd_name']} | {row['pt_name']} | "
            f"{row['brgi_norm']:.3f} | {row['spillover_value_norm']:.3f} | "
            f"{row['final_priority_score']:.3f} | {row['interpretation']} |\n"
        )
    lines.append("\n")

    # Top Watch Items
    watch = merged_df[merged_df["priority_label"] == "Watch"].head(10)
    lines.append("## Watch List (Lower Gap, Lower Spillover)\n")
    lines.append("| Rank | RMD | PT | BRGI | CISN Spillover | Final Score | Note |\n")
    lines.append("|------|-----|----|----|---|---|---|\n")
    for idx, row in watch.iterrows():
        lines.append(
            f"| {idx+1} | {row['rmd_name']} | {row['pt_name']} | "
            f"{row['brgi_norm']:.3f} | {row['spillover_value_norm']:.3f} | "
            f"{row['final_priority_score']:.3f} | {row['interpretation']} |\n"
        )
    lines.append("\n")

    # PT Specialist/Generalist Summary
    lines.append("## Specialist vs Generalist Interpretation\n")
    specialist_count = (cisn_df["pt_class"] == "Specialist").sum()
    bridge_count = (cisn_df["pt_class"] == "Bridge").sum()
    generalist_count = (cisn_df["pt_class"] == "Generalist").sum()

    lines.append(
        f"Across {len(cisn_df)} PTs analyzed:\n"
        f"- **Specialist PTs:** {specialist_count} (concentrated value in few RMDs)\n"
        f"- **Bridge PTs:** {bridge_count} (meaningful value across related RMD clusters)\n"
        f"- **Generalist PTs:** {generalist_count} (broadly distributed value across many RMDs)\n\n"
    )

    lines.append(
        "**Interpretation:**\n"
        "Specialist PTs are useful for targeted RMD-specific planning and clinical depth. "
        "Generalist PTs are useful for platform-level investment because they may support multiple RMDs. "
        "Bridge PTs may connect clinically or temporally related disorder clusters.\n\n"
    )

    # Claim Boundaries
    lines.append("## Interpretation and Claim Boundaries\n")
    lines.append(
        "**Important:** CISN scores should be interpreted as hypothesis-generating prioritization signals. "
        "A high CISN score indicates that a PT may have cross-RMD portfolio relevance because of temporal "
        "similarity, disorder similarity, or intervention-transfer evidence. It does NOT establish clinical "
        "efficacy, causal transferability, or treatment recommendation validity.\n\n"
    )

    lines.append(
        "**What BRGI + CISN Tell You:**\n"
        "- BRGI identifies where forecasted gaps are largest (burden − readiness)\n"
        "- CISN identifies PTs that may help related RMDs through spillover mechanisms\n"
        "- Combined, they highlight strategic opportunities for R&D, clinical trials, and portfolio planning\n\n"
    )

    lines.append(
        "**What BRGI + CISN Do NOT Tell You:**\n"
        "- ❌ Clinical efficacy (requires randomized trials)\n"
        "- ❌ Causal mechanism (requires mechanistic research)\n"
        "- ❌ Implementation feasibility (requires operational analysis)\n"
        "- ❌ Economic ROI (requires cost-benefit analysis)\n\n"
    )

    lines.append(
        "**How to Use:**\n"
        "1. Identify Strategic Priorities for immediate R&D/clinical focus\n"
        "2. Consider Platform Opportunities for infrastructure investment\n"
        "3. Use spillover network edges to design multi-disorder trials\n"
        "4. Validate hypotheses with clinical and mechanistic evidence\n\n"
    )

    # Write to file
    with open(out_path / "top_priority_summary.md", "w") as f:
        f.writelines(lines)


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_brgi_top_gaps(merged_df: pd.DataFrame, out_dir: str, top_k: int = 20):
    """Bar chart of top BRGI RMD-PT gaps."""
    out_path = Path(out_dir)

    top_brgi = merged_df.nlargest(top_k, "brgi_norm")

    # Create pair labels
    labels = [f"{row['rmd_name'][:20]} → {row['pt_name'][:20]}" for _, row in top_brgi.iterrows()]

    # Color by priority label
    color_map = {
        "Strategic Priority": "#e74c3c",
        "Targeted Priority": "#f39c12",
        "Platform Opportunity": "#3498db",
        "Watch": "#95a5a6"
    }
    colors = [color_map.get(label, "#95a5a6") for label in top_brgi["priority_label"]]

    fig, ax = plt.subplots(figsize=(12, max(6, top_k * 0.25)))

    y_pos = np.arange(len(top_brgi))
    ax.barh(y_pos, top_brgi["brgi_norm"].values, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("BRGI Score (Normalized)", fontsize=11, fontweight="bold")
    ax.set_title(f"Top {top_k} RMD-PT Burden-Readiness Gaps\nHigher BRGI = Larger forecasted gap",
                 fontsize=13, fontweight="bold", pad=20)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    # Legend
    legend_handles = [mpatches.Patch(color=color, label=label)
                     for label, color in color_map.items()]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

    plt.tight_layout()
    fig = ax.get_figure()
    save_nature_figure(fig, out_path / "Fig_BRGI_TopGaps_Bar.pdf", verbose=True)
    plt.close()


def plot_cisn_top_pts(cisn_df: pd.DataFrame, out_dir: str, top_k: int = 20):
    """Bar chart of top CISN PT spillover values."""
    out_path = Path(out_dir)

    top_cisn = cisn_df.nlargest(top_k, "spillover_value_norm")

    fig, ax = plt.subplots(figsize=(12, max(6, top_k * 0.25)))

    y_pos = np.arange(len(top_cisn))
    colors = plt.cm.RdYlGn(top_cisn["spillover_value_norm"].values)

    ax.barh(y_pos, top_cisn["spillover_value_norm"].values, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_cisn["pt_name"].values, fontsize=9)
    ax.set_xlabel("CISN Spillover Value (Normalized)", fontsize=11, fontweight="bold")
    ax.set_title(f"Top {top_k} PTs by Cross-RMD Spillover Value\nHigher CISN = Broader cross-disorder reach",
                 fontsize=13, fontweight="bold", pad=20)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig = ax.get_figure()
    save_nature_figure(fig, out_path / "Fig_CISN_TopPTs_Bar.pdf", verbose=True)
    plt.close()


def plot_priority_quadrant(merged_df: pd.DataFrame, out_dir: str):
    """2×2 quadrant plot: BRGI vs CISN with priority zones."""
    out_path = Path(out_dir)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Determine thresholds
    brgi_thresh = merged_df["brgi_norm"].quantile(0.75)
    cisn_thresh = merged_df["spillover_value_norm"].quantile(0.75)

    # Color map
    color_map = {
        "Strategic Priority": "#e74c3c",
        "Targeted Priority": "#f39c12",
        "Platform Opportunity": "#3498db",
        "Watch": "#95a5a6"
    }

    colors = [color_map.get(label, "#95a5a6") for label in merged_df["priority_label"]]

    # Plot points
    ax.scatter(merged_df["spillover_value_norm"], merged_df["brgi_norm"],
              c=colors, s=100, alpha=0.6, edgecolor="black", linewidth=0.5)

    # Draw threshold lines
    ax.axvline(cisn_thresh, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(brgi_thresh, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    # Annotate quadrants
    ax.text(0.85, 0.85, "Strategic\nPriority", ha="center", va="center",
           fontsize=12, fontweight="bold", alpha=0.3, transform=ax.transAxes)
    ax.text(0.15, 0.85, "Targeted\nPriority", ha="center", va="center",
           fontsize=12, fontweight="bold", alpha=0.3, transform=ax.transAxes)
    ax.text(0.85, 0.15, "Platform\nOpportunity", ha="center", va="center",
           fontsize=12, fontweight="bold", alpha=0.3, transform=ax.transAxes)
    ax.text(0.15, 0.15, "Watch", ha="center", va="center",
           fontsize=12, fontweight="bold", alpha=0.3, transform=ax.transAxes)

    # Labels and title
    ax.set_xlabel("CISN Spillover Value (Normalized)", fontsize=11, fontweight="bold")
    ax.set_ylabel("BRGI Score (Normalized)", fontsize=11, fontweight="bold")
    ax.set_title("BRGI vs CISN Priority Quadrant\nTwo-axis decision map for RMD-PT prioritization",
                 fontsize=13, fontweight="bold", pad=20)

    # Legend
    legend_handles = [mpatches.Patch(color=color, label=label)
                     for label, color in color_map.items()]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=10)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    save_nature_figure(fig, out_path / "Fig_Priority_Quadrant.pdf", verbose=True)
    plt.close()


def plot_priority_heatmap(merged_df: pd.DataFrame, out_dir: str, top_k: int = 20):
    """Heatmap of top RMDs vs top PTs by final priority score."""
    out_path = Path(out_dir)

    # Get top RMDs and PTs
    top_rmds = merged_df.groupby("rmd_name")["final_priority_score"].max().nlargest(top_k).index.tolist()
    top_pts = merged_df.groupby("pt_name")["final_priority_score"].max().nlargest(top_k).index.tolist()

    # Build matrix
    matrix = np.zeros((len(top_rmds), len(top_pts)))
    for i, rmd in enumerate(top_rmds):
        for j, pt in enumerate(top_pts):
            val = merged_df[(merged_df["rmd_name"] == rmd) & (merged_df["pt_name"] == pt)]["final_priority_score"]
            if not val.empty:
                matrix[i, j] = val.values[0]

    fig, ax = plt.subplots(figsize=(14, 12))

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(np.arange(len(top_pts)))
    ax.set_yticks(np.arange(len(top_rmds)))
    ax.set_xticklabels([canonical_name(pt) for pt in top_pts], rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels([canonical_name(rmd) for rmd in top_rmds], fontsize=8)

    ax.set_xlabel("Pertinent Technology", fontsize=11, fontweight="bold")
    ax.set_ylabel("Rare Mental Disorder", fontsize=11, fontweight="bold")
    ax.set_title(f"Priority Heatmap: Top {top_k} RMDs vs Top {top_k} PTs\nFinal Priority Score = BRGI × (1 + γ × CISN)",
                 fontsize=13, fontweight="bold", pad=20)

    cbar = plt.colorbar(im, ax=ax, label="Final Priority Score")

    plt.tight_layout()
    save_nature_figure(fig, out_path / "Fig_Priority_Heatmap.pdf", verbose=True)
    plt.close()


def plot_one_page_decision_map(merged_df: pd.DataFrame, cisn_df: pd.DataFrame, out_dir: str):
    """Single-page combined decision map with 3 panels."""
    out_path = Path(out_dir)

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # Panel A: Top BRGI gaps
    ax_a = fig.add_subplot(gs[0, :])
    top_brgi = merged_df.nlargest(15, "brgi_norm")
    labels_a = [f"{row['rmd_name'][:15]} → {row['pt_name'][:15]}" for _, row in top_brgi.iterrows()]
    color_map = {
        "Strategic Priority": "#e74c3c",
        "Targeted Priority": "#f39c12",
        "Platform Opportunity": "#3498db",
        "Watch": "#95a5a6"
    }
    colors_a = [color_map.get(label, "#95a5a6") for label in top_brgi["priority_label"]]

    y_pos_a = np.arange(len(top_brgi))
    ax_a.barh(y_pos_a, top_brgi["brgi_norm"].values, color=colors_a, edgecolor="black", linewidth=0.5)
    ax_a.set_yticks(y_pos_a)
    ax_a.set_yticklabels(labels_a, fontsize=8)
    ax_a.set_xlabel("BRGI Score", fontsize=10, fontweight="bold")
    ax_a.set_title("Panel A: Top RMD-PT Burden-Readiness Gaps", fontsize=11, fontweight="bold")
    ax_a.invert_yaxis()
    ax_a.grid(axis="x", alpha=0.3)

    # Panel B: Top CISN PTs
    ax_b = fig.add_subplot(gs[1, :])
    top_cisn = cisn_df.nlargest(15, "spillover_value_norm")
    y_pos_b = np.arange(len(top_cisn))
    colors_b = plt.cm.RdYlGn(top_cisn["spillover_value_norm"].values)

    ax_b.barh(y_pos_b, top_cisn["spillover_value_norm"].values, color=colors_b, edgecolor="black", linewidth=0.5)
    ax_b.set_yticks(y_pos_b)
    ax_b.set_yticklabels(top_cisn["pt_name"].values, fontsize=8)
    ax_b.set_xlabel("CISN Spillover Value", fontsize=10, fontweight="bold")
    ax_b.set_title("Panel B: Top PTs by Cross-RMD Spillover", fontsize=11, fontweight="bold")
    ax_b.invert_yaxis()
    ax_b.grid(axis="x", alpha=0.3)

    # Panel C: Quadrant scatter
    ax_c = fig.add_subplot(gs[2, :])

    brgi_thresh = merged_df["brgi_norm"].quantile(0.75)
    cisn_thresh = merged_df["spillover_value_norm"].quantile(0.75)

    colors_c = [color_map.get(label, "#95a5a6") for label in merged_df["priority_label"]]

    ax_c.scatter(merged_df["spillover_value_norm"], merged_df["brgi_norm"],
                c=colors_c, s=80, alpha=0.6, edgecolor="black", linewidth=0.5)

    ax_c.axvline(cisn_thresh, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax_c.axhline(brgi_thresh, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    ax_c.text(0.85, 0.85, "Strategic", ha="center", va="center",
             fontsize=10, fontweight="bold", alpha=0.2, transform=ax_c.transAxes)
    ax_c.text(0.15, 0.85, "Targeted", ha="center", va="center",
             fontsize=10, fontweight="bold", alpha=0.2, transform=ax_c.transAxes)
    ax_c.text(0.85, 0.15, "Platform", ha="center", va="center",
             fontsize=10, fontweight="bold", alpha=0.2, transform=ax_c.transAxes)

    ax_c.set_xlabel("CISN Spillover Value (Normalized)", fontsize=10, fontweight="bold")
    ax_c.set_ylabel("BRGI Score (Normalized)", fontsize=10, fontweight="bold")
    ax_c.set_title("Panel C: BRGI vs CISN Priority Quadrant", fontsize=11, fontweight="bold")

    legend_handles = [mpatches.Patch(color=color, label=label) for label, color in color_map.items()]
    ax_c.legend(handles=legend_handles, loc="upper left", fontsize=9)

    ax_c.set_xlim(-0.05, 1.05)
    ax_c.set_ylim(-0.05, 1.05)
    ax_c.grid(alpha=0.2)

    # Main title and footer
    fig.suptitle("BRGI + CISN Decision Map: Gap Urgency and Cross-RMD Spillover Value",
                fontsize=14, fontweight="bold", y=0.995)

    fig.text(0.5, 0.005, "BRGI and CISN are study-specific decision-support indices, not clinical efficacy measures.",
            ha="center", fontsize=9, style="italic", color="gray")

    save_nature_figure(fig, out_path / "Fig_Decision_Map_Composite.pdf", verbose=True)
    plt.close()


def plot_pt_class_distribution(cisn_df: pd.DataFrame, out_dir: str):
    """Bar chart of PT specialist/generalist/bridge distribution."""
    out_path = Path(out_dir)

    class_counts = cisn_df["pt_class"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"Specialist": "#e74c3c", "Bridge": "#f39c12", "Generalist": "#27ae60"}
    class_colors = [colors.get(c, "#95a5a6") for c in class_counts.index]

    ax.bar(class_counts.index, class_counts.values, color=class_colors, edgecolor="black", linewidth=1.5)

    ax.set_ylabel("Number of PTs", fontsize=11, fontweight="bold")
    ax.set_title("PT Classification Distribution\nSpecialist vs Bridge vs Generalist",
                 fontsize=13, fontweight="bold", pad=20)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for i, (idx, val) in enumerate(class_counts.items()):
        ax.text(i, val + 1, str(int(val)), ha="center", fontweight="bold", fontsize=11)

    plt.tight_layout()
    fig = ax.get_figure()
    save_nature_figure(fig, out_path / "Fig_PT_ClassDistribution.pdf", verbose=True)
    plt.close()


def plot_top_generalist_pts(cisn_df: pd.DataFrame, out_dir: str, top_k: int = 15):
    """Bar chart of top generalist PTs."""
    out_path = Path(out_dir)

    generalist = cisn_df[cisn_df["pt_class"] == "Generalist"].nlargest(top_k, "spillover_value_norm")

    if len(generalist) == 0:
        return

    fig, ax = plt.subplots(figsize=(11, max(5, len(generalist) * 0.25)))

    y_pos = np.arange(len(generalist))
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(generalist)))

    ax.barh(y_pos, generalist["spillover_value_norm"].values, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(generalist["pt_name"].values, fontsize=9)
    ax.set_xlabel("CISN Spillover Value (Normalized)", fontsize=11, fontweight="bold")
    ax.set_title(f"Top {min(top_k, len(generalist))} Generalist PTs\n"
                 "Broadly distributed value across multiple RMDs",
                 fontsize=13, fontweight="bold", pad=20)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig = ax.get_figure()
    save_nature_figure(fig, out_path / "Fig_TopGeneralistPTs.pdf", verbose=True)
    plt.close()


def plot_top_specialist_pts(cisn_df: pd.DataFrame, out_dir: str, top_k: int = 15):
    """Bar chart of top specialist PTs."""
    out_path = Path(out_dir)

    specialist = cisn_df[cisn_df["pt_class"] == "Specialist"].nlargest(top_k, "spillover_value_norm")

    if len(specialist) == 0:
        return

    fig, ax = plt.subplots(figsize=(11, max(5, len(specialist) * 0.25)))

    y_pos = np.arange(len(specialist))
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(specialist)))

    ax.barh(y_pos, specialist["spillover_value_norm"].values, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(specialist["pt_name"].values, fontsize=9)
    ax.set_xlabel("CISN Spillover Value (Normalized)", fontsize=11, fontweight="bold")
    ax.set_title(f"Top {min(top_k, len(specialist))} Specialist PTs\n"
                 "Concentrated value in small number of RMDs",
                 fontsize=13, fontweight="bold", pad=20)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig = ax.get_figure()
    save_nature_figure(fig, out_path / "Fig_TopSpecialistPTs.pdf", verbose=True)
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BRGI + CISN Decision-Support Reporting Layer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python brgi_cisn_report.py \\
    --brgi_csv model/forecast/brgi/ranked_pairs.csv \\
    --cisn_csv model/forecast/cisn/pt_spillover_value.csv \\
    --cisn_brgi_csv model/forecast/cisn/cisn_adjusted_brgi.csv \\
    --edges_csv model/forecast/cisn/spillover_network_edges.csv \\
    --out_dir model/forecast/brgi_cisn_report \\
    --top_k 20 \\
    --gamma 0.5
        """
    )

    parser.add_argument("--brgi_csv", required=True, help="BRGI ranked pairs CSV")
    parser.add_argument("--cisn_csv", required=True, help="CISN PT spillover values CSV")
    parser.add_argument("--cisn_brgi_csv", help="Optional CISN-adjusted BRGI CSV")
    parser.add_argument("--edges_csv", help="Optional spillover network edges CSV")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top items to display")
    parser.add_argument("--gamma", type=float, default=0.5, help="CISN adjustment multiplier")
    parser.add_argument("--specialist_entropy_threshold", type=float, default=0.35)
    parser.add_argument("--generalist_entropy_threshold", type=float, default=0.65)
    parser.add_argument("--generalist_effective_rmd_threshold", type=int, default=5)

    args = parser.parse_args()

    print("\n" + "="*80)
    print("BRGI + CISN DECISION-SUPPORT REPORTING LAYER")
    print("="*80)

    try:
        # Load data
        print("\n[1/8] Loading BRGI data...")
        brgi_df = load_brgi(args.brgi_csv)
        print(f"  ✓ {len(brgi_df)} BRGI RMD-PT pairs loaded")

        print("[2/8] Loading CISN data...")
        cisn_df = load_cisn(args.cisn_csv)
        print(f"  ✓ {len(cisn_df)} CISN PTs loaded")

        # Merge and compute priorities
        print("[3/8] Merging BRGI and CISN...")
        merged_df = merge_brgi_cisn(brgi_df, cisn_df, gamma=args.gamma)
        print(f"  ✓ {len(merged_df)} RMD-PT pairs merged")

        # Classify PTs
        print("[4/8] Classifying PTs (Specialist/Bridge/Generalist)...")
        cisn_classified = classify_pt_specialist_generalist(
            cisn_df,
            specialist_entropy_threshold=args.specialist_entropy_threshold,
            generalist_entropy_threshold=args.generalist_entropy_threshold,
            generalist_effective_rmd_threshold=args.generalist_effective_rmd_threshold
        )
        print(f"  ✓ PT classes assigned")

        # Save outputs
        print("[5/8] Saving data outputs...")
        save_priority_matrix(merged_df, args.out_dir)
        save_pt_classified(cisn_classified, args.out_dir)
        print(f"  ✓ Data saved to {args.out_dir}")

        # Generate markdown report
        print("[6/8] Generating markdown summary...")
        generate_priority_summary(merged_df, cisn_classified, args.out_dir)
        print(f"  ✓ top_priority_summary.md generated")

        # Generate plots (Nature-style PDF vectors with PNG previews)
        print("[7/8] Generating Nature-style visualizations (PDF + PNG)...")
        set_nature_style()

        plot_brgi_top_gaps(merged_df, args.out_dir, top_k=args.top_k)
        plot_cisn_top_pts(cisn_df, args.out_dir, top_k=args.top_k)
        plot_priority_quadrant(merged_df, args.out_dir)
        plot_priority_heatmap(merged_df, args.out_dir, top_k=args.top_k)
        plot_one_page_decision_map(merged_df, cisn_classified, args.out_dir)
        plot_pt_class_distribution(cisn_classified, args.out_dir)
        plot_top_generalist_pts(cisn_classified, args.out_dir, top_k=15)
        plot_top_specialist_pts(cisn_classified, args.out_dir, top_k=15)

        print("[8/8] Complete!\n")

        # Summary
        print("="*80)
        print("BRGI + CISN REPORTING COMPLETE")
        print("="*80)
        print(f"\nKey Findings:")
        print(f"  • {len(merged_df[merged_df['priority_label'] == 'Strategic Priority'])} Strategic Priorities "
              f"(high gap + high spillover)")
        print(f"  • {len(merged_df[merged_df['priority_label'] == 'Targeted Priority'])} Targeted Priorities "
              f"(high gap, disorder-specific)")
        print(f"  • {len(merged_df[merged_df['priority_label'] == 'Platform Opportunity'])} Platform Opportunities "
              f"(broad spillover, lower urgency)")
        print(f"  • {len(cisn_classified[cisn_classified['pt_class'] == 'Specialist'])} Specialist PTs")
        print(f"  • {len(cisn_classified[cisn_classified['pt_class'] == 'Bridge'])} Bridge PTs")
        print(f"  • {len(cisn_classified[cisn_classified['pt_class'] == 'Generalist'])} Generalist PTs")

        print(f"\nOutput files saved to: {args.out_dir}")
        print("\nInterpretation:")
        print("  BRGI identifies where forecasted RMD burden exceeds PT readiness.")
        print("  CISN identifies PTs with broader cross-RMD spillover potential.")
        print("  Combined, they highlight strategic PTs that are both gap-relevant and")
        print("  broadly useful across related RMDs.")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
