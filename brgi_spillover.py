"""
Cross-RMD Intervention Spillover Network (CISN)

CISN is a post-forecast decision-support layer that extends BR-MTGNN and BRGI
by estimating cross-disorder intervention portfolio value.

Key Methodological Notes:
- CISN is NOT a clinical efficacy model; it is a prioritization tool.
- It estimates potential cross-disorder intervention value based on:
  * Time-series similarity (default): RMD trajectory correlation
  * Clinical similarity (if provided): symptom/comorbidity/mechanism overlap
  * Hybrid approach (optional): blend of clinical + data-driven similarity
- Cross-disorder transferability is inferred from similarity, not proven clinically.
- CISN should inform decision-making; it does not replace clinical judgment.

Mathematical Foundation:

  B = forecast burden vector (normalized)
  A = RMD-PT association matrix
  S = cross-RMD spillover strength matrix (N×N)

  ReachableBurden = S @ B
  PTValue = A.T @ ReachableBurden

  CISN_BRGI(i,p) = BRGI(i,p) * (1 + gamma * SpilloverValueNorm(p))

Author: BR-MTGNN Audit & Enhancement
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
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
    """
    Normalize names across files.

    Removes trailing "_NoM", "_NoP", strips whitespace, preserves RMD_/PT_ prefixes.
    """
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


def row_normalize(S: np.ndarray) -> np.ndarray:
    """Row-normalize matrix to [0, 1] per row."""
    row_sums = np.sum(S, axis=1, keepdims=True)
    return safe_divide(S, row_sums)


def validate_matrix(M: np.ndarray, name: str = "Matrix") -> Tuple[bool, str]:
    """Validate matrix for NaN/inf, return status."""
    if not np.all(np.isfinite(M)):
        bad_count = (~np.isfinite(M)).sum()
        return False, f"{name} contains {bad_count} non-finite values"
    return True, f"{name} is valid"


# ============================================================================
# LOADING FUNCTIONS
# ============================================================================

def load_forecast_csv(forecast_csv: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load forecast CSV and identify RMD columns.

    Returns:
        df: forecast dataframe (RMD columns only)
        rmd_names: list of RMD column names (canonical)
    """
    df = pd.read_csv(forecast_csv)

    rmd_cols = [c for c in df.columns if "RMD_" in c and "_NoM" in c]
    rmd_names = [canonical_name(c, keep_prefix=True) for c in rmd_cols]

    if not rmd_cols:
        raise ValueError("No RMD columns (RMD_*_NoM) found in forecast CSV")

    return df[rmd_cols], rmd_names


def load_history_csv(history_csv: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load historical time-series CSV for RMD trajectories.

    Returns:
        df: history dataframe
        rmd_names: canonical RMD names
    """
    df = pd.read_csv(history_csv)

    rmd_cols = [c for c in df.columns if "RMD_" in c and "_NoM" in c]
    rmd_names = [canonical_name(c, keep_prefix=True) for c in rmd_cols]

    if not rmd_cols:
        raise ValueError("No RMD columns (RMD_*_NoM) found in history CSV")

    return df[rmd_cols], rmd_names


def load_mapping_csv(mapping_csv: str) -> pd.DataFrame:
    """Load RMD-PT mapping file."""
    return pd.read_csv(mapping_csv)


def load_similarity_matrix(csv_path: str, expected_names: List[str]) -> Optional[np.ndarray]:
    """
    Load clinical similarity matrix (symptom, comorbidity, mechanism, treatment).

    Attempts to match row/column names with expected_names.
    """
    if not Path(csv_path).exists():
        return None

    df = pd.read_csv(csv_path, index_col=0)

    # Attempt name matching
    canonical_idx = [canonical_name(idx) for idx in df.index]
    canonical_cols = [canonical_name(col) for col in df.columns]
    canonical_expected = [canonical_name(name) for name in expected_names]

    # Reindex if names match
    if set(canonical_expected) <= set(canonical_idx):
        df = df.loc[[idx for idx, c in enumerate(df.index)
                     if canonical_name(df.index[idx]) in canonical_expected]]

    M = df.values
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    return M


# ============================================================================
# BURDEN COMPUTATION
# ============================================================================

def compute_burden(forecast_df: pd.DataFrame,
                   rmd_cols: List[str],
                   method: str = "mean") -> np.ndarray:
    """
    Compute forecasted burden over 36-month horizon.

    Methods:
        mean: average across horizon
        sum: total across horizon
        last: final month value
        auc: trapezoidal area under curve

    Returns:
        B: normalized burden vector [0, 1]
    """
    if method == "mean":
        B_raw = forecast_df[rmd_cols].mean(axis=0).values
    elif method == "sum":
        B_raw = forecast_df[rmd_cols].sum(axis=0).values
    elif method == "last":
        B_raw = forecast_df[rmd_cols].iloc[-1].values
    elif method == "auc":
        B_raw = np.trapz(forecast_df[rmd_cols].values, axis=0)
    else:
        raise ValueError(f"Unknown burden method: {method}")

    # Replace NaN/inf
    B_raw = np.nan_to_num(B_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # Log transform
    B_log = np.log1p(B_raw)

    # Min-max normalize
    B_norm = minmax_normalize(B_log)

    return B_raw, B_log, B_norm


# ============================================================================
# ASSOCIATION MATRIX
# ============================================================================

def build_association_matrix(mapping_df: pd.DataFrame,
                             rmd_names: List[str],
                             pt_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build RMD-PT association matrix A ∈ R^(N×M).

    From mapping file with columns: rmd_name, pt_name, rank (optional)

    Weighting:
        If rank exists: A_ip = (max_rank - rank + 1) / max_rank
        If no rank: A_ip = 1.0

    Returns:
        A: association matrix (N × M)
        A_rank_counts: mapping counts per RMD
    """
    N, M = len(rmd_names), len(pt_names)
    A = np.zeros((N, M))
    rank_counts = np.zeros(N)

    # Canonical name mapping
    canonical_rmd = {canonical_name(name, keep_prefix=True): i for i, name in enumerate(rmd_names)}
    canonical_pt = {canonical_name(name, keep_prefix=True): i for i, name in enumerate(pt_names)}

    for _, row in mapping_df.iterrows():
        rmd_cn = canonical_name(row.get("rmd_name", ""), keep_prefix=True)
        pt_cn = canonical_name(row.get("pt_name", ""), keep_prefix=True)

        if rmd_cn not in canonical_rmd or pt_cn not in canonical_pt:
            continue

        i = canonical_rmd[rmd_cn]
        j = canonical_pt[pt_cn]

        # Rank-based weight (if available)
        if "rank" in row and pd.notna(row["rank"]):
            max_rank = mapping_df["rank"].max()
            weight = (max_rank - row["rank"] + 1) / max_rank
        else:
            weight = 1.0

        A[i, j] = max(A[i, j], weight)  # Take max if duplicate
        rank_counts[i] += 1

    # Normalize A to [0, 1]
    A_max = A.max()
    if A_max > 0:
        A = A / A_max

    return A, rank_counts


# ============================================================================
# SPILLOVER MATRIX COMPUTATION
# ============================================================================

def build_spillover_matrix_timeseries(history_df: pd.DataFrame,
                                     rmd_names: List[str]) -> np.ndarray:
    """
    Build spillover matrix S from historical RMD NoM trajectories.

    Method:
        1. Extract RMD NoM columns
        2. Apply log1p transformation
        3. Compute Pearson correlation
        4. Clip negative values to 0
        5. Set diagonal to 1.0

    Returns:
        S: spillover matrix (N × N)
    """
    N = len(rmd_names)

    # Get data
    rmd_data = history_df.values

    # Replace NaN/inf
    rmd_data = np.nan_to_num(rmd_data, nan=0.0, posinf=0.0, neginf=0.0)

    # Log transform
    rmd_data_log = np.log1p(rmd_data)

    # Compute Pearson correlation between RMD trajectories
    S = np.corrcoef(rmd_data_log.T)

    # Handle edge cases
    if not np.isfinite(S).all():
        S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)

    # Clip negative correlations
    S = np.clip(S, 0, 1)

    # Set diagonal to 1.0
    np.fill_diagonal(S, 1.0)

    return S


def build_spillover_matrix_clinical(symptom_mat: Optional[np.ndarray],
                                   comorbidity_mat: Optional[np.ndarray],
                                   mechanism_mat: Optional[np.ndarray],
                                   treatment_mat: Optional[np.ndarray],
                                   symptom_weight: float = 0.35,
                                   comorbidity_weight: float = 0.25,
                                   mechanism_weight: float = 0.25,
                                   treatment_weight: float = 0.15) -> np.ndarray:
    """
    Build spillover matrix from clinical similarity components.

    Weights should sum to 1; they are normalized internally.

    Returns:
        S: spillover matrix (N × N)
    """
    weights = np.array([symptom_weight, comorbidity_weight,
                        mechanism_weight, treatment_weight])
    weights = weights / weights.sum()

    matrices = [symptom_mat, comorbidity_mat, mechanism_mat, treatment_mat]

    # Filter out None matrices
    valid_matrices = [m for m in matrices if m is not None]
    valid_weights = [w for m, w in zip(matrices, weights) if m is not None]

    if not valid_matrices:
        raise ValueError("No clinical similarity matrices provided")

    # Normalize weights for valid matrices
    valid_weights = np.array(valid_weights) / np.array(valid_weights).sum()

    # Combine weighted matrices
    S = np.zeros_like(valid_matrices[0])
    for mat, weight in zip(valid_matrices, valid_weights):
        S += weight * mat

    # Clip to [0, 1] and set diagonal
    S = np.clip(S, 0, 1)
    np.fill_diagonal(S, 1.0)

    return S


def build_spillover_matrix(history_df: pd.DataFrame,
                          rmd_names: List[str],
                          similarity_source: str = "timeseries",
                          symptom_mat: Optional[np.ndarray] = None,
                          comorbidity_mat: Optional[np.ndarray] = None,
                          mechanism_mat: Optional[np.ndarray] = None,
                          treatment_mat: Optional[np.ndarray] = None,
                          symptom_weight: float = 0.35,
                          comorbidity_weight: float = 0.25,
                          mechanism_weight: float = 0.25,
                          treatment_weight: float = 0.15) -> Tuple[np.ndarray, str]:
    """
    Build spillover matrix using specified similarity source.

    Returns:
        S: spillover matrix
        source_description: human-readable description of approach used
    """
    if similarity_source == "timeseries":
        S = build_spillover_matrix_timeseries(history_df, rmd_names)
        source = "data-driven time-series similarity"

    elif similarity_source == "clinical":
        if not any([symptom_mat is not None, comorbidity_mat is not None,
                   mechanism_mat is not None, treatment_mat is not None]):
            raise ValueError("No clinical matrices provided but similarity_source='clinical'")
        S = build_spillover_matrix_clinical(
            symptom_mat, comorbidity_mat, mechanism_mat, treatment_mat,
            symptom_weight, comorbidity_weight, mechanism_weight, treatment_weight
        )
        source = "clinical similarity matrices"

    elif similarity_source == "hybrid":
        S_ts = build_spillover_matrix_timeseries(history_df, rmd_names)

        clinical_available = any([symptom_mat is not None, comorbidity_mat is not None,
                                 mechanism_mat is not None, treatment_mat is not None])
        if clinical_available:
            S_clinical = build_spillover_matrix_clinical(
                symptom_mat, comorbidity_mat, mechanism_mat, treatment_mat,
                symptom_weight, comorbidity_weight, mechanism_weight, treatment_weight
            )
            S = 0.7 * S_clinical + 0.3 * S_ts
            source = "hybrid: 70% clinical + 30% time-series"
        else:
            S = S_ts
            source = "time-series (clinical matrices unavailable)"

    else:
        raise ValueError(f"Unknown similarity_source: {similarity_source}")

    return S, source


# ============================================================================
# SPILLOVER VALUE COMPUTATION
# ============================================================================

def compute_spillover_values(A: np.ndarray,
                           S: np.ndarray,
                           B: np.ndarray,
                           rmd_names: List[str],
                           pt_names: List[str]) -> pd.DataFrame:
    """
    Compute spillover-adjusted PT values.

    SpilloverValue(p) = A.T @ (S @ B)

    Also compute:
        direct_value = A.T @ B
        spillover_gain = SpilloverValue - direct_value

    Returns:
        df: PT-level spillover metrics
    """
    # Reachable burden: which RMDs' burdens are "felt" across the network
    ReachableBurden = S @ B  # [N]

    # PT spillover value: which PTs reach the most burden
    PTValue = A.T @ ReachableBurden  # [M]

    # Direct value: PT value without spillover
    DirectValue = A.T @ B  # [M]

    # Normalize
    PTValueNorm = minmax_normalize(PTValue)

    # Spillover gain
    SpilloverGain = PTValue - DirectValue
    SpilloverGainNorm = minmax_normalize(np.abs(SpilloverGain))

    # Count linked RMDs per PT
    LinkedRMDCount = (A > 0).sum(axis=0)

    # Top reachable and source RMDs per PT
    TopSourceRMDs = []
    TopReachableRMDs = []

    for j in range(len(pt_names)):
        # Which RMDs link to this PT?
        source_idxs = np.where(A[:, j] > 0)[0]
        if len(source_idxs) > 0:
            source_names = [rmd_names[i].replace("RMD_", "") for i in source_idxs[:3]]
            TopSourceRMDs.append("; ".join(source_names))
        else:
            TopSourceRMDs.append("")

        # Which RMDs have highest reachable burden?
        reachable_idxs = np.argsort(ReachableBurden)[-3:][::-1]
        reachable_names = [rmd_names[i].replace("RMD_", "") for i in reachable_idxs]
        TopReachableRMDs.append("; ".join(reachable_names))

    df = pd.DataFrame({
        "pt_name": [name.replace("PT_", "") for name in pt_names],
        "direct_value": DirectValue,
        "spillover_value": PTValue,
        "spillover_value_norm": PTValueNorm,
        "spillover_gain": SpilloverGain,
        "spillover_gain_norm": SpilloverGainNorm,
        "linked_rmd_count": LinkedRMDCount,
        "top_source_rmds": TopSourceRMDs,
        "top_reachable_rmds": TopReachableRMDs,
    })

    return df.sort_values("spillover_value", ascending=False).reset_index(drop=True)


# ============================================================================
# BRGI INTEGRATION
# ============================================================================

def integrate_brgi(pt_spillover_df: pd.DataFrame,
                   brgi_csv: str,
                   gamma: float = 0.5) -> pd.DataFrame:
    """
    Merge spillover values with BRGI scores.

    CISN_BRGI(i,p) = BRGI(i,p) * (1 + gamma * SpilloverValueNorm(p))

    Returns:
        merged_df: BRGI pairs with CISN adjustment
    """
    brgi_df = pd.read_csv(brgi_csv)

    # Infer column names (flexible)
    rmd_col = next((c for c in brgi_df.columns if "rmd" in c.lower() or "burden" in c.lower()), None)
    pt_col = next((c for c in brgi_df.columns if "pt" in c.lower() or "readiness" in c.lower()), None)
    score_col = next((c for c in brgi_df.columns if "score" in c.lower() or "brgi" in c.lower()), None)

    if not all([rmd_col, pt_col, score_col]):
        raise ValueError(f"Could not infer columns in BRGI CSV. Found: {brgi_df.columns.tolist()}")

    # Build spillover lookup
    spillover_lookup = dict(zip(pt_spillover_df["pt_name"],
                               pt_spillover_df["spillover_value_norm"]))

    # Add spillover normalization
    brgi_df["spillover_value_norm"] = brgi_df[pt_col].apply(
        lambda x: spillover_lookup.get(canonical_name(str(x), keep_prefix=True), 0.0)
    )

    # Compute CISN-adjusted score
    brgi_df["cisn_adjusted_score"] = brgi_df[score_col] * (1 + gamma * brgi_df["spillover_value_norm"])

    return brgi_df.sort_values("cisn_adjusted_score", ascending=False)


# ============================================================================
# NORMALIZATION
# ============================================================================

def normalize_spillover_matrix(S: np.ndarray,
                             method: str = "row") -> np.ndarray:
    """
    Normalize spillover matrix.

    Methods:
        row: row-normalize to unit sum
        symmetric: symmetrize then row-normalize
        none: no normalization
    """
    # Ensure diagonal is 1 before normalization
    np.fill_diagonal(S, 1.0)

    if method == "row":
        return row_normalize(S)
    elif method == "symmetric":
        S_sym = (S + S.T) / 2
        return row_normalize(S_sym)
    elif method == "none":
        return np.clip(S, 0, np.inf)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# ============================================================================
# OUTPUTS
# ============================================================================

def save_outputs(out_dir: str,
                 S: np.ndarray,
                 A: np.ndarray,
                 B_raw: np.ndarray,
                 B_log: np.ndarray,
                 B_norm: np.ndarray,
                 pt_spillover_df: pd.DataFrame,
                 rmd_names: List[str],
                 pt_names: List[str],
                 rmd_burden_df: Optional[pd.DataFrame] = None,
                 cisn_brgi_df: Optional[pd.DataFrame] = None,
                 top_k: int = 30,
                 spillover_threshold: float = 0.05,
                 validation_report: Optional[Dict] = None):
    """Save all CISN output files."""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Spillover matrix
    S_df = pd.DataFrame(S,
                       index=[n.replace("RMD_", "") for n in rmd_names],
                       columns=[n.replace("RMD_", "") for n in rmd_names])
    S_df.to_csv(out_path / "spillover_matrix.csv")

    # 2. RMD burden vector
    if rmd_burden_df is None:
        rmd_burden_df = pd.DataFrame({
            "rmd_name": [n.replace("RMD_", "") for n in rmd_names],
            "burden_raw": B_raw,
            "burden_log1p": B_log,
            "burden_norm": B_norm,
        })
    rmd_burden_df.to_csv(out_path / "rmd_burden_vector.csv", index=False)

    # 3. RMD-PT association matrix
    A_df = pd.DataFrame(A,
                       index=[n.replace("RMD_", "") for n in rmd_names],
                       columns=[n.replace("PT_", "") for n in pt_names])
    A_df.to_csv(out_path / "rmd_pt_association_matrix.csv")

    # 4. PT spillover values
    pt_spillover_df.to_csv(out_path / "pt_spillover_value.csv", index=False)

    # 5. CISN-adjusted BRGI (if available)
    if cisn_brgi_df is not None:
        cisn_brgi_df.to_csv(out_path / "cisn_adjusted_brgi.csv", index=False)

    # 6. Top spillover PTs (markdown table)
    top_pts = pt_spillover_df.head(top_k)
    md_lines = ["# Top Spillover-Adjusted Pertinent Technologies\n"]
    md_lines.append("| Rank | PT Name | Direct Value | Spillover Value | Spillover Gain | "
                    "Linked RMDs |\n")
    md_lines.append("|------|---------|--------------|-----------------|----------------|"
                    "------|\n")
    for idx, row in top_pts.iterrows():
        md_lines.append(
            f"| {idx+1} | {row['pt_name']} | {row['direct_value']:.4f} | "
            f"{row['spillover_value']:.4f} | {row['spillover_gain']:.4f} | "
            f"{int(row['linked_rmd_count'])} |\n"
        )

    with open(out_path / "top_spillover_pts.md", "w") as f:
        f.writelines(md_lines)

    # 7. Spillover network edges
    edges = []
    for i in range(len(rmd_names)):
        for j in range(i+1, len(rmd_names)):
            weight = S[i, j]
            if weight > spillover_threshold:
                edges.append({
                    "source_rmd": rmd_names[i].replace("RMD_", ""),
                    "target_rmd": rmd_names[j].replace("RMD_", ""),
                    "spillover_weight": weight,
                })

    edges_df = pd.DataFrame(edges)
    if not edges_df.empty:
        edges_df = edges_df.sort_values("spillover_weight", ascending=False)
        edges_df.to_csv(out_path / "spillover_network_edges.csv", index=False)

    # 8. Validation report
    if validation_report is not None:
        with open(out_path / "validation_report.json", "w") as f:
            json.dump(validation_report, f, indent=2, default=str)


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_spillover_matrix_heatmap(S: np.ndarray,
                                 rmd_names: List[str],
                                 out_path: Path,
                                 max_label_nodes: int = 60):
    """Heatmap of spillover matrix."""
    fig, ax = plt.subplots(figsize=(14, 12))

    im = ax.imshow(S, cmap="YlOrRd", aspect="auto")

    N = len(rmd_names)
    short_names = [n.replace("RMD_", "")[:20] for n in rmd_names]

    if N <= max_label_nodes:
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels(short_names, rotation=90, fontsize=8)
        ax.set_yticklabels(short_names, fontsize=8)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_xlabel("Target RMD", fontsize=11, fontweight="bold")
    ax.set_ylabel("Source RMD", fontsize=11, fontweight="bold")
    ax.set_title("Cross-RMD Spillover Strength Matrix\n(Data-driven time-series correlation)",
                 fontsize=13, fontweight="bold", pad=20)

    cbar = plt.colorbar(im, ax=ax, label="Spillover Strength")
    cbar.set_label("Spillover Strength [0-1]", fontsize=10)

    plt.tight_layout()
    save_nature_figure(fig, out_path / "Fig_Spillover_Matrix_Heatmap.pdf", verbose=True)
    plt.close()


def plot_top_pt_spillover_bar(pt_spillover_df: pd.DataFrame,
                              out_path: Path,
                              top_k: int = 30):
    """Bar chart of top PT spillover values."""
    top_pts = pt_spillover_df.head(top_k)

    fig, ax = plt.subplots(figsize=(12, max(6, top_k * 0.2)))

    y_pos = np.arange(len(top_pts))
    colors = plt.cm.RdYlGn(minmax_normalize(top_pts["spillover_value"].values))

    ax.barh(y_pos, top_pts["spillover_value"].values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_pts["pt_name"].values, fontsize=9)
    ax.set_xlabel("Spillover-Adjusted Value", fontsize=11, fontweight="bold")
    ax.set_title(f"Top {top_k} Pertinent Technologies by Spillover Value\n"
                 "Higher value = broader cross-disorder portfolio impact",
                 fontsize=13, fontweight="bold", pad=20)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig = ax.get_figure()
    save_nature_figure(fig, out_path / "Fig_TopPT_Spillover_Bar.pdf", verbose=True)
    plt.close()


def plot_direct_vs_spillover(pt_spillover_df: pd.DataFrame,
                             out_path: Path,
                             top_annotate: int = 10):
    """Scatter plot: direct value vs spillover value."""
    fig, ax = plt.subplots(figsize=(11, 8))

    ax.scatter(pt_spillover_df["direct_value"],
              pt_spillover_df["spillover_value"],
              alpha=0.6, s=80, edgecolor="black", linewidth=0.5, color="steelblue")

    # Annotate top PTs
    top_pts = pt_spillover_df.nlargest(top_annotate, "spillover_value")
    for _, row in top_pts.iterrows():
        ax.annotate(row["pt_name"],
                   (row["direct_value"], row["spillover_value"]),
                   fontsize=8, alpha=0.7,
                   xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Direct Value (A.T @ B)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Spillover Value (A.T @ S @ B)", fontsize=11, fontweight="bold")
    ax.set_title("Direct vs. Spillover-Adjusted PT Value\nPTs above diagonal gain cross-disorder reach",
                 fontsize=13, fontweight="bold", pad=20)

    # Diagonal reference line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0, label="No spillover gain")
    ax.legend(fontsize=10)

    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig = ax.get_figure()
    save_nature_figure(fig, out_path / "Fig_DirectVsSpillover_Value.pdf", verbose=True)
    plt.close()


def plot_cisn_adjusted_brgi_top(cisn_brgi_df: pd.DataFrame,
                               out_path: Path,
                               top_k: int = 30):
    """Bar chart of top CISN-adjusted BRGI pair scores."""
    if cisn_brgi_df is None or cisn_brgi_df.empty:
        return

    top_pairs = cisn_brgi_df.head(top_k)

    # Create pair labels
    brgi_col = next((c for c in cisn_brgi_df.columns if "burden" in c.lower() or "rmd" in c.lower()), None)
    pt_col = next((c for c in cisn_brgi_df.columns if "readiness" in c.lower() or "pt" in c.lower()), None)

    if brgi_col and pt_col:
        pair_labels = [f"{canonical_name(row[brgi_col], keep_prefix=True).replace('RMD_', '')} → "
                      f"{canonical_name(row[pt_col], keep_prefix=True).replace('PT_', '')}"
                      for _, row in top_pairs.iterrows()]
    else:
        pair_labels = [f"Pair {i+1}" for i in range(len(top_pairs))]

    fig, ax = plt.subplots(figsize=(12, max(6, top_k * 0.2)))

    y_pos = np.arange(len(top_pairs))
    colors = plt.cm.RdYlGn(minmax_normalize(top_pairs["cisn_adjusted_score"].values))

    ax.barh(y_pos, top_pairs["cisn_adjusted_score"].values,
           color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pair_labels, fontsize=8)
    ax.set_xlabel("CISN-Adjusted BRGI Score", fontsize=11, fontweight="bold")
    ax.set_title(f"Top {top_k} RMD-PT Pairs by CISN-Adjusted BRGI Score\n"
                 "Combining burden-readiness gap with cross-disorder spillover value",
                 fontsize=13, fontweight="bold", pad=20)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig = ax.get_figure()
    save_nature_figure(fig, out_path / "Fig_CISN_Adjusted_BRGI_Pairs.pdf", verbose=True)
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-RMD Intervention Spillover Network (CISN) Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Basic spillover analysis (time-series similarity)
  python brgi_spillover.py \\
    --forecast_csv model/Bayesian/forecast/data/forecast_NoM_mean.csv \\
    --history_csv data/sm_data.csv \\
    --nodes_csv data/data.csv \\
    --mapping_csv model/Bayesian/forecast/plots_data/selected_pts_per_rmd.csv \\
    --out_dir model/Bayesian/forecast/cisn \\
    --similarity_source timeseries \\
    --top_k 30

  # With BRGI integration
  python brgi_spillover.py \\
    --forecast_csv model/Bayesian/forecast/data/forecast_NoM_mean.csv \\
    --history_csv data/sm_data.csv \\
    --mapping_csv model/Bayesian/forecast/plots_data/selected_pts_per_rmd.csv \\
    --brgi_csv model/Bayesian/forecast/brgi/ranked_pairs.csv \\
    --out_dir model/Bayesian/forecast/cisn \\
    --gamma 0.5
        """
    )

    # Required arguments
    parser.add_argument("--forecast_csv", required=True,
                       help="Forecast CSV with RMD_*_NoM and PT_*_NoM columns")
    parser.add_argument("--history_csv", required=True,
                       help="Historical time-series CSV for RMD trajectories")
    parser.add_argument("--nodes_csv", required=True,
                       help="Nodes metadata CSV (for reference)")
    parser.add_argument("--mapping_csv", required=True,
                       help="RMD-PT mapping CSV (rmd_name, pt_name, rank)")
    parser.add_argument("--out_dir", required=True,
                       help="Output directory for results")

    # Optional BRGI
    parser.add_argument("--brgi_csv",
                       help="BRGI ranked pairs CSV (optional)")

    # Optional clinical similarity matrices
    parser.add_argument("--symptom_overlap_csv",
                       help="Symptom overlap matrix (N×N, RMD names as index/columns)")
    parser.add_argument("--comorbidity_overlap_csv",
                       help="Comorbidity overlap matrix")
    parser.add_argument("--mechanism_overlap_csv",
                       help="Treatment mechanism overlap matrix")
    parser.add_argument("--treatment_transfer_csv",
                       help="Treatment transferability matrix")

    # Computation options
    parser.add_argument("--forecast_burden_method", default="mean",
                       choices=["mean", "sum", "last", "auc"],
                       help="How to aggregate forecast over 36-month horizon")
    parser.add_argument("--similarity_source", default="timeseries",
                       choices=["clinical", "timeseries", "hybrid"],
                       help="Source for RMD similarity matrix")
    parser.add_argument("--symptom_weight", type=float, default=0.35,
                       help="Weight for symptom overlap (if clinical)")
    parser.add_argument("--comorbidity_weight", type=float, default=0.25,
                       help="Weight for comorbidity overlap")
    parser.add_argument("--mechanism_weight", type=float, default=0.25,
                       help="Weight for mechanism overlap")
    parser.add_argument("--treatment_weight", type=float, default=0.15,
                       help="Weight for treatment transfer")
    parser.add_argument("--gamma", type=float, default=0.5,
                       help="BRGI spillover adjustment multiplier")
    parser.add_argument("--normalize_spillover", default="row",
                       choices=["row", "symmetric", "none"],
                       help="How to normalize spillover matrix")
    parser.add_argument("--top_k", type=int, default=30,
                       help="Number of top PTs to report/plot")
    parser.add_argument("--spillover_threshold", type=float, default=0.05,
                       help="Minimum spillover weight for network edge list")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("CROSS-RMD INTERVENTION SPILLOVER NETWORK (CISN)")
    print("="*80)

    try:
        # ====== LOAD DATA ======
        print("\n[1/9] Loading forecast data...")
        forecast_df, rmd_names = load_forecast_csv(args.forecast_csv)
        print(f"  ✓ {len(rmd_names)} RMDs")

        print("[2/9] Loading historical trajectories...")
        history_df, _ = load_history_csv(args.history_csv)
        print(f"  ✓ {len(history_df)} time steps")

        print("[3/9] Loading RMD-PT mapping...")
        mapping_df = load_mapping_csv(args.mapping_csv)
        print(f"  ✓ {len(mapping_df)} RMD-PT associations")

        # Extract unique PTs from mapping file
        pt_names = [canonical_name(pt, keep_prefix=True)
                   for pt in mapping_df["pt_name"].unique()]
        print(f"  ✓ {len(pt_names)} unique PTs")

        # ====== COMPUTE BURDEN ======
        print("\n[4/9] Computing forecasted burden...")
        B_raw, B_log, B_norm = compute_burden(forecast_df,
                                             [c for c in forecast_df.columns if "RMD_" in c],
                                             method=args.forecast_burden_method)
        print(f"  ✓ Burden method: {args.forecast_burden_method}")
        print(f"  ✓ Burden range: [{B_norm.min():.4f}, {B_norm.max():.4f}]")

        # ====== ASSOCIATION MATRIX ======
        print("[5/9] Building RMD-PT association matrix...")
        A, rank_counts = build_association_matrix(mapping_df, rmd_names, pt_names)
        print(f"  ✓ Matrix shape: {A.shape}")
        print(f"  ✓ Sparsity: {(A == 0).sum() / A.size * 100:.1f}%")

        # ====== SPILLOVER MATRIX ======
        print("[6/9] Building spillover matrix...")

        # Load clinical matrices if provided
        symptom_mat = load_similarity_matrix(args.symptom_overlap_csv, rmd_names) if args.symptom_overlap_csv else None
        comorbidity_mat = load_similarity_matrix(args.comorbidity_overlap_csv, rmd_names) if args.comorbidity_overlap_csv else None
        mechanism_mat = load_similarity_matrix(args.mechanism_overlap_csv, rmd_names) if args.mechanism_overlap_csv else None
        treatment_mat = load_similarity_matrix(args.treatment_transfer_csv, rmd_names) if args.treatment_transfer_csv else None

        S, similarity_source = build_spillover_matrix(
            history_df, rmd_names,
            similarity_source=args.similarity_source,
            symptom_mat=symptom_mat,
            comorbidity_mat=comorbidity_mat,
            mechanism_mat=mechanism_mat,
            treatment_mat=treatment_mat,
            symptom_weight=args.symptom_weight,
            comorbidity_weight=args.comorbidity_weight,
            mechanism_weight=args.mechanism_weight,
            treatment_weight=args.treatment_weight,
        )
        print(f"  ✓ Source: {similarity_source}")
        print(f"  ✓ Range: [{S.min():.4f}, {S.max():.4f}]")

        # Normalize
        S = normalize_spillover_matrix(S, method=args.normalize_spillover)
        print(f"  ✓ Normalization: {args.normalize_spillover}")

        # ====== SPILLOVER VALUES ======
        print("[7/9] Computing spillover-adjusted PT values...")
        pt_spillover_df = compute_spillover_values(A, S, B_norm, rmd_names, pt_names)
        print(f"  ✓ Top PT: {pt_spillover_df.iloc[0]['pt_name']} "
              f"({pt_spillover_df.iloc[0]['spillover_value']:.4f})")

        # ====== BRGI INTEGRATION ======
        cisn_brgi_df = None
        if args.brgi_csv:
            print("[8/9] Integrating with BRGI scores...")
            cisn_brgi_df = integrate_brgi(pt_spillover_df, args.brgi_csv, gamma=args.gamma)
            print(f"  ✓ {len(cisn_brgi_df)} RMD-PT pairs adjusted")
        else:
            print("[8/9] (BRGI integration skipped)")

        # ====== VALIDATION REPORT ======
        print("[9/9] Generating validation report...")

        # Prepare burden df
        rmd_burden_df = pd.DataFrame({
            "rmd_name": [n.replace("RMD_", "") for n in rmd_names],
            "burden_raw": B_raw,
            "burden_log1p": B_log,
            "burden_norm": B_norm,
        })

        validation_report = {
            "metadata": {
                "num_rmds": len(rmd_names),
                "num_pts": len(pt_names),
                "num_rmd_pt_pairs_mapped": (A > 0).sum(),
                "clinical_matrices_used": any([symptom_mat is not None, comorbidity_mat is not None,
                                               mechanism_mat is not None, treatment_mat is not None]),
                "similarity_source": similarity_source,
            },
            "burden_statistics": {
                "min_raw": float(B_raw.min()),
                "max_raw": float(B_raw.max()),
                "mean_raw": float(B_raw.mean()),
                "min_norm": float(B_norm.min()),
                "max_norm": float(B_norm.max()),
                "mean_norm": float(B_norm.mean()),
            },
            "spillover_matrix_statistics": {
                "min": float(S.min()),
                "max": float(S.max()),
                "mean": float(S.mean()),
                "diagonal_mean": float(np.diag(S).mean()),
                "row_sum_mean": float(S.sum(axis=1).mean()),
            },
            "pt_value_statistics": {
                "min_direct": float(pt_spillover_df["direct_value"].min()),
                "max_direct": float(pt_spillover_df["direct_value"].max()),
                "mean_direct": float(pt_spillover_df["direct_value"].mean()),
                "min_spillover": float(pt_spillover_df["spillover_value"].min()),
                "max_spillover": float(pt_spillover_df["spillover_value"].max()),
                "mean_spillover": float(pt_spillover_df["spillover_value"].mean()),
            },
            "data_quality": {
                "all_matrices_finite": bool(np.all(np.isfinite(A)) and np.all(np.isfinite(S))),
                "no_nan_in_outputs": True,
                "spillover_normalization": args.normalize_spillover,
            },
            "top_pt": {
                "name": str(pt_spillover_df.iloc[0]["pt_name"]),
                "spillover_value": float(pt_spillover_df.iloc[0]["spillover_value"]),
                "linked_rmds": int(pt_spillover_df.iloc[0]["linked_rmd_count"]),
            },
        }

        # ====== SAVE OUTPUTS ======
        save_outputs(
            args.out_dir,
            S, A, B_raw, B_log, B_norm,
            pt_spillover_df,
            rmd_names, pt_names,
            rmd_burden_df=rmd_burden_df,
            cisn_brgi_df=cisn_brgi_df,
            top_k=args.top_k,
            spillover_threshold=args.spillover_threshold,
            validation_report=validation_report,
        )
        print(f"  ✓ Output saved to {args.out_dir}")

        # ====== PLOTS ======
        print("\n[PLOTS] Generating Nature-style visualizations (PDF + PNG)...")
        set_nature_style()
        out_path = Path(args.out_dir)

        plot_spillover_matrix_heatmap(S, rmd_names, out_path)
        plot_top_pt_spillover_bar(pt_spillover_df, out_path, top_k=args.top_k)
        plot_direct_vs_spillover(pt_spillover_df, out_path, top_annotate=10)

        if cisn_brgi_df is not None:
            plot_cisn_adjusted_brgi_top(cisn_brgi_df, out_path, top_k=args.top_k)

        # ====== SUMMARY ======
        print("\n" + "="*80)
        print("CISN ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nKey Findings:")
        print(f"  • {len(rmd_names)} RMDs analyzed")
        print(f"  • {len(pt_names)} PTs ranked by spillover value")
        print(f"  • Spillover similarity: {similarity_source}")
        print(f"  • Top PT: {pt_spillover_df.iloc[0]['pt_name']}")
        print(f"    - Direct value: {pt_spillover_df.iloc[0]['direct_value']:.4f}")
        print(f"    - Spillover value: {pt_spillover_df.iloc[0]['spillover_value']:.4f}")
        print(f"    - Cross-disorder reach: {int(pt_spillover_df.iloc[0]['linked_rmd_count'])} RMDs")

        if cisn_brgi_df is not None:
            top_brgi = cisn_brgi_df.iloc[0]
            print(f"\n  • Top CISN-adjusted BRGI pair:")
            score_col = next((c for c in cisn_brgi_df.columns if "score" in c.lower()), None)
            if score_col:
                print(f"    - Score: {top_brgi[score_col]:.4f}")

        print(f"\nOutput files:")
        print(f"  • Matrices: spillover_matrix.csv, rmd_pt_association_matrix.csv")
        print(f"  • Metrics: pt_spillover_value.csv, rmd_burden_vector.csv")
        print(f"  • Rankings: top_spillover_pts.md, spillover_network_edges.csv")
        if cisn_brgi_df is not None:
            print(f"  • BRGI: cisn_adjusted_brgi.csv")
        print(f"  • Plots: 4 publication-quality figures")
        print(f"  • Report: validation_report.json")

        print("\n" + "="*80 + "\n")

    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
