"""
=============================================================================
PIPELINE STEP 5 of 5 — forecast.py
=============================================================================
Purpose : Load the production model, generate 36-month forecasts for all
          48 RMDs and their linked PTs, and produce publication-ready plots.
          Gemini validation runs automatically on nodes_csv before plotting
          (cache hit — no API call if data/validation_cache.json exists).

Run AFTER: train.py  (Step 4) — requires:
               model/Bayesian/o_model.pt
               model/Bayesian/hp.txt
               model/Bayesian/metadata.json
               data/sm_data.csv
               data/graph_sparse.csv
               data/data.csv  (raw, for historical plot lines)
Run BEFORE: Nothing — this is the final pipeline step.

HOW TO RUN
----------
    cd /home/sahsan03/B-MTGNN/BR_MTGNN_main

    python forecast.py \\
        --model_path model/Bayesian/o_model.pt \\
        --data_csv   data/sm_data.csv \\
        --nodes_csv  data/data.csv \\
        --graph_csv  data/graph_sparse.csv \\
        --output_dir model/Bayesian \\
        --use_tdb_input \\
        --plot_start_year 2007 \\
        --smooth_alpha 0.10

Key options:
    --model_path        Path to production model weights (o_model.pt)
    --data_csv          Smoothed time series for model input
    --nodes_csv         Raw data.csv — used for historical plot lines
    --graph_csv         RMD→PT adjacency from make_sparse_graph.py
    --use_tdb_input     Must match flag used in train_test.py and train.py
    --plot_start_year   First historical year shown on plots (default 2007)
    --smooth_alpha      EWMA smoothing on historical lines for plot clarity
    --mc_runs           MC-Dropout forward passes for uncertainty (default 10)
    --skip_validation   Skip Gemini validation (use if quota exhausted)
    --no_cache          Force fresh Gemini API call ignoring cache

Outputs:
    model/Bayesian/forecast/forecast_36m.csv        Long-format predictions
    model/Bayesian/forecast/plot_values_monthly.csv Historical + forecast
    model/Bayesian/forecast/gap_monthly.csv         RMD-PT gap analysis
    model/Bayesian/forecast/plots/*.png             48 plots at 600 DPI

Expected time: ~2 min on GPU (cuda:1)
=============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from net import gtnet
from util import (
    RMDPTData,
    ensure_dir,
    pick_device,
    reshape_model_output,
    set_random_seed,
)


# -----------------------------------------------------------------------------
# Plot style
# -----------------------------------------------------------------------------

PALETTE = [
    "#3366CC",  # blue
    "#DC3912",  # red
    "#FF9900",  # orange
    "#109618",  # green
    "#990099",  # purple
    "#0099C6",  # cyan
    "#DD4477",  # pink
    "#66AA00",  # lime
    "#B82E2E",  # dark red
    "#316395",  # steel blue
    "#994499",
    "#22AA99",
    "#AAAA11",
    "#6633CC",
    "#E67300",
    "#8B0707",
    "#651067",
    "#329262",
    "#5574A6",
    "#3B3EAC",
]

plt.rcParams.update(
    {
        "savefig.dpi": 600,
        "figure.dpi": 150,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# -----------------------------------------------------------------------------
# Argument handling
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BC-MTGNN RMD-PT forecast plotting")

    # Data / model paths
    p.add_argument("--model_path", type=str, default="model/Bayesian/o_model.pt")
    p.add_argument("--data_csv", type=str, default="data/sm_data.csv")
    p.add_argument("--nodes_csv", type=str, default=None,
                   help="Raw historical CSV for plot lines (default: data/data_validated.csv if it exists, else data/data.csv)")
    p.add_argument("--graph_csv", type=str, default="data/graph_sparse.csv")
    p.add_argument("--output_dir", type=str, default="model/Bayesian")

    # Runtime
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num_threads", type=int, default=1)

    # Data settings; these are overwritten from hp.txt when available
    p.add_argument("--seq_in_len", type=int, default=36)
    p.add_argument("--seq_out_len", type=int, default=36)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--global_regex", type=str, default="")
    p.add_argument("--no_log1p", action="store_true")
    p.add_argument("--no_clip_nonnegative", action="store_true")

    # Output mode; normally use the same flag used during training
    p.add_argument("--use_tdb_input", action="store_true", default=False)
    p.add_argument("--use_bai_input", action="store_true", default=False)
    p.add_argument("--bai_nom_weight", type=float, default=0.7)
    p.add_argument("--bai_nop_weight", type=float, default=0.3)

    # Model architecture; overwritten from hp.txt when available
    p.add_argument("--gcn_true", action="store_true", default=True)
    p.add_argument("--no_gcn", dest="gcn_true", action="store_false")
    p.add_argument("--buildA_true", action="store_true", default=True)
    p.add_argument("--static_graph_only", dest="buildA_true", action="store_false")
    p.add_argument("--gcn_depth", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.02)
    p.add_argument("--subgraph_size", type=int, default=10)
    p.add_argument("--node_dim", type=int, default=40)
    p.add_argument("--dilation_exponential", type=int, default=2)
    p.add_argument("--conv_channels", type=int, default=32)
    p.add_argument("--residual_channels", type=int, default=32)
    p.add_argument("--skip_channels", type=int, default=64)
    p.add_argument("--end_channels", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--propalpha", type=float, default=0.05)
    p.add_argument("--tanhalpha", type=float, default=3.0)
    p.add_argument("--graph_prior_weight", type=float, default=1.0)

    # Forecast generation
    p.add_argument(
        "--mc_runs",
        type=int,
        default=1,
        help="Use >1 for MC-dropout mean forecast. Plots still do not show CI shading.",
    )

    # Plot options
    p.add_argument(
        "--plot_start_year",
        type=int,
        default=2007,
        help="Start historical plotting from this year. Forecast still uses the full model input history.",
    )
    p.add_argument("--smooth_alpha", type=float, default=0.10)
    p.add_argument(
        "--normalize_scope",
        type=str,
        default="global",
        choices=["global", "panel", "none"],
        help="global = one max across all nodes; panel = per RMD plot max; none = no normalization",
    )
    p.add_argument(
        "--alarming_only",
        action="store_true",
        default=True,
        help="Only show PTs whose mean future trend is below the RMD mean future trend.",
    )
    p.add_argument(
        "--show_all_pts",
        dest="alarming_only",
        action="store_false",
        help="Show all linked PTs instead of only lower-than-RMD PTs.",
    )
    p.add_argument("--max_pts_per_plot", type=int, default=12)
    p.add_argument("--min_pts_per_plot", type=int, default=1)
    p.add_argument("--gap_shading", action="store_true", default=True)
    p.add_argument("--no_gap_shading", dest="gap_shading", action="store_false")
    p.add_argument("--forecast_boundary_line", action="store_true", default=True)
    p.add_argument("--plot_png", action="store_true", default=True)
    p.add_argument("--plot_pdf", action="store_true", default=True)
    p.add_argument(
        "--rmd_filter",
        type=str,
        default="",
        help="Optional case-insensitive substring or regex to plot only selected RMDs.",
    )

    # Gemini validation options
    p.add_argument("--gemini_api_key", type=str, default="",
                   help="Gemini API key for historical trend validation (overrides GEMINI_API_KEY env var)")
    p.add_argument("--no_gemini", action="store_true",
                   help="Skip Gemini API call; no anachronism corrections applied (Gemini is sole mechanism)")
    p.add_argument("--skip_validation", action="store_true",
                   help="Skip all historical validation (not recommended for production runs)")
    p.add_argument("--validation_report", type=str, default="",
                   help="Path to write the JSON validation report")
    p.add_argument("--cache_path", type=str, default="",
                   help="Path to validation cache JSON (default: data/validation_cache.json)")
    p.add_argument("--no_cache", action="store_true",
                   help="Disable cache; always call the Gemini API")

    return p.parse_args()


def _load_hp(hp_path: Path, args: argparse.Namespace) -> None:
    """Overwrite runtime/model args using hp.txt saved by train.py/train_test.py."""
    if not hp_path.exists():
        print(f"[forecast.py] hp.txt not found at {hp_path}; using CLI/default architecture.")
        return

    with open(hp_path, "r", encoding="utf-8") as f:
        hp = json.load(f)

    keys = [
        "gcn_true",
        "buildA_true",
        "gcn_depth",
        "dropout",
        "subgraph_size",
        "node_dim",
        "dilation_exponential",
        "conv_channels",
        "residual_channels",
        "skip_channels",
        "end_channels",
        "layers",
        "propalpha",
        "tanhalpha",
        "graph_prior_weight",
        "seq_in_len",
        "seq_out_len",
        "batch_size",
        "global_regex",
        "use_tdb_input",
        "use_bai_input",
        "bai_nom_weight",
        "bai_nop_weight",
    ]

    for k in keys:
        if k in hp:
            setattr(args, k, hp[k])

    if "no_log1p" in hp:
        args.no_log1p = bool(hp["no_log1p"])
    if "no_clip_nonnegative" in hp:
        args.no_clip_nonnegative = bool(hp["no_clip_nonnegative"])

    print(f"[forecast.py] Loaded architecture/data settings from {hp_path}")


# -----------------------------------------------------------------------------
# Name helpers
# -----------------------------------------------------------------------------

def clean_name(name: str) -> str:
    """Make RMD/PT names readable for plot legends."""
    s = str(name)

    s = re.sub(r"^RMD[_\-\s]+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^PT[_\-\s]+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[_\-\s]+NoM$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[_\-\s]+NoP$", "", s, flags=re.IGNORECASE)

    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()

    # Preserve common acronyms
    words = []
    for w in s.split():
        if w.upper() in {"AI", "MRI", "FMRI", "EEG", "VR", "AR", "NLP", "DBS", "TMS", "ECT"}:
            words.append(w.upper())
        elif len(w) <= 3 and w.isupper():
            words.append(w)
        else:
            words.append(w[:1].upper() + w[1:].lower())
    return " ".join(words)


def safe_file_name(name: str, max_len: int = 140) -> str:
    s = clean_name(name)
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] if s else "plot"


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def make_model(args: argparse.Namespace, data: RMDPTData, device: torch.device) -> gtnet:
    out_channels = 1 if bool(args.use_tdb_input) else 2

    model = gtnet(
        gcn_true=bool(args.gcn_true),
        buildA_true=bool(args.buildA_true),
        gcn_depth=int(args.gcn_depth),
        num_nodes=int(data.schema.node_count),
        device=device,
        predefined_A=data.adj,
        dropout=float(args.dropout),
        subgraph_size=min(int(args.subgraph_size), int(data.schema.node_count)),
        node_dim=int(args.node_dim),
        dilation_exponential=int(args.dilation_exponential),
        conv_channels=int(args.conv_channels),
        residual_channels=int(args.residual_channels),
        skip_channels=int(args.skip_channels),
        end_channels=int(args.end_channels),
        seq_length=int(args.seq_in_len),
        in_dim=len(data.schema.input_channels),
        out_dim=int(args.seq_out_len) * out_channels,
        out_channels=out_channels,
        layers=int(args.layers),
        propalpha=float(args.propalpha),
        tanhalpha=float(args.tanhalpha),
        graph_prior_weight=float(args.graph_prior_weight),
    ).to(device)

    return model


def torch_load_compatible(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(model_path: Path, args: argparse.Namespace, data: RMDPTData, device: torch.device) -> torch.nn.Module:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    state = torch_load_compatible(model_path, device)

    # Older workflow may have saved the full model object.
    if isinstance(state, torch.nn.Module):
        print(f"[forecast.py] Loaded full model object from {model_path}")
        return state.to(device)

    model = make_model(args, data, device)

    # New workflow saves state_dict directly.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise TypeError(f"Unsupported model file format at {model_path}")

    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        print("[forecast.py] Strict model loading failed. Trying strict=False.")
        print(f"[forecast.py] Strict-load error: {exc}")
        model.load_state_dict(state, strict=False)

    print(f"[forecast.py] Loaded model weights from {model_path}")
    return model


# -----------------------------------------------------------------------------
# Forecast helpers
# -----------------------------------------------------------------------------

def forecast_scaled(
    model: torch.nn.Module,
    data: RMDPTData,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return mean scaled forecast as [H, C, N] and optional MC samples [R, H, C, N]."""
    x = data.last_input_window().to(device)
    out_channels = 1 if bool(args.use_tdb_input) else 2

    mc_runs = max(1, int(args.mc_runs))
    samples = []

    if mc_runs == 1:
        model.eval()
        with torch.no_grad():
            raw = model(x)
            pred = reshape_model_output(
                raw,
                horizon=int(args.seq_out_len),
                out_channels=out_channels,
                node_count=int(data.schema.node_count),
            )
            mean_scaled = pred[0].detach().cpu().numpy()
        return mean_scaled, None

    # MC dropout mean forecast; no CI shading is plotted.
    model.train()
    with torch.no_grad():
        for _ in range(mc_runs):
            raw = model(x)
            pred = reshape_model_output(
                raw,
                horizon=int(args.seq_out_len),
                out_channels=out_channels,
                node_count=int(data.schema.node_count),
            )
            samples.append(pred[0].detach().cpu().numpy())

    arr = np.stack(samples, axis=0)
    mean_scaled = arr.mean(axis=0)
    return mean_scaled, arr


def get_history_scaled_for_plot(data: RMDPTData, args: argparse.Namespace) -> np.ndarray:
    """
    Return historical plot channel as [T, N].

    In TDB mode:
      plot target = scaled NoM + scaled NoP

    In normal mode:
      plot target = scaled NoM
    """
    if bool(args.use_tdb_input):
        hist = data.nom_scaled + data.nop_scaled
    else:
        hist = data.nom_scaled

    hist = np.asarray(hist, dtype=float)
    hist = np.clip(hist, 0.0, None)
    return hist


def get_forecast_scaled_for_plot(mean_scaled: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """
    Return future plot channel as [H, N].

    mean_scaled shape: [H, C, N]
    Channel 0 is TDB in TDB mode, otherwise NoM.
    """
    fut = np.asarray(mean_scaled[:, 0, :], dtype=float)
    fut = np.clip(fut, 0.0, None)
    return fut


def exponential_smoothing_1d(series: np.ndarray, alpha: float) -> np.ndarray:
    x = np.asarray(series, dtype=float)
    if x.size == 0:
        return x
    alpha = float(alpha)
    if alpha <= 0:
        return x.copy()
    if alpha >= 1:
        return x.copy()

    out = np.zeros_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def smooth_matrix(values: np.ndarray, alpha: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.zeros_like(arr, dtype=float)
    for j in range(arr.shape[1]):
        out[:, j] = exponential_smoothing_1d(arr[:, j], alpha)
    return out


def normalize_values(
    values: np.ndarray,
    scope: str,
    panel_indices: Optional[List[int]] = None,
    global_max: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    arr = np.asarray(values, dtype=float)

    if scope == "none":
        return arr.copy(), 1.0

    if scope == "global":
        denom = float(global_max) if global_max is not None else float(np.nanmax(arr))
    elif scope == "panel":
        if panel_indices:
            denom = float(np.nanmax(arr[:, panel_indices]))
        else:
            denom = float(np.nanmax(arr))
    else:
        raise ValueError(f"Invalid normalization scope: {scope}")

    if not np.isfinite(denom) or denom <= 1e-12:
        denom = 1.0

    return arr / denom, denom


# -----------------------------------------------------------------------------
# Graph helpers
# -----------------------------------------------------------------------------

def get_rmd_pt_links(data: RMDPTData) -> Dict[int, List[int]]:
    """Use graph_links built by RMDPTData. Only keeps valid RMD->PT links."""
    links: Dict[int, List[int]] = {}
    for rmd_i, pt_list in data.graph_links.items():
        clean = []
        for pt_i in pt_list:
            if (
                isinstance(pt_i, int)
                and 0 <= pt_i < data.schema.node_count
                and data.schema.entity_types[pt_i] == "PT"
            ):
                clean.append(pt_i)
        if clean:
            links[int(rmd_i)] = clean
    return links


def select_pts_for_plot(
    rmd_i: int,
    pt_indices: List[int],
    future_norm: np.ndarray,
    alarming_only: bool,
    max_pts: int,
    min_pts: int,
) -> List[int]:
    """Select PTs for the plot, optionally only PTs below the RMD forecast trend."""
    if not pt_indices:
        return []

    rmd_future = future_norm[:, rmd_i]
    rmd_mean = float(np.nanmean(rmd_future))

    rows = []
    for pt_i in pt_indices:
        pt_future = future_norm[:, pt_i]
        pt_mean = float(np.nanmean(pt_future))
        gap = rmd_mean - pt_mean
        rows.append((gap, pt_mean, pt_i))

    # Larger positive gap first
    rows.sort(key=lambda x: x[0], reverse=True)

    if alarming_only:
        chosen = [pt_i for gap, _, pt_i in rows if gap > 0]
    else:
        chosen = [pt_i for _, _, pt_i in rows]

    # Backfill if too few PTs were lower than RMD
    if len(chosen) < min_pts:
        for _, _, pt_i in rows:
            if pt_i not in chosen:
                chosen.append(pt_i)
            if len(chosen) >= min_pts:
                break

    return chosen[: max(1, int(max_pts))]


# -----------------------------------------------------------------------------
# Saving outputs
# -----------------------------------------------------------------------------

def save_forecast_csv(
    path: Path,
    future_dates: pd.DatetimeIndex,
    mean_scaled: np.ndarray,
    mean_raw: np.ndarray,
    data: RMDPTData,
    args: argparse.Namespace,
) -> None:
    """Save future forecast in long format."""
    ensure_dir(path.parent)

    rows = []
    channels = ["TDB"] if bool(args.use_tdb_input) else ["NoM", "NoP"]

    for t, d in enumerate(future_dates):
        for ch_i, ch_name in enumerate(channels):
            for node_i, entity in enumerate(data.schema.entity_names):
                rows.append(
                    {
                        "Date": pd.Timestamp(d).strftime("%Y-%m-%d"),
                        "Entity": entity,
                        "Display_Name": clean_name(entity),
                        "Entity_Type": data.schema.entity_types[node_i],
                        "Channel": ch_name,
                        "Forecast_Scaled": float(mean_scaled[t, ch_i, node_i]),
                        "Forecast_Raw": float(mean_raw[t, ch_i, node_i]),
                    }
                )

    pd.DataFrame(rows).to_csv(path, index=False)


def save_plot_values_csv(
    path: Path,
    dates_all: pd.DatetimeIndex,
    values_norm_smoothed: np.ndarray,
    hist_len: int,
    data: RMDPTData,
) -> None:
    ensure_dir(path.parent)

    rows = []
    for t, d in enumerate(dates_all):
        period = "Historical" if t < hist_len else "Forecast"
        for node_i, entity in enumerate(data.schema.entity_names):
            rows.append(
                {
                    "Date": pd.Timestamp(d).strftime("%Y-%m-%d"),
                    "Period": period,
                    "Entity": entity,
                    "Display_Name": clean_name(entity),
                    "Entity_Type": data.schema.entity_types[node_i],
                    "Trend_For_Plot": float(values_norm_smoothed[t, node_i]),
                }
            )

    pd.DataFrame(rows).to_csv(path, index=False)


def save_gap_csv(
    path: Path,
    future_dates: pd.DatetimeIndex,
    future_norm_smoothed: np.ndarray,
    links: Dict[int, List[int]],
    data: RMDPTData,
) -> None:
    ensure_dir(path.parent)

    rows = []
    for rmd_i, pt_list in links.items():
        for pt_i in pt_list:
            for t, d in enumerate(future_dates):
                rmd_val = float(future_norm_smoothed[t, rmd_i])
                pt_val = float(future_norm_smoothed[t, pt_i])
                rows.append(
                    {
                        "Date": pd.Timestamp(d).strftime("%Y-%m-%d"),
                        "RMD": data.schema.entity_names[rmd_i],
                        "RMD_Display": clean_name(data.schema.entity_names[rmd_i]),
                        "PT": data.schema.entity_names[pt_i],
                        "PT_Display": clean_name(data.schema.entity_names[pt_i]),
                        "RMD_Trend": rmd_val,
                        "PT_Trend": pt_val,
                        "Gap_RMD_minus_PT": rmd_val - pt_val,
                        "PT_Below_RMD": bool(pt_val < rmd_val),
                    }
                )

    pd.DataFrame(rows).to_csv(path, index=False)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def apply_reference_style(ax: plt.Axes, fig: plt.Figure) -> None:
    ax.set_facecolor("#EAEAF2")
    fig.patch.set_facecolor("white")

    ax.grid(True, color="white", linewidth=0.8, alpha=0.95)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#C8C8C8")

    ax.tick_params(axis="both", length=3, width=0.6, colors="black")


def plot_one_rmd(
    rmd_i: int,
    pt_indices: List[int],
    dates_hist: pd.DatetimeIndex,
    dates_future: pd.DatetimeIndex,
    hist_norm_smoothed: np.ndarray,
    future_norm_smoothed: np.ndarray,
    data: RMDPTData,
    out_plot_dir: Path,
    args: argparse.Namespace,
) -> Optional[Dict[str, object]]:
    if not pt_indices:
        return None

    rmd_name = data.schema.entity_names[rmd_i]
    rmd_label = clean_name(rmd_name)

    fig, ax = plt.subplots(figsize=(10.0, 6.6))
    apply_reference_style(ax, fig)

    # Give space for right-side legend
    fig.subplots_adjust(left=0.09, right=0.73, bottom=0.18, top=0.88)

    # Historical and future dates
    last_hist_date = dates_hist[-1]
    forecast_line_dates = pd.DatetimeIndex([last_hist_date]).append(dates_future)

    # Plot RMD first
    color = PALETTE[0]
    ax.plot(
        dates_hist,
        hist_norm_smoothed[:, rmd_i],
        color=color,
        linewidth=2.0,
        label=rmd_label,
        zorder=5,
    )
    rmd_future_line = np.concatenate(
        [[hist_norm_smoothed[-1, rmd_i]], future_norm_smoothed[:, rmd_i]]
    )
    ax.plot(
        forecast_line_dates,
        rmd_future_line,
        color=color,
        linewidth=2.0,
        zorder=6,
    )

    # Optional forecast boundary
    if bool(args.forecast_boundary_line):
        ax.axvline(
            dates_future[0],
            color="#444444",
            linewidth=0.8,
            linestyle="--",
            alpha=0.45,
            zorder=1,
        )

    # Plot PTs
    plotted = [rmd_i]
    for k, pt_i in enumerate(pt_indices, start=1):
        color = PALETTE[k % len(PALETTE)]
        pt_label = clean_name(data.schema.entity_names[pt_i])

        ax.plot(
            dates_hist,
            hist_norm_smoothed[:, pt_i],
            color=color,
            linewidth=1.15,
            label=pt_label,
            zorder=3,
        )

        pt_future_line = np.concatenate(
            [[hist_norm_smoothed[-1, pt_i]], future_norm_smoothed[:, pt_i]]
        )
        ax.plot(
            forecast_line_dates,
            pt_future_line,
            color=color,
            linewidth=1.15,
            zorder=4,
        )

        # Gap shading only in forecast period, no CI band
        if bool(args.gap_shading):
            rmd_f = future_norm_smoothed[:, rmd_i]
            pt_f = future_norm_smoothed[:, pt_i]
            mask = rmd_f > pt_f
            if np.any(mask):
                ax.fill_between(
                    dates_future,
                    pt_f,
                    rmd_f,
                    where=mask,
                    interpolate=True,
                    color=color,
                    alpha=0.13,
                    linewidth=0.0,
                    zorder=2,
                )

        plotted.append(pt_i)

    # Axis formatting
    ax.set_title(rmd_label, fontsize=13, pad=10)
    ax.set_ylabel("Trend", fontsize=10)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylim(bottom=0)

    y_max = np.nanmax(
        np.concatenate(
            [
                hist_norm_smoothed[:, plotted].reshape(-1),
                future_norm_smoothed[:, plotted].reshape(-1),
            ]
        )
    )
    if np.isfinite(y_max) and y_max > 0:
        ax.set_ylim(0, y_max * 1.12)

    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.02),
        frameon=False,
        borderaxespad=0.0,
        handlelength=2.0,
    )

    safe = safe_file_name(rmd_name)
    png_path = out_plot_dir / f"{safe}.png"
    pdf_path = out_plot_dir / f"{safe}.pdf"

    if bool(args.plot_png):
        fig.savefig(png_path, bbox_inches="tight")
    if bool(args.plot_pdf):
        fig.savefig(pdf_path, bbox_inches="tight", format="pdf")

    plt.close(fig)

    return {
        "rmd_index": rmd_i,
        "rmd_name": rmd_name,
        "pt_indices": pt_indices,
        "png": str(png_path),
        "pdf": str(pdf_path),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    set_random_seed(int(args.seed))

    if int(args.num_threads) > 0:
        torch.set_num_threads(int(args.num_threads))

    # Resolve nodes_csv default: prefer validated CSV if it exists
    if args.nodes_csv is None:
        _validated = Path("data/data_validated.csv")
        args.nodes_csv = str(_validated) if _validated.exists() else "data/data.csv"
        print(f"[forecast.py] nodes_csv auto-resolved: {args.nodes_csv}")

    output_dir = Path(args.output_dir)
    forecast_dir = output_dir / "forecast"
    plot_dir = forecast_dir / "plots"
    ensure_dir(forecast_dir)
    ensure_dir(plot_dir)

    # hp.txt must be loaded before data/model construction
    _load_hp(output_dir / "hp.txt", args)

    device = pick_device(str(args.device))

    print("=" * 72)
    print("BC-MTGNN forecast")
    print("=" * 72)
    print(f"[forecast.py] device          : {device}")
    print(f"[forecast.py] model_path      : {args.model_path}")
    print(f"[forecast.py] data_csv        : {args.data_csv}")
    print(f"[forecast.py] nodes_csv       : {args.nodes_csv}")
    print(f"[forecast.py] graph_csv       : {args.graph_csv}")
    print(f"[forecast.py] output_dir      : {args.output_dir}")
    print(f"[forecast.py] seq_in_len      : {args.seq_in_len}")
    print(f"[forecast.py] seq_out_len     : {args.seq_out_len}")
    print(f"[forecast.py] use_tdb_input   : {args.use_tdb_input}")
    print(f"[forecast.py] use_bai_input   : {args.use_bai_input}")
    print(f"[forecast.py] mc_runs         : {args.mc_runs}")
    print(f"[forecast.py] CI shading      : disabled")
    print(f"[forecast.py] gap shading     : {args.gap_shading}")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Gemini validation: audit historical PT data in nodes_csv (data.csv)
    # for anachronistic values and data duplication before plotting.
    # This corrects the raw display data used for legend labels / history
    # lines without altering the model weights or sm_data.csv.
    # ------------------------------------------------------------------
    import os as _os
    validated_nodes_csv = str(args.nodes_csv)
    if not args.skip_validation:
        try:
            from gemini_validator import validate_and_correct
            _raw_df = pd.read_csv(str(args.nodes_csv))
            _api_key = (
                getattr(args, "gemini_api_key", "")
                or _os.environ.get("GEMINI_API_KEY", "")
                or None
            )
            _no_cache = getattr(args, "no_cache", False)
            _cp = getattr(args, "cache_path", "")
            _cache_path = None if _no_cache else (Path(_cp) if _cp else None)
            _corrected_df, _val_report = validate_and_correct(
                _raw_df,
                api_key=_api_key,
                use_gemini=not getattr(args, "no_gemini", False),
                verbose=True,
                cache_path=_cache_path,
            )
            # Write corrected nodes_csv to a temp path alongside the original
            _validated_path = Path(str(args.nodes_csv)).with_suffix(".validated.csv")
            _corrected_df.to_csv(_validated_path, index=False)
            validated_nodes_csv = str(_validated_path)
            print(f"[forecast.py] Validated nodes_csv written to {_validated_path}")
            if getattr(args, "validation_report", ""):
                import json as _json
                Path(args.validation_report).parent.mkdir(parents=True, exist_ok=True)
                Path(args.validation_report).write_text(_json.dumps(_val_report, indent=2))
                print(f"[forecast.py] Validation report saved to {args.validation_report}")
        except ImportError:
            print("[forecast.py] WARNING: gemini_validator.py not found — skipping validation.")
    else:
        print("[forecast.py] Gemini validation skipped (--skip_validation).")

    # Use the same production split setup as train.py so scaling stays aligned.
    data = RMDPTData(
        data_csv=str(args.data_csv),
        graph_csv=str(args.graph_csv),
        nodes_csv=validated_nodes_csv,
        seq_in_len=int(args.seq_in_len),
        seq_out_len=int(args.seq_out_len),
        train_ratio=0.98,
        valid_ratio=0.01,
        batch_size=int(args.batch_size),
        device=device,
        global_regex=str(args.global_regex),
        log1p=not bool(args.no_log1p),
        clip_nonnegative=not bool(args.no_clip_nonnegative),
        use_tdb=bool(args.use_tdb_input),
        use_bai=bool(args.use_bai_input),
        bai_nom_weight=float(args.bai_nom_weight),
        bai_nop_weight=float(args.bai_nop_weight),
    )

    print(
        f"[forecast.py] rows={len(data.raw_df)} | nodes={data.schema.node_count} | "
        f"RMD nodes={len(data.schema.rmd_nom_indices)} | PT nodes={len(data.schema.pt_nom_indices)} | "
        f"input_channels={len(data.schema.input_channels)}"
    )

    model = load_model(Path(args.model_path), args, data, device)

    mean_scaled, mc_samples_scaled = forecast_scaled(model, data, args, device)

    # Inverse to raw units for saved CSV only.
    mean_raw = data.inverse_targets_array(mean_scaled[None, ...])[0]

    future_dates = data.future_dates()
    dates_hist = pd.DatetimeIndex(data.dates)
    dates_all = dates_hist.append(future_dates)
    # Plot display starts from selected year, but model forecast still uses full history
    plot_start_mask = dates_hist.year >= int(args.plot_start_year)

    if not np.any(plot_start_mask):
        raise ValueError(
            f"No historical dates found from plot_start_year={args.plot_start_year}. "
            f"Available range: {dates_hist.min()} to {dates_hist.max()}"
        )

    dates_hist_plot = dates_hist[plot_start_mask]

    # Historical and future plot channel in model-scaled space
    hist_plot = get_history_scaled_for_plot(data, args)              # [T, N]
    future_plot = get_forecast_scaled_for_plot(mean_scaled, args)    # [H, N]

    combined_plot = np.vstack([hist_plot, future_plot])
    combined_plot = np.clip(combined_plot, 0.0, None)

    # Normalize
    global_max = float(np.nanmax(combined_plot)) if combined_plot.size else 1.0
    if not np.isfinite(global_max) or global_max <= 1e-12:
        global_max = 1.0

    if args.normalize_scope == "global":
        combined_norm, denom = normalize_values(
            combined_plot,
            scope="global",
            global_max=global_max,
        )
        print(f"[forecast.py] Global trend normalization denominator: {denom:.6g}")
    elif args.normalize_scope == "none":
        combined_norm, _ = normalize_values(combined_plot, scope="none")
        print("[forecast.py] Trend normalization disabled.")
    else:
        # Panel normalization is applied inside selection/plot preparation.
        combined_norm = combined_plot.copy()
        print("[forecast.py] Panel-level trend normalization enabled.")

    # Smooth only for visualization
    combined_norm_smoothed = smooth_matrix(combined_norm, alpha=float(args.smooth_alpha))
    hist_norm_smoothed = combined_norm_smoothed[: len(dates_hist), :]
    future_norm_smoothed = combined_norm_smoothed[len(dates_hist) :, :]

    links = get_rmd_pt_links(data)

    # Save output files
    save_forecast_csv(
        forecast_dir / f"forecast_{int(args.seq_out_len)}m.csv",
        future_dates=future_dates,
        mean_scaled=mean_scaled,
        mean_raw=mean_raw,
        data=data,
        args=args,
    )

    save_plot_values_csv(
        forecast_dir / "plot_values_monthly.csv",
        dates_all=dates_all,
        values_norm_smoothed=combined_norm_smoothed,
        hist_len=len(dates_hist),
        data=data,
    )

    save_gap_csv(
        forecast_dir / "gap_monthly.csv",
        future_dates=future_dates,
        future_norm_smoothed=future_norm_smoothed,
        links=links,
        data=data,
    )

    if mc_samples_scaled is not None:
        np.save(forecast_dir / f"mc_samples_scaled_{int(args.seq_out_len)}m.npy", mc_samples_scaled)
        print(f"[forecast.py] MC samples saved: {forecast_dir / f'mc_samples_scaled_{int(args.seq_out_len)}m.npy'}")

    # Plot RMD groups
    plotted = []
    skipped_no_links = 0
    skipped_filter = 0

    for rmd_i in data.schema.rmd_nom_indices:
        rmd_name = data.schema.entity_names[rmd_i]

        if args.rmd_filter:
            try:
                if not re.search(args.rmd_filter, rmd_name, flags=re.IGNORECASE):
                    skipped_filter += 1
                    continue
            except re.error:
                if args.rmd_filter.lower() not in rmd_name.lower():
                    skipped_filter += 1
                    continue

        pt_candidates = links.get(rmd_i, [])
        if not pt_candidates:
            skipped_no_links += 1
            continue

        if args.normalize_scope == "panel":
            panel_indices = [rmd_i] + pt_candidates
            panel_combined_norm, panel_denom = normalize_values(
                combined_plot,
                scope="panel",
                panel_indices=panel_indices,
            )
            panel_smoothed = smooth_matrix(panel_combined_norm, alpha=float(args.smooth_alpha))
            panel_hist = panel_smoothed[: len(dates_hist), :]
            panel_future = panel_smoothed[len(dates_hist) :, :]
        else:
            panel_hist = hist_norm_smoothed
            panel_future = future_norm_smoothed

        selected_pts = select_pts_for_plot(
            rmd_i=rmd_i,
            pt_indices=pt_candidates,
            future_norm=panel_future,
            alarming_only=bool(args.alarming_only),
            max_pts=int(args.max_pts_per_plot),
            min_pts=int(args.min_pts_per_plot),
        )

        if not selected_pts:
            skipped_no_links += 1
            continue

        info = plot_one_rmd(
            rmd_i=rmd_i,
            pt_indices=selected_pts,
            dates_hist=dates_hist_plot,
            dates_future=future_dates,
            hist_norm_smoothed=panel_hist[plot_start_mask, :],
            future_norm_smoothed=panel_future,
            data=data,
            out_plot_dir=plot_dir,
            args=args,
        )
        if info is not None:
            plotted.append(info)

    # Save plotting manifest
    manifest_rows = []
    for item in plotted:
        manifest_rows.append(
            {
                "RMD": item["rmd_name"],
                "RMD_Display": clean_name(str(item["rmd_name"])),
                "PT_Count_Plotted": len(item["pt_indices"]),
                "PTs_Plotted": "|".join(data.schema.entity_names[i] for i in item["pt_indices"]),
                "PNG": item["png"],
                "PDF": item["pdf"],
            }
        )
    pd.DataFrame(manifest_rows).to_csv(forecast_dir / "plot_manifest.csv", index=False)

    print("=" * 72)
    print("[forecast.py] Forecast completed.")
    print(f"[forecast.py] Forecast CSV       : {forecast_dir / f'forecast_{int(args.seq_out_len)}m.csv'}")
    print(f"[forecast.py] Plot values CSV    : {forecast_dir / 'plot_values_monthly.csv'}")
    print(f"[forecast.py] Gap CSV            : {forecast_dir / 'gap_monthly.csv'}")
    print(f"[forecast.py] Plot manifest      : {forecast_dir / 'plot_manifest.csv'}")
    print(f"[forecast.py] Plot directory     : {plot_dir}")
    print(f"[forecast.py] RMD plots created  : {len(plotted)}")
    print(f"[forecast.py] Skipped no links   : {skipped_no_links}")
    print(f"[forecast.py] Skipped by filter  : {skipped_filter}")
    print("=" * 72)


if __name__ == "__main__":
    main()