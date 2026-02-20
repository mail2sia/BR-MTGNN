#!/usr/bin/env python3
"""
Generate grouped forecast plots (RMD + pertinent solutions) using the graph CSV.
Historical: 2004-2025 from data/sm_data_g.csv
Forecast: 2026-2028 from model/Bayesian/forecast/forecast_2026_2028*.csv
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def exponential_smoothing(series: np.ndarray, alpha: float) -> np.ndarray:
    result = [series[0]]
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1.0 - alpha) * result[n - 1])
    return np.stack(result, axis=0)


def _clip_spikes(values: np.ndarray, lower_q: float = 0.01, upper_q: float = 0.99) -> np.ndarray:
    if len(values) == 0:
        return values
    lo = np.quantile(values, lower_q)
    hi = np.quantile(values, upper_q)
    if lo == hi:
        return values
    return np.clip(values, lo, hi)


def _median_filter(values: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return values
    half = window // 2
    padded = np.pad(values, (half, window - 1 - half), mode="edge")
    out = np.empty_like(values, dtype=float)
    for i in range(len(values)):
        out[i] = np.median(padded[i : i + window])
    return out


def _smooth_series(values: np.ndarray, window: int = 17, median_window: int = 5) -> np.ndarray:
    clipped = _clip_spikes(values)
    filtered = _median_filter(clipped, window=median_window)
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(filtered, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _smooth_forecast_with_anchor(history: np.ndarray, forecast: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    if len(forecast) == 0:
        return forecast
    out = np.empty_like(forecast, dtype=float)
    prev = history[-1] if len(history) else forecast[0]
    for i, val in enumerate(forecast):
        prev = alpha * val + (1 - alpha) * prev
        out[i] = prev
    if len(history):
        offset = history[-1] - out[0]
        out = out + offset
    return out


def _smooth_forecast_series(values: np.ndarray, window: int = 21, median_window: int = 7) -> np.ndarray:
    if len(values) == 0:
        return values
    return _smooth_series(values, window=window, median_window=median_window)


def _bridge_hist_forecast(hist_s: np.ndarray, fut_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(fut_s) == 0:
        return hist_s, fut_s
    hist_with_bridge = np.concatenate([hist_s, fut_s[:1]], axis=0)
    return hist_with_bridge, fut_s


def _draw_join_connector(
    ax, x_hist_last: float, y_hist_last: float, x_fut_first: float, y_fut_first: float
) -> None:
    ax.plot(
        [x_hist_last, x_fut_first],
        [y_hist_last, y_fut_first],
        linestyle="--",
        color="0.45",
        linewidth=1.2,
        alpha=0.8,
        zorder=3,
    )


def _smooth_join_window(
    hist_s: np.ndarray, fut_s: np.ndarray, window: int = 11
) -> tuple[np.ndarray, np.ndarray]:
    if len(hist_s) == 0 or len(fut_s) == 0 or window <= 1:
        return hist_s, fut_s
    if window % 2 == 0:
        window += 1

    combined = np.concatenate([hist_s, fut_s], axis=0)
    half = window // 2
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(combined, (half, half), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")

    boundary = len(hist_s) - 1
    start = max(0, boundary - half)
    end = min(len(combined) - 1, boundary + half)

    combined[start : end + 1] = smoothed[start : end + 1]
    # Ensure perfect join: make first forecast point same as last history point
    boundary_val = combined[len(hist_s)]
    return combined[: len(hist_s)], combined[len(hist_s) - 1 :]


def _blend_forecast_start(history: np.ndarray, forecast: np.ndarray, months: int = 4) -> np.ndarray:
    if len(history) == 0 or len(forecast) == 0 or months <= 0:
        return forecast
    out = forecast.astype(float, copy=True)
    m = min(int(months), len(out))
    start = float(history[-1])
    for i in range(m):
        w = float(i + 1) / float(m + 1)
        out[i] = (1.0 - w) * start + w * out[i]
    return out


def _align_hist_plot(hist_plot: np.ndarray, x_hist: np.ndarray) -> np.ndarray:
    if len(hist_plot) == len(x_hist):
        return hist_plot
    if len(hist_plot) < len(x_hist):
        pad = np.full(len(x_hist) - len(hist_plot), hist_plot[-1], dtype=float)
        return np.concatenate([hist_plot, pad], axis=0)
    return hist_plot[: len(x_hist)]


def _positive_gap_mask(y_top: np.ndarray, y_bot: np.ndarray, min_months: int) -> np.ndarray:
    gap_pos = (y_top - y_bot) > 0
    if min_months <= 1:
        return gap_pos

    keep = np.zeros_like(gap_pos, dtype=bool)
    run_start = None
    for i, v in enumerate(gap_pos):
        if v and run_start is None:
            run_start = i
        if (not v or i == len(gap_pos) - 1) and run_start is not None:
            run_end = i if not v else i + 1
            if (run_end - run_start) >= min_months:
                keep[run_start:run_end] = True
            run_start = None
    return keep


def _fill_between_masked(ax, x, y1, y2, mask, **kwargs) -> None:
    y1m = np.where(mask, y1, np.nan)
    y2m = np.where(mask, y2, np.nan)
    ax.fill_between(x, y1m, y2m, **kwargs)


def _draw_gap_shading(ax, x, y1, y2, mask, **kwargs) -> None:
    y1m = np.where(mask, y1, np.nan)
    y2m = np.where(mask, y2, np.nan)
    ax.fill_between(x, y1m, y2m, **kwargs)


def _draw_ci_band(ax, x, y, ci, color, alpha=0.15) -> None:
    # CI ribbons intentionally disabled — drawing handled without confidence intervals
    return


def consistent_name(name: str) -> str:
    name = (
        name.replace("RMD", "")
        .replace("RMD_", "")
        .replace("PT_", "")
        .lstrip("_ ")
    )

    if "HIDDEN MARKOV MODEL" in name:
        return "Statistical HMM"

    if name in {"CAPTCHA", "DNSSEC", "RRAM"}:
        return name

    if "IZ" in name:
        name = name.replace("IZ", "IS")
    if "IOR" in name:
        name = name.replace("IOR", "IOUR")

    if not name.isupper():
        words = name.split(" ")
        result = ""
        for i, word in enumerate(words):
            if len(word) <= 2:
                result += word
            else:
                result += word[0].upper() + word[1:]
            if i < len(words) - 1:
                result += " "
        return result

    words = name.split(" ")
    result = ""
    for i, word in enumerate(words):
        if len(word) <= 3 or "/" in word or word in {"MITM", "SIEM"}:
            result += word
        else:
            result += word[0] + word[1:].lower()
        if i < len(words) - 1:
            result += " "
    return result


# Optional overrides for specific solution display names (post-normalization).
# Keys must match the output of consistent_name().
SOLUTION_COLOR_OVERRIDES: dict[str, str] = {
    "Cognitive Behavioral Therapy": "#1f77b4",
    "Mindfulness-Based Stress Reduction": "#ff7f0e",
    "Selective Serotonin Reuptake Inhibitors": "#2ca02c",
    "Brain-Computer Interface": "#d62728",
    "Neuroprotective Agents": "#9467bd",
    "Exposure Therapy": "#bcbd22",
}

# Gap shading overrides only (line colors remain unchanged).
SOLUTION_GAP_OVERRIDES: dict[str, str] = {
    "Neuromodulation Techniques": "#5EA3D9",
    "Neuroprotective Agents": "#D86A9A",
    "Eye Movement Desensitization And Reprocessing": "#38B48F",
    "Genetic Testing": "#D9B84A",
    "Antidepressants": "#c27ba0",
}


def _build_fixed_palette() -> list[str]:
    from matplotlib import cm

    palettes = ["tab20", "tab20b", "tab20c"]
    colors: list[str] = []
    for name in palettes:
        cmap = cm.get_cmap(name)
        entries = [cmap(i / 19.0) for i in range(20)]
        colors.extend(mcolors.to_hex(c) for c in entries)
    return colors


def _darken_color(col: str, amount: float = 0.25) -> str:
    """Return a darker hex color by blending `col` toward black by `amount` (0-1)."""
    try:
        rgb = mcolors.to_rgb(col)
    except Exception:
        rgb = (0.5, 0.5, 0.5)
    r, g, b = rgb
    r2 = max(0.0, r * (1.0 - amount))
    g2 = max(0.0, g * (1.0 - amount))
    b2 = max(0.0, b * (1.0 - amount))
    return mcolors.to_hex((r2, g2, b2))


def load_solution_color_map(nodes_path: Path) -> dict[str, str]:
    if not nodes_path.exists():
        return {}

    palette = _build_fixed_palette()
    color_map: dict[str, str] = {}
    with nodes_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            token = (row.get("token") or "").strip()
            category = (row.get("category") or "").strip()
            if not token:
                continue
            if category != "PT" and not token.startswith("PT_"):
                continue
            key = consistent_name(token)
            if key in color_map:
                continue
            color_map[key] = palette[len(color_map) % len(palette)]

    color_map.update(SOLUTION_COLOR_OVERRIDES)
    return color_map


def build_graph(file_name: Path) -> dict[str, list[str]]:
    graph: dict[str, list[str]] = defaultdict(list)
    with file_name.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            key_node = row[0]
            adjacent_nodes = [node for node in row[1:] if node]
            graph[key_node].extend(adjacent_nodes)
    print("Graph loaded with", len(graph), "RMDs...")
    return graph


def zero_negative_curves(data: np.ndarray, forecast: np.ndarray, indices: list[int]) -> None:
    for idx in indices:
        data[:, idx] = np.maximum(data[:, idx], 0.0)
        forecast[:, idx] = np.maximum(forecast[:, idx], 0.0)


def get_closest_curve_larger(
    c: np.ndarray,
    forecast: np.ndarray,
    confidence: np.ndarray,
    RMD: str,
    solutions: list[str],
    col: list[str],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    d = float("inf")
    cc = None
    cc_conf = None
    for j in range(forecast.shape[1]):
        name = col[j]
        if name not in solutions and name != RMD:
            continue
        f = forecast[:, j]
        f_conf = confidence[:, j]
        if float(np.mean(f)) <= float(np.mean(c)):
            continue
        diff = float(np.mean(f) - np.mean(c))
        if diff < d:
            d = diff
            cc = f.copy()
            cc_conf = f_conf.copy()
    return cc, cc_conf


def get_closest_curve_smaller(
    c: np.ndarray,
    forecast: np.ndarray,
    confidence: np.ndarray,
    RMD: str,
    solutions: list[str],
    col: list[str],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    d = float("inf")
    cc = None
    cc_conf = None
    for j in range(forecast.shape[1]):
        name = col[j]
        if name not in solutions and name != RMD:
            continue
        f = forecast[:, j]
        f_conf = confidence[:, j]
        if float(np.mean(f)) >= float(np.mean(c)):
            continue
        diff = float(np.abs(np.mean(f) - np.mean(c)))
        if diff < d:
            d = diff
            cc = f.copy()
            cc_conf = f_conf.copy()
    return cc, cc_conf


def get_closest_curve_larger_smoothed(
    c_fut: np.ndarray,
    fut_sm: dict[str, np.ndarray],
    conf_sm: dict[str, np.ndarray],
    RMD: str,
    solutions: list[str],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    d = float("inf")
    cc = None
    cc_conf = None
    for name in [RMD] + list(solutions):
        if name not in fut_sm:
            continue
        f = fut_sm[name]
        f_conf = conf_sm[name]
        if float(np.mean(f)) <= float(np.mean(c_fut)):
            continue
        diff = float(np.mean(f) - float(np.mean(c_fut)))
        if diff < d:
            d = diff
            cc = f.copy()
            cc_conf = f_conf.copy()
    return cc, cc_conf


def get_closest_curve_smaller_smoothed(
    c_fut: np.ndarray,
    fut_sm: dict[str, np.ndarray],
    conf_sm: dict[str, np.ndarray],
    RMD: str,
    solutions: list[str],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    d = float("inf")
    cc = None
    cc_conf = None
    for name in [RMD] + list(solutions):
        if name not in fut_sm:
            continue
        f = fut_sm[name]
        f_conf = conf_sm[name]
        if float(np.mean(f)) >= float(np.mean(c_fut)):
            continue
        diff = float(np.abs(np.mean(f) - float(np.mean(c_fut))))
        if diff < d:
            d = diff
            cc = f.copy()
            cc_conf = f_conf.copy()
    return cc, cc_conf


def plot_forecast(
    hist: np.ndarray,
    fut: np.ndarray,
    conf: np.ndarray,
    RMD: str,
    solutions: list[str],
    index: dict[str, int],
    col: list[str],
    out_dir: Path,
    start_year: int,
    alarming: bool = True,
    solution_color_map: dict[str, str] | None = None,
) -> None:
    colours = [
        "RoyalBlue",
        "Crimson",
        "DarkOrange",
        "MediumPurple",
        "MediumVioletRed",
        "DodgerBlue",
        "Indigo",
        "coral",
        "hotpink",
        "DarkMagenta",
        "SteelBlue",
        "brown",
        "MediumAquamarine",
        "SlateBlue",
        "SeaGreen",
        "MediumSpringGreen",
        "DarkOliveGreen",
        "Teal",
        "OliveDrab",
        "MediumSeaGreen",
        "DeepSkyBlue",
        "MediumSlateBlue",
        "MediumTurquoise",
        "FireBrick",
        "DarkCyan",
        "violet",
        "MediumOrchid",
        "DarkSalmon",
        "DarkRed",
    ]
    # Darken the fallback palette to increase contrast on plots
    colours = [_darken_color(c, amount=0.28) for c in colours]
    # Swap in a 12-color "love" hex palette (sweet, distinct hues).
    # These are slightly stronger pastels so edges and slight alpha differences
    # are more visible and separable.
    # Build an expanded distinct palette using matplotlib's Tab20 but reordered
    # to maximize contrast between adjacent entries. This yields 20 distinct hex colors.
    from matplotlib import cm

    # Use a high-contrast qualitative palette (Dark2) for plot fills/lines
    palette_name = "Dark2"
    cmap = cm.get_cmap(palette_name)
    # sample 12 distinct colors from the chosen colormap
    n_samples = 12
    colours_pattern = [mcolors.to_hex(cmap(i / float(max(1, n_samples - 1)))) for i in range(n_samples)]

    from matplotlib.colors import ListedColormap

    if solution_color_map is None:
        solution_color_map = {}

    shade_cmap = ListedColormap(colours_pattern)
    palette_len = len(colours_pattern)

    pyplot_style = "seaborn-v0_8-dark"
    try:
        plt.style.use(pyplot_style)
    except Exception:
        plt.style.use("seaborn-dark")

    fig, ax = plt.subplots(figsize=(10, 7))

    MIN_GAP_MONTHS = 1
    CI_VIS_SCALE = 10.0

    counter = 0
    RMD_idx = index[RMD]
    a_name = consistent_name(RMD)

    hist_len = hist.shape[0]
    fut_len = fut.shape[0]

    stretch = 2.0
    forecast_start = hist_len

    x_hist = np.arange(hist_len)
    x_fut = forecast_start - 1 + np.arange(fut_len + 1) * stretch
    forecast_mask = x_fut >= forecast_start - 1

    fut_sm: dict[str, np.ndarray] = {}
    conf_sm: dict[str, np.ndarray] = {}
    nodes = [RMD] + [s for s in solutions if s in index]
    for name in nodes:
        idx = index[name]
        hist_raw = hist[:, idx]
        fut_raw = fut[:, idx]
        conf_raw = conf[:, idx]

        hist_s = _smooth_series(hist_raw)
        # Minimal smoothing for forecast curves (preserve raw forecast detail)
        fut_s = _smooth_forecast_series(fut_raw)
        fut_s = _smooth_forecast_with_anchor(hist_s, fut_s, alpha=0.08)
        fut_s = _blend_forecast_start(hist_s, fut_s, months=4)
        conf_s = _smooth_series(conf_raw)

        fut_sm[name] = fut_s
        conf_sm[name] = conf_s

    hist_s = _smooth_series(hist[:, RMD_idx])
    hist_plot, fut_plot = _smooth_join_window(hist_s, fut_sm[RMD], window=9)
    conf_s = conf_sm[RMD] * CI_VIS_SCALE

    rmd_color = "black"
    ax.plot(x_hist, hist_plot, "-", color=rmd_color, label=a_name, linewidth=2.0)
    ax.plot(x_fut, fut_plot, "-", color=rmd_color, linewidth=2.0)
    _draw_ci_band(ax, x_fut, fut_plot, conf_s, rmd_color, alpha=0.6)
    f_RMD = fut_plot.copy()
    forecast_curves = [f_RMD]
    counter += 1

    if alarming:
        filtered = []
        for s in solutions:
            if s not in index:
                continue
            if float(np.mean(fut_sm[s])) < float(np.mean(f_RMD)):
                filtered.append(s)
        solutions = filtered

    max_pts = 5
    if len(solutions) > max_pts:
        ranked = sorted(
            ((s, float(np.mean(fut_sm[s]))) for s in solutions if s in fut_sm),
            key=lambda item: item[1],
            reverse=True,
        )
        solutions = [s for s, _ in ranked[:max_pts]]

    # Collect gap shading tasks so we can draw them in an order that
    # improves separability (draw larger fills first so smaller fills
    # appear on top). Each task stores the computed mean so we can sort.
    shading_tasks: list[dict] = []

    # separate index for shading colors so fills use a dedicated palette order
    shade_idx = 0
    for s in solutions:
        if s not in index:
            continue
        s_idx = index[s]
        s_name = consistent_name(s)

        hist_s = _smooth_series(hist[:, s_idx])
        hist_plot, fut_plot = _smooth_join_window(hist_s, fut_sm[s], window=9)
        conf_s = conf_sm[s] * CI_VIS_SCALE

        # Determine line color (use fixed mapping if available, else cycle 'colours')
        mapped_color = solution_color_map.get(s_name)
        line_color = mapped_color if mapped_color is not None else colours[counter]
        ax.plot(x_hist, hist_plot, "-", color=line_color, label=s_name, linewidth=1.0)
        ax.plot(x_fut, fut_plot, "-", color=line_color, linewidth=1.0)
        _draw_ci_band(ax, x_fut, fut_plot, conf_s, line_color, alpha=0.6)

        forecast_curves.append(fut_plot)
        gap_mask = forecast_mask
        if np.any(gap_mask):
            # Choose a distinct color for this solution's shading (independent index)
            # Prefer mapping color if provided, otherwise pick from shade_cmap
            if s_name in SOLUTION_GAP_OVERRIDES:
                gap_color = SOLUTION_GAP_OVERRIDES[s_name]
            elif s_name in solution_color_map:
                gap_color = solution_color_map[s_name]
            else:
                gap_color = shade_cmap(shade_idx % palette_len)
            # No border for shading fills per previous request
            edge_color = "none"

            shading_tasks.append(
                {
                    "x": x_fut,
                    "y_top": f_RMD,
                    "y_bot": fut_plot,
                    "mask": gap_mask,
                    "color": gap_color,
                    "edgecolor": edge_color,
                    "linewidth": 0.0,
                    "alpha": 0.10,
                    "zorder": 2,
                    "mean": float(np.mean(fut_plot)),
                }
            )
            shade_idx += 1

        counter = (counter + 1) % len(colours)

    # Draw gap fills after plotting all lines. Sort by mean descending
    # so larger fills are drawn first and smaller fills sit on top.
    if shading_tasks:
        shading_tasks_sorted = sorted(shading_tasks, key=lambda t: t["mean"], reverse=True)
        for t in shading_tasks_sorted:
            _draw_gap_shading(
                ax,
                t["x"],
                t["y_top"],
                t["y_bot"],
                t["mask"],
                color=t["color"],
                alpha=t["alpha"],
                edgecolor=t["edgecolor"],
                linewidth=t["linewidth"],
                zorder=t["zorder"],
            )

    if forecast_curves:
        lowest = np.min(np.vstack(forecast_curves), axis=0)
        _fill_between_masked(
            ax,
            x_fut,
            lowest,
            np.zeros_like(lowest),
            forecast_mask,
            color="#9ac2e6",
            alpha=0.35,
            zorder=1,
        )

    ax.set_ylabel("Trend", fontsize=15)
    ax.legend(loc="upper left", prop={"size": 10}, bbox_to_anchor=(1, 1.03))
    ax.axis("tight")
    ax.grid(True)
    plt.xticks(rotation=90, fontsize=13)
    plt.title(a_name, y=1.03, fontsize=18)

    history_end_year = 2025
    forecast_start_year = 2026
    forecast_end_year = 2029

    tick_positions = []
    tick_labels = []

    for year in range(start_year, history_end_year + 1, 2):
        month_index = (year - start_year) * 12
        if month_index < hist_len:
            tick_positions.append(month_index)
            tick_labels.append(str(year))

    for year in range(forecast_start_year, forecast_end_year + 1):
        month_index = hist_len + (year - forecast_start_year) * 12 * stretch
        tick_positions.append(month_index)
        tick_labels.append(str(year))

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    out_dir.mkdir(parents=True, exist_ok=True)
    safe = a_name.replace("/", "_")
    png_path = out_dir / f"{safe}.png"
    pdf_path = out_dir / f"{safe}.pdf"
    fig.set_size_inches(10, 7)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)


def save_gap(
    fut: np.ndarray,
    RMD: str,
    solutions: list[str],
    index: dict[str, int],
    out_dir: Path,
    start_year: int = 2026,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    years = [str(start_year + i) for i in range(3)]

    RMD_series = fut[:, index[RMD]]
    RMD_yearly = [np.mean(RMD_series[i : i + 12]) for i in range(0, len(RMD_series), 12)]

    rows = []
    for s in solutions:
        if s not in index:
            continue
        s_series = fut[:, index[s]]
        s_yearly = [np.mean(s_series[i : i + 12]) for i in range(0, len(s_series), 12)]
        gap = [a - b for a, b in zip(RMD_yearly, s_yearly)]
        rows.append([consistent_name(s)] + gap)

    rows = sorted(rows, key=lambda r: sum(r[1:]))
    out_path = out_dir / f"{consistent_name(RMD).replace('/', '_')}_gap.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Solution"] + years)
        writer.writerows(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_file = root / "data" / "sm_data_g.csv"
    graph_file = root / "data" / "graph.csv"
    point_file = root / "model" / "Bayesian" / "forecast" / "forecast_2026_2028.csv"
    lo_file = root / "model" / "Bayesian" / "forecast" / "forecast_2026_2028_pi_95_lower.csv"
    hi_file = root / "model" / "Bayesian" / "forecast" / "forecast_2026_2028_pi_95_upper.csv"

    out_plots = root / "model" / "Bayesian" / "forecast" / "plots_grouped"
    out_gaps = root / "model" / "Bayesian" / "forecast" / "gap"

    hist_df = pd.read_csv(data_file)
    point_df = pd.read_csv(point_file, index_col=0)
    lo_df = pd.read_csv(lo_file, index_col=0)
    hi_df = pd.read_csv(hi_file, index_col=0)

    col = list(hist_df.columns)
    index = {c: i for i, c in enumerate(col)}

    point_df = point_df[col]
    lo_df = lo_df[col]
    hi_df = hi_df[col]

    hist = hist_df.values.astype(float)
    fut = point_df.values.astype(float)
    conf = ((hi_df.values - lo_df.values) / 2.0).astype(float)

    if fut.shape[0] != 36:
        raise ValueError(f"Expected 36-month forecast, got {fut.shape[0]}")

    use_global_norm = os.environ.get("BMTGNN_GROUPED_GLOBAL_NORM", "0") == "1"
    use_exp_smooth = os.environ.get("BMTGNN_GROUPED_SMOOTH", "0") == "1"

    if use_global_norm:
        full = np.vstack([hist, fut])
        incident_max = -float("inf")
        mention_max = -float("inf")

        for i in range(full.shape[0]):
            for j in range(full.shape[1]):
                name = col[j]
                if "WAR" in name or "Holiday" in name or j in range(16, 32):
                    continue
                if "Mention" in name:
                    if full[i, j] > mention_max:
                        mention_max = full[i, j]
                else:
                    if full[i, j] > incident_max:
                        incident_max = full[i, j]

        if not np.isfinite(incident_max) or incident_max <= 0:
            incident_max = 1.0
        if not np.isfinite(mention_max) or mention_max <= 0:
            mention_max = 1.0

        all_n = np.zeros_like(full)
        conf_n = np.zeros_like(conf)
        for i in range(full.shape[0]):
            for j in range(full.shape[1]):
                if "Mention" in col[j]:
                    all_n[i, j] = full[i, j] / mention_max
                else:
                    all_n[i, j] = full[i, j] / incident_max

        for i in range(conf.shape[0]):
            for j in range(conf.shape[1]):
                if "Mention" in col[j]:
                    conf_n[i, j] = conf[i, j] / mention_max
                else:
                    conf_n[i, j] = conf[i, j] / incident_max

        smooth_all = exponential_smoothing(all_n, 0.1) if use_exp_smooth else all_n
        smooth_conf = exponential_smoothing(conf_n, 0.1) if use_exp_smooth else conf_n

        smooth_hist = smooth_all[:-fut.shape[0], :]
        smooth_fut = smooth_all[-fut.shape[0] :, :]
    else:
        smooth_hist = hist.copy()
        smooth_fut = fut.copy()
        smooth_conf = conf.copy()

    graph = build_graph(graph_file)
    solution_color_map = load_solution_color_map(root / "nodes.csv")

    for RMD, solutions in graph.items():
        if RMD not in index:
            continue
        indices = [index[RMD]] + [index[s] for s in solutions if s in index]
        zero_negative_curves(smooth_hist, smooth_fut, indices)
        plot_forecast(
            smooth_hist,
            smooth_fut,
            smooth_conf,
            RMD,
            solutions,
            index,
            col,
            out_plots,
            start_year=2004,
            alarming=True,
            solution_color_map=solution_color_map,
        )
        save_gap(smooth_fut, RMD, solutions, index, out_gaps, start_year=2026)

    print("Done. Plots:", out_plots)
    print("Gap tables:", out_gaps)


if __name__ == "__main__":
    main()
