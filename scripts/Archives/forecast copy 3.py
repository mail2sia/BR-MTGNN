import pickle
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import csv
from collections import defaultdict
from matplotlib import pyplot

# Make repo imports work when running from scripts/ or repo root
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.util import unwrap_model_output
from scripts.o_util import build_model_from_checkpoint, filter_state_dict_for_model

pyplot.rcParams["savefig.dpi"] = 1200
pyplot.rcParams["font.family"] = "sans-serif"


# ----------------------------
# Smoothing (ONLY exponential)
# ----------------------------
def exponential_smoothing(series, alpha: float):
    """
    Simple exponential smoothing over time dimension.
    series: torch.Tensor [T, N] or numpy array [T, N]
    returns: list[torch.Tensor] length T, each is [N]
    """
    result = [series[0]]
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n - 1])
    return result


def exponential_smoothing_with_init(series, alpha: float, init):
    """
    Exponential smoothing with an explicit initial value.
    series: torch.Tensor [T, N] or numpy array [T, N]
    init: torch.Tensor [N] or numpy array [N]
    returns: list[torch.Tensor] length T, each is [N]
    """
    result = []
    prev = init
    for n in range(len(series)):
        prev = alpha * series[n] + (1 - alpha) * prev
        result.append(prev)
    return result


# ----------------------------
# Naming
# ----------------------------
def consistent_name(name: str) -> str:
    name = (
        name.replace("RMD_", "")
        .replace("RMD", "")
        .replace("PT_", "")
        .replace("PT", "")
    )

    # special case
    if "HIDDEN MARKOV MODEL" in name:
        return "Statistical HMM"

    if name in {"CAPTCHA", "DNSSEC", "RRAM"}:
        return name

    if "IZ" in name:
        name = name.replace("IZ", "IS")  # British English in that dataset
    if "IOR" in name:
        name = name.replace("IOR", "IOUR")  # British English in that dataset

    # e.g., University of london
    if not name.isupper():
        words = name.split(" ")
        out = ""
        for i, w in enumerate(words):
            if len(w) <= 2:
                out += w
            else:
                out += w[0].upper() + w[1:]
            if i < len(words) - 1:
                out += " "
        return out

    words = name.split(" ")
    out = ""
    for i, w in enumerate(words):
        if len(w) <= 3 or "/" in w or w in {"MITM", "SIEM"}:
            out += w
        else:
            out += w[0] + w[1:].lower()
        if i < len(words) - 1:
            out += " "
    return out


# ----------------------------
# Gap helpers (mean-based)
# ----------------------------
def getClosestCurveLarger(c, forecast, confidence, RMD, solutions, col):
    """
    Closest curve (by mean distance) strictly larger than c.
    """
    d = 999999999
    cc = None
    cc_conf = None
    for j in range(forecast.shape[1]):
        f = forecast[:, j]
        f_conf = confidence[:, j]
        if (col[j] not in solutions) and (col[j] != RMD):
            continue
        if torch.mean(f) <= torch.mean(c):
            continue
        diff = torch.mean(f) - torch.mean(c)
        if diff < d:
            d = diff
            cc = f.clone()
            cc_conf = f_conf.clone()
    return cc, cc_conf


def getClosestCurveSmaller(c, forecast, confidence, RMD, solutions, col):
    """
    Closest curve (by mean distance) strictly smaller than c.
    """
    d = 999999999
    cc = None
    cc_conf = None
    for j in range(forecast.shape[1]):
        f = forecast[:, j]
        f_conf = confidence[:, j]
        if (col[j] not in solutions) and (col[j] != RMD):
            continue
        if torch.mean(f) >= torch.mean(c):
            continue
        diff = torch.abs(torch.mean(f) - torch.mean(c))
        if diff < d:
            d = diff
            cc = f.clone()
            cc_conf = f_conf.clone()
    return cc, cc_conf


# ----------------------------
# Clamp negatives to 0 (old behavior)
# ----------------------------
def zero_negative_curves(data, forecast, RMD, solutions, index):
    a = data[:, index[RMD]]
    f = forecast[:, index[RMD]]
    a[a < 0] = 0
    f[f < 0] = 0

    for s in solutions:
        if s not in index:
            continue
        a = data[:, index[s]]
        f = forecast[:, index[s]]
        a[a < 0] = 0
        f[f < 0] = 0
    return data, forecast


# ----------------------------
# Plot grouped forecast (graph-based)
# ----------------------------
def plot_forecast(data, forecast, confidence, RMD, solutions, index, col, alarming=True, start_year=2004):
    """
    Uses ONLY:
    - Exponential smoothing (already applied before calling)
    - Mean-based closest curve gap shading
    - CI band = mean ± 1.96 * std / sqrt(n) (already computed)
    - Grouped graph-based plot
    """
    data, forecast = zero_negative_curves(data, forecast, RMD, solutions, index)

    hist_len = data.shape[0]  # months
    fut_len = forecast.shape[0]
    fut_scale = 2.0

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

    try:
        pyplot.style.use("seaborn-dark")
    except OSError:
        try:
            pyplot.style.use("seaborn-v0_8-dark")
        except OSError:
            pyplot.style.use("default")
    fig = pyplot.figure()
    ax = fig.add_axes((0.1, 0.1, 0.7, 0.75))


    if RMD not in index:
        pyplot.close(fig)
        return

    # Plot the forecast of RMD
    counter = 0
    d = data[:, index[RMD]]
    f = forecast[:, index[RMD]]
    c = confidence[:, index[RMD]]
    a = consistent_name(RMD)

    x_d = np.arange(hist_len)
    x_f = hist_len + fut_scale * np.arange(fut_len)

    ax.plot(x_d, d, "-", color=colours[counter], label=a, linewidth=2)
    ax.plot(x_f, f, "-", color=colours[counter], linewidth=2)
    ax.fill_between(
        x_f,
        (f - c).numpy(),
        (f + c).numpy(),
        color=colours[counter],
        alpha=0.6,
    )
    f_RMD = f.clone()
    counter += 1

    # remove technologies that we are not worried about in the future
    if alarming:
        for s in list(solutions):
            if s not in index:
                solutions.remove(s)
                continue
            f_s = forecast[:, index[s]]
            if torch.mean(f_s) >= torch.mean(f_RMD):
                solutions.remove(s)

    # Plot the forecast of the solutions
    for s in solutions:
        if s not in index:
            continue

        d = data[:, index[s]]
        f = forecast[:, index[s]]
        c = confidence[:, index[s]]
        s_lbl = consistent_name(s)

        x_d = np.arange(hist_len)
        x_f = hist_len + fut_scale * np.arange(fut_len)

        ax.plot(x_d, d, "-", color=colours[counter], label=s_lbl, linewidth=1)
        ax.plot(x_f, f, "-", color=colours[counter], linewidth=1)
        ax.fill_between(
            x_f,
            (f - c).numpy(),
            (f + c).numpy(),
            color=colours[counter],
            alpha=0.6,
        )

        # gap shading (mean-based closest curve)
        if torch.mean(f_RMD) > torch.mean(f):
            cc, cc_conf = getClosestCurveLarger(f, forecast, confidence, RMD, solutions, col)
            if cc is not None and cc_conf is not None:
                ax.fill_between(
                    x_f,
                    (cc - cc_conf).numpy(),
                    (f + c).numpy(),
                    color=colours[counter],
                    alpha=0.3,
                )
        else:
            cc, cc_conf = getClosestCurveSmaller(f, forecast, confidence, RMD, solutions, col)
            if cc is not None and cc_conf is not None:
                ax.fill_between(
                    x_f,
                    (cc + cc_conf).numpy(),
                    (f - c).numpy(),
                    color=colours[counter],
                    alpha=0.3,
                )

        counter = (counter + 1) % len(colours)

    # X ticks: history every 2 years; forecast yearly
    total_len = hist_len + fut_len * fut_scale

    history_end_year = start_year + (hist_len // 12) - 1
    forecast_start_year = history_end_year + 1
    forecast_end_year = forecast_start_year + (fut_len // 12)

    tick_positions = []
    tick_labels = []

    for year in range(start_year, history_end_year + 1, 2):
        pos = (year - start_year) * 12
        if 0 <= pos <= total_len:
            tick_positions.append(pos)
            tick_labels.append(str(year))

    for year in range(forecast_start_year, forecast_end_year + 1, 1):
        pos = hist_len + fut_scale * ((year - forecast_start_year) * 12)
        if 0 <= pos <= total_len:
            tick_positions.append(pos)
            tick_labels.append(str(year))

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    ax.set_xlim(0, total_len)

    ax.set_ylabel("Trend", fontsize=15)
    pyplot.yticks(fontsize=13)
    ax.legend(loc="upper left", prop={"size": 10}, bbox_to_anchor=(1, 1.03))
    ax.axis("tight")
    ax.grid(True)
    pyplot.xticks(rotation=90, fontsize=13)
    pyplot.title(a, y=1.03, fontsize=18)

    fig = pyplot.gcf()
    fig.set_size_inches(10, 7)

    images_dir = "model/Bayesian/forecast/plots/"
    os.makedirs(images_dir, exist_ok=True)

    safe = a.replace("/", "_")
    pyplot.savefig(os.path.join(images_dir, safe + ".png"), bbox_inches="tight")
    pyplot.savefig(os.path.join(images_dir, safe + ".pdf"), bbox_inches="tight", format="pdf")
    pyplot.close(fig)


# ----------------------------
# Save numeric outputs (text + gap CSV)
# ----------------------------
def save_data(data, forecast, confidence, variance, col):
    file_dir = "model/Bayesian/forecast/data/"
    os.makedirs(file_dir, exist_ok=True)
    for i in range(data.shape[1]):
        d = data[:, i]
        f = forecast[:, i]
        c = confidence[:, i]
        v = variance[:, i]
        name = col[i]
        with open(os.path.join(file_dir, name.replace("/", "_") + ".txt"), "w") as ff:
            ff.write("Data: " + str(d.tolist()) + "\n")
            ff.write("Forecast: " + str(f.tolist()) + "\n")
            ff.write("95% Confidence: " + str(c.tolist()) + "\n")
            ff.write("Variance: " + str(v.tolist()) + "\n")


def save_gap(forecast, RMD, solutions, index, start_year_forecast=2026):
    out_dir = "model/Bayesian/forecast/gap/"
    os.makedirs(out_dir, exist_ok=True)

    # 3 years (36 months) -> 3 yearly means
    years = [str(start_year_forecast + i) for i in range(3)]
    out_path = os.path.join(out_dir, consistent_name(RMD).replace("/", "_") + "_gap.csv")

    with open(out_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Solution"] + years)

        table = []
        if RMD not in index:
            return

        a = forecast[:, index[RMD]].tolist()
        a_reduced = [sum(a[i : i + 12]) / 12 for i in range(0, len(a), 12)]

        for s in solutions:
            if s not in index:
                continue
            row_name = consistent_name(s)
            f = forecast[:, index[s]].tolist()
            f_reduced = [sum(f[i : i + 12]) / 12 for i in range(0, len(f), 12)]
            gap = [x - y for x, y in zip(a_reduced, f_reduced)]
            table.append((row_name, gap))

        sorted_table = sorted(table, key=lambda r: sum(r[1][-3:]))
        for row_name, gaps in sorted_table:
            writer.writerow([row_name] + [str(g) for g in gaps])


# ----------------------------
# CSV columns + graph
# ----------------------------
def create_columns(file_name):
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        header = [c for c in next(reader)]
        if header and "Date" in header[0]:
            header = header[1:]
        col_index = {c: i for i, c in enumerate(header)}
        return header, col_index


def _map_node_name(name: str, col_set: set) -> str:
    """
    Best-effort mapping for graph node names to dataset columns.
    """
    if name in col_set:
        return name
    # try with prefixes
    for cand in (f"RMD_{name}", f"PT_{name}"):
        if cand in col_set:
            return cand
    # try stripped
    stripped = name.replace("RMD_", "").replace("PT_", "")
    if stripped in col_set:
        return stripped
    return name


def build_graph(file_name: str, col: list[str], edge_thresh: float = 0.0) -> dict[str, list[str]]:
    """
    Supports:
      (A) Square numeric adjacency matrix CSV with NO header:
          - shape is [N, N]
          - row/col i corresponds to col[i]
      (B) Adjacency list CSV:
          - first field is key node name
          - remaining fields are adjacent node names

    Returns:
      dict {RMD_node: [PT_nodes...]} (keys filtered to RMD_* only)
    """
    col_set = set(col)

    # Case A: square numeric matrix with no header
    try:
        mat = np.loadtxt(file_name, delimiter=",", dtype=float)
        if mat.ndim == 2 and mat.shape[0] == mat.shape[1] and mat.shape[0] == len(col):
            graph = defaultdict(list)
            n = mat.shape[0]
            for i in range(n):
                src = col[i]
                if not (src.startswith("RMD") or src.startswith("RMD_")):
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    w = float(mat[i, j])
                    if w > edge_thresh:
                        dst = col[j]
                        graph[src].append(dst)
            print("Graph loaded (square-matrix) with", len(graph), "RMD nodes...")
            return dict(graph)
    except Exception:
        pass

    # Case B: adjacency list fallback
    graph = defaultdict(list)
    with open(file_name, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            key = row[0].strip()
            if key not in col_set:
                if f"RMD_{key}" in col_set:
                    key = f"RMD_{key}"
                elif f"PT_{key}" in col_set:
                    key = f"PT_{key}"
                else:
                    continue

            if not (key.startswith("RMD") or key.startswith("RMD_")):
                continue

            adj = []
            for node in row[1:]:
                node = node.strip()
                if not node:
                    continue
                if node in col_set:
                    adj.append(node)
                elif f"RMD_{node}" in col_set:
                    adj.append(f"RMD_{node}")
                elif f"PT_{node}" in col_set:
                    adj.append(f"PT_{node}")

            graph[key].extend(adj)

    print("Graph loaded (list) with", len(graph), "RMD nodes...")
    return dict(graph)


# ----------------------------
# Main (forecast + plot)
# ----------------------------
def main():
    # Files (edit if needed)
    data_file = "./data/sm_data_g.csv"
    model_file = "model/Bayesian/model.pt"
    nodes_file = "data/data.csv"
    graph_file = "data/graph_square.csv"
    hp_file = "hp.txt"  # read (for logging / checking), model.pt already contains hparams
    device = "cpu"      # set "cuda:0" etc on natogpu

    # read hp.txt (requested). model.pt also contains hparams and is what builds the model.
    if os.path.exists(hp_file):
        try:
            hp_txt = open(hp_file, "r", encoding="utf-8").read().strip()
            print("hp.txt loaded.")
            # do not parse/override here; model.pt hparams are used by build_model_from_checkpoint
        except Exception as e:
            print("hp.txt read failed:", e)

    # read the data
    with open(data_file, "r") as fin:
        rawdat = np.atleast_2d(np.loadtxt(fin, delimiter=",", skiprows=1))
    n, m = rawdat.shape

    # load column names and dictionary of (column name, index)
    col, index = create_columns(data_file)
    if len(col) != m:
        col, index = create_columns(nodes_file)
        if len(col) != m:
            col = col[:m]
            index = {c: i for i, c in enumerate(col)}

    # build the graph in the format {RMD:list of PTs}
    graph = build_graph(graph_file, col)

    # per-column max normalization (as old script)
    scale = np.ones(m, dtype=rawdat.dtype)
    dat = np.zeros((n, m), dtype=rawdat.dtype)
    for i in range(m):
        denom = np.max(np.abs(rawdat[:, i]))
        if denom == 0:
            denom = 1.0
        scale[i] = denom
        dat[:, i] = rawdat[:, i] / denom

    print("data shape:", dat.shape)

    # load the model (checkpoint with state_dict + hparams list)
    checkpoint = torch.load(model_file, map_location=device)
    if not (isinstance(checkpoint, dict) and "state_dict" in checkpoint):
        raise TypeError("model.pt must be a dict checkpoint with 'state_dict' and 'hparams'")

    model, inferred_seq_len, in_dim, state_dict = build_model_from_checkpoint(checkpoint, device)
    filtered_state, missing, unexpected, shape_mismatch = filter_state_dict_for_model(state_dict, model)
    if missing or unexpected or shape_mismatch:
        print("Model load warnings: missing", len(missing), "unexpected", len(unexpected), "shape_mismatch", len(shape_mismatch))
    model.load_state_dict(filtered_state, strict=False)
    model.eval()

    # input window
    P = inferred_seq_len
    if P > n:
        raise ValueError(f"Not enough history for seq_length={P}, got {n} rows")

    if in_dim == 1:
        X = torch.from_numpy(dat[-P:, :])
        X = torch.unsqueeze(X, dim=0)
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
    else:
        # optional 2-channel input (levels + pct change)
        levels = dat
        pct = np.zeros_like(levels)
        denom = np.abs(levels[:-1]) + 1e-6
        pct[1:] = (levels[1:] - levels[:-1]) / denom
        levels_tail = levels[-P:, :]
        pct_tail = pct[-P:, :]
        X_np = np.stack([levels_tail.T, pct_tail.T], axis=0)
        X = torch.from_numpy(X_np).unsqueeze(0)

    X = X.to(torch.float)

    # Bayesian estimation (MC forward passes)
    num_runs = 50
    outputs = []
    for _ in range(num_runs):
        with torch.no_grad():
            output = model(X)
            output_tensor = unwrap_model_output(output)
            y_pred = output_tensor[-1, :, :, -1].clone()  # [H, N]
        outputs.append(y_pred)

    outputs = torch.stack(outputs)  # [R, H, N]
    Y = torch.mean(outputs, dim=0)
    variance = torch.var(outputs, dim=0)
    std_dev = torch.std(outputs, dim=0)

    # Uncertainty (as requested): mean ± 1.96 * std / sqrt(n)
    z = 1.96
    confidence = z * std_dev / torch.sqrt(torch.tensor(num_runs))

    # scale back to original scale
    dat = dat * scale
    dat_tensor = torch.from_numpy(dat).to(torch.float)

    scale_tensor = torch.from_numpy(scale).to(torch.float)
    Y = Y * scale_tensor
    variance = variance * scale_tensor
    confidence = confidence * scale_tensor

    print("output shape:", Y.shape)

    # save numeric outputs
    save_data(dat_tensor, Y, confidence, variance, col)

    alpha = 0.05
    # exponential smoothing (ONLY smoothing used)
    # Apply separately to avoid a transition spike at the history/forecast boundary.
    sm_hist = torch.stack(exponential_smoothing(dat_tensor, alpha))                 # [T, N]
    sm_fut = torch.stack(exponential_smoothing_with_init(Y, alpha, sm_hist[-1]))    # [H, N]
    smoothed_conf = torch.stack(exponential_smoothing(confidence, alpha))           # [H, N]

    # grouped plots (graph-based)
    for RMD, solutions in graph.items():
        plot_forecast(sm_hist, sm_fut, smoothed_conf, RMD, solutions, index, col, alarming=True, start_year=2004)
        save_gap(sm_fut, RMD, solutions, index, start_year_forecast=2026)


if __name__ == "__main__":
    main()
