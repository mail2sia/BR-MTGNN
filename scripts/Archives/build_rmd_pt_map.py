import csv
import importlib.util
import math
from pathlib import Path

import pandas as pd

ROOT = Path(r"c:\B-MTGNN_31.01.2026\trustworthy-graph-neural-time-series-forecasts-main")
DATA_PATH = ROOT / "data" / "sm_data_g.csv"
NODES_PATH = ROOT / "data" / "nodes.csv"
MAP_PATH = ROOT / "data" / "RMD_PT_map.csv"

MIN_ABS_RHO = 0.35
MAX_Q = 0.05

if importlib.util.find_spec("scipy") is None:
    raise RuntimeError("scipy is required for Spearman p-values. Please install scipy.")

from scipy.stats import spearmanr
from typing import Any


def main():
    df = pd.read_csv(DATA_PATH)
    first_col = df.columns[0]
    if "date" in str(first_col).lower():
        df = df.drop(columns=[first_col])

    rmd_cols = [c for c in df.columns if c.startswith("RMD_")]
    pt_cols = [c for c in df.columns if c.startswith("PT_")]

    nodes_df = pd.read_csv(NODES_PATH)
    all_rmds = [t for t in nodes_df["token"].astype(str).tolist() if t.startswith("RMD_")]

    pairs = []  # (rmd, pt, rho, p)
    for rmd in rmd_cols:
        x = df[rmd].to_numpy()
        for pt in pt_cols:
            y = df[pt].to_numpy()
            result: Any = spearmanr(x, y, nan_policy="omit")
            rho_val = float(result[0])
            p_val = float(result[1])
            if math.isnan(rho_val) or math.isnan(p_val):
                continue
            pairs.append((rmd, pt, rho_val, p_val))

    pairs_sorted = sorted(pairs, key=lambda t: t[3])
    m = len(pairs_sorted)
    q_values = [0.0] * m
    min_coeff = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        p = pairs_sorted[i][3]
        q = (p * m) / rank
        if q < min_coeff:
            min_coeff = q
        q_values[i] = min(min_coeff, 1.0)

    q_lookup = {}
    rho_lookup = {}
    for (rmd, pt, rho, p), q in zip(pairs_sorted, q_values):
        q_lookup[(rmd, pt)] = q
        rho_lookup[(rmd, pt)] = rho

    map_rows = []
    kept_pairs = 0
    for rmd in all_rmds:
        if rmd in rmd_cols:
            candidates = []
            for pt in pt_cols:
                key = (rmd, pt)
                if key not in q_lookup:
                    continue
                rho = rho_lookup[key]
                q = q_lookup[key]
                if abs(rho) >= MIN_ABS_RHO and q <= MAX_Q:
                    candidates.append((pt, abs(rho)))
            candidates.sort(key=lambda t: t[1], reverse=True)
            pts = [pt for pt, _ in candidates]
            kept_pairs += len(pts)
        else:
            pts = []
        map_rows.append([rmd] + pts)

    with open(MAP_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(map_rows)

    print(f"Wrote {len(map_rows)} RMD rows to {MAP_PATH}")
    print(f"Total kept RMD-PT pairs: {kept_pairs}")


if __name__ == "__main__":
    main()
