#!/usr/bin/env python3
"""Build BRGI ranked RMD-PT pairs from forecast gap_monthly.csv."""
from pathlib import Path

import pandas as pd


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    forecast_dir = root / "model" / "Bayesian" / "forecast"
    gap_path = forecast_dir / "gap_monthly.csv"
    out_dir = forecast_dir / "brgi"
    out_path = out_dir / "ranked_pairs.csv"

    gap = pd.read_csv(gap_path)
    required = {"RMD", "PT", "Gap_RMD_minus_PT", "PT_Below_RMD", "RMD_Trend", "PT_Trend"}
    missing = sorted(required.difference(gap.columns))
    if missing:
        raise ValueError(f"Missing required columns in {gap_path}: {missing}")

    gap["Positive_Gap"] = gap["Gap_RMD_minus_PT"].clip(lower=0)
    ranked = (
        gap.groupby(["RMD", "PT"], as_index=False)
        .agg(
            BRGI_Score=("Positive_Gap", "mean"),
            Mean_Gap=("Gap_RMD_minus_PT", "mean"),
            Max_Gap=("Gap_RMD_minus_PT", "max"),
            Positive_Gap_Months=("PT_Below_RMD", "sum"),
            RMD_Burden=("RMD_Trend", "mean"),
            PT_Readiness=("PT_Trend", "mean"),
        )
        .sort_values(["BRGI_Score", "Max_Gap"], ascending=False)
        .reset_index(drop=True)
    )
    ranked.insert(0, "Rank", range(1, len(ranked) + 1))

    out_dir.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out_path, index=False)
    print(f"Saved {len(ranked)} BRGI ranked pairs to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
