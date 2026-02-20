import argparse
import re
from pathlib import Path

import pandas as pd


def _parse_month_label(x: str) -> pd.Timestamp:
    """
    Robust parser for common month labels:
      - 'YYYY-MM' / 'YYYY/MM' / 'YYYY-MM-DD'
      - 'MM-YY' / 'MM/YY' (assumes 20YY for YY<70 else 19YY)
    """
    s = str(x).strip()

    # YYYY-MM or YYYY/MM (optionally with -DD)
    m = re.match(r"^(\d{4})[-/](\d{1,2})(?:[-/](\d{1,2}))?$", s)
    if m:
        y = int(m.group(1))
        mo = int(m.group(2))
        return pd.Timestamp(year=y, month=mo, day=1)

    # MM-YY or MM/YY
    m = re.match(r"^(\d{1,2})[-/](\d{2})$", s)
    if m:
        mo = int(m.group(1))
        yy = int(m.group(2))
        y = 2000 + yy if yy < 70 else 1900 + yy
        return pd.Timestamp(year=y, month=mo, day=1)

    # fallback: let pandas try
    try:
        ts = pd.to_datetime(s, errors="raise")
        return pd.Timestamp(year=ts.year, month=ts.month, day=1)
    except Exception as e:
        raise ValueError(f"Unrecognized month label: {x!r}") from e


def load_any_forecast_csv(path: Path) -> pd.DataFrame:
    """
    Supports BOTH:
      A) 'wide' format: first column = month/date, remaining columns = nodes
      B) 'long' format: columns include [month/date, node, mean, (lower/upper...)]
    Returns a normalized LONG DataFrame with columns:
      ['date','node','mean', ...optional interval columns...]
    """
    df = pd.read_csv(path)

    # Detect long vs wide
    cols_lower = {c.lower(): c for c in df.columns}
    has_node = any(k in cols_lower for k in ["node", "series", "name"])
    has_mean = any(k in cols_lower for k in ["mean", "pred", "yhat", "median", "p50"])
    has_date = any(k in cols_lower for k in ["date", "month", "time"])

    if has_node and has_date and has_mean:
        # LONG
        date_col = cols_lower.get("date", cols_lower.get("month", cols_lower.get("time")))
        node_col = cols_lower.get("node", cols_lower.get("series", cols_lower.get("name")))
        mean_col = cols_lower.get("mean", cols_lower.get("pred", cols_lower.get("yhat", cols_lower.get("median", cols_lower.get("p50")))))

        out = df.copy()
        out["date"] = out[date_col].map(_parse_month_label)
        out.rename(columns={node_col: "node", mean_col: "mean"}, inplace=True)

        # keep any interval-like columns if present
        keep = ["date", "node", "mean"]
        for k in ["lower", "upper", "p05", "p95", "q05", "q95", "p10", "p90", "std"]:
            if k in cols_lower:
                keep.append(cols_lower[k])
        return out[keep].sort_values(["node", "date"]).reset_index(drop=True)

    # WIDE: assume first column is month/date index
    date_col = df.columns[0]
    wide = df.copy()
    wide["date"] = wide[date_col].map(_parse_month_label)
    wide = wide.drop(columns=[date_col])

    long = wide.melt(id_vars=["date"], var_name="node", value_name="mean")
    long = long.sort_values(["node", "date"]).reset_index(drop=True)
    return long


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast_csv", type=str, required=True,
                    help="Forecast CSV produced by scripts/forecast.py (wide or long).")
    ap.add_argument("--start", type=str, default="2026-01",
                    help="Start month (YYYY-MM). Default: 2026-01.")
    ap.add_argument("--months", type=int, default=36,
                    help="Number of months to extract. Default: 36.")
    ap.add_argument("--out_csv", type=str, default="forecast_2026_36m_long.csv",
                    help="Output CSV path (long format).")
    args = ap.parse_args()

    start = pd.Timestamp(args.start + "-01")
    # month-start range (MS) is the cleanest monthly index :contentReference[oaicite:1]{index=1}
    end_exclusive = (start + pd.offsets.MonthBegin(args.months))

    fc = load_any_forecast_csv(Path(args.forecast_csv))
    mask = (fc["date"] >= start) & (fc["date"] < end_exclusive)
    out = fc.loc[mask].copy()

    if out.empty:
        raise RuntimeError(
            f"No rows found in [{args.start}, {end_exclusive.strftime('%Y-%m')}) "
            f"from {args.forecast_csv}. Check your start_year/start_month and horizon."
        )

    out.to_csv(args.out_csv, index=False)
    print(f"[OK] Saved {len(out):,} rows to {args.out_csv}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
