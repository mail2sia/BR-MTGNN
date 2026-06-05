#!/usr/bin/env python3
"""
Generate Baseline Comparison Table

This script reads baseline/model output metrics and generates comparison
tables in CSV, Markdown, and HTML formats.

Usage:
    python generate_comparison_table.py
    python generate_comparison_table.py --sample
    python generate_comparison_table.py --results-dir ./results
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ComparisonTableGenerator:
    def __init__(self, results_dir: str | None = None):
        self.repo_root = Path(__file__).resolve().parent
        self.results_dir = Path(results_dir).resolve() if results_dir else (self.repo_root / "results")
        self.results: dict[str, dict[str, Any]] = {}

    def add_result(self, model_name: str, metrics: dict[str, Any]) -> None:
        """Add a model result."""
        self.results[model_name] = metrics

    def add_sample_results(self) -> None:
        """Add sample results for demonstration."""
        self.results = {
            "BMTGNN": {
                "RAE": 0.5123,
                "RSE": 0.4321,
                "Corr": 0.7654,
                "Coverage": 0.95,
                "elapsed_time": 145.32,
                "status": "completed",
            },
            "LSTM_M": {
                "RAE": 0.5678,
                "RSE": 0.4876,
                "Corr": 0.6789,
                "Coverage": 0.93,
                "elapsed_time": 89.45,
                "status": "completed",
            },
            "LSTM_U": {
                "RAE": 0.6234,
                "RSE": 0.5234,
                "Corr": 0.5432,
                "Coverage": 0.91,
                "elapsed_time": 67.23,
                "status": "completed",
            },
            "MTGNN": {
                "RAE": 0.4890,
                "RSE": 0.4123,
                "Corr": 0.7890,
                "Coverage": 0.96,
                "elapsed_time": 156.78,
                "status": "completed",
            },
            "Transformer_M": {
                "RAE": 0.5901,
                "RSE": 0.5012,
                "Corr": 0.6543,
                "Coverage": 0.94,
                "elapsed_time": 234.56,
                "status": "completed",
            },
            "Transformer_U": {
                "RAE": 0.6567,
                "RSE": 0.5456,
                "Corr": 0.5123,
                "Coverage": 0.92,
                "elapsed_time": 178.90,
                "status": "completed",
            },
            "VAR": {
                "RAE": 0.7234,
                "RSE": 0.6123,
                "Corr": 0.4321,
                "Coverage": 0.88,
                "elapsed_time": 45.67,
                "status": "completed",
            },
        }

    def load_results_from_files(self) -> None:
        """Load results from baseline output files or existing comparison CSV."""
        self.results = {}

        print(f"Repository root detected as: {self.repo_root}")

        # ------------------------------------------------------------
        # 1. First try loading existing generated comparison CSV
        # ------------------------------------------------------------
        existing_csv = self.repo_root / "baseline_comparison_results.csv"

        if existing_csv.exists():
            loaded = self._load_existing_comparison_csv(existing_csv)

            if loaded > 0:
                print(f"✅ Loaded {loaded} model results from existing CSV: {existing_csv}")
                return

        # ------------------------------------------------------------
        # 2. If no existing CSV is usable, search raw baseline outputs
        # ------------------------------------------------------------
        baseline_files = {
            "BMTGNN": [
                "BMTGNN/outb*.txt",
                "BMTGNN/out*.txt",
                "BMTGNN/**/*.txt",
                "BMTGNN/**/*.csv",
                "BR-MTGNN/model/**/*.csv",
                "BR-MTGNN/model/**/*.txt",
            ],
            "LSTM_M": [
                "Baselines/LSTM/out_m.txt",
                "Baselines/LSTM/**/*m*.txt",
                "Baselines/LSTM/**/*m*.csv",
            ],
            "LSTM_U": [
                "Baselines/LSTM/out_u.txt",
                "Baselines/LSTM/**/*u*.txt",
                "Baselines/LSTM/**/*u*.csv",
            ],
            "MTGNN": [
                "Baselines/MTGNN/out*.txt",
                "Baselines/MTGNN/**/*.txt",
                "Baselines/MTGNN/**/*.csv",
            ],
            "Transformer_M": [
                "Baselines/Transformer/out_m.txt",
                "Baselines/Transformer/**/*m*.txt",
                "Baselines/Transformer/**/*m*.csv",
            ],
            "Transformer_U": [
                "Baselines/Transformer/out_u.txt",
                "Baselines/Transformer/**/*u*.txt",
                "Baselines/Transformer/**/*u*.csv",
            ],
            "VAR": [
                "Baselines/VAR/forecast_results_VAR.csv",
                "Baselines/VAR/**/*.csv",
                "Baselines/VAR/**/*.txt",
            ],
        }

        for model_name, patterns in baseline_files.items():
            found_file = False

            for pattern in patterns:
                files = sorted(self.repo_root.glob(pattern))

                if not files:
                    continue

                for file in files:
                    if self._should_skip_file(file):
                        continue

                    metrics = self.load_metrics_from_any_file(file)

                    if metrics:
                        metrics.setdefault("status", "completed")
                        self.results[model_name] = metrics
                        print(f"✅ Loaded {model_name} from {file}")
                        found_file = True
                        break

                if found_file:
                    break

            if not found_file:
                print(f"⚠️ No usable result file found for {model_name}")

        if not self.results:
            print("\n❌ No baseline results were loaded.")
            print("Run the baselines first, or make sure baseline_comparison_results.csv contains real metric values.")

    def _should_skip_file(self, file: Path) -> bool:
        """Skip generated comparison files and obvious non-result files."""
        skip_names = {
            "baseline_comparison_results.csv",
            "BASELINE_COMPARISON.md",
            "baseline_comparison_results.html",
            "data.csv",
            "sm_data.csv",
            "graph_sparse.csv",
            "graph_sparse.scores.csv",
        }

        if file.name in skip_names:
            return True

        if "__pycache__" in file.parts:
            return True

        return False

    def _load_existing_comparison_csv(self, csv_path: Path) -> int:
        """Load model-level metrics from an existing baseline_comparison_results.csv."""
        try:
            df = pd.read_csv(csv_path)

            if df.empty:
                return 0

            # Handle pandas index column from df.to_csv()
            if "Unnamed: 0" in df.columns:
                df = df.rename(columns={"Unnamed: 0": "Model"})

            if "Model" not in df.columns:
                first_col = df.columns[0]
                df = df.rename(columns={first_col: "Model"})

            metric_cols = [
                "RAE",
                "RSE",
                "Corr",
                "Coverage",
                "MAE",
                "RMSE",
                "SMAPE",
                "sMAPE",
                "MAPE",
                "MeanIntervalWidth",
                "elapsed_time",
                "status",
            ]

            loaded = 0

            for _, row in df.iterrows():
                model_name = str(row.get("Model", "")).strip()

                if not model_name or model_name.lower() == "nan":
                    continue

                metrics: dict[str, Any] = {}

                for col in metric_cols:
                    if col in df.columns and pd.notna(row[col]):
                        out_col = "SMAPE" if col == "sMAPE" else col

                        if out_col == "status":
                            metrics[out_col] = str(row[col])
                        else:
                            try:
                                metrics[out_col] = float(row[col])
                            except Exception:
                                pass

                if metrics:
                    metrics.setdefault("status", "completed")
                    self.results[model_name] = metrics
                    loaded += 1

            return loaded

        except Exception as e:
            print(f"⚠️ Could not load existing comparison CSV {csv_path}: {e}")
            return 0

    def load_metrics_from_any_file(self, filepath: Path) -> dict[str, Any] | None:
        """Load metrics from TXT, LOG, MD, JSON, or CSV result files."""
        filepath = Path(filepath)

        try:
            suffix = filepath.suffix.lower()

            if suffix == ".csv":
                return self._parse_metrics_from_csv(filepath)

            if suffix == ".json":
                return self._parse_metrics_from_json(filepath)

            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            return self.parse_metrics(content)

        except Exception as e:
            print(f"⚠️ Could not parse {filepath}: {e}")
            return None

    def _parse_metrics_from_json(self, filepath: Path) -> dict[str, Any] | None:
        """Parse metrics from a JSON file."""
        try:
            data_raw = pd.read_json(filepath, typ="series").to_dict()
            data: dict[Any, Any] = dict(data_raw)
        except Exception:
            return None

        metrics: dict[str, Any] = {}
        aliases = self._metric_aliases()

        flat_data = self._flatten_dict(data)

        for metric_name, possible_names in aliases.items():
            for key, value in flat_data.items():
                if str(key).strip() in possible_names:
                    try:
                        metrics[metric_name] = float(value)
                    except Exception:
                        pass

        return metrics if metrics else None
    
    def _flatten_dict(self, data: dict[Any, Any], prefix: str = "") -> dict[str, Any]:
        """Flatten nested dictionaries."""
        out: dict[str, Any] = {}

        for key, value in data.items():
            key_str = str(key)
            new_key = f"{prefix}.{key_str}" if prefix else key_str

            if isinstance(value, dict):
                out.update(self._flatten_dict(dict(value), new_key))
            else:
                out[new_key] = value
                out[key_str] = value

        return out
    def _parse_metrics_from_csv(self, filepath: Path) -> dict[str, Any] | None:
        """Parse metrics from wide or long CSV result files."""
        try:
            df = pd.read_csv(filepath)
        except Exception:
            return None

        if df.empty:
            return None

        aliases = self._metric_aliases()
        metrics: dict[str, Any] = {}

        # ------------------------------------------------------------
        # Case A: wide CSV format, metric names are columns
        # Example: MAE, RMSE, RAE, RSE, Corr, Coverage
        # ------------------------------------------------------------
        for metric_name, possible_names in aliases.items():
            for col in df.columns:
                clean_col = str(col).strip()

                if clean_col in possible_names:
                    values = pd.to_numeric(df[col], errors="coerce").dropna()

                    if len(values) > 0:
                        metrics[metric_name] = float(values.iloc[-1])
                        break

        if metrics:
            return metrics

        # ------------------------------------------------------------
        # Case B: long CSV format
        # Example:
        # metric,value
        # RAE,0.51
        # RSE,0.43
        # ------------------------------------------------------------
        lower_cols = {str(c).strip().lower(): c for c in df.columns}

        metric_col = None
        value_col = None

        for candidate in ["metric", "metrics", "name", "measure"]:
            if candidate in lower_cols:
                metric_col = lower_cols[candidate]
                break

        for candidate in ["value", "score", "result"]:
            if candidate in lower_cols:
                value_col = lower_cols[candidate]
                break

        if metric_col is not None and value_col is not None:
            for _, row in df.iterrows():
                raw_metric_name = str(row.get(metric_col, "")).strip()

                for metric_name, possible_names in aliases.items():
                    if raw_metric_name in possible_names:
                        try:
                            metrics[metric_name] = float(row[value_col])
                        except Exception:
                            pass

        return metrics if metrics else None

    def _metric_aliases(self) -> dict[str, list[str]]:
        """Accepted metric column/name aliases."""
        return {
            "RAE": [
                "RAE",
                "rae",
                "Relative Absolute Error",
                "relative_absolute_error",
            ],
            "RSE": [
                "RSE",
                "rse",
                "Relative Squared Error",
                "relative_squared_error",
            ],
            "Corr": [
                "Corr",
                "corr",
                "Correlation",
                "correlation",
                "Pearson",
                "pearson",
            ],
            "Coverage": [
                "Coverage",
                "coverage",
                "Prediction Interval Coverage",
                "prediction_interval_coverage",
            ],
            "MAE": [
                "MAE",
                "mae",
                "Mean Absolute Error",
                "mean_absolute_error",
            ],
            "RMSE": [
                "RMSE",
                "rmse",
                "Root Mean Squared Error",
                "root_mean_squared_error",
            ],
            "SMAPE": [
                "SMAPE",
                "sMAPE",
                "smape",
                "Symmetric Mean Absolute Percentage Error",
            ],
            "MAPE": [
                "MAPE",
                "mape",
                "Mean Absolute Percentage Error",
            ],
            "MeanIntervalWidth": [
                "MeanIntervalWidth",
                "mean_interval_width",
                "IntervalWidth",
                "interval_width",
            ],
            "elapsed_time": [
                "elapsed_time",
                "Elapsed Time",
                "time",
                "runtime",
                "seconds",
                "Time (seconds)",
            ],
        }

    def parse_metrics(self, content: str) -> dict[str, Any] | None:
        """Parse metrics from text content."""
        metrics: dict[str, Any] = {}

        number = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"

        metric_patterns = {
            "RAE": [
                rf"\bRAE\b[:\s=,]*{number}",
                rf"\bRelative Absolute Error\b[:\s=,]*{number}",
            ],
            "RSE": [
                rf"\bRSE\b[:\s=,]*{number}",
                rf"\bRelative Squared Error\b[:\s=,]*{number}",
            ],
            "Corr": [
                rf"\bCorr\b[:\s=,]*{number}",
                rf"\bCorrelation\b[:\s=,]*{number}",
            ],
            "Coverage": [
                rf"\bCoverage\b[:\s=,]*{number}",
                rf"\bPrediction Interval Coverage\b[:\s=,]*{number}",
            ],
            "MAE": [
                rf"\bMAE\b[:\s=,]*{number}",
                rf"\bMean Absolute Error\b[:\s=,]*{number}",
            ],
            "RMSE": [
                rf"\bRMSE\b[:\s=,]*{number}",
                rf"\bRoot Mean Squared Error\b[:\s=,]*{number}",
            ],
            "SMAPE": [
                rf"\bSMAPE\b[:\s=,]*{number}",
                rf"\bsMAPE\b[:\s=,]*{number}",
            ],
            "MAPE": [
                rf"\bMAPE\b[:\s=,]*{number}",
            ],
            "MeanIntervalWidth": [
                rf"\bMeanIntervalWidth\b[:\s=,]*{number}",
                rf"\bMean Interval Width\b[:\s=,]*{number}",
            ],
            "elapsed_time": [
                rf"\belapsed_time\b[:\s=,]*{number}",
                rf"\bElapsed Time\b[:\s=,]*{number}",
                rf"\bruntime\b[:\s=,]*{number}",
                rf"\btime\b[:\s=,]*{number}",
            ],
        }

        for metric_name, patterns in metric_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE)

                if match:
                    try:
                        metrics[metric_name] = float(match.group(1))
                        break
                    except ValueError:
                        pass

        return metrics if metrics else None

    def calculate_rankings(self) -> pd.DataFrame:
        """Calculate rankings for each metric."""
        if not self.results:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(self.results, orient="index")
        rankings = pd.DataFrame(index=df.index)

        lower_is_better = ["MAE", "RMSE", "SMAPE", "MAPE", "RAE", "RSE", "MeanIntervalWidth"]
        higher_is_better = ["Corr", "Coverage"]

        for col in lower_is_better:
            if col in df.columns:
                rankings[f"{col}_rank"] = df[col].rank(ascending=True, method="min")

        for col in higher_is_better:
            if col in df.columns:
                rankings[f"{col}_rank"] = df[col].rank(ascending=False, method="min")

        return rankings

    def _sort_results_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort dataframe by best available error metric."""
        for metric in ["MAE", "RAE", "RMSE", "RSE", "SMAPE", "MAPE"]:
            if metric in df.columns:
                return df.sort_values(metric, ascending=True)

        if "Corr" in df.columns:
            return df.sort_values("Corr", ascending=False)

        return df

    def _dataframe_to_markdown(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        index_label: str = "Model",
    ) -> str:
        """Convert dataframe to markdown table without tabulate dependency."""
        if columns:
            existing_cols = [c for c in columns if c in df.columns]
            df = df[existing_cols]

        display_df = df.copy()
        display_df.insert(0, index_label, display_df.index)
        display_df.columns = [str(c) for c in display_df.columns]

        markdown = "| " + " | ".join(display_df.columns) + " |\n"
        markdown += "|" + "|".join(["-" * (len(str(col)) + 2) for col in display_df.columns]) + "|\n"

        for _, row in display_df.iterrows():
            values = []

            for v in row.values:
                if isinstance(v, (int, float, np.integer, np.floating)) and pd.notna(v):
                    values.append(str(round(float(v), 4)))
                else:
                    values.append(str(v))

            markdown += "| " + " | ".join(values) + " |\n"

        return markdown

    def generate_markdown_table(self) -> Path | None:
        """Generate markdown comparison table."""
        if not self.results:
            print("❌ No results to generate Markdown table")
            return None

        df = pd.DataFrame.from_dict(self.results, orient="index")
        df = self._sort_results_df(df)

        md_file = self.repo_root / "BASELINE_COMPARISON.md"

        with open(md_file, "w", encoding="utf-8") as f:
            f.write("# Baseline Comparison Results\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Configuration\n\n")
            f.write("- **Data:** 264 timesteps × 190 features\n")
            f.write("- **Train/Val/Test Split:** 43% / 30% / 27%\n")
            f.write("- **Features:** 96 RMD + 89 PT + 5 other\n\n")

            f.write("## Performance Metrics\n\n")

            metric_order = [
                "MAE",
                "RMSE",
                "SMAPE",
                "MAPE",
                "RAE",
                "RSE",
                "Corr",
                "Coverage",
                "MeanIntervalWidth",
                "elapsed_time",
                "status",
            ]

            metrics_cols = [c for c in metric_order if c in df.columns]
            f.write(self._dataframe_to_markdown(df, metrics_cols))
            f.write("\n\n")

            f.write("## Metric Definitions\n\n")
            f.write("| Metric | Full Name | Better |\n")
            f.write("|--------|-----------|--------|\n")
            f.write("| MAE | Mean Absolute Error | Lower |\n")
            f.write("| RMSE | Root Mean Squared Error | Lower |\n")
            f.write("| SMAPE | Symmetric Mean Absolute Percentage Error | Lower |\n")
            f.write("| MAPE | Mean Absolute Percentage Error | Lower |\n")
            f.write("| RAE | Relative Absolute Error | Lower |\n")
            f.write("| RSE | Relative Squared Error | Lower |\n")
            f.write("| Corr | Correlation Coefficient | Higher |\n")
            f.write("| Coverage | Prediction Interval Coverage | Target near 0.95 |\n")
            f.write("| MeanIntervalWidth | Mean Prediction Interval Width | Lower, if coverage remains acceptable |\n\n")

            f.write("## Model Descriptions\n\n")
            f.write("| Model | Type | Description |\n")
            f.write("|-------|------|-------------|\n")
            f.write("| BMTGNN | Graph-Temporal | Bayesian-Multivariate Temporal GNN |\n")
            f.write("| LSTM_M | Recurrent | LSTM with multivariate input |\n")
            f.write("| LSTM_U | Recurrent | LSTM with univariate input |\n")
            f.write("| MTGNN | Graph-Temporal | Multivariate Temporal GNN |\n")
            f.write("| Transformer_M | Attention | Transformer with multivariate input |\n")
            f.write("| Transformer_U | Attention | Transformer with univariate input |\n")
            f.write("| VAR | Classical | Vector AutoRegression |\n\n")

            rankings = self.calculate_rankings()

            if not rankings.empty:
                f.write("## Rankings by Metric\n\n")
                f.write("Lower rank is better. Error metrics are ranked ascending; Corr and Coverage are ranked descending.\n\n")
                f.write(self._dataframe_to_markdown(rankings))
                f.write("\n\n")

            numeric_df = df.select_dtypes(include=[np.number])

            if not numeric_df.empty:
                f.write("## Summary Statistics\n\n")
                f.write("```text\n")
                f.write(numeric_df.describe().to_string())
                f.write("\n```\n\n")

        print(f"✅ Markdown table saved to: {md_file}")
        return md_file

    def generate_csv_table(self) -> Path | None:
        """Generate CSV comparison table."""
        if not self.results:
            print("❌ No results to generate CSV table")
            return None

        df = pd.DataFrame.from_dict(self.results, orient="index")
        df = self._sort_results_df(df)
        df.index.name = "Model"

        csv_file = self.repo_root / "baseline_comparison_results.csv"
        df.to_csv(csv_file)

        print(f"✅ CSV table saved to: {csv_file}")
        return csv_file

    def generate_html_table(self) -> Path | None:
        """Generate HTML comparison table."""
        if not self.results:
            print("❌ No results to generate HTML table")
            return None

        df = pd.DataFrame.from_dict(self.results, orient="index")
        df = self._sort_results_df(df)
        df.index.name = "Model"

        html_file = self.repo_root / "baseline_comparison_results.html"

        rankings = self.calculate_rankings()

        rankings_html = ""
        if not rankings.empty:
            rankings_html = f"""
    <h2>Rankings by Metric</h2>
    <p>Lower rank is better. Error metrics are ranked ascending; Corr and Coverage are ranked descending.</p>
    {rankings.to_html(classes="results-table")}
"""

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Baseline Comparison Results</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 24px;
            color: #222;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            font-size: 14px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }}
        th {{
            background-color: #2f6f4e;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f7f7f7;
        }}
        h1, h2 {{
            color: #333;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
        }}
        .note {{
            background: #f2f6ff;
            border-left: 4px solid #4a6fa5;
            padding: 10px;
            margin: 15px 0;
        }}
    </style>
</head>
<body>
    <h1>Baseline Comparison Results</h1>
    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <h2>Configuration</h2>
    <ul>
        <li><strong>Data:</strong> 264 timesteps × 190 features</li>
        <li><strong>Train/Val/Test Split:</strong> 43% / 30% / 27%</li>
        <li><strong>Features:</strong> 96 RMD + 89 PT + 5 other</li>
    </ul>

    <h2>Performance Metrics</h2>
    {df.to_html(classes="results-table")}

    {rankings_html}

    <h2>Metric Legend</h2>
    <ul>
        <li><strong>MAE/RMSE/SMAPE/MAPE/RAE/RSE:</strong> Lower is better.</li>
        <li><strong>Corr:</strong> Higher is better.</li>
        <li><strong>Coverage:</strong> Target is usually near 0.95 for a 95% interval.</li>
        <li><strong>MeanIntervalWidth:</strong> Lower is better only when coverage remains acceptable.</li>
    </ul>
</body>
</html>
"""

        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"✅ HTML table saved to: {html_file}")
        return html_file

    def print_results(self) -> None:
        """Print results to console."""
        if not self.results:
            print("❌ No results to display")
            return

        df = pd.DataFrame.from_dict(self.results, orient="index")
        df = self._sort_results_df(df)

        print("\n" + "=" * 100)
        print("BASELINE COMPARISON RESULTS")
        print("=" * 100)
        print(df.to_string())
        print("=" * 100 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate baseline comparison table")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing result files",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sample results for demonstration",
    )

    args = parser.parse_args()

    generator = ComparisonTableGenerator(args.results_dir)

    if args.sample:
        print("Using sample results...")
        generator.add_sample_results()
    else:
        print("Loading results from files...")
        generator.load_results_from_files()

    generator.print_results()

    print("\nGenerating comparison tables...")
    generator.generate_markdown_table()
    generator.generate_csv_table()
    generator.generate_html_table()

    print("\n✅ Comparison table generation complete!")


if __name__ == "__main__":
    main()