#!/usr/bin/env python3
"""
Enhanced Visualization Suite for B-MTGNN Forecasts

Provides comprehensive visualizations including:
- Focused recent trends (2020-2028)
- Heatmap gap analysis
- Scenario analysis (optimistic/expected/pessimistic)
- Executive summary dashboard
- Interactive Plotly visualizations

Usage:
    python scripts/forecast_viz.py --mode all
    python scripts/forecast_viz.py --mode heatmap
    python scripts/forecast_viz.py --mode dashboard
    python scripts/forecast_viz.py --mode uncertainty
"""

import os
import sys
import argparse
import time
import csv
from pathlib import Path
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# Ensure project root is on sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import plotly.graph_objects as go  # type: ignore
    import plotly.express as px  # type: ignore
    from plotly.subplots import make_subplots  # type: ignore
    PLOTLY_AVAILABLE = True
except ImportError:
    # Provide type stubs for when plotly is not available
    if TYPE_CHECKING:
        import plotly.graph_objects as go  # type: ignore
        import plotly.express as px  # type: ignore
        from plotly.subplots import make_subplots  # type: ignore
    else:
        go: Any = None
        px: Any = None
        make_subplots: Any = None
    PLOTLY_AVAILABLE = False
    print('[viz] Warning: Plotly not available. Install with: pip install plotly')


class ForecastVisualizer:
    """Comprehensive visualization suite for forecast analysis"""
    
    def __init__(self, forecast_path, historical_path, nodes_path, rmd_pt_map_path, output_dir, forecast_metrics_path=None):
        """
        Initialize visualizer with data paths
        
        Args:
            forecast_path: Path to forecast CSV (2026-2028)
            historical_path: Path to historical data CSV (2004-2025)
            nodes_path: Path to nodes CSV
            rmd_pt_map_path: Path to RMD-PT mapping CSV
            output_dir: Output directory for visualizations
        """
        self.forecast_path = Path(forecast_path)
        self.historical_path = Path(historical_path)
        self.nodes_path = Path(nodes_path)
        self.rmd_pt_map_path = Path(rmd_pt_map_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.forecast_metrics_path = Path(forecast_metrics_path) if forecast_metrics_path else None
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load all required data"""
        print('[viz] Loading data...')
        
        # Load forecast
        self.forecast_df = pd.read_csv(self.forecast_path, parse_dates=False)
        if 'Date' in self.forecast_df.columns:
            self.forecast_df = self.forecast_df.set_index('Date')
        elif 't' in self.forecast_df.columns:
            self.forecast_df = self.forecast_df.set_index('t')
        else:
            # Fall back to the first column if it looks like a date label
            first_col = self.forecast_df.columns[0] if len(self.forecast_df.columns) else None
            if first_col is not None:
                self.forecast_df = self.forecast_df.set_index(first_col)

        # Load forecast metrics (variance + CI) if available
        self.forecast_metrics_df = None
        if self.forecast_metrics_path and self.forecast_metrics_path.exists():
            self.forecast_metrics_df = pd.read_csv(self.forecast_metrics_path)
        elif self.forecast_metrics_path:
            print(f"[viz] Warning: Forecast metrics not found at {self.forecast_metrics_path}")
        
        # Load historical
        self.historical_df = pd.read_csv(self.historical_path)
        
        # Load nodes metadata
        self.nodes_df = pd.read_csv(self.nodes_path)
        
        # Load RMD-PT mapping
        if not self.rmd_pt_map_path.exists():
            fallback_graph_path = Path(_ROOT) / 'data' / 'graph.csv'
            if fallback_graph_path.exists():
                print(f"[viz] Warning: Mapping file not found at {self.rmd_pt_map_path}. Using {fallback_graph_path} instead.")
                self.rmd_pt_map_path = fallback_graph_path
            else:
                raise FileNotFoundError(
                    f"Mapping file not found: {self.rmd_pt_map_path}. "
                    f"Also could not find fallback file: {fallback_graph_path}"
                )

        self.rmd_pt_map = {}
        with open(self.rmd_pt_map_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if parts:
                    RMD = parts[0]
                    solutions = [s for s in parts[1:] if s]
                    self.rmd_pt_map[RMD] = solutions
        
        # Combine historical and forecast
        historical_dates = pd.date_range(start='2004-01', periods=len(self.historical_df), freq='MS')
        self.historical_df.index = historical_dates.strftime('%Y-%m')
        
        self.combined_df = pd.concat([self.historical_df, self.forecast_df])
        
        print(f'[viz] Loaded {len(self.historical_df)} historical + {len(self.forecast_df)} forecast months')
        print(f'[viz] Total variables: {len(self.combined_df.columns)}')
        print(f'[viz] RMD conditions: {len(self.rmd_pt_map)}')

    def _get_node_uncertainty(self, metric="variance"):
        if self.forecast_metrics_df is None:
            return None
        metric = "variance" if metric not in ("variance", "ci_95") else metric
        if metric not in self.forecast_metrics_df.columns:
            return None
        
        # Map node indices to actual token names
        node_unc_raw = self.forecast_metrics_df.groupby("node")[metric].mean()
        
        # If nodes are generic (node_0, node_1, ...), map to actual names from nodes.csv
        if 'token' in self.nodes_df.columns:
            node_mapping = {}
            for idx, row in self.nodes_df.iterrows():
                node_key = f"node_{idx}"
                if node_key in node_unc_raw.index:
                    node_mapping[row['token']] = node_unc_raw[node_key]
            if node_mapping:
                return pd.Series(node_mapping)
        
        return node_unc_raw

    def create_uncertainty_heatmap(self, metric="variance", top_k=None, cluster=False, dpi=1200):
        """Create RMD-PT heatmap using node-level uncertainty metrics."""
        node_unc = self._get_node_uncertainty(metric=metric)
        if node_unc is None:
            print('[viz] Warning: Uncertainty metrics not available for heatmap')
            return

        rmd_vars = list(self.rmd_pt_map.keys())
        pt_vars = sorted({pt for pts in self.rmd_pt_map.values() for pt in pts})
        eps = 1e-12
        mat = np.full((len(rmd_vars), len(pt_vars)), np.nan, dtype=float)

        for i, rmd in enumerate(rmd_vars):
            if rmd not in node_unc:
                continue
            for j, pt in enumerate(pt_vars):
                if pt not in self.rmd_pt_map.get(rmd, []):
                    continue
                if pt not in node_unc:
                    continue
                mat[i, j] = float(node_unc[rmd] / (node_unc[pt] + eps))

        if top_k is not None and int(top_k) > 0:
            top_k = int(top_k)
            pt_counts = {}
            for i in range(len(rmd_vars)):
                row = mat[i]
                valid_idx = np.where(~np.isnan(row))[0]
                if len(valid_idx) == 0:
                    continue
                ranked = valid_idx[np.argsort(row[valid_idx])[::-1]]
                for j in ranked[:top_k]:
                    pt_counts[j] = pt_counts.get(j, 0) + 1

            if pt_counts:
                keep_idx = sorted(pt_counts.keys(), key=lambda j: (-pt_counts[j], j))
                pt_vars = [pt_vars[j] for j in keep_idx]
                mat = mat[:, keep_idx]

        finite_vals = mat[np.isfinite(mat)]
        if finite_vals.size:
            vmin = float(np.percentile(finite_vals, 5))
            vmax = float(np.percentile(finite_vals, 95))
        else:
            vmin, vmax = None, None

        title_suffix = f" (Top-{top_k} PTs per RMD)" if top_k is not None and int(top_k) > 0 else ""
        title = f'RMD vs PT Uncertainty Propagation Ratio{title_suffix}'

        mask = np.isnan(mat)
        row_labels = [v.replace('RMD_', '') for v in rmd_vars]
        col_labels = [v.replace('PT_', '') for v in pt_vars]

        output_path = self.output_dir / 'heatmap_uncertainty_rmd_pt.png'
        if cluster:
            # reduce white gaps and set NaN color to a light gray
            mat_cluster = np.nan_to_num(mat, nan=1.0, posinf=1.0, neginf=1.0)
            cmap = plt.cm.get_cmap('viridis')
            try:
                cmap.set_bad('#f2f2f2')
            except Exception:
                pass
            cg = sns.clustermap(
                mat_cluster,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                mask=mask,
                figsize=(20, 14),
                xticklabels=col_labels,
                yticklabels=row_labels,
                cbar_kws={'label': f'Uncertainty Ratio (RMD/PT) [{metric}]'},
                dendrogram_ratio=(0.12, 0.08),
                linewidths=0,
            )
            cg.ax_heatmap.set_title(title, fontsize=16, fontweight='bold', pad=16)
            cg.ax_heatmap.set_xlabel('Treatment Technologies (PT)', fontsize=12)
            cg.ax_heatmap.set_ylabel('Mental Disorders (RMD)', fontsize=12)
            cg.ax_heatmap.tick_params(axis='x', labelrotation=90, labelsize=8)
            cg.ax_heatmap.tick_params(axis='y', labelrotation=0, labelsize=8)
            # set light gray background to reduce white borders
            cg.ax_heatmap.set_facecolor('#f7f7f7')
            cg.fig.patch.set_facecolor('#f7f7f7')
            cg.fig.tight_layout()
            cg.savefig(output_path, dpi=int(dpi), bbox_inches='tight', pad_inches=0.02)
            cg.savefig(self.output_dir / 'heatmap_uncertainty_rmd_pt.pdf', bbox_inches='tight', pad_inches=0.02)
            plt.close(cg.fig)
        else:
            fig, ax = plt.subplots(figsize=(20, 12), facecolor='#f7f7f7')
            cmap = plt.cm.get_cmap('viridis')
            try:
                cmap.set_bad('#f2f2f2')
            except Exception:
                pass
            sns.heatmap(
                mat,
                xticklabels=col_labels,
                yticklabels=row_labels,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                mask=mask,
                annot=False,
                fmt='.2f',
                linewidths=0,
                cbar_kws={'label': f'Uncertainty Ratio (RMD/PT) [{metric}]'},
                ax=ax,
            )
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Treatment Technologies (PT)', fontsize=12)
            ax.set_ylabel('Mental Disorders (RMD)', fontsize=12)
            ax.set_facecolor('#f7f7f7')
            plt.xticks(rotation=90, ha='right', fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            plt.tight_layout()
            plt.savefig(output_path, dpi=int(dpi), bbox_inches='tight', pad_inches=0.02, facecolor=fig.get_facecolor())
            plt.savefig(self.output_dir / 'heatmap_uncertainty_rmd_pt.pdf', bbox_inches='tight', pad_inches=0.02, facecolor=fig.get_facecolor())
            plt.close()

        print(f'[viz] Saved uncertainty heatmap to {output_path}')

    def create_uncertainty_propagation(
        self,
        graph_path,
        metric="variance",
        threshold=0.1,
        max_hops=3,
        propagation_mode="row",
        spectral_gain=1.0,
        retain_ratio=0.0,
        retain_base=False,
    ):
        """Create hop-wise uncertainty propagation summary using adjacency diffusion.

        propagation_mode:
            - row: row-normalized adjacency (stable baseline)
            - spectral: A scaled by spectral radius (controlled amplification/attenuation)
            - none: raw adjacency (can amplify aggressively)
        retain_ratio:
            - 0.0 -> pure propagation each hop
            - >0  -> blend with previous hop value for smoother/plateau dynamics
        retain_base:
            - False -> blend with previous hop (recursive memory)
            - True  -> blend with initial state u0 (strong stabilizing anchor)
        """
        node_unc = self._get_node_uncertainty(metric=metric)
        if node_unc is None:
            print('[viz] Warning: Uncertainty metrics not available for propagation analysis')
            return

        graph_path = Path(graph_path)
        if not graph_path.exists():
            print(f"[viz] Warning: Graph file not found: {graph_path}")
            return

        try:
            adj = np.loadtxt(str(graph_path), delimiter=',')
        except Exception as e:
            print(f"[viz] Warning: Failed to load graph adjacency: {e}")
            return

        nodes_list = self.nodes_df['token'].tolist() if 'token' in self.nodes_df.columns else list(node_unc.index)
        size = min(len(nodes_list), adj.shape[0], adj.shape[1])
        nodes_list = nodes_list[:size]
        adj = adj[:size, :size]
        adj = np.where(adj > float(threshold), adj, 0.0)

        mode = str(propagation_mode).strip().lower()
        gain = float(spectral_gain)
        retain = float(retain_ratio)
        retain = max(0.0, min(0.99, retain))
        anchor_to_base = bool(retain_base)
        if mode == 'row':
            row_sum = adj.sum(axis=1, keepdims=True)
            row_sum = np.where(row_sum < 1e-12, 1.0, row_sum)
            adj_prop = adj / row_sum
        elif mode == 'spectral':
            try:
                eigvals = np.linalg.eigvals(adj)
                rho = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
            except Exception:
                rho = 0.0
            rho = rho if rho > 1e-12 else 1.0
            adj_prop = (gain / rho) * adj
        elif mode == 'none':
            adj_prop = adj
        else:
            print(f"[viz] Warning: unknown propagation mode '{propagation_mode}', falling back to row")
            row_sum = adj.sum(axis=1, keepdims=True)
            row_sum = np.where(row_sum < 1e-12, 1.0, row_sum)
            adj_prop = adj / row_sum

        u0 = np.array([float(node_unc.get(n, 0.0)) for n in nodes_list], dtype=float)
        base_mean = float(np.mean(u0))
        base_std = float(np.std(u0))

        output_path = self.output_dir / 'uncertainty'
        output_path.mkdir(parents=True, exist_ok=True)

        hop_csv = output_path / 'hop_uncertainty.csv'
        hop_summary_csv = output_path / 'hop_uncertainty_summary.csv'

        with open(hop_csv, 'w', newline='') as f_hop, open(hop_summary_csv, 'w', newline='') as f_sum:
            hop_writer = csv.writer(f_hop)
            sum_writer = csv.writer(f_sum)
            hop_writer.writerow(['node', 'hop', 'propagated_mean_variance', 'amplification_ratio'])
            sum_writer.writerow([
                'hop',
                'global_mean_variance',
                'global_mean_ratio',
                'global_std_variance',
                'global_std_ratio',
            ])

            u_prev = u0
            for hop in range(1, int(max_hops) + 1):
                propagated = adj_prop.T @ u_prev
                anchor = u0 if anchor_to_base else u_prev
                u_next = ((1.0 - retain) * propagated) + (retain * anchor)
                global_mean = float(np.mean(u_next))
                global_std = float(np.std(u_next))
                sum_writer.writerow([
                    hop,
                    global_mean,
                    float(global_mean / (base_mean + 1e-12)),
                    global_std,
                    float(global_std / (base_std + 1e-12)),
                ])
                for idx, name in enumerate(nodes_list):
                    hop_writer.writerow([
                        name,
                        hop,
                        float(u_next[idx]),
                        float(u_next[idx] / (u0[idx] + 1e-12)),
                    ])
                u_prev = u_next

        # NOTE: Removed flat amplification plot - replaced with plot_uncertainty_insights.py
        # which generates more informative node-level and edge-level uncertainty visualizations

        print(f'[viz] Saved uncertainty propagation outputs to {output_path}')
    
    def create_focused_trends(
        self,
        recent_years=None,
        open_plots=False,
        open_delay=1.0,
        zoom_forecast=False,
        zoom_years=4,
    ):
        """
        Create focused trend plots showing recent history + forecast
        
        Args:
            recent_years: Number of recent historical years to show (None = all years from 2004)
        """
        if recent_years is None:
            years_text = 'all years (2004-2028)'
        else:
            years_text = f'last {recent_years} years + forecast'
        print(f'[viz] Creating focused trends ({years_text})...')
        
        # Calculate start date for recent history
        forecast_start_idx = len(self.historical_df)
        if recent_years is None:
            # Show all data from 2004
            start_idx = 0
            months_to_show = forecast_start_idx
        else:
            months_to_show = recent_years * 12
            start_idx = max(0, forecast_start_idx - months_to_show)
        
        # Create plots for each RMD-PT pair
        output_path = self.output_dir / 'focused_trends'
        output_path.mkdir(parents=True, exist_ok=True)
        
        for RMD, solutions in self.rmd_pt_map.items():
            if RMD not in self.combined_df.columns:
                continue

            if zoom_forecast:
                fig, (ax, ax_zoom) = plt.subplots(
                    2,
                    1,
                    figsize=(14, 10),
                    gridspec_kw={'height_ratios': [2, 1]},
                )
            else:
                fig, ax = plt.subplots(figsize=(14, 7))
                ax_zoom = None
            
            # Plot RMD trend
            data_slice = self.combined_df.iloc[start_idx:]
            x = range(len(data_slice))
            
            ax.plot(
                x,
                data_slice[RMD],
                'o-',
                linewidth=2.5,
                label=RMD.replace('_', ' '),
                color='#e74c3c',
                markersize=4,
            )
            
            # Plot solutions
            colors = plt.cm.get_cmap('Set3')(np.linspace(0, 1, len(solutions)))
            for sol, color in zip(solutions, colors):
                if sol in self.combined_df.columns:
                    ax.plot(
                        x,
                        data_slice[sol],
                        '.-',
                        linewidth=1.5,
                        label=sol.replace('_', ' '),
                        color=color,
                        alpha=0.8,
                    )
            
            # Add vertical line at forecast boundary
            forecast_boundary = months_to_show
            # Get the forecast start date from the data
            forecast_start_date = self.forecast_df.index[0] if len(self.forecast_df) > 0 else '2026-01'
            ax.axvline(
                x=forecast_boundary,
                color='gray',
                linestyle='--',
                linewidth=2,
                label=f'Forecast Start ({forecast_start_date})',
            )
            
            # Shade forecast period
            ax.axvspan(forecast_boundary, len(data_slice), alpha=0.1, color='skyblue')
            
            # Formatting
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('Trend Value', fontsize=12)
            # Dynamic title based on date range
            start_year = str(data_slice.index[0]).split('-')[0]
            end_year = str(data_slice.index[-1]).split('-')[0]
            ax.set_title(f'{RMD.replace("_", " ")} - Trends & Forecast ({start_year}-{end_year})', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=9, framealpha=0.9)
            ax.grid(True, alpha=0.3)
            
            # X-axis labels - Show years from 2004 to 2028
            # Create year ticks at the start of each year
            year_positions = []
            year_labels = []
            for i, date_str in enumerate(data_slice.index):
                date_parts = str(date_str).split('-')
                if len(date_parts) < 2:
                    continue
                year, month = date_parts[0], date_parts[1]
                if month == '01':  # January of each year
                    year_positions.append(i)
                    year_labels.append(year)
            
            # Show yearly ticks through 2009, then spaced labels afterward
            display_positions = []
            display_labels = []
            for pos, label in zip(year_positions, year_labels):
                year_i = int(label)
                if year_i <= 2009 or year_i % 3 == 0:
                    display_positions.append(pos)
                    display_labels.append(label)

            ax.set_xticks(display_positions)
            ax.set_xticklabels(display_labels, rotation=0, ha='center')

            if ax_zoom is not None:
                ax_zoom.plot(
                    x,
                    data_slice[RMD],
                    'o-',
                    linewidth=2.0,
                    label=RMD.replace('_', ' '),
                    color='#e74c3c',
                    markersize=3,
                )
                for sol, color in zip(solutions, colors):
                    if sol in self.combined_df.columns:
                        ax_zoom.plot(
                            x,
                            data_slice[sol],
                            '.-',
                            linewidth=1.2,
                            label=sol.replace('_', ' '),
                            color=color,
                            alpha=0.8,
                        )

                ax_zoom.axvline(
                    x=forecast_boundary,
                    color='gray',
                    linestyle='--',
                    linewidth=1.5,
                )
                ax_zoom.axvspan(forecast_boundary, len(data_slice), alpha=0.1, color='skyblue')
                ax_zoom.set_xlabel('Year', fontsize=11)
                ax_zoom.set_ylabel('Trend', fontsize=11)
                ax_zoom.grid(True, alpha=0.3)

                zoom_months = max(12, int(zoom_years * 12))
                zoom_start = max(0, len(data_slice) - zoom_months)
                ax_zoom.set_xlim(zoom_start, len(data_slice) - 1)

                zoom_positions = []
                zoom_labels = []
                for i, date_str in enumerate(data_slice.index):
                    if i < zoom_start:
                        continue
                    date_parts = str(date_str).split('-')
                    if len(date_parts) < 2:
                        continue
                    year, month = date_parts[0], date_parts[1]
                    if month == '01':
                        zoom_positions.append(i)
                        zoom_labels.append(year)

                ax_zoom.set_xticks(zoom_positions)
                ax_zoom.set_xticklabels(zoom_labels, rotation=0, ha='center')
            
            plt.tight_layout()
            png_path = output_path / f'{RMD.replace("/", "_")}_focused.png'
            pdf_path = output_path / f'{RMD.replace("/", "_")}_focused.pdf'
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.savefig(pdf_path, bbox_inches='tight')
            plt.close()

            if open_plots:
                try:
                    img = mpimg.imread(png_path)
                    fig_view, ax_view = plt.subplots(figsize=(10, 6))
                    ax_view.imshow(img)
                    ax_view.axis('off')
                    ax_view.set_title(RMD.replace('_', ' '))
                    plt.show(block=False)
                    plt.pause(max(0.0, float(open_delay)))
                    plt.close(fig_view)
                except Exception as exc:
                    print(f'[viz] Warning: Failed to preview {png_path}: {exc}')
        
        print(f'[viz] Saved {len(self.rmd_pt_map)} focused trend plots to {output_path}')
    
    def create_heatmap_analysis(self):
        """Create heatmap showing gaps between RMDs and treatments"""
        print('[viz] Creating heatmap gap analysis...')
        
        # Calculate mean forecast values for RMDs and PTs
        forecast_means = self.forecast_df.mean()
        
        # Build gap matrix
        rmd_vars = [v for v in self.combined_df.columns if v.startswith('RMD_')]
        pt_vars = [v for v in self.combined_df.columns if v.startswith('PT_')]
        
        gap_matrix = np.zeros((len(rmd_vars), len(pt_vars)))
        
        for i, rmd in enumerate(rmd_vars):
            for j, pt in enumerate(pt_vars):
                if rmd in forecast_means and pt in forecast_means:
                    gap_matrix[i, j] = forecast_means[rmd] - forecast_means[pt]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(20, 12))
        
        sns.heatmap(gap_matrix, 
                    xticklabels=[v.replace('PT_', '') for v in pt_vars],
                    yticklabels=[v.replace('RMD_', '') for v in rmd_vars],
                    cmap='RdYlGn_r',  # Red=high gap, Green=covered
                    center=0,
                    annot=False,
                    fmt='.1f',
                    cbar_kws={'label': 'Gap (RMD - PT)'},
                    ax=ax)
        
        ax.set_title('RMD vs PT Treatment Gap Analysis (2026-2028 Forecast)', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Treatment Technologies (PT)', fontsize=12)
        ax.set_ylabel('Mental Disorders (RMD)', fontsize=12)
        
        plt.xticks(rotation=90, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        
        output_path = self.output_dir / 'heatmap_gap_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'heatmap_gap_analysis.pdf', bbox_inches='tight')
        plt.close()
        
        print(f'[viz] Saved heatmap to {output_path}')
    
    def create_scenario_analysis(self, confidence_level=0.95):
        """
        Create scenario analysis plots (best/expected/worst case)
        
        Args:
            confidence_level: Confidence level for scenarios (default 95%)
        """
        print('[viz] Creating scenario analysis...')
        
        # For demonstration, simulate scenarios using forecast std
        forecast_std = self.forecast_df.std()
        
        output_path = self.output_dir / 'scenario_analysis'
        output_path.mkdir(parents=True, exist_ok=True)
        
        for RMD in self.rmd_pt_map.keys():
            if RMD not in self.forecast_df.columns:
                continue
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Historical + Expected forecast
            historical = self.historical_df.iloc[-24:][RMD]  # Last 2 years
            expected = self.forecast_df[RMD]
            
            # Create scenarios
            optimistic = expected - 1.96 * forecast_std[RMD]
            pessimistic = expected + 1.96 * forecast_std[RMD]
            
            # Combine historical and scenarios
            x_hist = range(len(historical))
            x_forecast = range(len(historical), len(historical) + len(expected))
            
            # Plot
            ax.plot(x_hist, historical, 'o-', linewidth=2, label='Historical', color='#3498db')
            ax.plot(x_forecast, expected, 's-', linewidth=2.5, label='Expected (Mean)', color='#2ecc71')
            ax.fill_between(x_forecast, optimistic, pessimistic, alpha=0.2, color='#95a5a6', label='95% Confidence Range')
            ax.plot(x_forecast, optimistic, '--', linewidth=1.5, label='Optimistic Scenario', color='#27ae60', alpha=0.7)
            ax.plot(x_forecast, pessimistic, '--', linewidth=1.5, label='Pessimistic Scenario', color='#c0392b', alpha=0.7)
            
            # Vertical line at forecast start
            ax.axvline(x=len(historical), color='gray', linestyle='--', linewidth=2, alpha=0.5)
            
            # Formatting
            ax.set_title(f'{RMD.replace("_", " ")} - Scenario Analysis', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time Period', fontsize=12)
            ax.set_ylabel('Trend Value', fontsize=12)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / f'{RMD.replace("/", "_")}_scenarios.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f'[viz] Saved scenario analysis plots to {output_path}')
    
    def create_executive_summary(self):
        """Generate executive summary PDF report"""
        print('[viz] Creating executive summary report...')
        
        output_path = self.output_dir / 'Executive_Summary_Report.pdf'
        
        with PdfPages(output_path) as pdf:
            # Page 1: Overview Statistics
            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle('B-MTGNN Forecast Executive Summary (2026-2028)', fontsize=20, fontweight='bold', y=0.98)
            
            # Calculate key statistics
            rmd_means = self.forecast_df[[c for c in self.forecast_df.columns if c.startswith('RMD_')]].mean()
            pt_means = self.forecast_df[[c for c in self.forecast_df.columns if c.startswith('PT_')]].mean()
            has_rmd_pt = (len(rmd_means) > 0 and len(pt_means) > 0)
            
            # Top 10 RMDs with highest forecast values
            top_rmds = rmd_means.nlargest(10)
            
            # Create subplot grid
            gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
            
            # Top RMDs bar chart
            ax1 = fig.add_subplot(gs[0, :])
            if len(top_rmds) > 0:
                top_rmds.plot(kind='barh', ax=ax1, color='#e74c3c')
                ax1.set_title('Top 10 RMDs by Forecast Trend (Mean 2026-2028)', fontsize=12, fontweight='bold')
                ax1.set_xlabel('Mean Forecast Value')
                ax1.set_yticklabels([t.get_text().replace('RMD_', '') for t in ax1.get_yticklabels()], fontsize=9)
            else:
                ax1.text(0.5, 0.5, 'No RMD_* columns found in forecast data',
                         ha='center', va='center', fontsize=11)
                ax1.set_title('Top 10 RMDs by Forecast Trend (Mean 2026-2028)', fontsize=12, fontweight='bold')
                ax1.set_axis_off()
            
            # Gap analysis summary
            ax2 = fig.add_subplot(gs[1, 0])
            gaps = []
            gap_labels = []
            for RMD, solutions in list(self.rmd_pt_map.items())[:10]:
                if RMD in rmd_means:
                    RMD_mean = rmd_means[RMD]
                    solution_means = [pt_means[s] for s in solutions if s in pt_means]
                    if solution_means:
                        gap = RMD_mean - np.mean(solution_means)
                        gaps.append(gap)
                        gap_labels.append(RMD.replace('RMD_', ''))
            
            if gaps:
                ax2.barh(gap_labels, gaps, color=['#e74c3c' if g > 0 else '#2ecc71' for g in gaps])
                ax2.set_title('Treatment Gap Analysis (Top 10 RMDs)', fontsize=11, fontweight='bold')
                ax2.set_xlabel('Gap (RMD - Mean(PT))')
                ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
                ax2.tick_params(labelleft=True, labelsize=8)
            else:
                ax2.text(0.5, 0.5, 'Gap analysis unavailable for current column naming',
                         ha='center', va='center', fontsize=10)
                ax2.set_title('Treatment Gap Analysis (Top 10 RMDs)', fontsize=11, fontweight='bold')
                ax2.set_axis_off()
            
            # Statistics table
            ax3 = fig.add_subplot(gs[1, 1])
            ax3.axis('off')
            stats_data = [
                ['Metric', 'Value'],
                ['Total RMD Variables', f'{len(rmd_means)}'],
                ['Total PT Variables', f'{len(pt_means)}'],
                ['Forecast Period', '2026-01 to 2028-12'],
                ['Forecast Months', '36'],
                ['Mean RMD Trend', f'{rmd_means.mean():.2f}' if len(rmd_means) else 'N/A'],
                ['Mean PT Trend', f'{pt_means.mean():.2f}' if len(pt_means) else 'N/A'],
                ['Overall Gap', f'{rmd_means.mean() - pt_means.mean():.2f}' if has_rmd_pt else 'N/A'],
            ]
            
            table = ax3.table(cellText=stats_data, cellLoc='left', loc='center',
                             colWidths=[0.6, 0.4])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Format header row
            for i in range(2):
                table[(0, i)].set_facecolor('#3498db')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            ax3.set_title('Key Statistics', fontsize=11, fontweight='bold', pad=10)
            
            # Trend over time
            ax4 = fig.add_subplot(gs[2, :])
            monthly_rmd_mean = self.combined_df[[c for c in self.combined_df.columns if c.startswith('RMD_')]].mean(axis=1)
            monthly_pt_mean = self.combined_df[[c for c in self.combined_df.columns if c.startswith('PT_')]].mean(axis=1)
            
            # Plot only recent history + forecast
            recent_start = max(0, len(monthly_rmd_mean) - 96)  # Last 8 years + forecast
            x = range(len(monthly_rmd_mean[recent_start:]))
            forecast_boundary = len(self.historical_df) - recent_start
            
            ax4.plot(x, monthly_rmd_mean[recent_start:], linewidth=2, label='Mean RMD Trend', color='#e74c3c')
            ax4.plot(x, monthly_pt_mean[recent_start:], linewidth=2, label='Mean PT Trend', color='#2ecc71')
            ax4.axvline(x=forecast_boundary, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Forecast Start')
            ax4.axvspan(forecast_boundary, len(x), alpha=0.1, color='skyblue')
            ax4.set_title('Overall Trend Evolution (2018-2028)', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Time Period')
            ax4.set_ylabel('Mean Trend Value')
            ax4.legend(loc='best')
            ax4.grid(True, alpha=0.3)
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        print(f'[viz] Saved executive summary to {output_path}')
    
    def create_interactive_dashboard(self):
        """Create interactive Plotly dashboard"""
        if not PLOTLY_AVAILABLE:
            print('[viz] Skipping interactive dashboard - Plotly not installed')
            return
        
        print('[viz] Creating interactive dashboard...')
        
        # Create dashboard with multiple views
        for RMD, solutions in self.rmd_pt_map.items():
            if RMD not in self.combined_df.columns:
                continue
            
            # Create subplot figure
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    f'{RMD.replace("_", " ")} - Full History',
                    'Forecast Detail (2026-2028)',
                    'Treatment Comparison',
                    'Gap Trend'
                ),
                specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                       [{'type': 'bar'}, {'type': 'scatter'}]],
                vertical_spacing=0.15,
                horizontal_spacing=0.12
            )
            
            # Full history
            fig.add_trace(
                go.Scatter(x=self.combined_df.index, y=self.combined_df[RMD],
                          mode='lines', name=RMD.replace('_', ' '),
                          line=dict(color='red', width=2)),
                row=1, col=1
            )
            
            # Forecast detail
            fig.add_trace(
                go.Scatter(x=self.forecast_df.index, y=self.forecast_df[RMD],
                          mode='lines+markers', name='Forecast',
                          line=dict(color='red', width=3)),
                row=1, col=2
            )
            
            # Treatment comparison (mean forecast values)
            solution_means = []
            solution_labels = []
            for sol in solutions:
                if sol in self.forecast_df.columns:
                    solution_means.append(self.forecast_df[sol].mean())
                    solution_labels.append(sol.replace('PT_', ''))
            
            fig.add_trace(
                go.Bar(x=solution_labels, y=solution_means,
                      marker_color='lightblue', name='Treatments'),
                row=2, col=1
            )
            
            # Gap trend
            for sol in solutions:
                if sol in self.forecast_df.columns:
                    gap = self.forecast_df[RMD] - self.forecast_df[sol]
                    fig.add_trace(
                        go.Scatter(x=self.forecast_df.index, y=gap,
                                  mode='lines', name=sol.replace('PT_', ''),
                                  line=dict(width=1.5)),
                        row=2, col=2
                    )
            
            fig.add_shape(
                type='line',
                x0=0,
                x1=1,
                y0=0,
                y1=0,
                xref='x4 domain',
                yref='y4',
                line=dict(color='gray', dash='dash'),
            )
            
            # Update layout
            fig.update_layout(
                height=900,
                showlegend=True,
                title_text=f"Interactive Dashboard: {RMD.replace('_', ' ')}",
                title_font_size=18
            )
            
            # Save as HTML
            output_path = self.output_dir / 'interactive'
            output_path.mkdir(parents=True, exist_ok=True)
            fig.write_html(output_path / f'{RMD.replace("/", "_")}_dashboard.html')
        
        print(f'[viz] Saved interactive dashboards to {self.output_dir / "interactive"}')


def main():
    """Main visualization pipeline"""
    parser = argparse.ArgumentParser(description='Enhanced forecast visualization suite')
    parser.add_argument('--forecast', type=str, default='model/Bayesian/forecast/forecast_2026_2028.csv',
                        help='Path to forecast CSV')
    parser.add_argument('--forecast-metrics', type=str, default='model/Bayesian/forecast/forecast.csv',
                        help='Path to forecast metrics CSV (includes variance/CI)')
    parser.add_argument('--historical', type=str, default='data/sm_data_g.csv',
                        help='Path to historical data CSV')
    parser.add_argument('--nodes', type=str, default='data/nodes.csv',
                        help='Path to nodes CSV')
    parser.add_argument('--rmd-pt-map', type=str, default='data/graph.csv',
                        help='Path to RMD-PT mapping CSV')
    parser.add_argument('--graph', type=str, default='data/graph_topk_k12.csv',
                        help='Path to graph adjacency matrix')
    parser.add_argument('--output', type=str, default='model/Bayesian/forecast/enhanced_viz',
                        help='Output directory for visualizations')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'focused', 'heatmap', 'scenarios', 'summary', 'dashboard', 'uncertainty'],
                        help='Visualization mode')
    parser.add_argument('--uncertainty-metric', type=str, default='variance',
                        choices=['variance', 'ci_95'],
                        help='Uncertainty metric for heatmap/propagation')
    parser.add_argument('--uncertainty-topk', type=int, default=10,
                        help='Top-k PTs per RMD to include in uncertainty heatmap (0=all)')
    parser.add_argument('--uncertainty-dpi', type=int, default=300,
                        help='DPI for uncertainty heatmap PNG export')
    parser.add_argument('--uncertainty-cluster', action='store_true', default=False,
                        help='Enable clustered heatmap for uncertainty (default: off)')
    parser.add_argument('--no-uncertainty-cluster', action='store_false', dest='uncertainty_cluster',
                        help='Disable clustered heatmap for uncertainty')
    parser.add_argument('--uncertainty-threshold', type=float, default=0.1,
                        help='Adjacency threshold for uncertainty propagation analysis')
    parser.add_argument('--uncertainty-max-hops', type=int, default=3,
                        help='Max hops for uncertainty propagation analysis')
    parser.add_argument('--uncertainty-propagation-mode', type=str, default='row',
                        choices=['row', 'spectral', 'none'],
                        help='Propagation operator for uncertainty diffusion (default: row)')
    parser.add_argument('--uncertainty-spectral-gain', type=float, default=1.0,
                        help='Gain for spectral mode (A_scaled = gain/rho * A)')
    parser.add_argument('--uncertainty-retain-ratio', type=float, default=0.0,
                        help='Blend ratio with previous hop value (0..0.99) for smoother/plateau propagation')
    parser.add_argument('--uncertainty-retain-base', action='store_true', default=False,
                        help='Anchor retain blending to initial uncertainty state (u0) instead of previous hop')
    parser.add_argument('--recent-years', type=int, default=None,
                        help='Number of recent years to show in focused view (default: None = all years from 2004)')
    parser.add_argument('--open-plots', action='store_true',
                        help='Preview focused plots one by one after saving')
    parser.add_argument('--open-delay', type=float, default=1.0,
                        help='Seconds to show each plot before closing (default: 1.0)')
    parser.add_argument('--zoom-forecast', action='store_true',
                        help='Add a zoomed forecast panel for 2026+ period')
    parser.add_argument('--zoom-years', type=float, default=4.0,
                        help='Years to show in zoomed forecast panel (default: 4)')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    root = Path(_ROOT)
    forecast_path = root / args.forecast
    historical_path = root / args.historical
    nodes_path = root / args.nodes
    rmd_pt_map_path = root / args.rmd_pt_map
    output_dir = root / args.output
    
    print('=' * 80)
    print('B-MTGNN Enhanced Forecast Visualizations')
    print('=' * 80)
    
    # Initialize visualizer
    viz = ForecastVisualizer(
        forecast_path,
        historical_path,
        nodes_path,
        rmd_pt_map_path,
        output_dir,
        forecast_metrics_path=args.forecast_metrics,
    )
    
    # Generate visualizations based on mode
    if args.mode in ['all', 'focused']:
        viz.create_focused_trends(
            recent_years=args.recent_years,
            open_plots=args.open_plots,
            open_delay=args.open_delay,
            zoom_forecast=args.zoom_forecast,
            zoom_years=args.zoom_years,
        )
    
    if args.mode in ['all', 'heatmap']:
        viz.create_heatmap_analysis()
    
    if args.mode in ['all', 'scenarios']:
        viz.create_scenario_analysis()
    
    if args.mode in ['all', 'summary']:
        viz.create_executive_summary()
    
    if args.mode in ['all', 'dashboard']:
        viz.create_interactive_dashboard()

    if args.mode in ['all', 'uncertainty']:
        viz.create_uncertainty_heatmap(
            metric=args.uncertainty_metric,
            top_k=args.uncertainty_topk,
            cluster=args.uncertainty_cluster,
            dpi=args.uncertainty_dpi,
        )
        viz.create_uncertainty_propagation(
            graph_path=args.graph,
            metric=args.uncertainty_metric,
            threshold=args.uncertainty_threshold,
            max_hops=args.uncertainty_max_hops,
            propagation_mode=args.uncertainty_propagation_mode,
            spectral_gain=args.uncertainty_spectral_gain,
            retain_ratio=args.uncertainty_retain_ratio,
            retain_base=args.uncertainty_retain_base,
        )
    
    print('=' * 80)
    print('[viz] Visualization complete!')
    print(f'[viz] Output directory: {output_dir}')
    print('=' * 80)


if __name__ == '__main__':
    main()
