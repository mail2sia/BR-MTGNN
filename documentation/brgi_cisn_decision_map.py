#!/usr/bin/env python3
"""
BRGI + CISN Decision Map: Gap Urgency and Cross-RMD Spillover Value

A publication-ready visualization and reporting script that converts BRGI (forecasted
RMD–PT gaps) and CISN (cross-RMD spillover potential) into a two-step decision map.

Step 1 — BRGI finds the largest RMD–PT forecast gaps.
Step 2 — CISN finds PTs with cross-RMD spillover value.
Step 3 — The final matrix prioritizes PTs that are both gap-relevant and spillover-useful.

Usage:
    python brgi_cisn_decision_map.py \
      --brgi_csv model/Bayesian/forecast/brgi/ranked_pairs.csv \
      --cisn_csv model/Bayesian/forecast/cisn/pt_spillover_value.csv \
      --edges_csv model/Bayesian/forecast/cisn/spillover_network_edges.csv \
      --out_dir model/Bayesian/forecast/brgi_cisn_decision_map \
      --top_k 20 \
      --gamma 0.5 \
      --q_threshold 0.75
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from datetime import datetime
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add parent directory to path for nature_style_utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nature_style_utils import set_nature_style, save_nature_figure

warnings.filterwarnings('ignore')


# =============================================================================
# Utility Functions
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    # Get the project root (parent of documentation directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    default_brgi = os.path.join(project_root, 'model/Bayesian/forecast/brgi/ranked_pairs.csv')
    default_cisn = os.path.join(project_root, 'model/Bayesian/forecast/cisn/pt_spillover_value.csv')
    default_out = os.path.join(project_root, 'model/Bayesian/forecast/brgi_cisn_decision_map')

    parser = argparse.ArgumentParser(
        description='Generate BRGI + CISN decision map with priority matrix and visualization.'
    )
    parser.add_argument(
        '--brgi_csv',
        type=str,
        default=default_brgi,
        help=f'Path to BRGI ranked_pairs.csv (default: {default_brgi})'
    )
    parser.add_argument(
        '--cisn_csv',
        type=str,
        default=default_cisn,
        help=f'Path to CISN pt_spillover_value.csv (default: {default_cisn})'
    )
    parser.add_argument(
        '--edges_csv',
        type=str,
        default=None,
        help='Path to spillover_network_edges.csv (optional, for Panel B network visualization)'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default=default_out,
        help=f'Output directory for results (default: {default_out})'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=20,
        help='Number of top BRGI pairs for Panel A (default: 20)'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.5,
        help='Weight for CISN spillover in final priority score (default: 0.5)'
    )
    parser.add_argument(
        '--q_threshold',
        type=float,
        default=0.75,
        help='Quantile threshold for priority quadrant classification (default: 0.75)'
    )
    return parser.parse_args()


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def require_file(path, description):
    """Check that a required file exists; raise error if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{description} not found: {path}"
        )
    return path


def read_csv_checked(path, description):
    """Read CSV file with validation."""
    require_file(path, description)
    try:
        df = pd.read_csv(path)
        if len(df) == 0:
            raise ValueError(f"{description} has no rows")
        return df
    except Exception as e:
        raise ValueError(
            f"Failed to read {description} ({path}): {e}"
        )


def canonical_name(name):
    """
    Create a canonical merge key from a name.
    Rules:
    - strip whitespace
    - remove leading "RMD_" and "PT_" (with underscores)
    - lowercase
    - replace underscores with spaces
    - remove repeated spaces
    - remove suffix "_NoM" and "_NoP"
    """
    if not isinstance(name, str):
        return str(name)

    name = name.strip()
    # Remove prefixes first (before lowercasing)
    name = re.sub(r'^RMD_', '', name, flags=re.IGNORECASE)
    name = re.sub(r'^PT_', '', name, flags=re.IGNORECASE)

    name = name.lower()
    name = name.replace('_', ' ')
    while '  ' in name:
        name = name.replace('  ', ' ')
    name = re.sub(r'_nom$', '', name)
    name = re.sub(r'_nop$', '', name)
    return name.strip()


def display_name(name):
    """
    Format a name for display.
    Keep original case and spacing, but replace underscores with spaces
    (except scientific names).
    """
    if not isinstance(name, str):
        return str(name)
    name = name.strip()
    name = name.replace('_', ' ')
    # Remove RMD_ and PT_ prefix for display
    name = re.sub(r'^RMD_', '', name)
    name = re.sub(r'^PT_', '', name)
    return name.strip()


def minmax_norm(series, name=None):
    """
    Min-max normalization.
    If all values are constant, set to 0.5 and return (normalized, was_constant).
    """
    series = series.fillna(0)
    min_val = series.min()
    max_val = series.max()

    if min_val == max_val:
        return pd.Series(0.5, index=series.index), True

    return (series - min_val) / (max_val - min_val), False


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_brgi(path):
    """
    Load and validate BRGI data.
    Support multiple column name variations.
    """
    df = read_csv_checked(path, 'BRGI CSV')

    # Detect column names
    possible_rmd_cols = ['RMD', 'rmd_name', 'disorder', 'rmd']
    possible_pt_cols = ['PT', 'pt_name', 'technology', 'pt']
    possible_score_cols = ['BRGI_Score', 'brgi_score', 'final_score', 'score']

    rmd_col = next((c for c in possible_rmd_cols if c in df.columns), None)
    pt_col = next((c for c in possible_pt_cols if c in df.columns), None)
    score_col = next((c for c in possible_score_cols if c in df.columns), None)

    if not all([rmd_col, pt_col, score_col]):
        raise ValueError(
            f"BRGI CSV missing required columns. "
            f"Available: {df.columns.tolist()}. "
            f"Expected one of {possible_rmd_cols} for RMD, "
            f"{possible_pt_cols} for PT, {possible_score_cols} for score."
        )

    df_clean = df[[rmd_col, pt_col, score_col]].copy()
    df_clean.columns = ['rmd_name', 'pt_name', 'brgi_score']

    # Normalize BRGI scores
    df_clean['brgi_norm'], _ = minmax_norm(df_clean['brgi_score'], 'brgi_score')

    return df_clean


def load_cisn(path):
    """
    Load and validate CISN PT spillover data.
    Support multiple column name variations.
    """
    df = read_csv_checked(path, 'CISN CSV')

    # Detect column names
    possible_pt_cols = ['pt_name', 'PT', 'technology', 'pt']
    possible_spillover_cols = ['spillover_value', 'cisn_spillover_value', 'spillover_score']

    pt_col = next((c for c in possible_pt_cols if c in df.columns), None)
    spillover_col = next((c for c in possible_spillover_cols if c in df.columns), None)

    if not pt_col or not spillover_col:
        raise ValueError(
            f"CISN CSV missing required columns. "
            f"Available: {df.columns.tolist()}. "
            f"Expected one of {possible_pt_cols} for PT, "
            f"{possible_spillover_cols} for spillover value."
        )

    df_clean = df[[pt_col, spillover_col]].copy()
    df_clean.columns = ['pt_name', 'cisn_spillover_value']

    # Normalize CISN spillover scores
    df_clean['cisn_spillover_norm'], _ = minmax_norm(
        df_clean['cisn_spillover_value'], 'cisn_spillover_value'
    )

    return df_clean[['pt_name', 'cisn_spillover_value', 'cisn_spillover_norm']]


def load_edges(path):
    """
    Load spillover network edges.
    Return None if path is None or file doesn't exist or is empty.
    """
    if path is None or not os.path.exists(path):
        return None

    try:
        df = pd.read_csv(path)
        if len(df) == 0:
            return None
        return df
    except Exception:
        return None


# =============================================================================
# Merge and Priority Assignment
# =============================================================================

def merge_priority_matrix(brgi, cisn, gamma=0.5, q_threshold=0.75):
    """
    Merge BRGI and CISN data, compute final priority scores and labels.
    """
    # Create canonical keys for merging
    brgi['pt_canonical'] = brgi['pt_name'].apply(canonical_name)
    cisn['pt_canonical'] = cisn['pt_name'].apply(canonical_name)

    # Merge on PT canonical name
    merged = brgi.merge(cisn, on='pt_canonical', how='inner', suffixes=('_brgi', '_cisn'))

    # Use CISN pt_name (more canonical) for display
    merged['pt_name'] = merged['pt_name_cisn']

    if len(merged) == 0:
        raise ValueError(
            "BRGI and CISN have no common PTs after canonical name matching. "
            "Check data consistency."
        )

    # Compute final priority score: brgi_norm * (1 + gamma * cisn_spillover_norm)
    merged['final_priority_score'] = (
        merged['brgi_norm'] * (1 + gamma * merged['cisn_spillover_norm'])
    )

    # Sort by final priority score
    merged = merged.sort_values('final_priority_score', ascending=False).reset_index(drop=True)
    merged['rank'] = range(1, len(merged) + 1)

    return merged


def assign_priority_labels(merged, q_threshold=0.75):
    """
    Assign priority labels based on BRGI and CISN quantiles.
    """
    brgi_q = merged['brgi_norm'].quantile(q_threshold)
    cisn_q = merged['cisn_spillover_norm'].quantile(q_threshold)

    def label_row(row):
        brgi = row['brgi_norm']
        cisn = row['cisn_spillover_norm']

        if brgi >= brgi_q and cisn >= cisn_q:
            return 'Strategic Priority'
        elif brgi >= brgi_q and cisn < cisn_q:
            return 'Targeted Priority'
        elif brgi < brgi_q and cisn >= cisn_q:
            return 'Platform Opportunity'
        else:
            return 'Watch'

    merged['priority_label'] = merged.apply(label_row, axis=1)

    def interpretation(label):
        interpretations = {
            'Strategic Priority': 'Large forecasted gap and broad cross-RMD spillover value.',
            'Targeted Priority': 'Large forecasted gap but mostly disorder-specific value.',
            'Platform Opportunity': 'Broad cross-RMD value but lower immediate gap urgency.',
            'Watch': 'Lower gap urgency and lower spillover value.'
        }
        return interpretations.get(label, '')

    merged['interpretation'] = merged['priority_label'].apply(interpretation)

    return merged, brgi_q, cisn_q


# =============================================================================
# Visualization Functions
# =============================================================================

def make_combined_decision_figure(brgi, cisn, merged, edges=None, top_k=20):
    """
    Create the combined three-panel BRGI + CISN decision map.
    """
    brgi_q = merged['brgi_norm'].quantile(0.75)
    cisn_q = merged['cisn_spillover_norm'].quantile(0.75)

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: BRGI Bar Chart
    ax_a = fig.add_subplot(gs[0, 0])
    top_brgi = brgi.head(top_k).copy()
    top_brgi['label'] = (
        top_brgi['rmd_name'].apply(display_name) +
        ' → ' +
        top_brgi['pt_name'].apply(display_name)
    )
    y_pos = np.arange(len(top_brgi))
    colors = plt.cm.Blues(top_brgi['brgi_norm'].values)
    ax_a.barh(y_pos, top_brgi['brgi_score'].values, color=colors, edgecolor='black', linewidth=0.5)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(top_brgi['label'].values, fontsize=8)
    ax_a.set_xlabel('BRGI Gap Score', fontsize=10, fontweight='bold')
    ax_a.set_title('A. Largest Projected RMD–PT Gaps', fontsize=11, fontweight='bold', pad=10)
    ax_a.invert_yaxis()
    ax_a.grid(axis='x', alpha=0.3, linestyle='--')
    ax_a.text(
        0.98, 0.02,
        'Largest projected disorder–technology gaps',
        transform=ax_a.transAxes,
        ha='right', va='bottom',
        fontsize=8, style='italic', color='gray'
    )

    # Panel B: CISN Bar Chart
    ax_b = fig.add_subplot(gs[0, 1])
    top_cisn = cisn.head(15).copy()
    top_cisn['pt_display'] = top_cisn['pt_name'].apply(display_name)
    y_pos = np.arange(len(top_cisn))
    colors = plt.cm.Greens(top_cisn['cisn_spillover_norm'].values)
    ax_b.barh(y_pos, top_cisn['cisn_spillover_value'].values, color=colors, edgecolor='black', linewidth=0.5)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(top_cisn['pt_display'].values, fontsize=8)
    ax_b.set_xlabel('Spillover Value', fontsize=10, fontweight='bold')
    ax_b.set_title('B. PTs with Cross-RMD Spillover Potential', fontsize=11, fontweight='bold', pad=10)
    ax_b.invert_yaxis()
    ax_b.grid(axis='x', alpha=0.3, linestyle='--')
    ax_b.text(
        0.98, 0.02,
        'PTs with broader cross-RMD spillover potential',
        transform=ax_b.transAxes,
        ha='right', va='bottom',
        fontsize=8, style='italic', color='gray'
    )

    # Panel C: Priority Quadrant
    ax_c = fig.add_subplot(gs[1, :])

    color_map = {
        'Strategic Priority': '#d73027',
        'Targeted Priority': '#fee090',
        'Platform Opportunity': '#91bfdb',
        'Watch': '#e0e0e0'
    }

    for label in merged['priority_label'].unique():
        subset = merged[merged['priority_label'] == label]
        ax_c.scatter(
            subset['cisn_spillover_norm'],
            subset['brgi_norm'],
            s=subset['final_priority_score'] * 150,
            alpha=0.7,
            color=color_map.get(label, 'gray'),
            edgecolors='black',
            linewidth=0.5,
            label=label
        )

    ax_c.axvline(cisn_q, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax_c.axhline(brgi_q, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

    # Quadrant labels
    ax_c.text(
        0.95, 0.95, 'Strategic\n(High gap,\nhigh spillover)',
        transform=ax_c.transAxes, ha='right', va='top',
        fontsize=10, style='italic', color='darkred', fontweight='bold'
    )
    ax_c.text(
        0.05, 0.95, 'Targeted\n(High gap,\nspecific)',
        transform=ax_c.transAxes, ha='left', va='top',
        fontsize=10, style='italic', color='#b35806', fontweight='bold'
    )
    ax_c.text(
        0.95, 0.05, 'Platform\n(Low gap,\nhigh spillover)',
        transform=ax_c.transAxes, ha='right', va='bottom',
        fontsize=10, style='italic', color='#08519c', fontweight='bold'
    )
    ax_c.text(
        0.05, 0.05, 'Watch\n(Lower\npriority)',
        transform=ax_c.transAxes, ha='left', va='bottom',
        fontsize=10, style='italic', color='gray', fontweight='bold'
    )

    # Top 10 annotations
    top_10 = merged.nlargest(10, 'final_priority_score')
    for _, row in top_10.iterrows():
        ax_c.annotate(
            display_name(row['pt_name']),
            (row['cisn_spillover_norm'], row['brgi_norm']),
            fontsize=7, alpha=0.8,
            xytext=(5, 5), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.2),
            arrowprops=dict(arrowstyle='->', lw=0.5)
        )

    ax_c.set_xlabel('CISN Spillover (Normalized)', fontsize=11, fontweight='bold')
    ax_c.set_ylabel('BRGI Gap Score (Normalized)', fontsize=11, fontweight='bold')
    ax_c.set_title('C. Priority Matrix: Gap Urgency vs. Spillover Value', fontsize=12, fontweight='bold', pad=10)
    ax_c.set_xlim(-0.1, 1.1)
    ax_c.set_ylim(-0.1, 1.1)
    ax_c.grid(alpha=0.3, linestyle=':')
    ax_c.legend(loc='lower left', fontsize=9, framealpha=0.9)

    fig.suptitle(
        'BRGI + CISN Decision Map: Gap Urgency and Cross-RMD Spillover Value',
        fontsize=14, fontweight='bold', y=0.995
    )
    fig.text(
        0.5, 0.01,
        'Step 1: BRGI finds largest RMD–PT gaps. Step 2: CISN finds PTs with cross-RMD spillover. Step 3: Priority matrix combines both.',
        ha='center', fontsize=10, style='italic', color='gray'
    )

    return fig


# =============================================================================
# Output Functions
# =============================================================================

def save_priority_matrix(merged, out_dir):
    """
    Save priority matrix as CSV and markdown.
    """
    ensure_dir(out_dir)

    # Select and reorder columns for output
    output_cols = [
        'rank', 'rmd_name', 'pt_name', 'brgi_score', 'brgi_norm',
        'cisn_spillover_value', 'cisn_spillover_norm',
        'final_priority_score', 'priority_label', 'interpretation'
    ]

    output_df = merged[output_cols].copy()
    output_df['rmd_name'] = output_df['rmd_name'].apply(display_name)
    output_df['pt_name'] = output_df['pt_name'].apply(display_name)

    # CSV
    csv_path = os.path.join(out_dir, 'brgi_cisn_priority_matrix.csv')
    output_df.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")

    # Markdown
    md_path = os.path.join(out_dir, 'brgi_cisn_priority_matrix.md')
    with open(md_path, 'w') as f:
        f.write('# BRGI + CISN Priority Matrix\n\n')
        f.write('| Rank | RMD | PT | BRGI Score | BRGI Norm | Spillover | Spillover Norm | Final Priority | Label | Interpretation |\n')
        f.write('|------|-----|----|-----------:|----------:|----------:|---------------:|---------------:|-------|----------------|\n')

        for _, row in output_df.iterrows():
            f.write(
                f"| {int(row['rank'])} | {row['rmd_name']} | {row['pt_name']} | "
                f"{row['brgi_score']:.2f} | {row['brgi_norm']:.3f} | "
                f"{row['cisn_spillover_value']:.2f} | {row['cisn_spillover_norm']:.3f} | "
                f"{row['final_priority_score']:.3f} | {row['priority_label']} | {row['interpretation']} |\n"
            )

    print(f"✓ Saved: {md_path}")
    return csv_path, md_path


def write_summary_md(merged, out_dir):
    """
    Write plain-language summary markdown.
    """
    ensure_dir(out_dir)
    md_path = os.path.join(out_dir, 'brgi_cisn_decision_summary.md')

    with open(md_path, 'w') as f:
        f.write('# BRGI + CISN Decision Map Summary\n\n')

        f.write('## Plain-language Interpretation\n\n')
        f.write(
            'BRGI identifies where forecasted RMD burden exceeds PT readiness. '
            'CISN identifies PTs with broader cross-RMD spillover potential. '
            'Together, they convert many forecast curves into an actionable innovation-priority map.\n\n'
        )

        f.write('## How to Read the Figure\n\n')
        f.write('**Panel A** shows the largest RMD–PT gaps.\n\n')
        f.write('**Panel B** shows PTs with broader cross-RMD spillover value.\n\n')
        f.write('**Panel C** combines gap urgency and spillover value into four priority groups.\n\n')

        f.write('## Priority Categories\n\n')
        f.write('**Strategic Priority**: Large gap and broad cross-RMD utility.\n\n')
        f.write('**Targeted Priority**: Large gap but mostly disorder-specific value.\n\n')
        f.write('**Platform Opportunity**: Broad cross-RMD value but lower immediate gap urgency.\n\n')
        f.write('**Watch**: Lower immediate gap urgency and lower spillover value.\n\n')

        # Top 10 by priority label
        for label in ['Strategic Priority', 'Targeted Priority', 'Platform Opportunity', 'Watch']:
            subset = merged[merged['priority_label'] == label].head(10)
            if len(subset) > 0:
                f.write(f'## Top 10 {label}\n\n')
                f.write('| Rank | RMD | PT | BRGI | Spillover | Score |\n')
                f.write('|------|-----|----|----- |-----------|-------|\n')

                for _, row in subset.iterrows():
                    f.write(
                        f"| {int(row['rank'])} | {display_name(row['rmd_name'])} | "
                        f"{display_name(row['pt_name'])} | {row['brgi_score']:.2f} | "
                        f"{row['cisn_spillover_value']:.2f} | {row['final_priority_score']:.3f} |\n"
                    )

                f.write('\n')

        f.write('## Caution\n\n')
        f.write(
            'BRGI and CISN are study-specific decision-support indices. '
            'They are not clinical efficacy measures and should not be interpreted as proof '
            'that a PT works for a disorder.\n'
        )

    print(f"✓ Saved: {md_path}")
    return md_path


def write_manifest_json(out_dir, args, brgi, cisn, merged, edges, assumptions=None):
    """
    Write execution manifest JSON.
    """
    ensure_dir(out_dir)

    if assumptions is None:
        assumptions = []

    manifest = {
        'generation_timestamp': datetime.now().isoformat(),
        'input_files': {
            'brgi_csv': args.brgi_csv,
            'cisn_csv': args.cisn_csv,
            'edges_csv': args.edges_csv if args.edges_csv else 'not provided'
        },
        'data_summary': {
            'brgi_rows_loaded': len(brgi),
            'cisn_rows_loaded': len(cisn),
            'merged_rows': len(merged),
            'rows_per_priority_label': merged['priority_label'].value_counts().to_dict()
        },
        'parameters': {
            'gamma': args.gamma,
            'q_threshold': args.q_threshold,
            'top_k': args.top_k
        },
        'output_files': [
            'brgi_cisn_priority_matrix.csv',
            'brgi_cisn_priority_matrix.md',
            'Figure_BRGI_CISN_Decision_Map.png',
            'Figure_BRGI_CISN_Decision_Map.pdf',
            'brgi_cisn_decision_summary.md',
            'brgi_cisn_decision_manifest.json'
        ],
        'assumptions': assumptions,
        'warnings': []
    }

    if edges is None:
        manifest['warnings'].append('spillover_network_edges.csv not provided; Panel B uses bar chart only')

    json_path = os.path.join(out_dir, 'brgi_cisn_decision_manifest.json')
    with open(json_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ Saved: {json_path}")
    return json_path


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    args = parse_args()

    print('\n' + '='*70)
    print('BRGI + CISN Decision Map Generator')
    print('='*70 + '\n')

    try:
        # Load data
        print('Loading data...')
        brgi = load_brgi(args.brgi_csv)
        print(f"  ✓ BRGI: {len(brgi)} rows")

        cisn = load_cisn(args.cisn_csv)
        print(f"  ✓ CISN: {len(cisn)} rows")

        edges = load_edges(args.edges_csv)
        if edges is not None:
            print(f"  ✓ Edges: {len(edges)} rows")
        else:
            print(f"  ⊘ Edges: not provided or empty")

        # Merge and assign priorities
        print('\nMerging and computing priorities...')
        merged = merge_priority_matrix(brgi, cisn, gamma=args.gamma, q_threshold=args.q_threshold)
        print(f"  ✓ Merged: {len(merged)} RMD–PT pairs")

        merged, brgi_q, cisn_q = assign_priority_labels(merged, q_threshold=args.q_threshold)
        print(f"  ✓ Priority labels assigned")
        print(f"     - Strategic Priority: {len(merged[merged['priority_label'] == 'Strategic Priority'])}")
        print(f"     - Targeted Priority: {len(merged[merged['priority_label'] == 'Targeted Priority'])}")
        print(f"     - Platform Opportunity: {len(merged[merged['priority_label'] == 'Platform Opportunity'])}")
        print(f"     - Watch: {len(merged[merged['priority_label'] == 'Watch'])}")

        # Generate outputs
        print('\nGenerating outputs...')
        ensure_dir(args.out_dir)

        # Priority matrix
        csv_path, md_path = save_priority_matrix(merged, args.out_dir)

        # Summary markdown
        summary_path = write_summary_md(merged, args.out_dir)

        # Figure
        print('  Creating combined decision figure (Nature-style PDF + PNG)...')
        set_nature_style()
        fig = make_combined_decision_figure(brgi, cisn, merged, edges=edges, top_k=args.top_k)

        # Save as Nature-style vector PDF (primary) and PNG preview
        output_path = os.path.join(args.out_dir, 'Fig_BRGI_CISN_DecisionMap_Composite.pdf')
        save_nature_figure(fig, output_path, verbose=True)

        plt.close(fig)

        # Manifest
        manifest_path = write_manifest_json(args.out_dir, args, brgi, cisn, merged, edges)

        # Final message
        print('\n' + '='*70)
        print('✓ BRGI + CISN decision map generated successfully.')
        print('='*70)
        print('\nOutputs:')
        print(f'  - brgi_cisn_priority_matrix.csv')
        print(f'  - brgi_cisn_priority_matrix.md')
        print(f'  - Fig_BRGI_CISN_DecisionMap_Composite.pdf (vector format, primary)')
        print(f'  - Fig_BRGI_CISN_DecisionMap_Composite.png (preview)')
        print(f'  - brgi_cisn_decision_summary.md')
        print(f'  - brgi_cisn_decision_manifest.json')
        print(f'\nLocation: {args.out_dir}\n')

        return 0

    except Exception as e:
        print(f'\n✗ Error: {e}', file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
