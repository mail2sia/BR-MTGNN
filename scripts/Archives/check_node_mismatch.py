#!/usr/bin/env python
"""Validate node names match data columns exactly."""

import pandas as pd
import numpy as np

def check_node_column_match():
    print("=" * 80)
    print("NODE-COLUMN VALIDATION")
    print("=" * 80)
    
    # Load nodes
    nodes_df = pd.read_csv('nodes.csv')
    node_names = nodes_df['display'].tolist()
    
    # Load data columns
    data_df = pd.read_csv('data/sm_data_g.csv')
    data_cols = data_df.columns.tolist()
    
    print(f"\n📊 Count Summary:")
    print(f"  Nodes in nodes.csv: {len(node_names)}")
    print(f"  Columns in data: {len(data_cols)}")
    print(f"  Difference: {len(node_names) - len(data_cols)}")
    
    # Convert to sets for comparison
    nodes_set = set(node_names)
    data_set = set(data_cols)
    
    # Find mismatches
    in_nodes_not_data = nodes_set - data_set
    in_data_not_nodes = data_set - nodes_set
    common = nodes_set & data_set
    
    print(f"\n🔍 Match Analysis:")
    print(f"  Common (matching): {len(common)}")
    print(f"  In nodes.csv but NOT in data: {len(in_nodes_not_data)}")
    print(f"  In data but NOT in nodes.csv: {len(in_data_not_nodes)}")
    
    # Show mismatches
    if in_nodes_not_data:
        print(f"\n⚠️  NODES MISSING FROM DATA ({len(in_nodes_not_data)}):")
        for i, node in enumerate(sorted(in_nodes_not_data), 1):
            # Find the token
            token = nodes_df[nodes_df['display'] == node]['token'].values
            category = nodes_df[nodes_df['display'] == node]['category'].values
            token_str = token[0] if len(token) > 0 else 'N/A'
            cat_str = category[0] if len(category) > 0 else 'N/A'
            print(f"  {i:2d}. [{cat_str}] {node} ({token_str})")
    
    if in_data_not_nodes:
        print(f"\n⚠️  DATA COLUMNS MISSING FROM NODES ({len(in_data_not_nodes)}):")
        for i, col in enumerate(sorted(in_data_not_nodes), 1):
            # Check if it looks like a global/aggregated column
            prefix = col.split('_')[0] if '_' in col else col
            print(f"  {i:2d}. {col} (prefix: {prefix})")
    
    # Check order alignment (important for graph structure)
    print(f"\n📋 Order Alignment Check:")
    if len(node_names) == len(data_cols):
        misaligned = []
        for i, (node, col) in enumerate(zip(node_names, data_cols)):
            if node != col:
                misaligned.append((i, node, col))
        
        if misaligned:
            print(f"  ⚠️  {len(misaligned)} positions have different names:")
            for idx, node, col in misaligned[:10]:  # Show first 10
                print(f"    Position {idx}: nodes='{node}' vs data='{col}'")
            if len(misaligned) > 10:
                print(f"    ... and {len(misaligned) - 10} more")
        else:
            print(f"  ✓ All {len(node_names)} positions match perfectly!")
    else:
        print(f"  ⚠️  Cannot check order - different lengths!")
    
    # Category breakdown
    print(f"\n📁 Node Categories:")
    for cat in nodes_df['category'].unique():
        count = (nodes_df['category'] == cat).sum()
        in_data = len([n for n in nodes_df[nodes_df['category'] == cat]['display'] if n in data_set])
        print(f"  {cat}: {count} nodes ({in_data} in data, {count - in_data} missing)")
    
    # Check data file structure
    print(f"\n📄 Data File Info:")
    print(f"  First 5 columns: {data_cols[:5]}")
    print(f"  Last 5 columns: {data_cols[-5:]}")
    
    # Global columns check (columns that might be aggregated metrics)
    global_cols = [c for c in data_cols if c.startswith('Global_')]
    if global_cols:
        print(f"\n🌍 Global/Aggregated Columns Found ({len(global_cols)}):")
        for col in global_cols:
            print(f"  - {col}")
    
    # Recommendations
    print(f"\n" + "=" * 80)
    if len(in_nodes_not_data) == 0 and len(in_data_not_nodes) == 0:
        print("✅ PERFECT MATCH - All nodes have corresponding data columns!")
    else:
        print("⚠️  MISMATCH DETECTED - Action Required:")
        if in_nodes_not_data:
            print(f"\n  1. {len(in_nodes_not_data)} nodes in nodes.csv are missing from data")
            print(f"     → Update nodes.csv to remove these, OR")
            print(f"     → Add missing columns to data with zeros/defaults")
        if in_data_not_nodes:
            print(f"\n  2. {len(in_data_not_nodes)} data columns are missing from nodes.csv")
            print(f"     → Add these to nodes.csv, OR")
            print(f"     → Remove from data if not needed")
        
        print(f"\n  Suggested Fix:")
        print(f"  Create nodes_from_data.csv with exact data columns:")
        print(f"  python -c \"import pandas as pd; df=pd.read_csv('data/sm_data_g.csv'); ")
        print(f"  nodes=pd.DataFrame({{'token':['N'+str(i).zfill(4) for i in range(len(df.columns))], ")
        print(f"  'display':df.columns, 'category':['AUTO' for _ in df.columns]}}); ")
        print(f"  nodes.to_csv('nodes_from_data.csv', index=False)\"")
    
    print("=" * 80)
    
    return {
        'nodes_count': len(node_names),
        'data_count': len(data_cols),
        'match_count': len(common),
        'missing_from_data': list(in_nodes_not_data),
        'missing_from_nodes': list(in_data_not_nodes),
        'perfect_match': len(in_nodes_not_data) == 0 and len(in_data_not_nodes) == 0
    }

if __name__ == '__main__':
    result = check_node_column_match()
    
    # Return exit code based on match status
    import sys
    sys.exit(0 if result['perfect_match'] else 1)
