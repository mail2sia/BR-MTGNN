#!/usr/bin/env python
"""Create graph adjacency matrix for sm_data_g.csv."""

import pandas as pd
import numpy as np

def create_graph_adjacency():
    print("=" * 80)
    print("CREATING GRAPH ADJACENCY MATRIX")
    print("=" * 80)
    
    # Load nodes
    nodes = pd.read_csv('data/nodes.csv')
    data = pd.read_csv('data/sm_data_g.csv')
    
    N = len(nodes)
    print(f"\nNodes: {N}")
    print(f"Data columns: {len(data.columns)}")
    
    # Create adjacency matrix
    adj = np.zeros((N, N), dtype=np.float32)
    
    # Get category for each node
    categories = nodes['category'].values
    
    # Build graph structure:
    # 1. Self-connections (diagonal)
    np.fill_diagonal(adj, 1.0)
    
    # 2. Global nodes connect to all nodes
    global_mask = (categories == 'Global')
    global_indices = np.where(global_mask)[0]
    for i in global_indices:
        adj[i, :] = 0.5  # Global to all
        adj[:, i] = 0.5  # All to global
        adj[i, i] = 1.0  # Self-connection stronger
    
    # 3. PT (treatments) connect to RMD (disorders) - bidirectional
    pt_mask = (categories == 'PT')
    rmd_mask = (categories == 'RMD')
    pt_indices = np.where(pt_mask)[0]
    rmd_indices = np.where(rmd_mask)[0]
    
    for pt_idx in pt_indices:
        for rmd_idx in rmd_indices:
            adj[pt_idx, rmd_idx] = 0.3  # PT -> RMD
            adj[rmd_idx, pt_idx] = 0.3  # RMD -> PT
    
    # 4. Within-category connections (weaker)
    for cat in ['PT', 'RMD']:
        cat_indices = np.where(categories == cat)[0]
        for i in cat_indices:
            for j in cat_indices:
                if i != j and adj[i, j] == 0:  # Don't override existing
                    adj[i, j] = 0.1
    
    # Statistics
    print(f"\n📊 Graph Statistics:")
    print(f"  Total edges: {np.sum(adj > 0)}")
    print(f"  Density: {np.sum(adj > 0) / (N * N) * 100:.2f}%")
    print(f"  Self-loops: {np.sum(np.diag(adj) > 0)}")
    print(f"  Non-zero values: {np.unique(adj[adj > 0])}")
    
    # Category breakdown
    print(f"\n📁 Category Connections:")
    print(f"  Global nodes: {np.sum(global_mask)} (connect to all)")
    print(f"  PT nodes: {np.sum(pt_mask)} (treatments)")
    print(f"  RMD nodes: {np.sum(rmd_mask)} (disorders)")
    print(f"  PT-RMD connections: {np.sum(adj[pt_indices][:, rmd_indices] > 0)}")
    
    # Save graph
    output_path = 'data/graph.csv'
    pd.DataFrame(adj).to_csv(output_path, index=False, header=False)
    print(f"\n✅ Saved: {output_path}")
    
    # Also create square graph (A @ A) for stronger multi-hop connections
    adj_square = adj @ adj
    # Normalize to [0, 1]
    if adj_square.max() > 0:
        adj_square = adj_square / adj_square.max()
    
    output_square = 'data/graph_square.csv'
    pd.DataFrame(adj_square).to_csv(output_square, index=False, header=False)
    print(f"✅ Saved: {output_square} (A @ A normalized)")
    
    # Symmetric normalized graph
    # D^(-1/2) A D^(-1/2)
    rowsum = adj.sum(axis=1)
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(d_inv_sqrt)
    adj_symnorm = D_inv_sqrt @ adj @ D_inv_sqrt
    
    output_symnorm = 'data/graph_symnorm.csv'
    pd.DataFrame(adj_symnorm).to_csv(output_symnorm, index=False, header=False)
    print(f"✅ Saved: {output_symnorm} (symmetric normalized)")
    
    # Top-k graph (keep strongest k connections per node)
    k = 12
    adj_topk = np.zeros_like(adj)
    for i in range(N):
        row = adj[i, :]
        if np.sum(row > 0) > k:
            # Keep top-k strongest connections
            top_indices = np.argsort(row)[-k:]
            adj_topk[i, top_indices] = row[top_indices]
        else:
            adj_topk[i, :] = row
    
    output_topk = f'data/graph_topk_k{k}.csv'
    pd.DataFrame(adj_topk).to_csv(output_topk, index=False, header=False)
    print(f"✅ Saved: {output_topk} (top-{k} per node)")
    
    print("\n" + "=" * 80)
    print("GRAPH CREATION COMPLETE")
    print("=" * 80)
    print("\nAvailable graph files:")
    print("  - data/graph.csv (base adjacency)")
    print("  - data/graph_square.csv (A @ A for multi-hop)")
    print("  - data/graph_symnorm.csv (symmetric normalized)")
    print(f"  - data/graph_topk_k{k}.csv (top-{k} connections)")
    print("\nRecommended for training: graph_square.csv or graph_topk_k12.csv")
    print("=" * 80)
    
    return adj

if __name__ == '__main__':
    create_graph_adjacency()
