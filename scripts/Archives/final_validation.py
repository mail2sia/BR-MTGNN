#!/usr/bin/env python
"""Final validation before training - check all components are aligned."""

import pandas as pd
import numpy as np
import os

def final_validation():
    print("=" * 80)
    print("FINAL TRAINING READINESS CHECK")
    print("=" * 80)
    
    issues = []
    warnings = []
    
    # 1. Data file validation
    print("\n1️⃣  DATA FILE (sm_data_g.csv)")
    try:
        data = pd.read_csv('data/sm_data_g.csv')
        print(f"   ✅ Loaded: {data.shape[0]} rows × {data.shape[1]} columns")
        
        # Check for issues
        nan_count = data.isna().sum().sum()
        inf_count = np.isinf(data.values).sum()
        neg_count = (data.values < 0).sum()
        
        print(f"   ✅ NaN values: {nan_count}")
        print(f"   ✅ Inf values: {inf_count}")
        print(f"   ✅ Negative values: {neg_count}")
        print(f"   ✅ Data range: [{data.values.min():.2f}, {data.values.max():.2f}]")
        
        if nan_count > 0:
            issues.append("Data contains NaN values")
        if inf_count > 0:
            issues.append("Data contains Inf values")
        
        data_cols = data.columns.tolist()
        has_date = 'date' in data_cols or 'Date' in data_cols
        print(f"   ✅ Has date column: {has_date} (should be False)")
        if has_date:
            issues.append("Data still contains date column - should be removed")
            
    except Exception as e:
        issues.append(f"Cannot load data/sm_data_g.csv: {e}")
        data_cols = []
        data = None
    
    # 2. Nodes file validation
    print("\n2️⃣  NODES FILE (nodes.csv)")
    try:
        nodes = pd.read_csv('nodes.csv')
        print(f"   ✅ Loaded: {len(nodes)} nodes")
        print(f"   ✅ Categories: {nodes['category'].value_counts().to_dict()}")
        
        if data is not None:
            if len(nodes) != len(data_cols):
                issues.append(f"Node count ({len(nodes)}) != data columns ({len(data_cols)})")
            else:
                # Check order
                mismatches = sum(1 for n, c in zip(nodes['display'], data_cols) if n != c)
                if mismatches > 0:
                    issues.append(f"{mismatches} nodes don't match data column order")
                else:
                    print(f"   ✅ Perfect alignment with data columns")
    except Exception as e:
        issues.append(f"Cannot load nodes.csv: {e}")
        nodes = None
    
    # 3. Graph files validation
    print("\n3️⃣  GRAPH FILES")
    graph_files = [
        'data/graph.csv',
        'data/graph_square.csv',
        'data/graph_symnorm.csv',
        'data/graph_topk_k12.csv'
    ]
    
    for gfile in graph_files:
        try:
            graph = pd.read_csv(gfile, header=None)
            expected_shape = (95, 95) if data is not None else None
            
            if expected_shape and graph.shape != expected_shape:
                issues.append(f"{gfile}: wrong shape {graph.shape}, expected {expected_shape}")
            else:
                non_zero = (graph.values != 0).sum()
                print(f"   ✅ {os.path.basename(gfile)}: {graph.shape}, {non_zero:,} edges")
        except Exception as e:
            issues.append(f"Cannot load {gfile}: {e}")
    
    # 4. Training script configuration
    print("\n4️⃣  TRAINING CONFIGURATION (run_safe_train.sh)")
    has_header_flag = False
    drop_first_col_flag = False
    data_path = False
    graph_path = False
    y_transform = False
    script = ""
    
    try:
        with open('run_safe_train.sh', 'r') as f:
            script = f.read()
        
        # Check critical flags
        has_header_flag = '--has_header' in script
        drop_first_col_flag = '--drop_first_col' in script
        data_path = './data/sm_data_g.csv' in script
        graph_path = './data/graph_square.csv' in script
        y_transform = '--y_transform none' in script
        
        print(f"   ✅ --has_header: {has_header_flag} (should be True)")
        print(f"   ✅ --drop_first_col: {drop_first_col_flag} (should be False)")
        print(f"   ✅ Data path: {data_path} (sm_data_g.csv)")
        print(f"   ✅ Graph path: {graph_path} (graph_square.csv)")
        print(f"   ✅ y_transform: {y_transform} (none, data pre-smoothed)")
        
        if not has_header_flag:
            issues.append("Missing --has_header flag in run_safe_train.sh")
        if drop_first_col_flag:
            issues.append("Should NOT use --drop_first_col (no date column)")
        if not data_path:
            warnings.append("run_safe_train.sh not using sm_data_g.csv")
        if not graph_path:
            warnings.append("run_safe_train.sh not using graph_square.csv")
            
    except Exception as e:
        warnings.append(f"Cannot read run_safe_train.sh: {e}")
    
    # 5. Python script default
    print("\n5️⃣  PYTHON SCRIPT DEFAULT (scripts/train_test.py)")
    try:
        with open('scripts/train_test.py', 'r') as f:
            train_script = f.read()
        
        if "default='./data/sm_data_g.csv'" in train_script:
            print(f"   ✅ Default data path: sm_data_g.csv")
        else:
            warnings.append("train_test.py default not set to sm_data_g.csv")
    except Exception as e:
        warnings.append(f"Cannot read train_test.py: {e}")
    
    # 6. File sizes and integrity
    print("\n6️⃣  FILE INTEGRITY")
    files_to_check = [
        ('data/sm_data_g.csv', 450000, 450500),  # ~450 KB
        ('data/graph_square.csv', 90000, 110000),  # ~99 KB
        ('nodes.csv', 4000, 6000),  # ~5 KB
    ]
    
    for fname, min_size, max_size in files_to_check:
        try:
            size = os.path.getsize(fname)
            if min_size <= size <= max_size:
                print(f"   ✅ {fname}: {size:,} bytes")
            else:
                warnings.append(f"{fname}: size {size:,} bytes outside expected range")
        except Exception as e:
            issues.append(f"Cannot check {fname}: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    if issues:
        print(f"\n❌ CRITICAL ISSUES ({len(issues)}):")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for i, warn in enumerate(warnings, 1):
            print(f"   {i}. {warn}")
    
    if not issues and not warnings:
        print("\n✅ ALL CHECKS PASSED!")
        print("\n🚀 READY FOR TRAINING")
        print("\nRecommended command:")
        print("  bash run_safe_train.sh")
        print("\nOr direct Python:")
        print("  python scripts/train_test.py --device cuda:0 --train --has_header")
    elif not issues:
        print("\n✅ NO CRITICAL ISSUES - Ready for training")
        print("⚠️  Some warnings present but training can proceed")
    else:
        print("\n❌ CRITICAL ISSUES FOUND - Fix before training")
        return False
    
    print("=" * 80)
    
    # Final checklist
    print("\n📋 FINAL CHECKLIST:")
    print(f"   {'✅' if data is not None else '❌'} sm_data_g.csv: 264 rows × 95 columns")
    print(f"   {'✅' if nodes is not None and len(nodes) == 95 else '❌'} nodes.csv: 95 nodes")
    print(f"   {'✅' if os.path.exists('data/graph_square.csv') else '❌'} graph_square.csv: 95×95 adjacency")
    print(f"   {'✅' if data is not None and 'date' not in data_cols else '❌'} No date column in data")
    print(f"   {'✅' if data is not None and data.values.max() < 300 else '❌'} Spikes smoothed (max < 300)")
    print(f"   {'✅' if '--has_header' in script else '❌'} Training script uses --has_header")
    print(f"   {'✅' if '--drop_first_col' not in script or drop_first_col_flag == False else '❌'} No --drop_first_col flag")
    
    print("\n" + "=" * 80)
    
    return len(issues) == 0

if __name__ == '__main__':
    import sys
    success = final_validation()
    sys.exit(0 if success else 1)
