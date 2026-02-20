import pandas as pd
import os

graphs = [
    'data/graph.csv',
    'data/graph_square.csv', 
    'data/graph_symnorm.csv',
    'data/graph_topk_k12.csv'
]

print('Graph File Verification:\n')
for g in graphs:
    df = pd.read_csv(g, header=None)
    size = os.path.getsize(g)
    print(f'{g}:')
    print(f'  Shape: {df.shape}')
    print(f'  Size: {size:,} bytes')
    print(f'  Non-zero: {(df.values != 0).sum():,}')
    print()
