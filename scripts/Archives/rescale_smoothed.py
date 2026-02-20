import numpy as np
import pandas as pd
import os


def read_csv(path):
    df = pd.read_csv(path)
    if df.shape[1] > 1 and str(df.columns[0]).lower() in ("date","month","time"):
        df = df.iloc[:,1:]
    return df.to_numpy(dtype=float)


def slice_indices(start_year=2004, steps_per_year=12, train_end=2017, valid_end=2021, test_end=2024):
    base = start_year
    train_len = (train_end - base + 1) * steps_per_year
    val_len = (valid_end - train_end) * steps_per_year
    test_len = (test_end - valid_end) * steps_per_year
    return (slice(0, train_len), slice(train_len, train_len+val_len), slice(train_len+val_len, train_len+val_len+test_len))


def rescale(orig_path, smooth_path, out_path):
    Xo = read_csv(orig_path)
    Xs = read_csv(smooth_path)
    # align lengths by trimming to minimum available rows
    minT = min(Xo.shape[0], Xs.shape[0])
    Xo = Xo[:minT]
    Xs = Xs[:minT]
    tr, va, te = slice_indices()
    # ensure slices fit into trimmed length
    Xo_tr = Xo[tr]
    Xs_tr = Xs[tr]
    T,N = Xs.shape
    S_fix = np.empty_like(Xs)
    for j in range(N):
        mean_o = np.nanmean(Xo_tr[:,j])
        std_o = np.nanstd(Xo_tr[:,j])
        mean_s = np.nanmean(Xs_tr[:,j])
        std_s = np.nanstd(Xs_tr[:,j])
        if std_s < 1e-12 or np.isnan(std_s):
            S_fix[:,j] = Xs[:,j]
        else:
            if std_o < 1e-12 or np.isnan(std_o):
                # if original node had near-zero std, preserve original series
                S_fix[:,j] = Xo[:,j]
            else:
                S_fix[:,j] = (Xs[:,j] - mean_s) * (std_o / std_s) + mean_o
    # write csv
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path,'w',newline='') as f:
        for row in S_fix:
            f.write(','.join([f"{x:.6f}" for x in row])+'\n')
    print('Wrote', out_path)
    return S_fix

if __name__ == '__main__':
    import sys
    orig = sys.argv[1] if len(sys.argv)>1 else 'data/data.csv'
    smooth = sys.argv[2] if len(sys.argv)>2 else 'data/sm_aggr_v4.csv'
    out = sys.argv[3] if len(sys.argv)>3 else 'data/sm_aggr_v4_fix.csv'
    rescale(orig, smooth, out)
