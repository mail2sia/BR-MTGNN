# Unified DataLoaderS (global + rolling norm)
import os
import sys
import csv
import pickle
import json
from collections import defaultdict
import numpy as np

# -----------------------------
# Numerical-stability helpers
# -----------------------------
def _np_finite_float64(a):
    """Return float64 array with NaN/Inf replaced by 0 (transform/metrics safety)."""
    a = np.asarray(a, dtype=np.float64)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

def _safe_nanstd(a, axis=None, eps: float = 1e-12):
    """Compute nanstd without overflow by max-abs rescaling (scale-invariant)."""
    x = _np_finite_float64(a)
    # max over finite values
    scale = np.nanmax(np.abs(x), axis=axis, keepdims=True)
    scale = np.where(scale < eps, 1.0, scale)
    xs = x / scale
    s = np.nanstd(xs, axis=axis)
    return s * np.squeeze(scale, axis=axis)

def _safe_var(a, axis=None, eps: float = 1e-12):
    s = _safe_nanstd(a, axis=axis, eps=eps)
    return np.square(s, dtype=np.float64)
import pandas as pd
import torch
from torch.autograd import Variable
from typing import Union, cast, Dict
from datetime import datetime
from dateutil import parser as date_parser

def trend_aware_smooth_np(
    x: np.ndarray,
    *,
    alpha: float = 0.25,
    beta: float = 0.05,
    resid_alpha: float = 0.20,
    clamp_min: float | None = 0.0,
    spike_impute: bool = False,
    spike_window: int = 5,
    spike_sigma: float = 3.0,
    spike_min_mad: float = 1e-6,
) -> np.ndarray:
    """
        Trend-aware smoothing:
      1) Holt (level+trend) estimate
            2) optional spike impute vs trend (robust residual outliers)
            3) smooth residuals only (EMA)
            4) recombine trend + smoothed residual

    Works well for sparse/heavy-tailed monthly series:
      - preserves upward ramps
      - reduces noise without flattening peaks as much
    """
    X = np.asarray(x, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    T, M = X.shape

    out = np.zeros_like(X)

    for j in range(M):
        y = X[:, j].copy()
        if clamp_min is not None:
            y = np.maximum(y, clamp_min)

        def _holt_trend(series: np.ndarray) -> np.ndarray:
            l = series[0]
            b = series[1] - series[0] if T > 1 else 0.0
            tr = np.zeros(T, dtype=np.float64)
            tr[0] = l
            for t in range(1, T):
                l_new = alpha * series[t] + (1.0 - alpha) * (l + b)
                b_new = beta * (l_new - l) + (1.0 - beta) * b
                l, b = l_new, b_new
                tr[t] = l
            return tr

        # --- Holt level/trend (initial) ---
        trend = _holt_trend(y)

        # --- optional: spike imputation vs trend ---
        if spike_impute and T > 2:
            resid = y - trend
            # global robust scale fallback
            global_med = np.median(resid)
            global_mad = np.median(np.abs(resid - global_med))
            global_scale = 1.4826 * max(global_mad, spike_min_mad)

            win = max(3, int(spike_window))
            half = win // 2
            y_imputed = y.copy()
            for t in range(T):
                lo = max(0, t - half)
                hi = min(T, t + half + 1)
                rwin = resid[lo:hi]
                med = np.median(rwin)
                mad = np.median(np.abs(rwin - med))
                scale = 1.4826 * mad
                if not np.isfinite(scale) or scale < spike_min_mad:
                    scale = global_scale
                if np.isfinite(scale) and scale > 0:
                    if abs(resid[t] - med) > spike_sigma * scale:
                        y_imputed[t] = trend[t] + med
            y = y_imputed
            # Recompute trend after imputation so spikes don't bias the trend
            trend = _holt_trend(y)

        # --- smooth residuals only ---
        resid = y - trend
        sr = resid[0]
        for t in range(1, T):
            sr = resid_alpha * resid[t] + (1.0 - resid_alpha) * sr
            resid[t] = sr

        out[:, j] = trend + resid

    return out.astype(np.float32)

def normal_std(x):
    # robust: if not enough samples, return 0.0
    try:
        n = len(x)
        if n <= 1:
            return 0.0
        return x.std() * np.sqrt((n - 1.)/n)
    except Exception:
        return 0.0

def parse_dates_from_csv(file_name: str) -> tuple[list, int, int]:
    """
    Parse dates from CSV file if the first column contains date strings.
    
    Returns:
        tuple: (dates_list, start_year, start_month)
               dates_list: List of parsed dates
               start_year: Year of the first date (0 if no dates parsed)
               start_month: Month of the first date (0 if no dates parsed)
    """
    dates = []
    start_year = 0
    start_month = 0
    
    try:
        _delim = ',' if str(file_name).lower().endswith('.csv') else '\t'
        
        with open(file_name, 'r') as f:
            reader = csv.reader(f, delimiter=_delim)
            header_row = next(reader, None)
            
            # Check if first column header suggests it's a date column
            if header_row and header_row[0].lower().strip() in ['date', 'dates', 'time', 'timestamp']:
                # Read data rows
                for row in reader:
                    if row and row[0]:
                        try:
                            # Parse date string (handles formats like "2004-01", "2004-01-15", etc.)
                            date_str = row[0].strip()
                            parsed_date = date_parser.parse(date_str, default=datetime(2000, 1, 1))
                            dates.append(parsed_date)
                        except Exception:
                            # If parsing fails, skip this row
                            continue
                
                # Extract start year and month from first date
                if dates:
                    start_year = dates[0].year
                    start_month = dates[0].month
                    print(f"[DateParsing] Parsed {len(dates)} dates from {file_name}, starting from {start_year:04d}-{start_month:02d}")
    
    except Exception as e:
        print(f"[DateParsing] Could not parse dates from {file_name}: {e}")
    
    return dates, start_year, start_month

class DataLoaderS(object):
    # CLASS-LEVEL TYPE HINTS for static analysis / IDEs
    # These are populated at runtime in __init__ but declaring them here
    # avoids false-positive 'unknown attribute' warnings in editors.
    col: list = []
    adj: Union[torch.Tensor, None] = None
    active_mask: Union[torch.Tensor, None] = None
    rse: float = 0.0
    rae: float = 0.0
    # Declared here for static analysis (populated in __init__)
    use_log1p: bool
    use_yj: bool
    yj_lambdas: Union[np.ndarray, None]
    rolling: bool
    raw_levels: Union[np.ndarray, None]
    per_window_mu: Dict[str, np.ndarray]
    per_window_std: Dict[str, np.ndarray]
    _inv_override: Union[tuple, None]
    y_offset: float
    y_transform_selective: bool
    y_transform_std_min: float
    y_transform_skip_neg: bool
    clip_outliers: bool
    clip_percentile: float
    clip_thresholds: Union[np.ndarray, None]
    dat_in: Union[np.ndarray, None]
    dat: Union[np.ndarray, None]
    diff_dat: Union[np.ndarray, None]
    in_dim: int
    dual: Union[str, None]
    mu: Union[torch.Tensor, None]
    std: Union[torch.Tensor, None]

    @staticmethod
    def get_plot_node_name(col_name: str) -> str:
        """Remove RMD_ and PT_ prefixes from column names for clean plot titles.
        
        Args:
            col_name: Original column name (e.g., "RMD_Node1", "PT_Tech2")
        
        Returns:
            Cleaned name without prefix (e.g., "Node1", "Tech2")
        """
        name = str(col_name)
        # Remove RMD_ prefix
        if name.startswith("RMD_"):
            name = name[4:]
        # Remove PT_ prefix
        if name.startswith("PT_"):
            name = name[3:]
        return name

    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, device, horizon, window, normalize=2, out=1,
                 chronological: bool = False,
                 start_year=None,
                 steps_per_year=None,
                 train_end_year=None,
                 valid_end_year=None,
                 test_end_year=None,
                 dual_channel: str | None = None,
                 pct_clip: float | None = None,
                 y_transform: Union[str, None] = None,
                 trend_smooth: bool = False,
                 trend_alpha: float = 0.25,
                 trend_beta: float = 0.05,
                 resid_alpha: float = 0.20,
                 has_header: bool = False,
                 drop_first_col: bool = False,
                 exclude_names: Union[str, list] | None = None,
                 y_offset: float = 0.0,
                 auto_y_offset: bool = False,
                 y_transform_selective: bool = False,
                 y_transform_std_min: float = 1e-6,
                 y_transform_skip_neg: bool = True,
                 clip_outliers: bool = False,
                 clip_percentile: float = 95.0):
        # Basic window/horizon/device
        self.P = window
        self.h = horizon
        self.device = device
        
        # Parse dates from the CSV file (if available)
        self.dates, self.start_year, self.start_month = parse_dates_from_csv(file_name)
        self.steps_per_year = 12  # Assuming monthly data as confirmed by user

        # Load data and ensure 2D shape (choose delimiter by extension)
        _delim = ',' if str(file_name).lower().endswith('.csv') else '\t'
        
        # Auto-detect header if not explicitly specified
        detected_header = has_header
        if not detected_header:
            # Try to detect if first row is a header by checking if it's numeric
            try:
                with open(file_name) as fin:
                    first_line = fin.readline().strip()
                    if first_line:
                        # Try to parse first line as floats
                        try:
                            _parts = first_line.split(_delim)
                            for part in _parts[:min(3, len(_parts))]:  # Check first 3 values
                                float(part.strip())
                        except (ValueError, IndexError):
                            # First line contains non-numeric data -> it's a header
                            detected_header = True
            except Exception:
                pass
        
        if detected_header:
            df = pd.read_csv(file_name)
            if drop_first_col and df.shape[1] > 1:
                df = df.iloc[:, 1:]
            raw = df.to_numpy()
        else:
            with open(file_name) as fin:
                raw = np.loadtxt(fin, delimiter=_delim)
        self.rawdat = np.atleast_2d(raw)
        self.rawdat_orig = self.rawdat.copy()

        # --- OPTIONAL: trend-aware smoothing (preserves trend, reduces noise) ---
        if trend_smooth:
            try:
                self.rawdat = trend_aware_smooth_np(
                    self.rawdat,
                    alpha=float(trend_alpha),
                    beta=float(trend_beta),
                    resid_alpha=float(resid_alpha),
                    clamp_min=0.0,
                )
            except Exception as _e:
                print(f"[warn] trend_smooth failed, continuing without it: {_e}")

        # selective transform options
        self.y_transform_selective = bool(y_transform_selective)
        self.y_transform_std_min = float(y_transform_std_min)
        self.y_transform_skip_neg = bool(y_transform_skip_neg)

        # Optional: clip outliers per column at specified percentile
        self.clip_outliers = bool(clip_outliers)
        self.clip_percentile = float(clip_percentile)
        self.clip_thresholds = None
        if self.clip_outliers:
            try:
                # Compute per-column clip threshold at specified percentile
                clip_thresholds = np.percentile(self.rawdat, self.clip_percentile, axis=0)
                # Clip each column
                for col_idx in range(self.rawdat.shape[1]):
                    self.rawdat[:, col_idx] = np.clip(self.rawdat[:, col_idx], 
                                                       a_min=None, 
                                                       a_max=clip_thresholds[col_idx])
                clipped_cols = np.sum(clip_thresholds < np.max(self.rawdat_orig, axis=0))
                print(f"[DataLoaderS] Clipped outliers at {self.clip_percentile}th percentile in {clipped_cols}/{self.rawdat.shape[1]} columns")
                self.clip_thresholds = clip_thresholds
            except Exception as e:
                print(f"[DataLoaderS] Warning: Outlier clipping failed: {e}")
                self.clip_thresholds = None

        # Optional: apply a global positive offset so all series are >= 0
        # Useful when you want non-negative outputs but the data contains negatives.
        self.y_offset = float(y_offset) if y_offset is not None else 0.0
        if bool(auto_y_offset):
            try:
                min_val = float(np.nanmin(self.rawdat))
                if np.isfinite(min_val) and min_val < 0.0:
                    # ensure strictly non-negative after shift
                    self.y_offset = -min_val + 1e-6
            except Exception:
                pass
        if self.y_offset != 0.0:
            try:
                self.rawdat = self.rawdat + self.y_offset
            except Exception:
                pass

        # --- NEW: normalization config ---
        # normalize modes: legacy values kept; 4 -> rolling per-window normalization
        # Allow explicit request to apply a target transform (e.g., log1p) via
        # the `y_transform` argument. This makes applying log1p in the loader
        # explicit (recommended) instead of encoding it in `normalize==3` only.
        ytf = str(y_transform).lower() if y_transform is not None else 'none'
        # Yeo-Johnson transform reference:
        # Yeo, I.-K., & Johnson, R. A. (2000). A new family of power transformations to improve
        # normality or symmetry. Biometrika, 87(4), 954–959. https://doi.org/10.1093/biomet/87.4.954
        self.use_yj = ytf in ('yeo-johnson', 'yeo_johnson', 'yj')
        self.use_log1p = (normalize == 3) or (ytf == 'log1p')  # 3 := log1p + z-score
        if self.use_yj:
            # Yeo-Johnson is mutually exclusive with log1p
            self.use_log1p = False
        self.rolling = (normalize == 4)
        # Optional attributes populated later; declare defaults for static analyzers
        self.raw_levels = None
        self.yj_lambdas = None
        # per-split per-window stats stored as dicts: keys 'train','valid','test' -> arrays shape (n_windows, m)
        self.per_window_mu = {}
        self.per_window_std = {}
        self._inv_override = None

        # Raw numeric matrix as a torch tensor on the right device
        # Apply optional transforms BEFORE normalization.
        if self.use_yj:
            rd = self.rawdat.copy()
            try:
                # selective transform mask: keep originals for flat/zero/negative cols
                if self.y_transform_selective:
                    col_min = np.nanmin(self.rawdat_orig, axis=0)
                    col_std = _safe_nanstd(self.rawdat_orig, axis=0)
                    skip_mask = (col_std < self.y_transform_std_min)
                    if self.y_transform_skip_neg:
                        skip_mask = skip_mask | (col_min <= 0)
                else:
                    skip_mask = np.zeros(rd.shape[1], dtype=bool)

                self.yj_lambdas = self._fit_yeo_johnson_lambdas(rd)
                if self.yj_lambdas is not None and skip_mask.any():
                    # lambda=1 gives identity for Yeo-Johnson
                    self.yj_lambdas[skip_mask] = 1.0
                rd = self._apply_yeo_johnson(rd, self.yj_lambdas)
            except Exception:
                # fallback to raw if fitting fails
                self.yj_lambdas = None
            Y = torch.as_tensor(rd, dtype=torch.float32, device=device)
        elif self.use_log1p:
            try:
                if self.y_transform_selective:
                    col_min = np.nanmin(self.rawdat_orig, axis=0)
                    col_std = _safe_nanstd(self.rawdat_orig, axis=0)
                    skip_mask = (col_std < self.y_transform_std_min)
                    if self.y_transform_skip_neg:
                        skip_mask = skip_mask | (col_min <= 0)
                else:
                    skip_mask = np.zeros(self.rawdat.shape[1], dtype=bool)

                rd = self.rawdat.copy()
                if skip_mask.any():
                    idx = np.where(~skip_mask)[0]
                    rd[:, idx] = np.log1p(np.clip(rd[:, idx], 0.0, None))
                else:
                    rd = np.log1p(np.clip(rd, 0.0, None))
                Y = torch.as_tensor(rd, dtype=torch.float32, device=device)
            except Exception:
                Y = torch.as_tensor(self.rawdat, dtype=torch.float32, device=device)
        else:
            Y = torch.as_tensor(self.rawdat, dtype=torch.float32, device=device)

        # --- NEW: chronological split ---
        self.chronological = chronological
        if self.chronological:
            # When using a fixed calendar split, the validation set should cover the
            # entire period specified by the year ranges, not a fractional ratio.
            # We set `valid` to 1.0 to signal that the validation data loader should
            # use all available data within its designated year range. The test set
            # is handled separately by the sliding window evaluation.
            valid = 1.0

        # Train/valid/test cut points
        n = Y.size(0)
        n_train = int(round(train * n))

        # --- NEW: train-only mean/std (only used for global normalization) ---
        # If chronological split is fully specified, compute stats from the
        # exact train-year segment; otherwise, fall back to the ratio slice.
        if self.chronological and all(v is not None for v in (start_year, steps_per_year, train_end_year, valid_end_year, test_end_year)):
            try:
                sy = int(start_year)  # type: ignore[arg-type]
                spy = int(steps_per_year)  # type: ignore[arg-type]
                tey = int(train_end_year)  # type: ignore[arg-type]

                def year_last_index(y: int) -> int:
                    return (y - sy + 1) * spy - 1

                global_last = int(n - 1)
                train_first = 0
                train_last = min(year_last_index(tey), global_last)
                if train_last >= train_first:
                    Y_train = Y[train_first:train_last + 1]
                else:
                    # degenerate boundaries -> fallback
                    Y_train = Y[:n_train]
            except Exception:
                Y_train = Y[:n_train]
        else:
            Y_train = Y[:n_train]

        self.mu = Y_train.mean(dim=0)  # (m,)
        std_raw = Y_train.std(dim=0)
        # Guard against near-constant series in the training set which would
        # produce extremely small std and explode z-space values after division.
        # For such series, replace std with 1.0 (no scaling) and mark them inactive.
        tiny_thresh = 1e-6
        small_mask = (std_raw < tiny_thresh)
        if small_mask.any():
            # log how many constant/near-constant nodes we saw
            try:
                print(f"[Warning] {int(small_mask.sum().item())} node(s) have tiny train-std; treating as inactive")
            except Exception:
                pass

        std_safe = std_raw.clone()
        # Clamp very small stds to avoid exploding z-scores, then set constant-series to 1.0
        std_safe = torch.clamp(std_safe, min=1e-6)
        std_safe[small_mask] = 1.0
        # Apply configurable median-based floor to per-node std to keep
        # inversion behaviour consistent across codepaths. This mirrors the
        # logic used in StandardScaler.inverse_transform and is tunable
        # via env vars `BMTGNN_STD_FLOOR_MULT` and `BMTGNN_STD_FLOOR_MIN`.
        try:
            std_abs = std_safe.abs()
            try:
                mult = float(os.environ.get('BMTGNN_STD_FLOOR_MULT', '0.05'))
            except Exception:
                mult = 0.05
            try:
                min_floor = float(os.environ.get('BMTGNN_STD_FLOOR_MIN', '1e-3'))
            except Exception:
                min_floor = 1e-3
            try:
                med = float(torch.median(std_abs))
            except Exception:
                med = 0.0
            floor = max(med * mult, min_floor)
            # clamp in-place
            num_low = int((std_abs < floor).sum().item()) if std_abs.numel() > 0 else 0
            if num_low > 0 and os.environ.get('BMTGNN_DEBUG_SCALE', '0') == '1':
                try:
                    print(f'[BMTGNN_DEBUG_SCALE] clamped {num_low} std entries to floor={floor}')
                except Exception:
                    pass
            std_safe = std_safe.clamp(min=floor)
        except Exception:
            pass
        self.std = std_safe
        # Mask nodes that are constant on train (we marked them above)
        self.active_mask = ~small_mask

        # --- Optional: exclude named series from targets/training ---
        # Accept either an explicit `exclude_names` parameter (list or comma-separated
        # string) or the environment variable `BMTGNN_EXCLUDE_NAMES` (comma-separated).
        try:
            env_ex = os.environ.get('BMTGNN_EXCLUDE_NAMES', None)
        except Exception:
            env_ex = None
        ex_val = exclude_names if exclude_names is not None else env_ex
        if ex_val:
            if isinstance(ex_val, str):
                names = [n.strip() for n in ex_val.split(',') if n.strip()]
            else:
                names = list(ex_val)
            try:
                cols = self.create_columns()
            except Exception:
                cols = []
            name_to_idx = {name: idx for idx, name in enumerate(cols)}
            exclude_idx = [name_to_idx[n] for n in names if n in name_to_idx]
            if len(exclude_idx) > 0:
                try:
                    am = self.active_mask.clone()
                except Exception:
                    am = self.active_mask
                for idx in exclude_idx:
                    if 0 <= idx < am.numel():
                        am[idx] = False
                self.active_mask = am
                # Neutralize stats for excluded nodes so they don't influence inversion
                try:
                    std_tmp = self.std.clone()
                    mu_tmp = self.mu.clone()
                except Exception:
                    std_tmp = self.std
                    mu_tmp = self.mu
                for idx in exclude_idx:
                    if 0 <= idx < std_tmp.numel():
                        std_tmp[idx] = 1.0
                        mu_tmp[idx] = 0.0
                        try:
                            self.rawdat[:, idx] = 0.0
                        except Exception:
                            pass
                self.std = std_tmp
                self.mu = mu_tmp
                self.exclude_idx = exclude_idx
                if os.environ.get('BMTGNN_DEBUG_SCALE', '0') == '1':
                    try:
                        print(f"[BMTGNN_DEBUG_SCALE] excluded {len(exclude_idx)} series: {exclude_idx}")
                    except Exception:
                        pass

        # Normalize ENTIRE series with train stats (global z-score) unless
        # rolling normalization requested. For rolling mode we keep original
        # levels in `self.raw_levels` and will compute per-window z during
        # batchification.
        def _norm(Z):
            return (Z - self.mu) / self.std

        if not self.rolling:
            Yz = _norm(Y)
        else:
            # keep raw levels (after optional log1p) in numpy for per-window ops
            Yz = None
            self.raw_levels = Y.cpu().numpy()

        # Allocate arrays and metadata
        self.n = int(Y.size(0))
        self.m = int(Y_train.size(1))
        self.normalize = normalize
        self.out_len = out

        # Use normalized data for splits
        dat = Yz.cpu().numpy() if Yz is not None else None  # (T, N) level in z-space or None for rolling
        # Clip threshold for pct channel (None or <=0 disables)
        try:
            pct_clip_val = float(pct_clip) if pct_clip is not None else 0.0
        except Exception:
            pct_clip_val = 0.0
        self.pct_clip = pct_clip_val if pct_clip_val > 0 else 0.0

        # Build dual-channel inputs: channel 0 = level, channel 1 = short-term movement
        if not self.rolling:
            # dat must be populated in non-rolling mode (global normalization)
            assert dat is not None, "Internal error: dat is None in non-rolling mode"
            if dual_channel in ('diff', 'pct'):
                mov = np.zeros_like(dat)
                mov[1:, :] = dat[1:, :] - dat[:-1, :]
                if dual_channel == 'pct':
                    mov[1:, :] = (dat[1:, :] - dat[:-1, :]) / (np.abs(dat[:-1, :]) + 1e-6)
                    if self.pct_clip > 0:
                        mov = np.clip(mov, -self.pct_clip, self.pct_clip)
                self.dat_in = np.stack([dat, mov], axis=-1)  # (T, N, 2)
                self.in_dim = 2
                self.dual = dual_channel
            else:
                self.dat_in = dat[..., None]  # (T, N, 1)
                self.in_dim = 1
                self.dual = None
        else:
            # For rolling mode we'll build per-window standardized inputs in _batchify
            self.dat_in = None
            self.in_dim = 2 if dual_channel in ('diff', 'pct') else 1
            self.dual = dual_channel

        self.dat = dat
        self.diff_dat = np.zeros_like(self.dat) if self.dat is not None else None

        # Perform either ratio-based split (legacy) or explicit chronological split
        if chronological and all(v is not None for v in (start_year, steps_per_year, train_end_year, valid_end_year, test_end_year)):
            try:
                # Assign after presence check; typing tools now know they're not None
                sy = int(start_year)  # type: ignore[arg-type]
                spy = int(steps_per_year)  # type: ignore[arg-type]
                tey = int(train_end_year)  # type: ignore[arg-type]
                vey = int(valid_end_year)  # type: ignore[arg-type]
                tey2 = int(test_end_year)  # type: ignore[arg-type]
                self._split_chronological(sy, spy, tey, vey, tey2)
            except Exception as e:
                print(f"[ChronologicalSplitWarning] Falling back to ratio split due to error: {e}")
                self._split(int(train * self.n), int((train + valid) * self.n), self.n)
        else:
            self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        # Prepare metrics (robust to Tensor/ndarray) using TEST split just created
        tmp = self.test[1]
        if isinstance(tmp, np.ndarray):  # from earlier design; now usually tensor
            tmp = torch.from_numpy(tmp)
        tmp = tmp.float()
        self.rse = float(normal_std(tmp))
        self.rae = float(torch.mean(torch.abs(tmp - torch.mean(tmp))).item())

        # Graph and columns
        self.adj = self.build_predefined_adj()
        DataLoaderS.col = self.create_columns()

    # --- NEW: inverse/broadcast helpers ---
    def _expand_like(self, stat, x):
        # broadcasts (m,) -> (..., m) to match x
        # If stat has a batch dimension matching x's batch, expand per-sample
        # For torch tensors
        if torch.is_tensor(stat):
            # stat: (m,) -> expand to (..., m)
            if stat.dim() == 1:
                view = [1] * x.dim()
                view[-1] = -1
                return stat.view(*view).expand_as(x)
            # stat: (B, m) and x: (B, ..., m)
            if stat.dim() == x.dim() - 1 and stat.size(0) == x.size(0):
                # reshape to (B, 1, ..., 1, m) where the middle dims match x
                shape = [stat.size(0)] + [1] * (x.dim() - 2) + [stat.size(1)]
                return stat.view(*shape).expand_as(x)
            # stat already matches x dims
            if stat.dim() == x.dim():
                return stat.expand_as(x)
            # fallback: attempt reshape to last-dim broadcast
            view = [1] * x.dim(); view[-1] = -1
            return stat.view(*view).expand_as(x)

        # For numpy arrays
        stat_arr = np.asarray(stat)
        if stat_arr.ndim == 1:
            shape = list(x.shape)
            view = [1] * len(shape); view[-1] = -1
            return np.broadcast_to(stat_arr.reshape(*view), shape)
        if stat_arr.ndim == x.ndim - 1 and stat_arr.shape[0] == x.shape[0]:
            # reshape to (B, 1, ..., 1, m)
            shape = (stat_arr.shape[0],) + (1,) * (x.ndim - 2) + (stat_arr.shape[1],)
            return np.broadcast_to(stat_arr.reshape(shape), x.shape)
        # fallback
        shape = list(x.shape)
        view = [1] * len(shape); view[-1] = -1
        return np.broadcast_to(stat_arr.reshape(*view), shape)

    def inv_transform_like(self, x, idx=None, split=None):
        """Invert z-score (and log1p if enabled) for any tensor whose last dim is m."""
        # Backward-compatible: allow optional per-sample mu/std via attributes
        # If caller passed a tuple (x, mu, std) use mu/std; but keep original
        # signature for existing callers by supporting named args below.
        # This function also accepts mu/std as attributes of self when rolling.
        # x: tensor or numpy array; mu/std can be tensors or numpy arrays shaped (m,) or (B,m)
        # If rolling mode, callers should provide mu/std arrays for per-sample inversion.
        # Default behaviour (global): use self.mu/self.std
        # Allow per-batch per-window stats when rolling: provide idx (array-like) and split name
        if self.rolling and idx is not None and split is not None:
            try:
                mu_arr = self.per_window_mu.get(split, None)
                std_arr = self.per_window_std.get(split, None)
                if mu_arr is None or std_arr is None:
                    raise RuntimeError('per-window stats missing for split')
                # idx may be tensor or numpy
                if torch.is_tensor(idx):
                    sel = idx.cpu().numpy()
                else:
                    sel = np.asarray(idx)
                mu_t = mu_arr[sel]
                std_t = std_arr[sel]
            except Exception:
                # fallback to global behaviour
                if hasattr(self, '_inv_override') and self._inv_override is not None:
                    mu_t, std_t = self._inv_override
                else:
                    mu_t = self.mu
                    std_t = self.std
        else:
            if hasattr(self, '_inv_override') and self._inv_override is not None:
                mu_t, std_t = self._inv_override
            else:
                mu_t = self.mu
                std_t = self.std

        if mu_t is None or std_t is None:
            raise RuntimeError('inv_transform_like called before mu/std are set')

        # Convert mu/std to match x's type/device
        if torch.is_tensor(x):
            if torch.is_tensor(mu_t):
                mu_t = mu_t.to(x.device).type_as(x)
            else:
                mu_t = torch.as_tensor(mu_t, device=x.device, dtype=x.dtype)
            if torch.is_tensor(std_t):
                std_t = std_t.to(x.device).type_as(x)
            else:
                std_t = torch.as_tensor(std_t, device=x.device, dtype=x.dtype)
        else:
            # x is numpy; ensure mu/std are numpy arrays
            if torch.is_tensor(mu_t):
                mu_t = mu_t.cpu().numpy()
            if torch.is_tensor(std_t):
                std_t = std_t.cpu().numpy()

        try:
            mu  = self._expand_like(mu_t, x)
            std = self._expand_like(std_t, x)
        except Exception as e:
            # Replace transient prints with an informative exception containing
            # the relevant shapes to aid debugging without noisy stdout.
            xs = getattr(x, 'shape', None)
            muts = getattr(mu_t, 'shape', None)
            stdts = getattr(std_t, 'shape', None)
            raise RuntimeError(f"inv_transform_like failed to expand mu/std to match x. x.shape={xs}, mu.shape={muts}, std.shape={stdts}") from e
        y = x * std + mu
        if self.use_yj and self.yj_lambdas is not None:
            y = self._inv_yeo_johnson(y, self.yj_lambdas)
        elif self.use_log1p:
            # torch.expm1 works on tensors; for numpy arrays, use np.expm1
            if torch.is_tensor(y):
                y = torch.expm1(y)
            else:
                y = np.expm1(y)
        # Remove optional global offset after inversion
        if getattr(self, 'y_offset', 0.0) not in (0.0, None):
            if torch.is_tensor(y):
                y = y - float(self.y_offset)
            else:
                y = y - float(self.y_offset)
        return y

    # Helper API for callers that need per-batch inversion when rolling.
    def inv_transform_with_stats(self, x, mu, std):
        """Invert using provided per-sample mu/std. mu/std shape: (B,m) or (m,)"""
        if torch.is_tensor(x):
            mu_t = torch.as_tensor(mu, device=x.device, dtype=x.dtype) if not torch.is_tensor(mu) else mu.to(x.device).type_as(x)
            std_t = torch.as_tensor(std, device=x.device, dtype=x.dtype) if not torch.is_tensor(std) else std.to(x.device).type_as(x)
        else:
            mu_t = mu
            std_t = std
        # Expand to match x
        view = [1] * x.dim()
        # place -1 on last dim
        view[-1] = -1
        # if mu_t has batch dim, try to align with x
        if torch.is_tensor(x) and mu_t.dim() == 2 and x.dim() >= 3:
            # x shape expected (B, L, m) or (B, m)
            # mu_t shape (B, m) -> reshape to (B,1,m)
            mu_expand = mu_t.view(mu_t.size(0), 1, mu_t.size(1)).expand_as(x)
            std_expand = std_t.view(std_t.size(0), 1, std_t.size(1)).expand_as(x)
            y = x * std_expand + mu_expand
            if self.use_log1p:
                y = torch.expm1(y)
            if self.use_yj and self.yj_lambdas is not None:
                y = self._inv_yeo_johnson(y, self.yj_lambdas)
            if getattr(self, 'y_offset', 0.0) not in (0.0, None):
                y = y - float(self.y_offset)
            return y
        else:
            return self.inv_transform_like(x)

    # ---- Yeo-Johnson transform helpers ----
    # Reference: Yeo, I.-K., & Johnson, R. A. (2000). Biometrika, 87(4), 954–959.
    # https://doi.org/10.1093/biomet/87.4.954
    def _yeo_johnson(self, x: np.ndarray, lmbda: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        pos = x >= 0
        neg = ~pos
        if lmbda != 0:
            out[pos] = ((x[pos] + 1.0) ** lmbda - 1.0) / lmbda
        else:
            out[pos] = np.log1p(x[pos])
        if lmbda != 2:
            out[neg] = -(((1.0 - x[neg]) ** (2.0 - lmbda) - 1.0) / (2.0 - lmbda))
        else:
            out[neg] = -np.log1p(-x[neg])
        return out

    def _inv_yeo_johnson(self, y, lambdas):
        # y can be torch or numpy; lambdas is ndarray of shape (m,)
        if torch.is_tensor(y):
            lmbda = torch.as_tensor(lambdas, device=y.device, dtype=y.dtype)
            view = [1] * y.dim(); view[-1] = -1
            lmbda = lmbda.view(*view)
            
            # Identity shortcut: lambda=1 -> inverse is identity (y = x)
            identity_mask = (lmbda == 1)
            if identity_mask.all():
                return y.clone()
            
            pos = y >= 0
            out = torch.zeros_like(y)

            # pos branch
            pos_l0 = (lmbda == 0)
            pos_l1 = (lmbda == 1)
            lmbda_safe = torch.where(pos_l0 | pos_l1, torch.ones_like(lmbda), lmbda)
            out_pos = torch.where(
                pos_l0,
                torch.expm1(y),
                torch.where(
                    pos_l1,
                    y,  # identity for lambda=1
                    torch.pow(torch.clamp(y * lmbda_safe + 1.0, min=1e-10), 1.0 / lmbda_safe) - 1.0
                )
            )
            # neg branch
            neg_l2 = (lmbda == 2)
            neg_l1 = (lmbda == 1)
            denom = (2.0 - lmbda)
            denom_safe = torch.where(neg_l2 | neg_l1, torch.ones_like(denom), denom)
            out_neg = torch.where(
                neg_l2,
                1.0 - torch.expm1(-y),
                torch.where(
                    neg_l1,
                    y,  # identity for lambda=1
                    1.0 - torch.pow(torch.clamp(-denom_safe * y + 1.0, min=1e-10), 1.0 / denom_safe)
                )
            )

            out = torch.where(pos, out_pos, out_neg)
            # Guard non-finite outputs
            if not torch.isfinite(out).all():
                out = torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
            return out
        else:
            y = np.asarray(y, dtype=float)
            lmbda = np.asarray(lambdas, dtype=float)
            shape = [1] * y.ndim; shape[-1] = -1
            lmbda = lmbda.reshape(*shape)
            
            # Identity shortcut: lambda=1 -> inverse is identity (y = x)
            if np.all(lmbda == 1):
                return y.copy()
            
            pos = y >= 0
            out = np.zeros_like(y, dtype=float)

            identity_mask = (lmbda == 1)
            lmbda_safe = np.where((lmbda == 0) | identity_mask, 1.0, lmbda)
            out_pos = np.where(lmbda == 0, np.expm1(y), 
                              np.where(identity_mask, y, 
                                      np.power(np.maximum(y * lmbda_safe + 1.0, 1e-10), 1.0 / lmbda_safe) - 1.0))
            denom = (2.0 - lmbda)
            denom_safe = np.where((lmbda == 2) | identity_mask, 1.0, denom)
            out_neg = np.where(lmbda == 2, 1.0 - np.expm1(-y), 
                              np.where(identity_mask, y,
                                      1.0 - np.power(np.maximum(-denom_safe * y + 1.0, 1e-10), 1.0 / denom_safe)))
            out = np.where(pos, out_pos, out_neg)
            # Guard non-finite outputs
            if not np.isfinite(out).all():
                out = np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
            return out

    def _yj_loglike(self, x: np.ndarray, lmbda: float) -> float:
        # Gaussian log-likelihood (up to constant) for YJ-transformed data
        y = self._yeo_johnson(x, lmbda)
        var = float(_safe_var(y))
        if not np.isfinite(var) or var <= 0:
            return -np.inf
        pos = x >= 0
        neg = ~pos
        s_pos = np.sum(np.log1p(x[pos])) if np.any(pos) else 0.0
        s_neg = np.sum(np.log1p(-x[neg])) if np.any(neg) else 0.0
        jac = (lmbda - 1.0) * (s_pos - s_neg)
        n = x.size
        return -0.5 * n * np.log(var) + jac

    def _fit_yeo_johnson_lambdas(self, X: np.ndarray) -> np.ndarray:
        # Fit lambda per column via grid search (robust, no extra deps)
        X = np.asarray(X, dtype=float)
        m = X.shape[1]
        lambdas = np.zeros(m, dtype=float)
        grid = np.linspace(-3.0, 3.0, 121)
        n_constant = 0
        for i in range(m):
            xi = X[:, i]
            if not np.isfinite(xi).all():
                xi = np.nan_to_num(xi, nan=0.0, posinf=0.0, neginf=0.0)
            # If nearly constant, keep identity
            if float(_safe_nanstd(xi)) < 1e-8:
                lambdas[i] = 1.0
                n_constant += 1
                continue
            best_l = 1.0
            best_ll = -np.inf
            for l in grid:
                ll = self._yj_loglike(xi, l)
                if ll > best_ll:
                    best_ll = ll
                    best_l = l
            lambdas[i] = best_l
        # Debug output
        try:
            import os
            if os.environ.get('BMTGNN_DEBUG_YJ', '0') == '1':
                print(f"[YJ-FIT] Fitted {m} lambdas: min={lambdas.min():.3f}, max={lambdas.max():.3f}, mean={lambdas.mean():.3f}")
                print(f"[YJ-FIT] {n_constant} constant series (lambda=1.0)")
                print(f"[YJ-FIT] Lambda distribution: {np.unique(np.round(lambdas, 1), return_counts=True)}")
        except Exception:
            pass
        return lambdas

    def _apply_yeo_johnson(self, X: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.zeros_like(X, dtype=float)
        for i in range(X.shape[1]):
            out[:, i] = self._yeo_johnson(X[:, i], float(lambdas[i]))
        # Guard against non-finite values
        n_bad = (~np.isfinite(out)).sum()
        if n_bad > 0:
            try:
                import os
                if os.environ.get('BMTGNN_DEBUG_YJ', '0') == '1':
                    print(f"[YJ-APPLY] Warning: {n_bad} non-finite values after transform, clamping to 0")
            except Exception:
                pass
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    # (batch-aware handling implemented inside single std_expand_like below)

    def std_expand_like(self, x, idx=None, split=None):
        """Broadcast std to the same shape as x (useful for approx CI scaling).
        If rolling and idx+split are provided, return per-sample std expanded
        using stored per-window stats. Otherwise use global train std.
        """
        if self.rolling and idx is not None and split is not None:
            mu_arr = self.per_window_mu.get(split, None)
            std_arr = self.per_window_std.get(split, None)
            if mu_arr is None or std_arr is None:
                raise RuntimeError('per-window stats missing for split')
            sel = idx.cpu().numpy() if torch.is_tensor(idx) else np.asarray(idx)
            std_np = std_arr[sel]
            if torch.is_tensor(x):
                std_t = torch.as_tensor(std_np, device=x.device, dtype=x.dtype)
                if std_t.ndim == 2 and x.dim() >= 3:
                    return std_t.view(std_t.shape[0], 1, std_t.shape[1]).expand_as(x)
                else:
                    view = [1] * x.dim(); view[-1] = -1
                    return std_t.view(*view).expand_as(x)
            else:
                shape = list(x.shape)
                view = [1] * len(shape); view[-1] = -1
                return np.broadcast_to(std_np.reshape(*view), shape)

        # fallback: original global behavior
        if self.std is None:
            raise RuntimeError('std_expand_like called before std is set')
        if torch.is_tensor(x):
            if torch.is_tensor(self.std):
                std_t = self.std.to(x.device).type_as(x)
            else:
                std_t = torch.as_tensor(self.std, device=x.device, dtype=x.dtype)
        else:
            if torch.is_tensor(self.std):
                std_t = self.std.cpu().numpy()
            else:
                std_t = self.std
        return self._expand_like(std_t, x)


    def _split(self, train, valid, test):
        """Robust window-aware split that builds training/validation/test
        window start indices and produces batched tensors. Inputs are taken
        from self.dat_in (shape (T, N, C)) while targets are from self.dat
        (shape (T, N)). The produced test_window is a contiguous slice of
        self.dat_in and has shape (T*, N, C).
        """
        # earliest index we can start a window from (we need P history and h offset)
        min_start = self.P + self.h - 1
        # latest start index such that the out window fits in the data
        max_start = max(0, self.n - self.out_len)

        total_windows = max(0, max_start - min_start + 1)
        if total_windows <= 0:
            # no feasible windows; return zero-shaped datasets
            self.train = (np.zeros((0, self.P, self.m, self.in_dim), dtype=np.float32),
                          np.zeros((0, self.out_len, self.m), dtype=np.float32))
            self.valid = (np.zeros((0, self.P, self.m, self.in_dim), dtype=np.float32),
                          np.zeros((0, self.out_len, self.m), dtype=np.float32))
            self.test  = (np.zeros((0, self.P, self.m, self.in_dim), dtype=np.float32),
                          np.zeros((0, self.out_len, self.m), dtype=np.float32))
            self.test_window = np.zeros((0, self.m, self.in_dim), dtype=np.float32)
            return

        # allocate windows by ratios (at least 1 where possible)
        # ---- FIX: accept ratios in (0,1] as ratios, not absolute indices ----
        # train_test.py passes train_ratio/valid_ratio (e.g., 0.57, 0.15).
        # This class expects boundaries in "time steps" when not using ratios.
        if 0 < train <= 1.0 and 0 < valid <= 1.0:
            if (train + valid) >= 1.0:
                print(
                    "[DataLoaderS] Warning: train_ratio + valid_ratio >= 1.0; "
                    "test split may be empty or forced to 1 window."
                )
            train_end = int(round(train * self.n))
            valid_end = int(round((train + valid) * self.n))
            # Keep sane ordering within valid time-step bounds.
            train = max(1, min(train_end, self.n - 2))
            valid = max(train + 1, min(valid_end, self.n - 1))
        tr = max(1, int(round((train / float(self.n)) * total_windows)))
        vr = max(1, int(round(((valid - train) / float(self.n)) * total_windows)))
        # ensure we leave at least 1 for test when possible
        if tr + vr >= total_windows:
            vr = max(1, min(vr, total_windows - tr - 1))
            tr = max(1, min(tr, total_windows - vr - 1))
        te = total_windows - tr - vr
        if te < 1:
            te = 1
            if vr > 1:
                vr -= 1
            elif tr > 1:
                tr -= 1

        train_low = min_start
        train_high = train_low + tr
        valid_low = train_high
        valid_high = valid_low + vr
        test_low = valid_high
        test_high = test_low + te

        train_set = list(range(train_low, train_high))
        valid_set = list(range(valid_low, valid_high))
        test_set = list(range(test_low, test_high))

        # Store the index sets so callers can map local batch indices back to
        # global window start indices. This is necessary for rolling mode
        # so we can find the correct per-window mu/std when evaluating.
        self._train_idx_set = train_set
        self._valid_idx_set = valid_set
        self._test_idx_set  = test_set

        # Create batched datasets. X shape: (n, P, m, C); Y shape: (n, out_len, m)
        self.train = self._batchify(train_set, self.h, split='train')
        self.valid = self._batchify(valid_set, self.h, split='valid')
        self.test  = self._batchify(test_set, self.h, split='test')

        # test_window: contiguous segment covering the test targets and required history
        # Use dat_in (T, N, C) so sliding-window evaluation can read both channels.
        if not self.rolling:
            assert self.dat_in is not None, "dat_in missing in non-rolling mode"
            self.test_window = self.dat_in[test_low : test_high + self.out_len, :, :]
        else:
            # build from raw_levels and movement channel if needed
            assert self.raw_levels is not None, "raw_levels missing in rolling mode"
            start = test_low
            end = test_high + self.out_len
            levels = self.raw_levels[start:end, :]
            if self.in_dim > 1:
                mov = np.zeros_like(levels)
                mov[1:, :] = levels[1:, :] - levels[:-1, :]
                self.test_window = np.stack([levels, mov], axis=-1)
            else:
                self.test_window = levels[..., None]

    # --- New chronological split ---
    def _split_chronological(self, start_year: int, steps_per_year: int,
                              train_end_year: int, valid_end_year: int, test_end_year: int):
        """Deterministic year-range based split (no leakage) using calendar boundaries.

        Assumptions:
          - Time index 0 corresponds to January (or first period) of start_year.
          - Data are evenly spaced with steps_per_year periods per calendar year.
          - train spans start_year .. train_end_year (inclusive),
            valid spans (train_end_year+1) .. valid_end_year (inclusive),
            test  spans (valid_end_year+1) .. test_end_year (inclusive).
          - Windows may draw historical context from earlier periods (e.g., validation
            inputs can include training-era history) but target prediction timesteps
            must lie fully within their respective segment.
        """
        if steps_per_year <= 0:
            raise ValueError("steps_per_year must be positive for chronological split")

        def year_last_index(y: int) -> int:
            # Inclusive index of the final step of year y
            return (y - start_year + 1) * steps_per_year - 1

        n_total = self.n
        global_last = n_total - 1

        # Compute raw boundary indices (inclusive) and clamp to data length
        train_last = min(year_last_index(train_end_year), global_last)
        valid_last = min(year_last_index(valid_end_year), global_last)
        test_last  = min(year_last_index(test_end_year),  global_last)

        # Derive segment start indices (inclusive)
        train_first = 0
        valid_first = min(train_last + 1, global_last)
        test_first  = min(valid_last + 1, global_last)

        if not (train_first <= train_last < valid_first <= valid_last < test_first <= test_last):
            raise ValueError("Inconsistent chronological boundaries after clamping; check year ranges or data length")

        # Feasible target start indices constraints
        min_start = self.P + self.h - 1          # earliest target index ensuring enough history
        max_start = max(0, n_total - self.out_len)  # latest target start so output fits

        def build_idx_set(seg_first: int, seg_last: int):
            # Targets must start within segment AND satisfy min_start
            first = max(min_start, seg_first)
            last  = min(seg_last - self.out_len + 1, max_start)
            if last < first:
                return []
            return list(range(first, last + 1))

        train_set = build_idx_set(train_first, train_last)
        valid_set = build_idx_set(valid_first, valid_last)
        test_set  = build_idx_set(test_first,  test_last)

        # Fallback guarantee: if earlier segments empty but later non-empty, leave them; else ensure at least test has something
        if len(test_set) == 0 and len(valid_set) == 0 and len(train_set) == 0:
            # No feasible windows at all -> mimic _split empty handling
            self.train = (np.zeros((0, self.P, self.m), dtype=np.float32),
                          np.zeros((0, self.out_len, self.m), dtype=np.float32))
            self.valid = (np.zeros((0, self.P, self.m), dtype=np.float32),
                          np.zeros((0, self.out_len, self.m), dtype=np.float32))
            self.test  = (np.zeros((0, self.P, self.m), dtype=np.float32),
                          np.zeros((0, self.out_len, self.m), dtype=np.float32))
            self.test_window = np.zeros((0, self.m), dtype=np.float32)
            return

        # Batchify inside this method
        self.train = self._batchify(train_set, self.h, split='train')
        self.valid = self._batchify(valid_set, self.h, split='valid')
        self.test  = self._batchify(test_set,  self.h, split='test')

        # Build test_window analogous to ratio split (in NORMALIZED space)
        if len(test_set) > 0:
            test_low = test_set[0]
            test_high_exclusive = test_set[-1] + 1
            if not self.rolling:
                assert self.dat_in is not None, "dat_in is None in non-rolling mode"
                self.test_window = cast(np.ndarray, self.dat_in)[test_low : test_high_exclusive + self.out_len, :, :]
            else:
                start = test_low
                end = test_high_exclusive + self.out_len
                assert self.raw_levels is not None, "raw_levels missing in rolling mode"
                levels = cast(np.ndarray, self.raw_levels)[start:end, :]
                if self.in_dim > 1:
                    mov = np.zeros_like(levels)
                    mov[1:, :] = levels[1:, :] - levels[:-1, :]
                    self.test_window = np.stack([levels, mov], axis=-1)
                else:
                    self.test_window = levels[..., None]
        else:
            self.test_window = np.zeros((0, self.m, self.in_dim), dtype=np.float32)

    def _batchify(self, idx_set, horizon, split=None):
        # Guard against out_len overrun at tail indices (can occur with rounding)
        data_len = None
        if self.rolling and self.raw_levels is not None:
            data_len = int(self.raw_levels.shape[0])
        elif self.dat is not None:
            data_len = int(self.dat.shape[0])
        if data_len is not None and len(idx_set) > 0:
            idx_set = [i for i in idx_set if (i + self.out_len) <= data_len]

        n = len(idx_set)

        # If there are no valid start indices, return empty tensors with the
        # correct trailing dimensions so callers can handle zero samples.
        in_dim = int(getattr(self, 'in_dim', 1))
        if n == 0:
            X = torch.zeros((0, self.P, self.m, in_dim))
            Y = torch.zeros((0, self.out_len, self.m))
            return [X, Y]

        # Allocate one sample per valid start index with channel dim
        X = torch.zeros((n, self.P, self.m, in_dim))  # n samples x P lookback x m x C
        Y = torch.zeros((n, self.out_len, self.m))

        per_window_mu = []
        per_window_std = []
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            # For rolling mode compute per-window mu/std on levels from original data
            if self.rolling:
                # Ensure raw_levels is available (lazy-init from raw data if needed)
                if self.raw_levels is None:
                    base = self.rawdat.astype(np.float32, copy=False)
                    if self.use_log1p:
                        base = np.log1p(np.clip(base, 0.0, None))
                    self.raw_levels = base
                # levels from raw_levels: shape (T, m)
                levels = self.raw_levels[start:end, :]
                mu = levels.mean(axis=0)
                std = _safe_nanstd(levels, axis=0)
                std = np.maximum(std, 1e-6)
                per_window_mu.append(mu.astype(np.float32))
                per_window_std.append(std.astype(np.float32))
                # Normalize inputs
                norm_levels = (levels - mu.reshape(1, -1)) / std.reshape(1, -1)
                if self.dual == 'diff' or self.dual == 'pct':
                    mov = np.zeros_like(norm_levels)
                    # movement on normalized levels: first lag is diff from previous (unknown here), use zeros for first
                    mov[1:, :] = norm_levels[1:, :] - norm_levels[:-1, :]
                    if self.dual == 'pct':
                        denom = np.abs(norm_levels[:-1, :]) + 1e-6
                        mov[1:, :] = (norm_levels[1:, :] - norm_levels[:-1, :]) / denom
                        if getattr(self, 'pct_clip', 0.0) > 0:
                            mov = np.clip(mov, -self.pct_clip, self.pct_clip)
                    xin = np.stack([norm_levels, mov], axis=-1)
                else:
                    xin = norm_levels[..., None]
                X[i, :, :, :] = torch.from_numpy(xin)
                # Targets: take raw levels for future window and normalize with same mu/std
                assert self.raw_levels is not None, "raw_levels missing in rolling mode"
                y_raw = cast(np.ndarray, self.raw_levels)[idx_set[i]:idx_set[i] + self.out_len, :]
                y_norm = (y_raw - mu.reshape(1, -1)) / std.reshape(1, -1)
                Y[i, :, :] = torch.from_numpy(y_norm)
            else:
                # Inputs from level+movement (normalized space)
                assert self.dat_in is not None and self.dat is not None, "dat_in/dat missing in non-rolling mode"
                X[i, :, :, :] = torch.from_numpy(cast(np.ndarray, self.dat_in)[start:end, :, :])
                # Targets: level only (normalized space)
                Y[i, :, :] = torch.from_numpy(cast(np.ndarray, self.dat)[idx_set[i]:idx_set[i] + self.out_len, :])

        # store per-window stats for rolling mode to support inversion later
        if self.rolling:
            mu_arr = np.stack(per_window_mu, axis=0) if len(per_window_mu) > 0 else np.zeros((0, self.m), dtype=np.float32)
            std_arr = np.stack(per_window_std, axis=0) if len(per_window_std) > 0 else np.ones((0, self.m), dtype=np.float32)
            key = split if split is not None else 'unspecified'
            self.per_window_mu[key] = mu_arr
            self.per_window_std[key] = std_arr

        return [X, Y]


    def get_batches(self, inputs, targets, batch_size, shuffle=True, return_indices=False):
        """Yield minibatches. If return_indices is True, also yield the sample indices
        (numpy int array) corresponding to each minibatch so callers can map back
        to per-window stats (useful in rolling mode).
        """
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.arange(length, dtype=torch.long)
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            # convert excerpt to numpy indices for indexing numpy arrays
            idx_np = excerpt.cpu().numpy() if torch.is_tensor(excerpt) else np.asarray(excerpt)
            # safely index inputs/targets (they may be numpy or torch)
            if isinstance(inputs, np.ndarray):
                X_batch = inputs[idx_np]
            else:
                X_batch = inputs[excerpt]
            if isinstance(targets, np.ndarray):
                Y_batch = targets[idx_np]
            else:
                Y_batch = targets[excerpt]
            # ensure tensors on device
            Xb = torch.as_tensor(X_batch).to(self.device)
            Yb = torch.as_tensor(Y_batch).to(self.device)
            if return_indices:
                yield Variable(Xb), Variable(Yb), idx_np
            else:
                yield Variable(Xb), Variable(Yb)
            start_idx += batch_size

    
    #builds the graph of threats and pertinent technologies
    def build_predefined_adj(self):
        # If a precomputed square adjacency exists, load it to save time.
        cache_path = 'data/graph_square.csv'
        if os.path.exists(cache_path):
            try:
                adj_np = np.loadtxt(cache_path, delimiter=',')
                adj_t = torch.from_numpy(adj_np.astype(np.float32)).to(self.device)
                print(f'Loaded cached adjacency from {cache_path} ({adj_t.shape[0]} nodes)')
                return adj_t
            except Exception:
                # fall through to rebuild if loading fails
                print(f'Failed to load cached adjacency at {cache_path}, rebuilding...')

        # Read bipartite graph CSV into a dict of sets for O(1) membership tests
        graph = defaultdict(set)
        with open('data/graph.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                key_node = row[0]
                adjacent_nodes = {node for node in row[1:] if node}
                if adjacent_nodes:
                    graph[key_node].update(adjacent_nodes)

        print('Graph loaded with', len(graph), 'RMDs...')

        # Read column headers from the dataset CSV. If the header file is
        # missing, fall back to synthetic numeric column names so we can still
        # construct a placeholder adjacency matrix of the right shape.
        col_file_candidates = ['data/sm_data_g.csv', 'data/data.csv']
        col = None
        for cf in col_file_candidates:
            if os.path.exists(cf):
                try:
                    with open(cf, 'r') as f:
                        reader = csv.reader(f)
                        col = [c for c in next(reader)]
                    print(f'Loaded column headers from {cf}')
                    break
                except Exception:
                    col = None

        if col is None:
            # fallback: generate synthetic column names (col_0 ... col_{m-1})
            col = [f'col_{i}' for i in range(self.m)]
            print(f'Column header CSV not found; using synthetic {len(col)} columns')

        # Use numpy for adjacency building (faster than repeated tensor ops)
        adj_np = np.zeros((len(col), len(col)), dtype=np.float32)
        name_to_index = {name: idx for idx, name in enumerate(col)}

        for i, name in enumerate(col):
            if name in graph:
                neigh = graph[name]
                for nbr in neigh:
                    j = name_to_index.get(nbr)
                    if j is not None:
                        adj_np[i, j] = 1.0
                        adj_np[j, i] = 1.0

        # Save the square adjacency for future runs (best-effort)
        try:
            np.savetxt(cache_path, adj_np, delimiter=',', fmt='%.6g')
            print(f'Wrote cached adjacency to {cache_path}')
        except Exception:
            pass

        adj_t = torch.from_numpy(adj_np).to(self.device)
        print('Adjacency created...')
        return adj_t

    #by Zaid et al.
    # returns set of column names within dataset  
    def create_columns(self):

        file_name='data/data.csv'
        if self.m==123: #no external features
            file_name='data/sm_data_g.csv'
        # Read the CSV file of the dataset. If it's missing, generate synthetic
        # column names to avoid crashing callers that expect a list of column ids.
        if os.path.exists(file_name):
            try:
                with open(file_name, 'r') as f:
                    reader = csv.reader(f)
                    col = [c for c in next(reader)]
                    if col and col[0].strip().lower() == 'date':
                        return col[1:]
                    return col
            except Exception:
                pass

        # Fallback: synthetic column names
        return [f'col_{i}' for i in range(self.m)]


def resolve_split_and_build_data(args, device):
    """Build a DataLoaderS instance using ratio-based splitting only.
    
    Chronological splitting has been removed. All splits are now based on
    train_ratio and valid_ratio arguments.

    Returns: (Data, use_chrono, steps_py, required_months)
             where use_chrono is always False
    """
    # Ratio-based splitting only
    tr = max(0.05, min(0.9, float(getattr(args, 'train_ratio', 0.60))))
    vr = max(0.05, min(0.9, float(getattr(args, 'valid_ratio', 0.20))))
    if tr + vr >= 0.95:
        scale = 0.95 / (tr + vr)
        tr *= scale
        vr *= scale
    args.train_ratio = tr
    args.valid_ratio = vr
    test_ratio = 1.0 - tr - vr
    
    print(f'[SPLIT] Ratio-based splitting:')
    print(f'  Train: {tr:.2%}, Valid: {vr:.2%}, Test: {test_ratio:.2%}')
    
    # Always use ratio-based splitting
    Data = DataLoaderS(
        args.data, float(args.train_ratio), float(args.valid_ratio), device,
        args.horizon, args.seq_in_len, args.normalize, args.seq_out_len,
        chronological=False,
        dual_channel=args.dual_channel,
        pct_clip=float(getattr(args, 'pct_clip', 0.0)),
        y_transform=(getattr(args, 'y_transform', None)),
        trend_smooth=bool(getattr(args, 'trend_smooth', False)),
        trend_alpha=float(getattr(args, 'trend_alpha', 0.25)),
        trend_beta=float(getattr(args, 'trend_beta', 0.05)),
        resid_alpha=float(getattr(args, 'resid_alpha', 0.20)),
        has_header=bool(getattr(args, 'has_header', False)),
        drop_first_col=bool(getattr(args, 'drop_first_col', False)),
        exclude_names=(args.exclude_names if getattr(args, 'exclude_names', '') != '' else None),
        y_offset=float(getattr(args, 'y_offset', 0.0)),
        auto_y_offset=bool(getattr(args, 'auto_y_offset', False)),
        y_transform_selective=bool(getattr(args, 'y_transform_selective', False)),
        y_transform_std_min=float(getattr(args, 'y_transform_std_min', 1e-6)),
        y_transform_skip_neg=bool(getattr(args, 'y_transform_skip_neg', False)),
        clip_outliers=bool(getattr(args, 'clip_outliers', False)),
        clip_percentile=float(getattr(args, 'clip_percentile', 95.0)),
    )

    def _warn_empty(name: str, XY):
        try:
            n = int(getattr(XY[0], 'shape')[0]) if XY and XY[0] is not None else 0
        except Exception:
            n = 0
        if n == 0:
            print(f'[SplitGuard] {name} has 0 windows. Adjust seq_in_len/seq_out_len or try different split ratios.')

    _warn_empty('Validation', getattr(Data, 'valid', None))
    _warn_empty('Test', getattr(Data, 'test', None))

    # Always return False for use_chrono since we removed chronological splitting
    return Data, False, 12, int(args.seq_in_len) + int(args.seq_out_len)


def prepare_graph_and_subgraph(args, device, data_obj=None):
    """Apply subgraph_size defaults and load a predefined adjacency if provided.

    Returns: predefined_A (torch.Tensor or None)
    """
    if getattr(args, 'subgraph_size', 0) <= 0 and getattr(args, 'num_nodes', 0) > 0:
        if args.num_nodes < 20:
            args.subgraph_size = args.num_nodes
        elif args.num_nodes <= 100:
            args.subgraph_size = 20
        else:
            args.subgraph_size = 10
        print(f"[Graph] Auto-set subgraph_size to {args.subgraph_size} for {args.num_nodes} nodes.")

    predefined_A = None
    if getattr(args, 'graph', None):
        try:
            print(f"[Graph] Loading graph from {args.graph}")
            # Pass num_nodes to ensure correct shape
            num_nodes = int(getattr(args, 'num_nodes', 0)) if getattr(args, 'num_nodes', 0) > 0 else None
            predefined_A = load_adj(args.graph) if num_nodes is None else load_graph(args.graph, num_nodes=num_nodes)
            if predefined_A is not None and predefined_A.shape[0] > 0:
                predefined_A = torch.tensor(predefined_A) - torch.eye(args.num_nodes)
                predefined_A = predefined_A.to(device)
            else:
                print(f"[Graph] Warning: loaded adjacency has invalid shape {predefined_A.shape if predefined_A is not None else 'None'}")
                predefined_A = None
        except Exception as e:
            print(f"[Graph] Error loading adjacency matrix: {e}", file=sys.stderr)
            predefined_A = None

    return predefined_A

class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def transform(self, data):
        # Handle device/dtype mismatches for torch tensors
        if torch.is_tensor(data):
            mean = self.mean
            std = self.std
            # Ensure mean and std are on same device as data
            if torch.is_tensor(mean) and mean.device != data.device:
                mean = mean.to(device=data.device, dtype=data.dtype)
            elif not torch.is_tensor(mean):
                mean = torch.as_tensor(mean, dtype=data.dtype, device=data.device)
            if torch.is_tensor(std) and std.device != data.device:
                std = std.to(device=data.device, dtype=data.dtype)
            elif not torch.is_tensor(std):
                std = torch.as_tensor(std, dtype=data.dtype, device=data.device)
            return (data - mean) / std
        return (data - self.mean) / self.std
    def inverse_transform(self, data):
        # Accept either torch.Tensor or numpy array as `data`.
        # Rely on the module-level import of torch so the name is always bound;
        # catch any runtime errors when checking tensor-ness.
        is_torch = False
        try:
            is_torch = torch.is_tensor(data)
        except Exception:
            is_torch = False

        if is_torch:
            # Ensure mean/std are tensors on the correct device/dtype
            mean = self.mean
            std = self.std
            if not torch.is_tensor(mean):
                mean = torch.as_tensor(mean, dtype=data.dtype, device=data.device)
            else:
                mean = mean.to(device=data.device, dtype=data.dtype)
            if not torch.is_tensor(std):
                std = torch.as_tensor(std, dtype=data.dtype, device=data.device)
            else:
                std = std.to(device=data.device, dtype=data.dtype)
            # Clamp tiny stds to avoid division/mul explosions
            try:
                # compute a sensible floor based on the median std to avoid
                # tiny per-node stds causing huge original-unit amplification.
                std_abs = None
                try:
                    std_abs = std.abs()
                    med = float(torch.median(std_abs))
                except Exception:
                    med = 0.0
                    std_abs = None
                # allow tuning via env var BMTGNN_STD_FLOOR_MULT (default 0.05)
                try:
                    mult = float(os.environ.get('BMTGNN_STD_FLOOR_MULT', '0.05'))
                except Exception:
                    mult = 0.05
                try:
                    min_floor = float(os.environ.get('BMTGNN_STD_FLOOR_MIN', '1e-3'))
                except Exception:
                    min_floor = 1e-3
                floor = max(med * mult, min_floor)
                std = std.clamp(min=floor)
                if os.environ.get('BMTGNN_DEBUG_SCALE', '0') == '1':
                    # report if any stds were raised above the floor
                    try:
                        num_low = 0
                        if std_abs is not None:
                            mask = std_abs < floor
                            if torch.is_tensor(mask):
                                num_low = int(mask.sum().item())
                            else:
                                num_low = int(np.sum(mask))
                        if num_low > 0:
                            print(f'[BMTGNN_DEBUG_SCALE] clamped {num_low} std entries to floor={floor}')
                    except Exception:
                        pass
            except Exception:
                pass
            out = data * std + mean
            # Detect NaN/inf early and raise for upstream handling/logging
            if torch.isnan(out).any() or torch.isinf(out).any():
                raise ValueError('inverse_transform produced NaN or Inf in torch tensor')
            return out
        else:
            arr = _to_numpy_array(data)
            mean = np.asarray(self.mean, dtype=np.float32)
            std = np.asarray(self.std, dtype=np.float32)
            # compute median-based floor to avoid tiny stds blowing up predictions
            std_abs = np.abs(std)
            med = float(np.median(std_abs)) if std_abs.size > 0 else 0.0
            floor = max(med * 0.05, 1e-3)
            # clamp stds to the floor
            try:
                low_mask = std_abs < floor
                if np.any(low_mask):
                    std = np.maximum(std, floor)
                    if os.environ.get('BMTGNN_DEBUG_SCALE', '0') == '1':
                        print(f'[BMTGNN_DEBUG_SCALE] clamped {int(low_mask.sum())} std entries to floor={floor}')
            except Exception:
                std[std_abs < 1e-6] = 1.0
            out = (arr * std) + mean
            if np.isnan(out).any() or np.isinf(out).any():
                raise ValueError('inverse_transform produced NaN or Inf in numpy array')
            return out


def _to_numpy_array(a: Union[np.ndarray, torch.Tensor, list]) -> np.ndarray:
    """Best-effort conversion to a NumPy ndarray (float32)."""
    if isinstance(a, np.ndarray):
        return a.astype(np.float32, copy=False)
    if torch.is_tensor(a):
        return a.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(a, dtype=np.float32)


def sym_adj(adj: Union[np.ndarray, torch.Tensor, list]) -> np.ndarray:
    """Symmetrically normalize adjacency matrix: D^{-1/2} A D^{-1/2}. Returns dense ndarray."""
    adj_arr = _to_numpy_array(adj)
    rowsum = adj_arr.sum(axis=1)
    # Avoid division by zero
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return (D_inv_sqrt @ adj_arr @ D_inv_sqrt).astype(np.float32)

def asym_adj(adj: Union[np.ndarray, torch.Tensor, list]) -> np.ndarray:
    """Asymmetrically normalize adjacency matrix: D^{-1} A. Returns dense ndarray."""
    adj_arr = _to_numpy_array(adj)
    rowsum = adj_arr.sum(axis=1)
    with np.errstate(divide='ignore'):
        d_inv = np.power(rowsum, -1.0)
    d_inv[np.isinf(d_inv)] = 0.0
    D_inv = np.diag(d_inv)
    return (D_inv @ adj_arr).astype(np.float32)

def calculate_normalized_laplacian(adj: Union[np.ndarray, torch.Tensor, list]) -> np.ndarray:
    """
    L = I - D^{-1/2} A D^{-1/2}
    """
    adj_arr = _to_numpy_array(adj)
    rowsum = adj_arr.sum(axis=1)
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.eye(adj_arr.shape[0], dtype=np.float32) - (D_inv_sqrt @ adj_arr @ D_inv_sqrt).astype(np.float32)
    return L

def calculate_scaled_laplacian(adj_mx: Union[np.ndarray, torch.Tensor, list], lambda_max: Union[float, None] = 2, undirected: bool = True) -> np.ndarray:
    """
    Scaled Laplacian: L_tilde = 2 / lambda_max * L - I
    Returns dense ndarray.
    """
    A = _to_numpy_array(adj_mx)
    if undirected:
        A = np.maximum(A, A.T)
    L = calculate_normalized_laplacian(A)
    if lambda_max is None:
        # For small graphs this is fine; avoids scipy.sparse typing woes
        eigvals = np.linalg.eigvals(L)
        lambda_max = float(np.max(np.real(eigvals)))
    M = L.shape[0]
    I = np.eye(M, dtype=np.float32)
    L_scaled = (2.0 / float(lambda_max)) * L - I
    return L_scaled.astype(np.float32)


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_graph(path: str, num_nodes: int | None = None) -> np.ndarray:
    """Load adjacency from various formats.

    Supports:
    - .pkl/.pickle: unpickle; accept ndarray directly, or tuple/dict where adjacency is extractable
    - .npy: numpy binary array
    - .csv: try numeric square CSV first; else treat as edge list with 2-3 columns
    Returns a dense square ndarray (float32). If num_nodes is provided, result is sized to (num_nodes, num_nodes)
    by truncating or zero-padding as needed.
    """
    ext = os.path.splitext(path)[1].lower()

    def _ensure_square(a: np.ndarray, N: int | None = None) -> np.ndarray:
        a = np.asarray(a, dtype=np.float32)
        if a.ndim != 2:
            raise ValueError(f"Adjacency must be 2D, got shape {a.shape}")
        if N is None:
            if a.shape[0] != a.shape[1]:
                raise ValueError(f"Adjacency not square: {a.shape}")
            return a.astype(np.float32)
        # resize to NxN (truncate or pad)
        out = np.zeros((N, N), dtype=np.float32)
        n0 = min(N, a.shape[0]); n1 = min(N, a.shape[1])
        out[:n0, :n1] = a[:n0, :n1]
        return out

    if ext in {".pkl", ".pickle"}:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, np.ndarray):
            return _ensure_square(obj, num_nodes)
        if isinstance(obj, (list, tuple)) and len(obj) == 3:
            # common (ids, id_to_ind, adj) layout
            return _ensure_square(np.asarray(obj[2]), num_nodes)
        if isinstance(obj, dict):
            for k in ("adj", "A", "adjacency"):
                if k in obj:
                    return _ensure_square(np.asarray(obj[k]), num_nodes)
        raise ValueError("Unsupported pickle content for adjacency")

    if ext == ".npy":
        arr = np.load(path)
        return _ensure_square(arr, num_nodes)

    # CSV path
    try:
        arr = np.loadtxt(path, delimiter=",")
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            return _ensure_square(arr, num_nodes)
    except Exception:
        pass

    # Edge list CSV fallback (2-3 columns, headerless).
    # Also supports a row-wise adjacency list: first col = src, remaining cols = dsts.
    df = pd.read_csv(path, header=None)
    src = []
    dst = []
    w = []
    if df.shape[1] in (2, 3):
        src = df.iloc[:, 0].astype(str).tolist()
        dst = df.iloc[:, 1].astype(str).tolist()
        if df.shape[1] == 3:
            try:
                w = df.iloc[:, 2].astype(float).to_numpy()
            except Exception:
                w = np.ones(len(src), dtype=np.float32)
        else:
            w = np.ones(len(src), dtype=np.float32)
    else:
        # Treat each row as: src, dst1, dst2, ... (ignore empty cells)
        for _, row in df.iterrows():
            if row.empty:
                continue
            src_val = row.iloc[0]
            if pd.isna(src_val):
                continue
            s = str(src_val).strip()
            for d_val in row.iloc[1:]:
                if pd.isna(d_val):
                    continue
                d = str(d_val).strip()
                if not d:
                    continue
                src.append(s)
                dst.append(d)
                w.append(1.0)
        w = np.asarray(w, dtype=np.float32)

    # map node labels to indices
    labels = sorted(set(src) | set(dst))
    if num_nodes is None:
        N = len(labels)
    else:
        N = int(num_nodes)
        # if provided N is less than the number of labels, keep only first N labels deterministically
        labels = labels[:N]
    index = {lab: i for i, lab in enumerate(labels)}
    A = np.zeros((N, N), dtype=np.float32)
    for s, d, wt in zip(src, dst, w):
        if s in index and d in index:
            i = index[s]; j = index[d]
            A[i, j] = float(wt)
    return A


def load_adj(path: str) -> np.ndarray:
    """Backward-compatible wrapper around load_graph."""
    return load_graph(path)


def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data



def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse


def load_node_feature(path):
    fi = open(path)
    x = []
    for li in fi:
        li = li.strip()
        li = li.split(",")
        e = [float(t) for t in li[1:]]
        x.append(e)
    x = np.array(x)
    mean = np.mean(x,axis=0)
    std = np.std(x,axis=0)
    z = torch.tensor((x-mean)/std,dtype=torch.float)
    return z

# ============================================================================
# Calendar and date utilities
# ============================================================================
def ym_to_int(ym: str) -> tuple[int, int]:
    """Convert 'YYYY-MM' -> (YYYY, MM)."""
    y, m = ym.split("-")
    return int(y), int(m)

def months_between(y0: int, m0: int, y1: int, m1: int) -> int:
    """Number of whole months between (y0,m0) inclusive start and (y1,m1) inclusive end."""
    return (y1 - y0) * 12 + (m1 - m0)

# ============================================================================
# Tensor transformation utilities
# ============================================================================
def ensure_btn(x):
    """Ensure forecast array/tensor is [B, T, N]. Accepts [B,C,N,T] or [B,T,N] or [T,N]."""
    import numpy as _np
    import torch as _torch
    if _torch.is_tensor(x):
        t = x
        if t.ndim == 4:  # [B,C,N,T] -> [B,T,N]
            return t[:, 0, :, :].permute(0, 2, 1).contiguous()
        if t.ndim == 3:  # [B,T,N]
            return t
        if t.ndim == 2:  # [T,N] -> [1,T,N]
            return t.unsqueeze(0)
        raise RuntimeError(f"ensure_btn: unsupported tensor shape {tuple(t.shape)}")
    else:
        a = _np.asarray(x)
        if a.ndim == 4:  # [B,C,N,T]
            a = a[:, 0, :, :].transpose(0, 3, 1)  # -> [B,T,N]
            return a
        if a.ndim == 3:  # [B,T,N]
            return a
        if a.ndim == 2:  # [T,N]
            return a[None, ...]
        raise RuntimeError(f"ensure_btn: unsupported array shape {a.shape}")

def flatten_weights(w):
    """Return 1D list[float] regardless of tensor/list nesting."""
    import numpy as _np
    import torch as _torch
    if _torch.is_tensor(w):
        return _np.asarray(w.detach().cpu()).reshape(-1).tolist()
    w = _np.asarray(w, dtype=float)
    return w.reshape(-1).tolist()

def to_float(x):
    """Convert tensor, array, or scalar to float, with fallback."""
    try:
        import torch, numpy as np
        if isinstance(x, torch.Tensor):
            return float(x.detach().reshape(-1)[0].cpu().item())
        if isinstance(x, (list, tuple)) and x:
            return float(x[0])
        if isinstance(x, np.ndarray) and x.size:
            return float(x.reshape(-1)[0])
        if isinstance(x, (int, float)):
            return float(x)
        return None
    except Exception:
        try:
            return float(str(x))
        except Exception:
            return None

def unwrap_model_output(o) -> torch.Tensor:
    """Normalize model output to a Tensor.
    
    Supports:
      - Tensor: returned directly
      - Dict with key 'mean' (quantile head) -> returns value
      - Dict with any tensor value -> first tensor value
    """
    if torch.is_tensor(o):
        return o
    if isinstance(o, dict):
        if 'mean' in o and torch.is_tensor(o['mean']):
            return o['mean']
        for v in o.values():
            if torch.is_tensor(v):
                return v
    raise TypeError(f"Model output is not a Tensor or expected dict; got {type(o)}")

def norm_mode_name(n: int) -> str:
    """Convert normalization mode integer to name."""
    return {0: "none", 1: "global", 2: "per_node"}.get(int(n), "unknown")

def maybe_inv_scale(t: torch.Tensor, scaler):
    """Inverse-transform tensor t if a scaler is provided, else return t.
    
    Accepts either sklearn-like scaler with inverse_transform, or a dict
    with 'mean'/'std' tensors (per-node).
    """
    if scaler is None:
        return t
    # Dict( 'mean': [N], 'std':[N] ) used in this repo
    if isinstance(scaler, dict):
        if 'mean' in scaler and 'std' in scaler:
            mean, std = scaler['mean'], scaler['std']
            # broadcast over [B, T, N] or [B, N]
            while mean.dim() < t.dim():
                mean = mean.unsqueeze(0)
                std = std.unsqueeze(0)
            return t * std + mean
    # sklearn-like or custom scaler with inverse_transform
    if hasattr(scaler, 'inverse_transform') and callable(getattr(scaler, 'inverse_transform', None)):
        sh = t.shape
        # Flatten to 2D, inverse, reshape back
        t_np = t.detach().cpu().numpy().reshape(-1, 1)
        t_inv = scaler.inverse_transform(t_np)  # type: ignore[union-attr]
        return torch.from_numpy(t_inv.reshape(sh)).to(t.device)
    return t

# ============================================================================
# Metrics computation
# ============================================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, MAPE, RRSE metrics.
    
    Args:
        y_true: ground truth [B, T, N] or flattened
        y_pred: predictions [B, T, N] or flattened
    Returns:
        dict with mae, rmse, mape, rrse keys
    """
    import numpy as np
    y_true_flat = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred_flat = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    finite = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
    if not finite.any():
        return {'mae': float('nan'), 'rmse': float('nan'), 'mape': float('nan'), 'rrse': float('nan')}
    y_true_flat = y_true_flat[finite]
    y_pred_flat = y_pred_flat[finite]
    # Prevent overflow in variance/mean for extreme values.
    max_abs = np.sqrt(np.finfo(np.float64).max) * 0.5
    if (np.abs(y_true_flat) > max_abs).any() or (np.abs(y_pred_flat) > max_abs).any():
        y_true_flat = np.clip(y_true_flat, -max_abs, max_abs)
        y_pred_flat = np.clip(y_pred_flat, -max_abs, max_abs)
    mae = np.mean(np.abs(y_true_flat - y_pred_flat))
    # Scale before squaring to avoid overflow for extreme values.
    scale = max(1.0, float(np.max(np.abs(np.concatenate([y_true_flat, y_pred_flat])))))
    if scale > 1.0:
        try:
            import os
            if os.environ.get('BMTGNN_DEBUG_METRICS_SCALE', '0') == '1':
                print(f"[Metrics] scaling active (scale={scale:.3e}, n={y_true_flat.size})")
        except Exception:
            pass
    diff_scaled = (y_true_flat - y_pred_flat) / scale
    rmse = np.sqrt(np.mean(diff_scaled ** 2)) * scale
    denom = np.abs(y_true_flat) + 1e-8
    mape = np.mean(np.abs((y_true_flat - y_pred_flat) / denom)) * 100.0
    t_scaled = y_true_flat / scale
    var = np.var(t_scaled) * (scale ** 2)
    rrse = rmse / np.sqrt(var + 1e-8)
    return {'mae': float(mae), 'rmse': float(rmse), 'mape': float(mape), 'rrse': float(rrse)}

# ============================================================================
# Tensor operations and transformations
# ============================================================================
def robust_range(t: torch.Tensor, q_low: float = 0.05, q_high: float = 0.95, dim: int = 1, eps: float = 1e-6):
    """Quantile range as amplitude proxy. t: (B,L,N) or (B,L) -> returns (B,N) or (B,)"""
    # quantile needs float32/64; AMP may produce float16/bfloat16
    t = t.to(torch.float32)
    ql = torch.quantile(t, q_low, dim=dim)
    qh = torch.quantile(t, q_high, dim=dim)
    return torch.clamp(qh - ql, min=eps)

def exp_smooth_2d(arr, alpha: float = 0.1):
    """Simple exponential smoothing for 1D or 2D numpy arrays used in plotting only."""
    import numpy as np
    a = np.array(arr, dtype=float).copy()
    if a.ndim == 1:
        for t in range(1, a.shape[0]):
            a[t] = alpha * a[t] + (1.0 - alpha) * a[t - 1]
    else:
        for n in range(a.shape[1]):
            for t in range(1, a.shape[0]):
                a[t, n] = alpha * a[t, n] + (1.0 - alpha) * a[t - 1, n]
    return a

def last_level_baseline_expand(X: torch.Tensor, T_out: int) -> torch.Tensor:
    """Build a per-sample, per-node baseline from the last observed input level.
    
    Expand it across the forecast horizon so shapes match Y: [B, Tout, N].
    Works for X shaped [B, Tin, N, C], [B, C, N, Tin], or [B, Tin, N].
    
    Args:
        X: Input tensor [B, Tin, N, C], [B, C, N, Tin], or [B, Tin, N]
        T_out: Output horizon length
    Returns:
        Baseline tensor [B, T_out, N]
    """
    if T_out <= 0:
        raise ValueError("T_out must be positive for baseline expansion")
    if X.dim() == 4:
        if X.size(-1) <= 4:
            # channels-last layout [B, Tin, N, C]
            last_level = X[:, -1, :, 0]
        elif X.size(1) <= 4:
            # channels-first layout [B, C, N, Tin]
            last_level = X[:, 0, :, -1]
        else:
            # default to treating as channels-first if layout is ambiguous
            last_level = X[:, 0, :, -1]
    elif X.dim() == 3:
        last_level = X[:, -1, :]
    else:
        raise RuntimeError(f"Unexpected X dims for residual baseline: {tuple(X.shape)}")
    return last_level.unsqueeze(1).expand(-1, T_out, -1).contiguous()

# ============================================================================
# Logging utilities
# ============================================================================
class AnalysisLogger:
    """Write newline-delimited JSON (JSON Lines) records to a file."""
    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        # truncate the file
        with open(self.log_path, "w") as f:
            pass

    def log(self, record: dict):
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            import sys
            print(f"[AnalysisLogger Warning] Failed to write to log: {e}", file=sys.stderr)

_RUNLOG_PATH = None

def start_runlog(args, out_dir="model/Bayesian/logs", version="0.2.0"):
    """Initialize run logging to JSONL file."""
    import platform
    from datetime import datetime
    global _RUNLOG_PATH
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"run_{ts}.jsonl")
    _RUNLOG_PATH = path

    hdr = {
        "event": "run_started",
        "ts": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": __import__("torch").__version__,
        "cuda_available": __import__("torch").cuda.is_available(),
        "version": version,
        "args": vars(args)
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(hdr) + "\n")
    print(f"[RunLog] Logging to: {path}")
    return path

def jlog(event: str, **kv):
    """Append a small JSON record (scalars/short strings only)."""
    from datetime import datetime
    global _RUNLOG_PATH
    if _RUNLOG_PATH is None:
        return
    rec = {}
    rec["event"] = event
    rec["ts"] = datetime.now().isoformat(timespec="seconds")
    for k, v in kv.items():
        try:
            if isinstance(v, (int, float, str, bool)):
                rec[k] = v
            else:
                f = to_float(v)
                rec[k] = f if f is not None else str(v)[:200]
        except Exception:
            rec[k] = str(v)[:200]
    with open(_RUNLOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

# ============================================================================
# Model utilities
# ============================================================================
import contextlib

@contextlib.contextmanager
def MCDropoutContext(model):
    """Context manager to enable dropout during evaluation for MC sampling."""
    was_training = model.training if model is not None else False
    try:
        if model is not None:
            model.train()
        yield
    finally:
        if model is not None and not was_training:
            model.eval()

def to_model_layout(x: torch.Tensor, seq_len: int, debug: bool = False) -> torch.Tensor:
    """Convert tensor to model expected layout [B, C, N, T].
    
    Handles multiple input formats:
    - [B, T, N, C] -> [B, C, N, T]
    - [B, T, N] -> [B, 1, N, T]
    - [B, C, N, T] -> unchanged (or pad/crop T to seq_len)
    """
    if x.dim() == 4:
        B, dim1, dim2, dim3 = x.shape
        # Guess layout based on sizes
        if dim1 <= 4 and dim3 > dim1:
            # Likely [B, C, N, T]
            C, N, T = dim1, dim2, dim3
            if T < seq_len:
                pad = seq_len - T
                x = torch.nn.functional.pad(x, (pad, 0), mode='replicate')
            elif T > seq_len:
                x = x[..., -seq_len:]
            return x
        elif dim3 <= 4 and dim1 > dim3:
            # Likely [B, T, N, C]
            x = x.permute(0, 3, 2, 1)
            T = dim1
            if T < seq_len:
                pad = seq_len - T
                x = torch.nn.functional.pad(x, (pad, 0), mode='replicate')
            elif T > seq_len:
                x = x[..., -seq_len:]
            return x
        else:
            # Ambiguous - try as [B, C, N, T]
            return x
    elif x.dim() == 3:
        # [B, T, N] -> [B, 1, N, T]
        B, T, N = x.shape
        x = x.unsqueeze(1).permute(0, 1, 3, 2)
        if T < seq_len:
            pad = seq_len - T
            x = torch.nn.functional.pad(x, (pad, 0), mode='replicate')
        elif T > seq_len:
            x = x[..., -seq_len:]
        return x
    else:
        raise RuntimeError(f"Unexpected input dim {x.dim()} for to_model_layout")

def set_random_seed(seed, cudnn_benchmark=False):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def fit_linear_calibration(pred, true):
    """Fit linear calibration y_cal = a*pred + b to minimize MSE.
    
    Returns: (a, b) tuple
    """
    import numpy as np
    p = np.asarray(pred, dtype=float).ravel()
    t = np.asarray(true, dtype=float).ravel()
    if len(p) < 2:
        return (1.0, 0.0)
    p_mean = p.mean()
    t_mean = t.mean()
    num = np.sum((p - p_mean) * (t - t_mean))
    denom = np.sum((p - p_mean) ** 2)
    if abs(denom) < 1e-12:
        return (1.0, t_mean - p_mean)
    a = num / denom
    b = t_mean - a * p_mean
    return (float(a), float(b))
