import numpy as np

from src.util import compute_metrics, jlog


def _np_finite_float64(a):
    """Return float64 array with NaN/Inf replaced by 0 (metrics-only path)."""
    a = np.asarray(a, dtype=np.float64)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_corr_np(p, y, eps: float = 1e-12) -> float:
    """Scale-invariant Pearson corr computed safely (avoids overflow in std/var)."""
    p = _np_finite_float64(p)
    y = _np_finite_float64(y)

    # Rescale (corr is scale-invariant), so std/var won't overflow.
    p_scale = np.max(np.abs(p), axis=0, keepdims=True)
    y_scale = np.max(np.abs(y), axis=0, keepdims=True)
    p_scale = np.where(p_scale < eps, 1.0, p_scale)
    y_scale = np.where(y_scale < eps, 1.0, y_scale)

    ps = p / p_scale
    ys = y / y_scale

    mp = ps.mean(axis=0)
    my = ys.mean(axis=0)

    sp = ps.std(axis=0)
    sy = ys.std(axis=0)

    denom = (sp * sy) + eps
    corr_vec = ((ps - mp) * (ys - my)).mean(axis=0) / denom

    valid = (sy > 0) & (sp > 0)
    return float(corr_vec[valid].mean()) if valid.any() else float("nan")


# Use compute_metrics from util module, but add extra sMAPE, RSE, RAE for compatibility
def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # Get base metrics from util
    base = compute_metrics(y_true, y_pred)
    # Diagnostic logging: if compute_metrics returned NaNs, record shapes and finite counts
    try:
        rrse_base = base.get("rrse", None)
        if rrse_base is None or (isinstance(rrse_base, float) and np.isnan(rrse_base)):
            y_a = np.asarray(y_true)
            p_a = np.asarray(y_pred)
            finite_mask = np.isfinite(y_a) & np.isfinite(p_a)
            try:
                jlog(
                    "metrics_base_nan",
                    y_shape=int(y_a.size),
                    p_shape=int(p_a.size),
                    finite_count=int(finite_mask.sum()),
                )
            except Exception:
                pass
            print(
                f"[MetricsDiag] base metrics NaN: y_size={int(y_a.size)} p_size={int(p_a.size)} finite={int(finite_mask.sum())}"
            )
    except Exception:
        pass
    # Add extra metrics for full compatibility
    t = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_pred, dtype=np.float64).ravel()
    finite = np.isfinite(t) & np.isfinite(p)
    if finite.any():
        t = t[finite]
        p = p[finite]
        max_abs = np.sqrt(np.finfo(np.float64).max) * 0.5
        if (np.abs(t) > max_abs).any() or (np.abs(p) > max_abs).any():
            t = np.clip(t, -max_abs, max_abs)
            p = np.clip(p, -max_abs, max_abs)
    else:
        # No overlapping finite entries
        try:
            jlog(
                "metrics_no_finite_overlap",
                y_shape=int(np.asarray(y_true).size),
                p_shape=int(np.asarray(y_pred).size),
                finite_count=0,
            )
        except Exception:
            pass
        print(
            f"[MetricsDiag] no finite overlap: y_size={int(np.asarray(y_true).size)} p_size={int(np.asarray(y_pred).size)}"
        )
        t = np.array([], dtype=np.float64)
        p = np.array([], dtype=np.float64)
    eps = 1e-9
    smape = (
        float(np.mean(2.0 * np.abs(p - t) / np.maximum(eps, (np.abs(t) + np.abs(p))))) * 100.0
        if t.size
        else float("nan")
    )
    # RAE = sum |t-p| / sum |t-mean(t)|
    if t.size > 0:
        diff = t - p
        centered = t - t.mean()
        denom_abs = np.sum(np.abs(centered))
        rae = float(np.sum(np.abs(diff)) / (denom_abs + eps)) if denom_abs > 0 else float("nan")
    else:
        rae = float("nan")
    # Combine: capitalize keys for consistency
    return {
        "MAE": base["mae"],
        "RMSE": base["rmse"],
        "MAPE": base["mape"],
        "sMAPE": smape,
        "RSE": base["rrse"],
        "RAE": rae,
    }
