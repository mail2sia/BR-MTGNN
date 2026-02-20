# src/uncertainty.py
# Lightweight uncertainty utilities: Negative Binomial (and ZINB) NLL,
# weighted quantiles, conformal calibration, and MC variance decomposition.
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

# -----------------------------
# Numerical-stability helpers
# -----------------------------
def _np_finite_float64(a):
    a = np.asarray(a, dtype=np.float64)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

def _safe_var(a, axis=None, eps: float = 1e-12):
    x = _np_finite_float64(a)
    scale = np.max(np.abs(x), axis=axis, keepdims=True)
    scale = np.where(scale < eps, 1.0, scale)
    xs = x / scale
    v = np.var(xs, axis=axis, ddof=0)
    return v * np.squeeze(np.square(scale, dtype=np.float64), axis=axis)

# ---------------------------
# Negative Binomial utilities
# ---------------------------

def _softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)

def nb_nll(y_true: torch.Tensor,
           mu: torch.Tensor,
           log_alpha: torch.Tensor,
           eps: float = 1e-8) -> torch.Tensor:
    """Negative Binomial NLL with mean/dispersion parameterization.
    Var(Y) = mu + alpha * mu^2, where alpha = softplus(log_alpha).
    Accepts real-valued y_true via lgamma continuity.
    Broadcasts over shapes; returns mean NLL.
    """
    alpha = _softplus(log_alpha) + eps
    mu = torch.clamp(mu, min=eps)
    r = 1.0 / (alpha + eps)                 # shape as broadcasted
    log_coef = torch.lgamma(y_true + r) - torch.lgamma(r) - torch.lgamma(y_true + 1.0)
    log_p = torch.log(r) - torch.log(r + mu)
    log_1mp = torch.log(mu) - torch.log(r + mu)
    log_nb = log_coef + r * log_p + y_true * log_1mp
    return (-log_nb).mean()

def zinb_nll(y_true: torch.Tensor,
             mu: torch.Tensor,
             log_alpha: torch.Tensor,
             logit_pi: torch.Tensor,
             eps: float = 1e-8) -> torch.Tensor:
    """Zero-inflated NB NLL.
    pi = sigmoid(logit_pi) is the zero-inflation prob.
    For y==0: loglik = log(pi + (1-pi)*NB(y=0)); for y>0: loglik = log(1-pi)+log NB(y).
    """
    alpha = _softplus(log_alpha) + eps
    mu = torch.clamp(mu, min=eps)
    r = 1.0 / (alpha + eps)
    # NB at y=0
    log_p = torch.log(r) - torch.log(r + mu)
    log_nb0 = -torch.lgamma(r) + r * log_p
    pi = torch.sigmoid(logit_pi)
    is_zero = (y_true <= 0.0).to(mu.dtype)
    log_pi = torch.log(pi + eps)
    log1m_pi = torch.log1p(-pi + eps)
    log_mix0 = torch.logaddexp(log_pi, log1m_pi + log_nb0)
    # NB for y>0
    log_coef = torch.lgamma(y_true + r) - torch.lgamma(r) - torch.lgamma(y_true + 1.0)
    log_1mp = torch.log(mu) - torch.log(r + mu)
    log_nb = log_coef + r * log_p + y_true * log_1mp
    loglik = is_zero * log_mix0 + (1.0 - is_zero) * (log1m_pi + log_nb)
    return (-loglik).mean()

# ---------------------------------
# MC variance decomposition (numpy)
# ---------------------------------

def mc_decompose(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose total variance into epistemic and aleatoric.
    samples: [S,T,N] with S MC or ensemble draws.
    Returns (mean, epistemic_var, aleatoric_var) each [T,N].
    """
    if samples.ndim != 3:
        raise ValueError(f"samples must be [S,T,N], got {samples.shape}")
    mean = samples.mean(axis=0)
    total_var = _safe_var(samples, axis=0)
    S = samples.shape[0]
    K = max(1, int(np.sqrt(S)))
    chunk = max(1, S // K)
    means = np.stack([samples[i:i+chunk].mean(axis=0) for i in range(0, S, chunk)], axis=0)
    epistemic = _safe_var(means, axis=0)
    aleatoric = np.clip(total_var - epistemic, 0.0, None)
    return mean, epistemic, aleatoric

# -----------------------------
# Weighted quantiles (numpy)
# -----------------------------

def weighted_quantile(values: np.ndarray, quantile: float, weights: Optional[np.ndarray] = None) -> float:
    v = np.asarray(values, dtype=float).ravel()
    if weights is None:
        return float(np.quantile(v, quantile))
    w = np.asarray(weights, dtype=float).ravel()
    idx = np.argsort(v)
    v = v[idx]; w = w[idx]
    cw = np.cumsum(w) / np.sum(w)
    j = np.searchsorted(cw, quantile, side='right')
    j = np.clip(j, 0, len(v)-1)
    return float(v[j])

# ----------------------------------------
# Conformal: CQR (Mondrian by node, decay)
# ----------------------------------------

class ConformalCalibrator:
    def __init__(self, alpha: float=0.05, window: int=60, decay: float=0.015):
        self.alpha = float(alpha)
        self.window = int(window)
        self.decay = float(decay)
        self.qhat_lo = None
        self.qhat_hi = None

    def fit(self, y_true_hist: np.ndarray, q_lo_hist: np.ndarray, q_hi_hist: np.ndarray) -> None:
        """Nonconformity s = max(q_lo - y, y - q_hi), per-node; weighted by recency."""
        y = np.asarray(y_true_hist, dtype=float)
        lo = np.asarray(q_lo_hist, dtype=float)
        hi = np.asarray(q_hi_hist, dtype=float)
        T, N = y.shape
        y = y[-min(self.window, T):]
        lo = lo[-len(y):]; hi = hi[-len(y):]
        s = np.maximum(lo - y, y - hi)
        s = np.clip(s, 0.0, None)
        if self.decay > 0:
            ages = np.arange(len(y))[::-1]
            w = np.exp(-self.decay * ages)
        else:
            w = np.ones(len(y), dtype=float)
        q = 1.0 - self.alpha
        self.qhat_lo = np.zeros(y.shape[1], dtype=float)
        self.qhat_hi = np.zeros(y.shape[1], dtype=float)
        for n in range(y.shape[1]):
            self.qhat_hi[n] = weighted_quantile(s[:, n], q, weights=w)
            self.qhat_lo[n] = self.qhat_hi[n]

    def calibrate(self, q_lo: np.ndarray, q_hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert self.qhat_lo is not None and self.qhat_hi is not None, "Call fit() first."
        lo = np.asarray(q_lo, dtype=float).copy()
        hi = np.asarray(q_hi, dtype=float).copy()
        return lo - self.qhat_lo[None, :], hi + self.qhat_hi[None, :]
