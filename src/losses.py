# src/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Optional

class PairwiseTemporalRankLoss(nn.Module):
    """Adjacent-pair directional rank loss.

    Keeps backward compatibility with earlier margin-based form but now uses
    a temperatured logistic (softplus) when margin == 0 for smoother grads.
    Inputs: y_hat, y_true: [B,T,N]
    """
    def __init__(self, margin: float = 0.0, reduction: str = "mean", temperature: float = 1.0):
        super().__init__()
        self.margin = float(margin)
        self.reduction = reduction
        self.temperature = float(max(1e-6, temperature))

    def forward(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_hat.ndim != 3 or y_true.ndim != 3:
            raise ValueError("PairwiseTemporalRankLoss expects [B,T,N] tensors")
        d_true = y_true[:, 1:, :] - y_true[:, :-1, :]
        d_hat  = y_hat[:, 1:, :] - y_hat[:, :-1, :]
        sgn = torch.sign(d_true)
        if self.margin == 0.0:
            loss = torch.nn.functional.softplus(-sgn * d_hat / self.temperature)
        else:
            loss = torch.relu(self.margin - sgn * d_hat)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

def change_point_magnitude_loss(y_hat: torch.Tensor,
                                y_true: torch.Tensor,
                                k_points: int = 2,
                                tau: float = 1.0) -> torch.Tensor:
    """
    Emphasize top-|Δy_true| change points: penalize under-response of |Δy_hat|.
    y_*: [B,T,N] in z-space. Returns scalar.
    """
    B, T, N = y_true.shape
    if T < 2 or k_points <= 0:
        return y_hat.new_tensor(0.0)

    d_true = torch.abs(y_true[:, 1:, :] - y_true[:, :-1, :])  # [B,T-1,N]
    d_hat  = torch.abs(y_hat[:, 1:, :]  - y_hat[:, :-1, :])   # [B,T-1,N]

    k = max(1, min(int(k_points), T - 1))
    idx = torch.topk(d_true, k=k, dim=1).indices  # [B,k,N]

    b_idx = torch.arange(B, device=y_true.device)[:, None, None].expand(B, k, N)
    n_idx = torch.arange(N, device=y_true.device)[None, None, :].expand(B, k, N)
    dt_sel = d_true[b_idx, idx, n_idx]  # [B,k,N]
    dh_sel = d_hat[b_idx, idx, n_idx]   # [B,k,N]

    loss = F.relu(dt_sel - dh_sel)
    if tau > 0:
        loss = loss / float(tau)
    return loss.mean()

# --- Additional probabilistic / quantile utilities -------------------------
def pinball_loss(pred_q: torch.Tensor, y_true: torch.Tensor, quantiles: Sequence[float]) -> torch.Tensor:
    """Pinball (quantile) loss.
    pred_q: [B,T,N,Q] or [T,N,Q]; y_true: [B,T,N] or [T,N].
    """
    if y_true.ndim == 2:
        y_true = y_true.unsqueeze(0)
    if pred_q.ndim == 3:
        pred_q = pred_q.unsqueeze(0)
    if pred_q.ndim != 4:
        raise ValueError("pred_q must be 4D [B,T,N,Q]")
    Q = pred_q.shape[-1]
    if Q != len(quantiles):
        raise ValueError("Quantile dimension mismatch")
    y = y_true.unsqueeze(-1).expand_as(pred_q)
    e = y - pred_q
    qs = torch.tensor(quantiles, device=pred_q.device, dtype=pred_q.dtype).view(1,1,1,-1)
    loss = torch.maximum(qs * e, (qs - 1.0) * e)
    return loss.mean()

def gaussian_nll(mu: torch.Tensor, log_var: torch.Tensor, y_true: torch.Tensor, temperature: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    if mu.ndim == 2: mu = mu.unsqueeze(0)
    if log_var.ndim == 2: log_var = log_var.unsqueeze(0)
    if y_true.ndim == 2: y_true = y_true.unsqueeze(0)
    var = (log_var.exp() + eps) * (temperature ** 2)
    return 0.5 * (log_var + (y_true - mu) ** 2 / (var + eps)).mean()

__all__ = [
    'PairwiseTemporalRankLoss',
    'change_point_magnitude_loss',
    'pinball_loss',
    'gaussian_nll',
    'weighted_huber_horizon_loss'
]

def weighted_huber_horizon_loss(
    yhat_t: torch.Tensor,
    ytrue_t: torch.Tensor,
    *,
    delta: float = 1.0,
    nonzero_weight: float = 4.0,
    horizon_gamma: float = 1.5,
    node_weights: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Time-horizon weighted Huber loss with optional node-wise weights.

    Args:
        yhat_t:  (B, H, N) predictions in normalized space.
        ytrue_t:(B, H, N) targets in normalized space.
        delta:  Huber threshold.
        nonzero_weight: extra weight for strictly positive targets.
        horizon_gamma: exponent for horizon weighting (later steps heavier).
        node_weights: optional per-node weights, shape (N,) or (1, N) or
            broadcastable to (1, 1, N). Typically used to up-weight RMD
            nodes relative to PT and auxiliary nodes.
        eps:    numerical epsilon.

    Returns:
        Scalar Huber loss.
    """
    if yhat_t.dim() == 4:
        yhat_t = yhat_t.squeeze(-1)
    if ytrue_t.dim() == 4:
        ytrue_t = ytrue_t.squeeze(-1)

    if yhat_t.shape != ytrue_t.shape:
        raise ValueError(f"Shape mismatch yhat={yhat_t.shape} ytrue={ytrue_t.shape}")

    B, H, N = ytrue_t.shape

    # Standard Huber
    diff = yhat_t - ytrue_t
    abs_diff = diff.abs()
    quadratic = 0.5 * diff.pow(2)
    linear = delta * (abs_diff - 0.5 * delta)
    huber = torch.where(abs_diff <= delta, quadratic, linear)

    # (1) non-zero mask weights
    nz = (ytrue_t > 0).float()
    w_nz = 1.0 + nonzero_weight * nz

    # (2) horizon weights: later horizons get higher weight
    t = torch.arange(H, device=ytrue_t.device, dtype=ytrue_t.dtype)
    w_h = ((t + 1.0) / float(H + eps)) ** horizon_gamma
    w_h = w_h.view(1, H, 1)  # (1, H, 1) broadcast over batch and nodes

    # (3) optional node-wise weights (RMD/PT/other)
    if node_weights is not None:
        # Accept shapes (N,), (1,N), (1,1,N); broadcast to (1,1,N)
        if node_weights.dim() == 1:
            w_node = node_weights.view(1, 1, N)
        elif node_weights.dim() == 2:
            w_node = node_weights.view(1, 1, N)
        else:
            # try to broadcast; if it fails, drop back to uniform weights
            try:
                w_node = node_weights.view(1, 1, N)
            except Exception:
                w_node = torch.ones(1, 1, N, device=ytrue_t.device, dtype=ytrue_t.dtype)
        w_node = w_node.to(device=ytrue_t.device, dtype=ytrue_t.dtype)
    else:
        w_node = torch.ones(1, 1, N, device=ytrue_t.device, dtype=ytrue_t.dtype)

    # Combine all weights
    w = w_nz * w_h * w_node  # (B, H, N)

    return (huber * w).mean()

# --- NB/ZINB losses (imported from src.uncertainty) ---
# Wrappers that adapt parameter order: losses.py uses (mu, log_alpha, y_true)
# but uncertainty.py expects (y_true, mu, log_alpha)
try:
    from src.uncertainty import nb_nll as _nb_nll_impl, zinb_nll as _zinb_nll_impl
    
    def nb_nll(mu: torch.Tensor, log_alpha: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Negative Binomial NLL (parameter order adapted from src.uncertainty)."""
        return _nb_nll_impl(y_true=y_true, mu=mu, log_alpha=log_alpha)
    
    def zinb_nll(mu: torch.Tensor, log_alpha: torch.Tensor, logit_pi: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Zero-Inflated NB NLL (parameter order adapted from src.uncertainty)."""
        return _zinb_nll_impl(y_true=y_true, mu=mu, log_alpha=log_alpha, logit_pi=logit_pi)
    
except ImportError as _import_err:
    # Fallback: define stubs that raise informative errors
    def nb_nll(mu: torch.Tensor, log_alpha: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:  # type: ignore[misc]
        raise RuntimeError(f"nb_nll unavailable: src.uncertainty import failed ({_import_err})")
    
    def zinb_nll(mu: torch.Tensor, log_alpha: torch.Tensor, logit_pi: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:  # type: ignore[misc]
        raise RuntimeError(f"zinb_nll unavailable: src.uncertainty import failed ({_import_err})")

__all__ += ['nb_nll', 'zinb_nll']
