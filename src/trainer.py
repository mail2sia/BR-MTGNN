# Training logic for B-MTGNN
import torch
import torch.nn as nn
import os
import torch.optim as optim
import math
import torch.nn.functional as F
from src.net import *
from src.losses import (
    nb_nll,
    zinb_nll,
    weighted_huber_horizon_loss,
    PairwiseTemporalRankLoss,
    change_point_magnitude_loss,
    pinball_loss,
    gaussian_nll,
)
from src import util
from typing import Optional, List, Dict
import numpy as np


# ------------------ Target transform helpers ------------------
def _apply_y_transform(y: torch.Tensor, mode: str) -> torch.Tensor:
    if mode is None or mode == 'none':
        return y
    if mode == 'log1p':
        return torch.log1p(torch.clamp(y, min=0.0))
    raise ValueError(f"Unknown y_transform: {mode}")


def _invert_y_transform(y_t: torch.Tensor, mode: str) -> torch.Tensor:
    if mode is None or mode == 'none':
        return y_t
    if mode == 'log1p':
        return torch.expm1(y_t)
    raise ValueError(f"Unknown y_transform: {mode}")

# ------------------------------------------------------------------------------------

class Trainer():
    def __init__(self, model, lrate, wdecay, clip, step_size, seq_out_len, scaler, device,
                 cl=True,
                 # NEW knobs (all optional / default off)
                 lambda_rank: float = 0.0,
                 lambda_cp: float = 0.0,
                 cp_k: int = 2,
                 cp_tau: float = 1.0,
                 use_ordinal: bool = False,
                 ordinal_levels: int = 5,
                 lambda_ord: float = 0.0,
                 use_gauss: bool = False,
                 lambda_nll: float = 0.0,
                 quantiles: Optional[List[float]] = None,
                 lambda_q: float = 0.0,
                 use_nb_head: bool = False, use_zinb: bool = False,
                 # transform & loss knobs
                 y_transform: str = 'none',
                 nonzero_weight: float = 4.0,
                 horizon_gamma: float = 1.5,
                 huber_delta: float = 1.0):
        self.model = model
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = clip
        self.step = step_size
        self.iter = 1
        self.task_level = 1
        self.seq_out_len = seq_out_len
        self.cl = cl
        self.device = device
        
        # Learning rate warmup for stability (especially with temporal transformers)
        self.base_lr = lrate
        self.warmup_steps = int(os.environ.get('BMTGNN_WARMUP_STEPS', '0'))
        if self.warmup_steps > 0:
            print(f'[Trainer] LR warmup enabled: {self.warmup_steps} steps')
        self.grad_warn_every = int(os.environ.get('BMTGNN_GRAD_WARN_EVERY', '50'))
        self.grad_warn_mult = float(os.environ.get('BMTGNN_GRAD_WARN_MULT', '50'))
        self.grad_warn_min = float(os.environ.get('BMTGNN_GRAD_WARN_MIN', '1000'))

        # Directional and change-point losses
        self.lambda_rank = float(lambda_rank)
        self.lambda_cp   = float(lambda_cp)
        self.cp_k        = int(cp_k)
        self.cp_tau      = float(cp_tau)
        self._rank_loss_fn = PairwiseTemporalRankLoss(margin=0.0, reduction='mean')

        # Ordinal surrogate (self-supervised; no labels)
        self.use_ordinal    = bool(use_ordinal)
        self.ordinal_levels = int(ordinal_levels)
        self.lambda_ord     = float(lambda_ord)
        if self.use_ordinal:
            k = max(2, self.ordinal_levels)
            self._theta = torch.nn.Parameter(torch.randn(k - 1, device=device))

        # Lightweight Gaussian NLL + temperature (uncertainty)
        self.use_gauss = bool(use_gauss)
        self.lambda_nll = float(lambda_nll)
        if self.use_gauss:
            self._log_var    = torch.nn.Parameter(torch.zeros(1, device=device))
            self._temperature = torch.nn.Parameter(torch.ones(1, device=device))
            # ensure the new params are optimized
            self.optimizer.add_param_group({'params': [self._log_var, self._temperature]})

        # Quantile regression (pinball)
        self.taus = torch.tensor(quantiles, device=device).float() if quantiles else None
        self.lambda_q = float(lambda_q)

        # Negative Binomial / ZINB (optional)
        self.use_nb_head = bool(use_nb_head)
        self.use_zinb = bool(use_zinb)
        # Transform + loss tuning (applied in original units via scaler.inverse_transform)
        self.y_transform = y_transform
        self.nonzero_weight = float(nonzero_weight)
        self.horizon_gamma = float(horizon_gamma)
        self.huber_delta = float(huber_delta)
        if self.use_nb_head:
            # Fallback global params if model head absent
            self._log_alpha = torch.nn.Parameter(torch.zeros(1, device=device))
            self._logit_pi  = torch.nn.Parameter(torch.zeros(1, device=device)) if self.use_zinb else None
            self.optimizer.add_param_group({'params': [p for p in [self._log_alpha, self._logit_pi] if p is not None]})

    def _pinball_loss(self, q_pred: torch.Tensor, y_true: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
        """
        q_pred: [B, K, N, T]
        y_true: [B, 1, N, T]  (broadcast over K)
        taus  : [K]
        """
        t = taus.view(1, -1, 1, 1)
        err = y_true - q_pred
        loss = torch.maximum(t * err, (t - 1.0) * err)
        return loss.mean()

    def _ordinal_thresholds(self):
        # ordered positive increments via softplus, then cumulative sum
        return torch.cumsum(F.softplus(self._theta), dim=0)  # [K-1]

    def train(self, input, real_val, idx=None):
        self.model.train()
        self.optimizer.zero_grad()
        try:
            if bool(int(os.environ.get('BMTGNN_DEBUG_LAYOUT','0'))):
                print(f"[Trainer.train] input.shape={tuple(input.shape)} model.seq_length={getattr(self.model,'seq_length','?')}")
        except Exception:
            pass
        output = self.model(input, idx=idx)

        # accept dict from model (mean + optional quantiles)
        if isinstance(output, dict):
            pred_mean = output['mean']
            pred_q    = output.get('quantiles', None)
            pred_logvar = output.get('logvar', None)
            taus      = output.get('taus', None)
        else:
            pred_mean = output
            pred_q, pred_logvar, taus = None, None, None

        pred_mean = pred_mean.transpose(1,3)
        # Both output and real_val are in normalized (z) space already
        # Shapes:
        #  output: (B, 1, N, out_len)
        #  real_val: (B, out_len, N) -> (B, 1, out_len, N) -> (B, 1, N, out_len)
        real = torch.unsqueeze(real_val, dim=1).transpose(2, 3)
        pred = pred_mean
        pred_use = pred[:, :, :, :self.seq_out_len]
        real_use = real[:, :, :, :self.seq_out_len]

        logvar_use = None
        if pred_logvar is not None:
            pred_logvar = pred_logvar.transpose(1,3)
            logvar_use = pred_logvar[:, :, :, :self.seq_out_len]
            logvar_use = logvar_use.squeeze(1).transpose(1, 2)

        # Compute primary loss in transformed ORIGINAL units (if scaler available)
        # Collapse to [B,T,N] in z-space first
        y_hat = pred_use.squeeze(1).transpose(1, 2)  # (B, out, N) -> (B,T,N)
        y_true = real_use.squeeze(1).transpose(1, 2)

        total = 0.0
        try:
            # Optional debug: print scaler & batch stats when troubleshooting large metrics
            if os.environ.get('BMTGNN_DEBUG_SCALE', '0') == '1':
                try:
                    sm = getattr(self.scaler, 'mean', None)
                    ss = getattr(self.scaler, 'std', None)
                    print('[BMTGNN_DEBUG_SCALE] y_hat.shape=', tuple(y_hat.shape), 'y_true.shape=', tuple(y_true.shape))
                    if sm is not None and ss is not None:
                        try:
                            print('[BMTGNN_DEBUG_SCALE] scaler.mean.shape=', tuple(sm.shape) if hasattr(sm, 'shape') else type(sm),
                                  'scaler.std.shape=', tuple(ss.shape) if hasattr(ss, 'shape') else type(ss))
                            # Print scaler percentiles and top std nodes
                            try:
                                # ensure tensors on GPU are moved to CPU before numpy conversion
                                if hasattr(ss, 'detach') and hasattr(ss, 'cpu'):
                                    ss_a = ss.detach().cpu().numpy()
                                else:
                                    ss_a = np.asarray(ss)
                                if hasattr(sm, 'detach') and hasattr(sm, 'cpu'):
                                    sm_a = sm.detach().cpu().numpy()
                                else:
                                    sm_a = np.asarray(sm)
                                ss_a_flat = ss_a.ravel()
                                pcts = np.percentile(ss_a_flat, [0, 25, 50, 75, 90, 95, 99])
                                print('[BMTGNN_DEBUG_SCALE] scaler.std percentiles 0/25/50/75/90/95/99= ', tuple(float(x) for x in pcts))
                                # show top-5 largest stds and their node indices
                                if ss_a_flat.size > 0:
                                    top_idx = np.argsort(-ss_a_flat)[:5]
                                    top_vals = ss_a_flat[top_idx]
                                    print('[BMTGNN_DEBUG_SCALE] scaler.std top5 (idx,val)=', list(zip(top_idx.tolist(), top_vals.tolist())))
                                    # nodes with very large std
                                    large_idx = np.where(ss_a_flat > (np.median(ss_a_flat) * 10))[0]
                                    if large_idx.size > 0:
                                        print('[BMTGNN_DEBUG_SCALE] scaler.std large-mult10 median: count=', large_idx.size,
                                              'examples=', large_idx[:10].tolist())
                            except Exception as _ee:
                                print('[BMTGNN_DEBUG_SCALE] scaler percentile debug failed:', _ee)
                        except Exception:
                            print('[BMTGNN_DEBUG_SCALE] scaler.mean/std types', type(sm), type(ss))
                    with torch.no_grad():
                        print('[BMTGNN_DEBUG_SCALE] y_hat z-space stats min,mean,max,std=',
                              float(y_hat.min()), float(y_hat.mean()), float(y_hat.max()), float(y_hat.std()))
                        print('[BMTGNN_DEBUG_SCALE] y_true z-space stats min,mean,max,std=',
                              float(y_true.min()), float(y_true.mean()), float(y_true.max()), float(y_true.std()))
                except Exception as _e:
                    print('[BMTGNN_DEBUG_SCALE] debug print failed:', _e)

            # Optional z-space clipping to limit extreme z predictions before inverse transform
            z_clip_val = os.environ.get('BMTGNN_Z_CLIP', None)
            if z_clip_val is not None:
                try:
                    clip_v = float(z_clip_val)
                    # Apply clamp in normal autograd mode so gradients are preserved.
                    y_hat = y_hat.clamp(min=-clip_v, max=clip_v)
                    y_true = y_true.clamp(min=-clip_v, max=clip_v)
                    if os.environ.get('BMTGNN_DEBUG_SCALE', '0') == '1':
                        print(f'[BMTGNN_DEBUG_SCALE] applied z-space clip={clip_v}')
                except Exception as _e:
                    print('[BMTGNN_DEBUG_SCALE] z-clip parse/apply failed:', _e)

            # inverse transform to original units using provided scaler
            y_hat_o = self.scaler.inverse_transform(y_hat)
            y_true_o = self.scaler.inverse_transform(y_true)
            if os.environ.get('BMTGNN_DEBUG_SCALE', '0') == '1':
                try:
                    with torch.no_grad():
                        print('[BMTGNN_DEBUG_SCALE] y_hat_o (post-inv) min,mean,max,std=',
                              float(y_hat_o.min()), float(y_hat_o.mean()), float(y_hat_o.max()), float(y_hat_o.std()))
                        print('[BMTGNN_DEBUG_SCALE] y_true_o (post-inv) min,mean,max,std=',
                              float(y_true_o.min()), float(y_true_o.mean()), float(y_true_o.max()), float(y_true_o.std()))
                except Exception as _e:
                    print('[BMTGNN_DEBUG_SCALE] post-inv stats failed:', _e)
            # apply target transform (e.g., log1p)
            y_hat_t = _apply_y_transform(y_hat_o, self.y_transform)
            y_true_t = _apply_y_transform(y_true_o, self.y_transform)
            # Optional: mask or down-weight top-K nodes by scaler.std during TRAINING
            try:
                mask_k = os.environ.get('BMTGNN_MASK_TOP_STD', None)
                dw = os.environ.get('BMTGNN_DOWNWEIGHT_TOP_STD', None)
                if (mask_k is not None) or (dw is not None):
                    try:
                        ss = getattr(self.scaler, 'std', None)
                        if ss is not None:
                            # get numpy array of stds
                            if hasattr(ss, 'detach') and hasattr(ss, 'cpu'):
                                ss_a = ss.detach().cpu().numpy().ravel()
                            else:
                                import numpy as _np
                                ss_a = _np.asarray(ss).ravel()
                            # determine top-k indices
                            if mask_k is not None:
                                k = int(mask_k)
                            else:
                                k = 0
                            if k > 0 and ss_a.size > 0:
                                import numpy as _np
                                top_idx = _np.argsort(-ss_a)[:k]
                                top_idx_t = torch.as_tensor(top_idx, dtype=torch.long, device=y_hat_t.device)
                                if dw is not None:
                                    try:
                                        weight = float(dw)
                                    except Exception:
                                        weight = 0.0
                                else:
                                    weight = 0.0
                                # weight==0 -> full mask (no loss for those nodes)
                                # Apply by shrinking the prediction towards truth for masked nodes
                                if weight <= 0.0:
                                    if os.environ.get('BMTGNN_DEBUG_SCALE', '0') == '1':
                                        print(f'[BMTGNN_DEBUG_SCALE] masking top-{k} std nodes: {top_idx.tolist()}')
                                    # set prediction equal to truth for those nodes (no loss)
                                    y_hat_t.index_copy_(2, top_idx_t, y_true_t.index_select(2, top_idx_t))
                                else:
                                    if os.environ.get('BMTGNN_DEBUG_SCALE', '0') == '1':
                                        print(f'[BMTGNN_DEBUG_SCALE] downweighting top-{k} std nodes by factor {weight}: {top_idx.tolist()}')
                                    # shrink residual on those nodes
                                    res = y_hat_t.index_select(2, top_idx_t) - y_true_t.index_select(2, top_idx_t)
                                    new_res = res * float(weight)
                                    new_pred = y_true_t.index_select(2, top_idx_t) + new_res
                                    y_hat_t.index_copy_(2, top_idx_t, new_pred)
                    except Exception as _e:
                        if os.environ.get('BMTGNN_DEBUG_SCALE', '0') == '1':
                            print('[BMTGNN_DEBUG_SCALE] mask/downweight apply failed:', _e)
            except Exception:
                pass
            # compute weighted huber horizon loss in transformed original units
            base = weighted_huber_horizon_loss(
                y_hat_t,
                y_true_t,
                delta=self.huber_delta,
                nonzero_weight=self.nonzero_weight,
                horizon_gamma=self.horizon_gamma,
            )
            total = base
        except Exception:
            # fallback to existing z-space loss
            if self.cl:
                base = self.loss(pred_use[:, :, :, :self.task_level], real_use[:, :, :, :self.task_level], 0.0)
            else:
                base = self.loss(pred_use, real_use, 0.0)
            total = base

        # Directional (pairwise temporal ranking)
        if self.lambda_rank > 0.0:
            total = total + self.lambda_rank * self._rank_loss_fn(y_hat, y_true)

        # Change-point magnitude response
        if self.lambda_cp > 0.0:
            total = total + self.lambda_cp * change_point_magnitude_loss(
                y_hat, y_true, k_points=self.cp_k, tau=self.cp_tau
            )

        # Ordinal surrogate (no external labels)
        if self.use_ordinal and self.lambda_ord > 0.0:
            B, T, N = y_true.shape
            y_flat = y_true.reshape(B*T*N)
            K = max(2, self.ordinal_levels)
            qs = torch.quantile(y_flat, torch.linspace(0, 1, K+1, device=y_true.device))
            qs[0] -= 1e-6; qs[-1] += 1e-6
            y_exp = y_true.unsqueeze(-1)                          # [B,T,N,1]
            levels = ((y_exp >= qs[:-1]) & (y_exp < qs[1:])).float()  # [B,T,N,K]
            cum_targets = levels.flip(-1).cumsum(dim=-1).flip(-1)[...,:-1]  # [B,T,N,K-1]

            th = self._ordinal_thresholds().view(1,1,1,-1)  # [1,1,1,K-1]
            s  = y_hat.unsqueeze(-1)                        # [B,T,N,1]
            cum_logits = s - th                             # [B,T,N,K-1]
            cum_probs  = torch.sigmoid(cum_logits)
            loss_ord = F.binary_cross_entropy(cum_probs, cum_targets)
            total = total + self.lambda_ord * loss_ord

        # Lightweight Gaussian NLL with temperature (uncertainty calibration)
        if self.use_gauss and self.lambda_nll > 0.0:
            mu = y_hat
            if logvar_use is not None:
                log_var = logvar_use + torch.log(self._temperature.clamp_min(1e-6))
                log_var = torch.nan_to_num(log_var, nan=0.0, posinf=10.0, neginf=-10.0)
                log_var = log_var.clamp(min=-10.0, max=10.0)
            else:
                log_var = self._log_var * self._temperature
            var = torch.exp(log_var).clamp_min(1e-6)
            nll = 0.5 * ((y_true - mu)**2 / var + log_var)
            loss_nll = nll.mean()
            total = total + self.lambda_nll * loss_nll

        # NB/ZINB NLL (original units), if enabled
        if self.use_nb_head and (self.lambda_nll > 0.0):
            try:
                mu_o = self.scaler.inverse_transform(pred_use).clamp_min(0.0)
                y_o  = self.scaler.inverse_transform(real_use).clamp_min(0.0)
            except Exception:
                mu_o = pred_use.clamp_min(0.0)
                y_o  = real_use.clamp_min(0.0)
            mu_o = mu_o.squeeze(1).transpose(1,2)  # [B,T,N]
            y_o  = y_o.squeeze(1).transpose(1,2)
            log_alpha = None; logit_pi = None
            if isinstance(output, dict):
                da = output.get('dispersion', None)
                if da is not None:
                    log_alpha = (da.squeeze(1).transpose(1,2) if da.ndim == 4 else da)
                else:
                    log_alpha = self._log_alpha
                pi = output.get('pi', None)
                if pi is not None:
                    logit_pi = (pi.squeeze(1).transpose(1,2) if pi.ndim == 4 else pi)
                else:
                    logit_pi = self._logit_pi
            else:
                log_alpha = self._log_alpha
                logit_pi  = self._logit_pi
            if self.use_zinb and (logit_pi is not None):
                loss_nb = zinb_nll(mu=mu_o, log_alpha=log_alpha, logit_pi=logit_pi, y_true=y_o)
            else:
                loss_nb = nb_nll(mu=mu_o, log_alpha=log_alpha, y_true=y_o)
            total = total + self.lambda_nll * loss_nb

        # Quantile pinball loss (z-space), if provided
        if (self.taus is not None) and (self.lambda_q > 0.0) and (pred_q is not None):
            B, KC, N, _ = pred_q.shape
            K = int(self.taus.numel())
            out_len = int(KC // K)
            q = pred_q.view(B, K, out_len, N, 1).permute(0,1,3,2,4).squeeze(-1)   # [B,K,N,T]
            y = real_use.squeeze(1)                                               # [B,N,T]
            y = y.unsqueeze(1)                                                    # [B,1,N,T]
            loss_q = self._pinball_loss(q, y, self.taus)
            total = total + self.lambda_q * loss_q

        total.backward()

        # === CRITICAL: NaN/inf detection and gradient explosion prevention ===
        # Check for NaN/inf in loss value
        if not torch.isfinite(total):
            print(f'[GRAD_EXPLOSION] Loss is {total.item()}, skipping optimizer step')
            self.optimizer.zero_grad()
            self.iter += 1
            return float('nan'), float('nan'), float('nan')

        # Check for NaN/inf in gradients and compute pre-clip norm
        has_nan_grad = False
        total_norm_sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    has_nan_grad = True
                    break
                total_norm_sq += float(p.grad.data.norm(2).item() ** 2)
        
        pre_clip_norm = float(total_norm_sq ** 0.5)
        
        # Skip update if NaN/inf detected in gradients
        if has_nan_grad:
            print(f'[GRAD_EXPLOSION] NaN/inf detected in gradients, skipping optimizer step')
            self.optimizer.zero_grad()
            self.iter += 1
            return float('nan'), float('nan'), float('nan')
        
        # Warn if gradient norm is extremely large (potential explosion)
        warn_threshold = self.grad_warn_min
        if self.clip is not None:
            warn_threshold = max(warn_threshold, float(self.clip) * self.grad_warn_mult)
        if pre_clip_norm > warn_threshold:
            if self.grad_warn_every <= 1 or (self.iter % self.grad_warn_every == 0):
                print(f'[GRAD_WARNING] Very large gradient norm: {pre_clip_norm:.2f} (clipping to {self.clip})')
        
        # Optional detailed gradient debug
        if os.environ.get('BMTGNN_DEBUG_GRAD', '0') == '1':
            print(f'[BMTGNN_DEBUG_GRAD] pre_clip_grad_norm={pre_clip_norm:.4f}')

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        # Compute post-clip norm
        post_clip_total_norm_sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                post_clip_total_norm_sq += float(p.grad.data.norm(2).item() ** 2)
        post_clip_norm = float(post_clip_total_norm_sq ** 0.5)
        
        if os.environ.get('BMTGNN_DEBUG_GRAD', '0') == '1':
            print(f'[BMTGNN_DEBUG_GRAD] post_clip_grad_norm={post_clip_norm:.4f}')

        # Apply learning rate warmup if enabled
        if self.warmup_steps > 0 and self.iter <= self.warmup_steps:
            warmup_factor = float(self.iter) / float(self.warmup_steps)
            current_lr = self.base_lr * warmup_factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            if os.environ.get('BMTGNN_DEBUG_GRAD', '0') == '1' and self.iter % 10 == 0:
                print(f'[LR_WARMUP] step {self.iter}/{self.warmup_steps}, lr={current_lr:.6f}')

        self.optimizer.step()
        
        # === Check parameter norms after update to detect weight explosion ===
        param_norm_sq = 0.0
        max_param_norm = 0.0
        for p in self.model.parameters():
            if p.data is not None:
                pnorm = float(p.data.norm(2).item())
                param_norm_sq += pnorm ** 2
                max_param_norm = max(max_param_norm, pnorm)
        
        total_param_norm = float(param_norm_sq ** 0.5)
        
        # Warn if parameters are exploding
        if total_param_norm > 10000.0 or max_param_norm > 1000.0:
            print(f'[PARAM_EXPLOSION] Parameter norms very large: total={total_param_norm:.2f}, max={max_param_norm:.2f}')
        
        if os.environ.get('BMTGNN_DEBUG_GRAD', '0') == '1':
            print(f'[BMTGNN_DEBUG_GRAD] total_param_norm={total_param_norm:.4f}, max_param_norm={max_param_norm:.4f}')
        # Optional: compute metrics in original units for logging ONLY
        try:
            pred_o = self.scaler.inverse_transform(y_hat)
            real_o = self.scaler.inverse_transform(y_true)
            # If transform was applied during training, invert it for metrics
            try:
                pred_o_plot = _invert_y_transform(pred_o, self.y_transform)
                real_o_plot = _invert_y_transform(real_o, self.y_transform)
            except Exception:
                pred_o_plot = pred_o
                real_o_plot = real_o
            mape = util.masked_mape(pred_o_plot, real_o_plot, 0.0).item()
            rmse = util.masked_rmse(pred_o_plot, real_o_plot, 0.0).item()
        except Exception:
            # fallback to z-space metrics if scaler not provided
            mape = util.masked_mape(y_hat, y_true, 0.0).item()
            rmse = util.masked_rmse(y_hat, y_true, 0.0).item()
        self.iter += 1
        return base.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        if isinstance(output, dict):
            output = output['mean']
        output = output.transpose(1,3)
        
        # === CRITICAL: Check for NaN/inf in model output ===
        if not torch.isfinite(output).all():
            print('[EVAL_EXPLOSION] NaN/inf detected in model output, returning fallback metrics')
            return float('nan'), float('nan'), float('nan')
        
        # Clamp z-space predictions to prevent extreme values
        output = output.clamp(min=-50.0, max=50.0)
        
        # Keep loss in z-space for consistency
        real = torch.unsqueeze(real_val, dim=1).transpose(2, 3)
        pred = output
        # Try to compute evaluation loss using the same transformed original-unit loss
        try:
            y_hat = pred.squeeze(1).transpose(1, 2)  # [B,T,N]
            y_true = real.squeeze(1).transpose(1, 2)
            if os.environ.get('BMTGNN_DEBUG_SCALE', '0') == '1':
                try:
                    print('[BMTGNN_DEBUG_SCALE][EVAL] y_hat.shape=', tuple(y_hat.shape), 'y_true.shape=', tuple(y_true.shape))
                    print('[BMTGNN_DEBUG_SCALE][EVAL] y_hat z-space stats min,mean,max,std=',
                          float(y_hat.min()), float(y_hat.mean()), float(y_hat.max()), float(y_hat.std()))
                except Exception as _e:
                    print('[BMTGNN_DEBUG_SCALE][EVAL] debug pre-inv failed:', _e)

            # Optional z-space clipping (eval)
            z_clip_val = os.environ.get('BMTGNN_Z_CLIP', None)
            if z_clip_val is not None:
                try:
                    clip_v = float(z_clip_val)
                    with torch.no_grad():
                        y_hat = y_hat.clamp(min=-clip_v, max=clip_v)
                        y_true = y_true.clamp(min=-clip_v, max=clip_v)
                    if os.environ.get('BMTGNN_DEBUG_SCALE', '0') == '1':
                        print(f'[BMTGNN_DEBUG_SCALE][EVAL] applied z-space clip={clip_v}')
                except Exception as _e:
                    print('[BMTGNN_DEBUG_SCALE][EVAL] z-clip parse/apply failed:', _e)

            y_hat_o = self.scaler.inverse_transform(y_hat)
            y_true_o = self.scaler.inverse_transform(y_true)
            
            # === Check for explosions after inverse transform ===
            if not torch.isfinite(y_hat_o).all() or not torch.isfinite(y_true_o).all():
                print('[EVAL_EXPLOSION] NaN/inf after inverse transform, returning fallback')
                return float('nan'), float('nan'), float('nan')
            
            # Clamp to reasonable range in original units (adjust based on your data scale)
            y_hat_o = y_hat_o.clamp(min=-1e6, max=1e6)
            
            if os.environ.get('BMTGNN_DEBUG_SCALE', '0') == '1':
                try:
                    print('[BMTGNN_DEBUG_SCALE][EVAL] y_hat_o (post-inv) min,mean,max,std=',
                          float(y_hat_o.min()), float(y_hat_o.mean()), float(y_hat_o.max()), float(y_hat_o.std()))
                except Exception as _e:
                    print('[BMTGNN_DEBUG_SCALE][EVAL] post-inv stats failed:', _e)
            y_hat_t = _apply_y_transform(y_hat_o, self.y_transform)
            y_true_t = _apply_y_transform(y_true_o, self.y_transform)
            loss = weighted_huber_horizon_loss(
                y_hat_t,
                y_true_t,
                delta=self.huber_delta,
                nonzero_weight=self.nonzero_weight,
                horizon_gamma=self.horizon_gamma,
            )
            # metrics in original units (invert transform for plots/metrics)
            try:
                pred_o_plot = _invert_y_transform(y_hat_o, self.y_transform)
                real_o_plot = _invert_y_transform(y_true_o, self.y_transform)
            except Exception:
                pred_o_plot = y_hat_o
                real_o_plot = y_true_o
            mape = util.masked_mape(pred_o_plot, real_o_plot, 0.0).item()
            rmse = util.masked_rmse(pred_o_plot, real_o_plot, 0.0).item()
            return float(loss.item()), mape, rmse
        except Exception:
            loss = self.loss(pred, real, 0.0)
            try:
                pred_o = self.scaler.inverse_transform(pred)
                real_o = self.scaler.inverse_transform(real)
                mape = util.masked_mape(pred_o, real_o, 0.0).item()
                rmse = util.masked_rmse(pred_o, real_o, 0.0).item()
            except Exception:
                mape = util.masked_mape(pred, real, 0.0).item()
                rmse = util.masked_rmse(pred, real, 0.0).item()
            return float(loss.item()), mape, rmse



class Optim(object):

    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adamw':
            self.optimizer = optim.AdamW(self.params, lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, params, method, lr, clip, weight_decay=0.0, lr_decay=1, start_decay_at=None):
        self.params = params  # careful: params may be a generator
        self.last_ppl = None
        self.lr = lr
        self.clip = clip
        self.method = method
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False

        self._makeOptimizer()

    def step(self):
        # Compute gradients norm.
        grad_norm = 0
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip)

        # for param in self.params:
        #     grad_norm += math.pow(param.grad.data.norm(), 2)
        #
        # grad_norm = math.sqrt(grad_norm)
        # if grad_norm > 0:
        #     shrinkage = self.max_grad_norm / grad_norm
        # else:
        #     shrinkage = 1.
        #
        # for param in self.params:
        #     if shrinkage < 1:
        #         param.grad.data.mul_(shrinkage)
        self.optimizer.step()
        return  grad_norm

    # decay learning rate if val perf does not improve or we hit the start_decay_at limit
    def updateLearningRate(self, ppl, epoch):
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)
        #only decay for one epoch
        self.start_decay = False

        self.last_ppl = ppl

        self._makeOptimizer()


def run_trainer_path(
    *,
    args,
    Data,
    device,
    q_list,
    has_quantiles: bool,
    q_weight: bool,
    use_trainer_path: bool,
    evaluate_fn,
    evaluate_sliding_window_fn=None,
    to_model_layout_fn,
    jlog_fn,
):
    """Run the single-pass Trainer path when probabilistic features are enabled."""
    if not use_trainer_path:
        return

    # define evaluation losses locally for Trainer path
    evaluateL2 = nn.MSELoss(reduction='sum').to(device)
    evaluateL1 = nn.L1Loss(reduction='sum').to(device)

    # Guard: ensure seq_in_len matches Data windows (may have been auto-adjusted)
    try:
        _data_in_len = int(getattr(Data.train[0], 'shape', [0, 0, 0, 0])[1]) if Data and Data.train and Data.train[0] is not None else None
        if _data_in_len and _data_in_len != int(args.seq_in_len):
            print(f"[TrainerGuard] Adjusting seq_in_len from {args.seq_in_len} -> {_data_in_len} to match Data windows")
            args.seq_in_len = _data_in_len
        if hasattr(args, '_final_seq_in_len') and int(args._final_seq_in_len) != int(args.seq_in_len):
            print(f"[TrainerGuard] Overriding seq_in_len with recorded final {_data_in_len}->{args._final_seq_in_len}")
            args.seq_in_len = int(args._final_seq_in_len)
    except Exception as _e:
        print(f"[TrainerGuard] Warning: could not reconcile seq_in_len: {_e}")

    try:
        in_dim_use = getattr(Data, 'in_dim', args.in_dim)
    except NameError:
        in_dim_use = args.in_dim

    model: nn.Module | None = gtnet(
        args.gcn_true, args.buildA_true, int(args.gcn_depth), int(Data.m),
        device, Data.adj, dropout=float(args.dropout), subgraph_size=int(args.subgraph_size),
        node_dim=int(args.node_dim), dilation_exponential=int(args.dilation_exponential),
        conv_channels=int(args.conv_channels), residual_channels=int(args.residual_channels),
        skip_channels=int(args.skip_channels), end_channels=int(args.end_channels),
        seq_length=int(args.seq_in_len), in_dim=in_dim_use, out_dim=int(args.seq_out_len),
        layers=int(args.layers), propalpha=float(getattr(args, 'propalpha', 0.05)),
        tanhalpha=float(getattr(args, 'tanhalpha', 3)), layer_norm_affline=False,
        temporal_attn=bool(getattr(args, 'temporal_attn', False)),
        attn_dim=int(getattr(args, 'attn_dim', 64)),
        attn_heads=int(getattr(args, 'attn_heads', 2)),
        attn_dropout=float(getattr(args, 'attn_dropout', 0.1)),
        attn_window=int(getattr(args, 'attn_window', 0)),
        attn_math_mode=bool(getattr(args, 'attn_math_mode', False)),
        attn_bn_chunk=int(getattr(args, 'attn_bn_chunk', 0)),
        attn_gate_threshold=int(getattr(args, 'attn_gate_threshold', 0)),
        temporal_transformer=bool(getattr(args, 'temporal_transformer', 0)),
        tt_layers=int(getattr(args, 'tt_layers', 2)),
        graph_mix=float(getattr(args, 'graph_mix', 0.0)),
        dropedge_p=float(getattr(args, 'dropedge_p', 0.0)),
        quantiles=q_list,
        nb_head=bool(int(getattr(args, 'use_nb_head', 0)) == 1),
        zinb=bool(int(getattr(args, 'use_zinb', 0)) == 1),
    )
    model.to(device)

    try:
        seq_attr = getattr(model, 'seq_length', None)
        if seq_attr is not None and int(seq_attr) != int(args.seq_in_len):
            print(f"[TrainerGuard] Detected model.seq_length={seq_attr} != args.seq_in_len={args.seq_in_len}; rebuilding model")
            del model
            model = None
            model = gtnet(
                args.gcn_true, args.buildA_true, int(args.gcn_depth), int(Data.m),
                device, Data.adj, dropout=float(args.dropout), subgraph_size=int(args.subgraph_size),
                node_dim=int(args.node_dim), dilation_exponential=int(args.dilation_exponential),
                conv_channels=int(args.conv_channels), residual_channels=int(args.residual_channels),
                skip_channels=int(args.skip_channels), end_channels=int(args.end_channels),
                seq_length=int(args.seq_in_len), in_dim=in_dim_use, out_dim=int(args.seq_out_len),
                layers=int(args.layers), propalpha=float(getattr(args, 'propalpha', 0.05)),
                tanhalpha=float(getattr(args, 'tanhalpha', 3)), layer_norm_affline=False,
                temporal_attn=bool(getattr(args, 'temporal_attn', False)),
                attn_dim=int(getattr(args, 'attn_dim', 64)),
                attn_heads=int(getattr(args, 'attn_heads', 2)),
                attn_dropout=float(getattr(args, 'attn_dropout', 0.1)),
                attn_window=int(getattr(args, 'attn_window', 0)),
                attn_math_mode=bool(getattr(args, 'attn_math_mode', False)),
                attn_bn_chunk=int(getattr(args, 'attn_bn_chunk', 0)),
                attn_gate_threshold=int(getattr(args, 'attn_gate_threshold', 0)),
                temporal_transformer=bool(getattr(args, 'temporal_transformer', 0)),
                tt_layers=int(getattr(args, 'tt_layers', 2)),
                graph_mix=float(getattr(args, 'graph_mix', 0.0)),
                dropedge_p=float(getattr(args, 'dropedge_p', 0.0)),
                quantiles=q_list,
                nb_head=bool(int(getattr(args, 'use_nb_head', 0)) == 1),
                zinb=bool(int(getattr(args, 'use_zinb', 0)) == 1),
            )
            model.to(device)
        print(f"[TrainerGuard] Final model seq_length={getattr(model, 'seq_length', '?')} in_len={args.seq_in_len}")
    except Exception as _e:
        print(f"[TrainerGuard] seq_length validation skipped: {_e}")

    if model is None:
        raise RuntimeError("Model construction failed (model is None)")

    compile_mode = getattr(args, 'compile', 'auto')
    if getattr(torch, 'compile', None) is not None and compile_mode != 'off':
        try:
            from typing import cast
            compiled: nn.Module
            if compile_mode == 'eager':
                compiled = cast(nn.Module, torch.compile(model, backend='eager'))
            elif compile_mode == 'inductor':
                compiled = cast(nn.Module, torch.compile(model))
            else:
                compiled = cast(nn.Module, torch.compile(model))
            model = compiled
            print(f'[compile] backend={compile_mode}')
        except Exception as _e:
            print(f'[compile] disabled (fallback): {_e}')

    try:
        y_scaler = util.StandardScaler(Data.mu, Data.std)
    except Exception:
        y_scaler = util.StandardScaler(torch.tensor(0.0), torch.tensor(1.0))

    try:
        requested_y_transform = getattr(args, 'y_transform', 'none')
        data_applied_log1p = getattr(Data, 'use_log1p', False)
        if data_applied_log1p and (str(requested_y_transform).lower() == 'log1p'):
            print('[TransformGuard] DataLoader applied log1p (normalize==3) and --y_transform=log1p was requested.')
            print('[TransformGuard] To avoid double-transform, overriding trainer y_transform to "none".')
            setattr(args, 'y_transform', 'none')
    except Exception:
        pass

    trainer = Trainer(
        model=model,
        lrate=float(args.lr),
        wdecay=float(args.weight_decay),
        clip=float(getattr(args, 'clip', 0.0)),
        step_size=int(getattr(args, 'step_size', 100)),
        seq_out_len=int(args.seq_out_len),
        scaler=y_scaler,
        device=device,
        cl=True,
        quantiles=q_list if (has_quantiles and q_weight) else None,
        lambda_q=float(getattr(args, 'lambda_quantile', 0.0)),
        use_gauss=bool(int(getattr(args, 'use_gauss', 0)) == 1),
        lambda_nll=float(getattr(args, 'lambda_nll', 0.0)),
        use_nb_head=bool(int(getattr(args, 'use_nb_head', 0)) == 1),
        use_zinb=bool(int(getattr(args, 'use_zinb', 0)) == 1),
        y_transform=getattr(args, 'y_transform', 'none'),
        nonzero_weight=float(getattr(args, 'nonzero_weight', 4.0)),
        horizon_gamma=float(getattr(args, 'horizon_gamma', 1.5)),
        huber_delta=float(getattr(args, 'huber_delta', 1.0)),
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_val_rrse = float('inf')
    no_improve = 0
    def _arg_source(name: str) -> str:
        try:
            cli = set(getattr(args, '_cli_keys', []) or [])
            met = set(getattr(args, '_metrics_keys', []) or [])
            if name in cli:
                return 'cli'
            if name in met:
                return 'metrics'
        except Exception:
            pass
        return 'default'

    patience = int(getattr(args, 'early_stop_patience', 0))
    E = int(getattr(args, 'epochs', 50))
    B = int(getattr(args, 'batch_size', 64))
    q_print = q_list if (has_quantiles and q_weight) else []
    g_on = bool(int(getattr(args, "use_gauss", 0)) == 1)
    lam_nll = float(getattr(args, "lambda_nll", 0.0))
    print(
        f'[trainer] epochs={E}({_arg_source("epochs")}) '
        f'batch_size={B}({_arg_source("batch_size")}) '
        f'patience={patience}({_arg_source("early_stop_patience")}) '
        f'quantiles={q_print}({_arg_source("quantiles")}) '
        f'gauss={g_on}({_arg_source("use_gauss")}) '
        f'lambda_nll={lam_nll}({_arg_source("lambda_nll")})'
    )

    for ep in range(1, E + 1):
        model.train()
        train_losses = []
        for batch in Data.get_batches(Data.train[0], Data.train[1], B, shuffle=True):
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                Xb, Yb = batch[0], batch[1]
            else:
                continue
            try:
                Xb = to_model_layout_fn(
                    Xb,
                    int(getattr(model, 'seq_length', args.seq_in_len)),
                    debug=getattr(args, 'debug_layout', False),
                )
            except Exception as _e:
                print(f"[TrainerLayout] FATAL layout issue: {_e}")
                raise
            loss_item, _, _ = trainer.train(Xb, Yb)
            train_losses.append(loss_item)

        train_loss = float(sum(train_losses) / max(1, len(train_losses)))
        if (Data.valid[0] is not None and Data.valid[1] is not None):
            # Avoid per-epoch plot generation; only plot once after training.
            do_plots = False
            val_metrics = evaluate_fn(
                Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                B, do_plots, mc_runs=getattr(args, 'mc_runs', 30), kind='Validation'
            )
            if isinstance(val_metrics, (list, tuple)) and len(val_metrics) >= 2:
                val_rrse = val_metrics[0]
                val_rae = val_metrics[1]
            elif isinstance(val_metrics, (list, tuple)) and len(val_metrics) >= 1:
                val_rrse = val_metrics[0]
                val_rae = float('nan')
            else:
                val_rrse = float('inf')
                val_rae = float('nan')
        else:
            val_rrse = float('inf')
            val_rae = float('nan')

        if getattr(args, 'runlog', False):
            jlog_fn("trainer_epoch", epoch=ep, train_mae=train_loss, val_rrse=val_rrse, val_rae=val_rae)

        print(f'[trainer][epoch {ep:03d}] loss={train_loss:.6f} val_rse={val_rrse:.6f} val_rae={val_rae:.6f}')
        improved = val_rrse < best_val_rrse
        if improved:
            best_val_rrse = val_rrse
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(f'[trainer] Early stop at epoch {ep} (patience={patience}).')
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
        try:
            os.makedirs(os.path.dirname(args.save), exist_ok=True)
            # Persist enough hyperparams to rebuild the exact model on reload.
            hparams = [
                int(getattr(args, 'gcn_depth', 2)),
                float(getattr(args, 'lr', 0.001)),
                int(getattr(args, 'conv_channels', 32)),
                int(getattr(args, 'residual_channels', 32)),
                int(getattr(args, 'skip_channels', 64)),
                int(getattr(args, 'end_channels', 128)),
                int(getattr(args, 'subgraph_size', 20)),
                float(getattr(args, 'dropout', 0.3)),
                int(getattr(args, 'dilation_exponential', 1)),
                int(getattr(args, 'node_dim', 40)),
                float(getattr(args, 'propalpha', 0.05)),
                float(getattr(args, 'tanhalpha', 3)),
                int(getattr(args, 'layers', 3)),
                -1,
            ]
            torch.save({'state_dict': best_state, 'hparams': hparams}, args.save)
        except Exception as _e:
            print(f'[trainer][warn] checkpoint save failed: {_e}')

    # Optional: final evaluation with plots (uses same helpers as main)
    try:
        do_plots = not bool(getattr(args, 'no_plots', False))
        if do_plots and evaluate_fn is not None:
            if Data.valid[0] is not None and Data.valid[1] is not None:
                evaluate_fn(
                    Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                    B, do_plots, mc_runs=getattr(args, 'mc_runs', 30), kind='Validation'
                )
            if evaluate_sliding_window_fn is not None and getattr(Data, 'test_window', None) is not None:
                evaluate_sliding_window_fn(
                    Data, Data.test_window, model, evaluateL2, evaluateL1,
                    int(args.seq_in_len), do_plots, mc_runs=getattr(args, 'mc_runs', 30)
                )
    except Exception as _e:
        print(f"[trainer][warn] final plotting skipped: {_e}")

    return model