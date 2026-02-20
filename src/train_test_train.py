import torch
from torch.amp.autocast_mode import autocast

from src.losses import weighted_huber_horizon_loss
from src.util import last_level_baseline_expand, maybe_inv_scale, to_model_layout


def train_impl(
    data,
    X,
    Y,
    model,
    criterion,
    optim,
    batch_size,
    data_scaler,
    alpha: float = 1.0,
    beta: float = 0.4,
    gamma: float = 0.8,
    clip: float = 10.0,
    mae_weight: float = 0.2,
    grad_scaler: "torch.amp.grad_scaler.GradScaler|None" = None,
    scheduler=None,
    *,
    args,
    device,
):
    """
    Composite loss:
      total = alpha * L1(z-space) + beta * (1 - corr) + gamma * sMAPE(original)
    """
    model.train()
    total_loss = 0.0
    n_elems = 0
    it = 0
    out = None
    loss = None

    if not bool(getattr(args, "amp", False)):
        grad_scaler = None

    try:
        use_amp = (grad_scaler is not None) and (
            not hasattr(grad_scaler, "is_enabled") or grad_scaler.is_enabled()
        )
    except Exception:
        use_amp = grad_scaler is not None

    movement_weight = float(getattr(args, "movement_loss_weight", 0.0))

    def _movement_loss(y_true_o: torch.Tensor, y_pred_o: torch.Tensor) -> torch.Tensor | None:
        if y_true_o.dim() < 3 or y_pred_o.dim() < 3:
            return None
        if y_true_o.shape[1] < 2 or y_pred_o.shape[1] < 2:
            return None
        dy = y_true_o[:, 1:] - y_true_o[:, :-1]
        dp = y_pred_o[:, 1:] - y_pred_o[:, :-1]
        return torch.mean(torch.abs(dy - dp))

    grad_accum = max(1, int(getattr(args, "grad_accum_steps", 1)))
    accum_step = 0
    for batch in data.get_batches(X, Y, batch_size, True, return_indices=True):
        if len(batch) == 3:
            Xb_raw, Yb_raw, idxs = batch
        else:
            Xb_raw, Yb_raw = batch
            idxs = None
        if accum_step == 0:
            if hasattr(optim, "zero_grad"):
                optim.zero_grad()
            else:
                optim.optimizer.zero_grad()
        Xb_tensor = torch.as_tensor(Xb_raw)
        if Xb_tensor.dim() == 3:
            Xb_tensor = Xb_tensor.unsqueeze(-1)
        try:
            expected_len = int(getattr(model, "seq_length", Xb_tensor.shape[1]))
            Xb = to_model_layout(Xb_tensor, expected_len, debug=getattr(args, "debug_layout", False))
        except Exception as _e:
            print(f"[LayoutTrain] Failed to reconcile layout: {_e}")
            raise
        Xb = Xb.to(device, dtype=torch.float)
        Yb = Yb_raw.to(device)

        if use_amp:
            with autocast(device_type="cuda", enabled=use_amp):
                out_raw = model(Xb)
                q_pred = None
                taus_tensor = None
                if isinstance(out_raw, dict):
                    out = out_raw["mean"]
                    q_pred = out_raw.get("quantiles", None)
                    taus_tensor = out_raw.get("taus", None)
                else:
                    out = out_raw
                if out.dim() == 4:
                    out = out.squeeze(3)

            if getattr(args, "use_weighted_horizon_loss", False):
                out_o = maybe_inv_scale(out, data_scaler)
                y_o = maybe_inv_scale(Yb, data_scaler)

                if getattr(args, "residual_head", False):
                    baseline_z = last_level_baseline_expand(Xb_raw, out.shape[1]).detach()
                    baseline_o = maybe_inv_scale(baseline_z, data_scaler)
                    out_o = out_o + baseline_o

                loss = weighted_huber_horizon_loss(
                    out_o,
                    y_o,
                    delta=float(getattr(args, "huber_delta", 1.0)),
                    nonzero_weight=float(getattr(args, "nonzero_weight", 4.0)),
                    horizon_gamma=float(getattr(args, "horizon_gamma", 1.5)),
                    node_weights=(
                        torch.from_numpy(getattr(data, "node_weights")).to(
                            out_o.device, dtype=out_o.dtype
                        )
                        if getattr(data, "node_weights", None) is not None
                        else None
                    ),
                )
                if movement_weight > 0.0:
                    mv = _movement_loss(y_o, out_o)
                    if mv is not None:
                        loss = loss + movement_weight * mv
            else:
                out_o = maybe_inv_scale(out, data_scaler)
                y_o = maybe_inv_scale(Yb, data_scaler)

                if getattr(args, "residual_head", False):
                    baseline_z = last_level_baseline_expand(Xb_raw, out.shape[1]).detach()
                    baseline_o = maybe_inv_scale(baseline_z, data_scaler)
                    out_o = out_o + baseline_o

                smape_denom = torch.clamp(torch.abs(y_o) + torch.abs(out_o), min=1e-6)
                s_mape = torch.mean(torch.abs(y_o - out_o) / smape_denom)

                l1_loss = criterion(out, Yb)

                out_centered = out - out.mean(dim=[1, 2], keepdim=True)
                yb_centered = Yb - Yb.mean(dim=[1, 2], keepdim=True)
                corr_num = torch.mean(out_centered * yb_centered, dim=[1, 2])
                corr_den = torch.sqrt(
                    torch.mean(out_centered**2, dim=[1, 2])
                    * torch.mean(yb_centered**2, dim=[1, 2])
                )
                corr_loss = 1.0 - torch.mean(corr_num / torch.clamp(corr_den, min=1e-6))

                loss = alpha * l1_loss + beta * corr_loss + gamma * s_mape
                if mae_weight > 0:
                    loss = loss + mae_weight * torch.mean(torch.abs(y_o - out_o))
                if movement_weight > 0.0:
                    mv = _movement_loss(y_o, out_o)
                    if mv is not None:
                        loss = loss + movement_weight * mv
            if loss is not None:
                if grad_scaler is not None:
                    grad_scaler.scale(loss / grad_accum).backward()
                else:
                    (loss / grad_accum).backward()
        else:
            out_raw = model(Xb)
            q_pred = None
            taus_tensor = None
            if isinstance(out_raw, dict):
                out = out_raw["mean"]
                q_pred = out_raw.get("quantiles", None)
                taus_tensor = out_raw.get("taus", None)
            else:
                out = out_raw
            if out.dim() == 4:
                out = out.squeeze(3)

            out_o = maybe_inv_scale(out, data_scaler)
            y_o = maybe_inv_scale(Yb, data_scaler)

            if getattr(args, "residual_head", False):
                baseline_z = last_level_baseline_expand(Xb_raw, out.shape[1]).detach()
                baseline_o = maybe_inv_scale(baseline_z, data_scaler)
                out_o = out_o + baseline_o

            smape_denom = torch.clamp(torch.abs(y_o) + torch.abs(out_o), min=1e-6)
            s_mape = torch.mean(torch.abs(y_o - out_o) / smape_denom)
            l1_loss = criterion(out, Yb)
            out_centered = out - out.mean(dim=[1, 2], keepdim=True)
            yb_centered = Yb - Yb.mean(dim=[1, 2], keepdim=True)
            corr_num = torch.mean(out_centered * yb_centered, dim=[1, 2])
            corr_den = torch.sqrt(
                torch.mean(out_centered**2, dim=[1, 2]) * torch.mean(yb_centered**2, dim=[1, 2])
            )
            corr_loss = 1.0 - torch.mean(corr_num / torch.clamp(corr_den, min=1e-6))
            loss = alpha * l1_loss + beta * corr_loss + gamma * s_mape
            if mae_weight > 0:
                loss = loss + mae_weight * torch.mean(torch.abs(y_o - out_o))
            if movement_weight > 0.0:
                mv = _movement_loss(y_o, out_o)
                if mv is not None:
                    loss = loss + movement_weight * mv
            (loss / grad_accum).backward()

        if loss is not None:
            if (accum_step + 1) % grad_accum == 0:
                if clip is not None and clip > 0:
                    if grad_scaler is not None:
                        grad_scaler.unscale_(optim.optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                if grad_scaler is not None:
                    grad_scaler.step(optim.optimizer)
                    grad_scaler.update()
                else:
                    optim.step()
                if scheduler is not None and scheduler.__class__.__name__ == "OneCycleLR":
                    scheduler.step()
                accum_step = 0
            else:
                accum_step += 1

        if loss is not None:
            total_loss += float(loss.item()) * out.numel()
        n_elems += out.numel()

        if it % 20 == 0 and out is not None and loss is not None:
            denom = max(1, out.size(0) * out.size(1) * data.m)
            print(f"iter:{it:3d} | loss: {float(loss.item())/denom:.6f}")
        it += 1

    if "grad_accum" in locals() and "accum_step" in locals():
        if accum_step != 0:
            try:
                clip = float(clip)
            except Exception:
                pass
            if clip is not None and clip > 0:
                if grad_scaler is not None:
                    grad_scaler.unscale_(optim.optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            if grad_scaler is not None:
                grad_scaler.step(optim.optimizer)
                grad_scaler.update()
            else:
                optim.step()
            if scheduler is not None and scheduler.__class__.__name__ == "OneCycleLR":
                scheduler.step()
    return total_loss / max(1.0, n_elems)
