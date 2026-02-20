import math
import os
import sys

import numpy as np
import torch

from src.train_test_metrics import _compute_metrics, _safe_corr_np
from src.train_test_plotting import _offset_year_month, plot_predicted_actual
from src.util import (
    DataLoaderS,
    MCDropoutContext,
    jlog,
    last_level_baseline_expand,
    maybe_inv_scale,
    to_model_layout,
    unwrap_model_output,
)
from torch.amp.autocast_mode import autocast


def evaluate_sliding_window_impl(
    data,
    test_window,
    model,
    evaluateL2,
    evaluateL1,
    n_input,
    is_plot,
    *,
    args,
    device,
    state,
    mc_runs=None,
    save_metrics_fn=None,
):
    """Sliding-window testing with MC Dropout and correct 95% CI in original units."""
    z = float(state.get("conf_z", 1.96))

    model.eval()

    try:
        total_len = test_window.shape[0]
    except Exception:
        total_len = len(test_window)
    if total_len < n_input + data.out_len:
        batch_sz = int(getattr(args, "batch_size", 64)) if "args" in globals() else 64
        return evaluate_impl(
            data,
            data.test[0],
            data.test[1],
            model,
            evaluateL2,
            evaluateL1,
            batch_sz,
            False,
            args=args,
            device=device,
            state=state,
            mc_runs=mc_runs,
            save_metrics_fn=save_metrics_fn,
        )

    try:
        model_dev = next(model.parameters()).device
    except StopIteration:
        model_dev = torch.device("cpu")
    x_input = torch.as_tensor(test_window[0:n_input, ...], dtype=torch.float32, device=model_dev)

    preds_o_list, trues_o_list, conf_o_list = [], [], []

    if mc_runs is None:
        mc_runs = int(getattr(args, "mc_runs", 30))
    with torch.no_grad():
        for i in range(n_input, test_window.shape[0], data.out_len):
            X_raw = x_input.unsqueeze(0)
            if X_raw.dim() == 4:
                pass
            elif X_raw.dim() == 3:
                X_raw = X_raw.unsqueeze(-1)
            try:
                exp_len = int(getattr(model, "seq_length", X_raw.shape[1]))
                X = to_model_layout(X_raw, exp_len, debug=getattr(args, "debug_layout", False)).to(device)
            except Exception as _e:
                print(f"[LayoutEvalSlide] layout error: {_e}")
                raise
            y_true_z = torch.as_tensor(
                (
                    test_window[i : i + data.out_len, :, 0]
                    if test_window.ndim == 3
                    else test_window[i : i + data.out_len, :]
                ),
                dtype=torch.float32,
                device=device,
            )

            analysis_logger = state.get("analysis_logger")
            with MCDropoutContext(model):
                amp_on = bool(getattr(args, "amp", False)) and device.type == "cuda"
                if args.vectorized_mc:
                    B, C, N, T = X.shape
                    est_elems = mc_runs * B * C * N * T
                    max_elems = int(getattr(args, "mc_vec_max_elems", 120_000_000))
                    use_vec = est_elems <= max_elems
                    outs = None
                    if use_vec:
                        try:
                            Xrep = X.repeat(mc_runs, 1, 1, 1)
                            with autocast(device_type="cuda", enabled=amp_on):
                                raw_out = model(Xrep)
                            raw_out = unwrap_model_output(raw_out)
                            if raw_out.dim() == 4:
                                raw_out = raw_out.squeeze(3)
                            outs = raw_out[:, -1, ...]
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                torch.cuda.empty_cache()
                                print(
                                    f"[MC Fallback-SW] OOM vectorized MC (est_elems={est_elems:,}); using looped mode."
                                )
                                use_vec = False
                            else:
                                raise
                    if not use_vec or outs is None:
                        out_list = []
                        for _ in range(mc_runs):
                            with autocast(device_type="cuda", enabled=amp_on):
                                o = model(X)
                            o = unwrap_model_output(o)
                            if o.dim() == 4:
                                o = o.squeeze(3)
                            out_list.append(o[-1])
                        outs = torch.stack(out_list, dim=0)
                else:
                    out_list = []
                    for _ in range(mc_runs):
                        with autocast(device_type="cuda", enabled=amp_on):
                            o = model(X)
                        o = unwrap_model_output(o)
                        if o.dim() == 4:
                            o = o.squeeze(3)
                        out_list.append(o[-1])
                    outs = torch.stack(out_list, dim=0)

            if analysis_logger:
                try:
                    log_record = {
                        "type": "sliding_window_diagnostics",
                        "slide_start_index": i,
                        "input_mean": float(x_input.mean()),
                        "input_std": float(x_input.std()),
                        "true_target_mean_z": float(y_true_z.mean()),
                        "true_target_std_z": float(y_true_z.std()),
                        "pred_mean_z": float(outs.mean()),
                        "pred_std_z": float(outs.std()),
                    }
                    analysis_logger.log(log_record)
                except Exception as e:
                    print(
                        f"[AnalysisLogger Warning] Failed during sliding window logging: {e}",
                        file=sys.stderr,
                    )
            try:
                if getattr(args, "persist_mc", ""):
                    slides_arg = str(getattr(args, "persist_mc_slides", "")).strip()
                    persist_all = slides_arg == ""
                    persist_set = set()
                    if not persist_all:
                        for token in slides_arg.split(","):
                            try:
                                persist_set.add(int(token.strip()))
                            except Exception:
                                pass
                    if persist_all or i in persist_set:
                        try:
                            os.makedirs(os.path.dirname(args.persist_mc) or ".", exist_ok=True)
                            fn = f"{args.persist_mc}_slide_{i}.npz"
                            np.savez_compressed(fn, mc_samples=outs.detach().cpu().numpy())
                            if analysis_logger:
                                analysis_logger.log(
                                    {
                                        "type": "persist_mc_saved",
                                        "slide_start_index": i,
                                        "path": fn,
                                    }
                                )
                        except Exception as _e:
                            print(
                                f"[persist_mc] failed to write MC samples for slide {i}: {_e}",
                                file=sys.stderr,
                            )
            except Exception:
                pass

            if getattr(args, "nan_debug", False):
                if torch.isnan(outs).any() or torch.isinf(outs).any():
                    bad = torch.isnan(outs) | torch.isinf(outs)
                    idx_flat = bad.nonzero(as_tuple=False)[0].tolist() if bad.any() else []
                    print(
                        f"[NaNDebug][SW] Detected NaN/Inf in MC samples at slide {i}, first index={idx_flat}"
                    )
            mean_z = outs.mean(dim=0)
            std_z = outs.std(dim=0) + 1e-8
            if not torch.isfinite(mean_z).all() or not torch.isfinite(std_z).all():
                try:
                    bad_mean = int((~torch.isfinite(mean_z)).sum().item())
                    bad_std = int((~torch.isfinite(std_z)).sum().item())
                    if getattr(args, "nan_debug", False):
                        print(
                            f"[NanGuard] slide={i} mean_z bad={bad_mean} std_z bad={bad_std} (sanitizing)"
                        )
                except Exception:
                    pass
                mean_z = torch.nan_to_num(mean_z, nan=0.0, posinf=0.0, neginf=0.0)
                std_z = torch.nan_to_num(std_z, nan=0.0, posinf=0.0, neginf=0.0)

            half_z = z * std_z

            out_eff = y_true_z.shape[0]
            if mean_z.shape[0] > out_eff:
                mean_z = mean_z[:out_eff]
                std_z = std_z[:out_eff]
                half_z = half_z[:out_eff]

            w_idx = i - n_input
            if getattr(data, "rolling", False) and hasattr(data, "per_window_mu") and "test" in data.per_window_mu:
                if isinstance(w_idx, torch.Tensor):
                    w_idx = int(w_idx.item())
                if w_idx < 0 or w_idx >= len(data.per_window_mu["test"]):
                    mean_o = data.inv_transform_like(mean_z)
                    true_o = data.inv_transform_like(y_true_z)
                    if getattr(data, "use_log1p", False):
                        lower_o = data.inv_transform_like(mean_z - half_z)
                        upper_o = data.inv_transform_like(mean_z + half_z)
                        conf_o = 0.5 * (upper_o - lower_o)
                    else:
                        conf_o = half_z * data.std_expand_like(half_z)
                else:
                    mu = data.per_window_mu["test"][w_idx]
                    std = data.per_window_std["test"][w_idx]
                    mu_t = (
                        torch.as_tensor(mu, dtype=mean_z.dtype, device=mean_z.device)
                        .unsqueeze(0)
                        .expand_as(mean_z)
                    )
                    std_t = (
                        torch.as_tensor(std, dtype=mean_z.dtype, device=mean_z.device)
                        .unsqueeze(0)
                        .expand_as(mean_z)
                    )
                    mean_o = mean_z * std_t + mu_t
                    true_o = y_true_z * std_t + mu_t
                    if getattr(data, "use_log1p", False):
                        mean_o = torch.expm1(mean_o)
                        true_o = torch.expm1(true_o)
                        lower_o = torch.expm1((mean_z - half_z) * std_t + mu_t)
                        upper_o = torch.expm1((mean_z + half_z) * std_t + mu_t)
                        conf_o = 0.5 * (upper_o - lower_o)
                    else:
                        conf_o = half_z * std_t
            else:
                if getattr(data, "use_log1p", False):
                    lower_o = data.inv_transform_like(mean_z - half_z)
                    upper_o = data.inv_transform_like(mean_z + half_z)
                    mean_o = data.inv_transform_like(mean_z)
                    true_o = data.inv_transform_like(y_true_z)
                    conf_o = 0.5 * (upper_o - lower_o)
                else:
                    mean_o = data.inv_transform_like(mean_z)
                    conf_o = half_z * data.std_expand_like(half_z)
                    true_o = data.inv_transform_like(y_true_z)
                    conf_o = half_z * data.std_expand_like(half_z)

            level_z_for_roll = mean_z
            if getattr(args, "residual_head", False):
                try:
                    baseline_z = last_level_baseline_expand(X, mean_z.shape[0])
                    level_z_for_roll = mean_z + baseline_z
                    if (
                        getattr(data, "rolling", False)
                        and "test" in getattr(data, "per_window_mu", {})
                        and 0 <= w_idx < len(data.per_window_mu["test"])
                    ):
                        mu = data.per_window_mu["test"][w_idx]
                        std = data.per_window_std["test"][w_idx]
                        mu_t = (
                            torch.as_tensor(mu, dtype=baseline_z.dtype, device=baseline_z.device)
                            .unsqueeze(0)
                            .expand_as(baseline_z)
                        )
                        std_t = (
                            torch.as_tensor(std, dtype=baseline_z.dtype, device=baseline_z.device)
                            .unsqueeze(0)
                            .expand_as(baseline_z)
                        )
                        base_o = baseline_z * std_t + mu_t
                        if getattr(data, "use_log1p", False):
                            base_o = torch.expm1(base_o)
                    else:
                        base_o = data.inv_transform_like(baseline_z)
                    base_o = base_o.squeeze(0).type_as(mean_o)
                    mean_o = mean_o + base_o
                except Exception:
                    pass

            level_z_for_roll = level_z_for_roll.detach()
            if x_input.ndim == 2:
                if data.P <= data.out_len:
                    take = min(data.P, level_z_for_roll.shape[0])
                    x_input = level_z_for_roll[-take:].clone()
                else:
                    keep = data.P - level_z_for_roll.shape[0]
                    if keep > 0:
                        tgt_dev = level_z_for_roll.device
                        x_input = torch.cat(
                            [x_input[-keep:, :].clone().to(tgt_dev), level_z_for_roll.clone()],
                            dim=0,
                        )
                    else:
                        x_input = level_z_for_roll.clone()
            else:
                L_out = level_z_for_roll.shape[0]
                level_seq = level_z_for_roll
                if data.in_dim > 1:
                    prev_last = x_input[-1, :, 0].to(level_seq.device)
                    m0 = (level_seq[0] - prev_last).unsqueeze(0).unsqueeze(-1)
                    md = (
                        torch.diff(level_seq, dim=0).unsqueeze(-1)
                        if L_out > 1
                        else torch.zeros((0, level_seq.shape[1], 1), device=level_seq.device)
                    )
                    mov_tail = torch.cat([m0, md], dim=0)
                    new_tail = torch.cat([level_seq.unsqueeze(-1), mov_tail], dim=-1)
                else:
                    new_tail = level_seq.unsqueeze(-1)
                if data.P <= L_out:
                    take = min(data.P, L_out)
                    x_input = new_tail[-take:].clone()
                else:
                    keep = data.P - L_out
                    if keep > 0:
                        tgt_dev = new_tail.device
                        x_input = torch.cat(
                            [x_input[-keep:, :].clone().to(tgt_dev), new_tail.clone()],
                            dim=0,
                        )
                    else:
                        x_input = new_tail.clone()

            if (
                (not torch.isfinite(mean_o).all())
                or (not torch.isfinite(true_o).all())
                or (not torch.isfinite(conf_o).all())
            ):
                if getattr(args, "nan_debug", False):
                    print(f"[NanGuard] slide={i} sanitizing non-finite outputs for metrics")
                mean_o = torch.nan_to_num(mean_o, nan=0.0, posinf=0.0, neginf=0.0)
                true_o = torch.nan_to_num(true_o, nan=0.0, posinf=0.0, neginf=0.0)
                conf_o = torch.nan_to_num(conf_o, nan=0.0, posinf=0.0, neginf=0.0)

            preds_o_list.append(mean_o.detach().cpu())
            trues_o_list.append(true_o.detach().cpu())
            conf_o_list.append(conf_o.detach().cpu())

    pred_o = torch.cat(preds_o_list, dim=0)
    true_o = torch.cat(trues_o_list, dim=0)
    conf_o = torch.cat(conf_o_list, dim=0)
    try:
        calib_ab = state.get("calib_ab")
        if getattr(args, "calibration", "none") in ("test", "both") and isinstance(calib_ab, tuple) and len(calib_ab) == 2:
            a_t, b_t = calib_ab
            a = a_t.to(pred_o.device)
            b = b_t.to(pred_o.device)
            pred_o = a.unsqueeze(0) * pred_o + b.unsqueeze(0)
    except Exception as _e:
        jlog("warn_calibration_linear_test", error=str(_e)[:160])
    try:
        p = pred_o.numpy()
        y = true_o.numpy()
        _m = _compute_metrics(y, p)
        rrse = float(_m.get("RSE", float("nan")))
        rae = float(_m.get("RAE", float("nan")))
        sm = float(_m.get("sMAPE", float("nan")))
        correlation = _safe_corr_np(p, y, eps=1e-12)
    except Exception:
        rrse = float("nan")
        rae = float("nan")
        correlation = float("nan")
        sm = float("nan")
    if is_plot:
        is_chrono = bool(getattr(args, "use_chronological_split", False)) or bool(
            getattr(args, "chronological_split", False)
        )
        steps_per_year = int(getattr(args, "steps_per_year", 12))
        base_year_test = None
        base_month_test = 1

        if is_chrono:
            base_year_test = args.valid_end_year + 1
            base_month_test = 1
        else:
            data_start_year = getattr(data, "start_year", None)
            data_start_month = getattr(data, "start_month", None)

            if data_start_year is not None and data_start_month is not None:
                idx_set = getattr(data, "_test_idx_set", None)

                if idx_set and len(idx_set) > 0:
                    first_window_idx = int(idx_set[0])
                    total_months_from_start = first_window_idx
                    total_months = (data_start_month - 1) + total_months_from_start
                    base_year_test = data_start_year + (total_months // steps_per_year)
                    base_month_test = (total_months % steps_per_year) + 1
                    print(
                        f"[DateMapping] Testing split starts at {base_year_test:04d}-{base_month_test:02d} (window index {first_window_idx})"
                    )

        if base_year_test is not None and is_chrono:
            try:
                idx_set = getattr(data, "_test_idx_set", None)
                if idx_set:
                    base_year_test, base_month_test = _offset_year_month(
                        base_year_test,
                        base_month_test,
                        int(idx_set[0]),
                        steps_per_year,
                    )
            except Exception:
                pass

        n_cols = getattr(data, "m", pred_o.shape[1])

        print(f"[Plot] Testing: plotting {n_cols} nodes")

        for col in range(n_cols):
            node_name = str(DataLoaderS.col[col])
            plot_node_name = DataLoaderS.get_plot_node_name(node_name)
            pred_col = pred_o[:, col]
            true_col = true_o[:, col]
            ci_col = conf_o[:, col] if conf_o is not None else None
            plot_predicted_actual(
                pred_col,
                true_col,
                plot_node_name,
                "Testing",
                ci=ci_col,
                base_year=getattr(data, "start_year", None),
                steps_per_year=getattr(data, "steps_per_year", None),
                base_month=getattr(data, "base_month", 1),
            )
    return (
        float(rrse if rrse is not None else 0.0),
        float(rae if rae is not None else 0.0),
        correlation,
        float(sm if sm is not None else 0.0),
    )


def evaluate_impl(
    data,
    X,
    Y,
    model,
    evaluateL2,
    evaluateL1,
    batch_size,
    is_plot,
    *,
    args,
    device,
    state,
    mc_runs=None,
    kind="Validation",
    save_metrics_fn=None,
):
    model.eval()
    eps = 1e-12
    z = float(state.get("conf_z", 1.96))
    predict = []
    target = []
    conf95 = []
    pred_full = torch.empty(0)
    true_full = torch.empty(0)
    labels = []
    r_collect = None
    if X is None or (hasattr(X, "shape") and getattr(X, "shape")[0] == 0):
        print("[evaluate] No windows provided; skipping metrics for", kind)
        return float("nan"), float("nan"), float("nan"), float("nan")
    if mc_runs is None:
        mc_runs = int(getattr(args, "mc_runs", 50))
    with torch.no_grad():
        r_collect = [] if (kind == "Validation" and getattr(args, "conf_calibrate", False)) else None
        for b_idx, batch in enumerate(data.get_batches(X, Y, batch_size, False, return_indices=True)):
            if len(batch) == 3:
                Xb_raw, Yb_raw, idxs = batch
            else:
                Xb_raw, Yb_raw = batch
                idxs = None
            Xb_tensor = torch.as_tensor(Xb_raw)
            if Xb_tensor.dim() == 3:
                Xb_tensor = Xb_tensor.unsqueeze(-1)
            try:
                exp_len = int(getattr(model, "seq_length", Xb_tensor.shape[1]))
                Xb = to_model_layout(Xb_tensor, exp_len, debug=getattr(args, "debug_layout", False)).to(
                    device, dtype=torch.float
                )
            except Exception as _e:
                print(f"[LayoutEval] layout error: {_e}")
                raise
            Yb = Yb_raw.to(device)
            with MCDropoutContext(model):
                amp_on = bool(getattr(args, "amp", False)) and device.type == "cuda"
                if args.vectorized_mc:
                    B, C, N, T = Xb.shape
                    est_elems = mc_runs * B * C * N * T
                    max_elems = int(getattr(args, "mc_vec_max_elems", 120_000_000))
                    use_vec = est_elems <= max_elems
                    outs = None
                    if use_vec:
                        try:
                            Xrep = Xb.repeat(mc_runs, 1, 1, 1)
                            with autocast(device_type="cuda", enabled=amp_on):
                                raw_out = model(Xrep)
                            raw_out = unwrap_model_output(raw_out)
                            if raw_out.dim() == 4:
                                raw_out = raw_out[:, :, :, -1]
                            outs = raw_out.view(mc_runs, B, *raw_out.shape[1:])
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                torch.cuda.empty_cache()
                                use_vec = False
                                print(
                                    f"[MC Fallback] OOM during vectorized MC (est_elems={est_elems:,}); falling back to looped mode."
                                )
                            else:
                                raise
                    if not use_vec or outs is None:
                        out_list = []
                        for _ in range(mc_runs):
                            with autocast(device_type="cuda", enabled=amp_on):
                                o = model(Xb)
                            o = unwrap_model_output(o)
                            if o.dim() == 4:
                                o = o[:, :, :, -1]
                            out_list.append(o)
                        outs = torch.stack(out_list, dim=0)
                else:
                    out_list = []
                    for _ in range(mc_runs):
                        with autocast(device_type="cuda", enabled=amp_on):
                            o = model(Xb)
                        o = unwrap_model_output(o)
                        if o.dim() == 4:
                            o = o[:, :, :, -1]
                        out_list.append(o)
                    outs = torch.stack(out_list, dim=0)
            if getattr(args, "nan_debug", False):
                if torch.isnan(outs).any() or torch.isinf(outs).any():
                    bad = torch.isnan(outs) | torch.isinf(outs)
                    loc = bad.nonzero(as_tuple=False)[0].tolist() if bad.any() else []
                    print(
                        f"[NaNDebug][{kind}] NaN/Inf detected in MC outs batch={b_idx} first_index={loc}"
                    )
            mean_z = outs.mean(dim=0)
            std_z = outs.std(dim=0) + 1e-8

            debug_eval = bool(getattr(args, "nan_debug", False)) or os.environ.get("BMTGNN_DEBUG_EVAL", "0") == "1"
            if debug_eval and b_idx == 0 and kind == "Validation":
                outs_finite = torch.isfinite(outs).sum().item()
                mean_z_finite = torch.isfinite(mean_z).sum().item()
                print(
                    f"[MCDiag] batch=0 outs: size={outs.numel()} finite={outs_finite} mean_z: size={mean_z.numel()} finite={mean_z_finite}"
                )

            half_z = z * std_z
            if r_collect is not None:
                with torch.no_grad():
                    r = torch.abs(Yb - mean_z) / torch.clamp(std_z, min=1e-8)
                    r_collect.append(r.detach().flatten().cpu())

            analysis_logger = state.get("analysis_logger")
            if analysis_logger and kind == "Validation":
                try:
                    norm_mode = "rolling" if getattr(data, "rolling", False) else "global"
                    try:
                        if getattr(data, "rolling", False) and hasattr(data, "per_window_mu") and idxs is not None:
                            split = "valid"
                            mu = data.per_window_mu[split][idxs]
                            std = data.per_window_std[split][idxs]
                            pred_o = data.inv_transform_with_stats(mean_z.cpu(), mu, std)
                            true_o = data.inv_transform_with_stats(Yb.cpu(), mu, std)
                        else:
                            pred_o = data.inv_transform_like(mean_z.cpu())
                            true_o = data.inv_transform_like(Yb.cpu())
                        pred_orig_mean = float(pred_o.mean())
                        true_orig_mean = float(true_o.mean())
                    except Exception:
                        pred_orig_mean = None
                        true_orig_mean = None

                    analysis_logger.log(
                        {
                            "type": "validation_batch_diagnostics",
                            "batch_index": b_idx,
                            "input_shape": list(Xb.shape),
                            "normalize_mode": norm_mode,
                            "pred_mean_z": float(mean_z.mean()),
                            "pred_std_z": float(mean_z.std()),
                            "true_mean_z": float(Yb.mean()),
                            "true_std_z": float(Yb.std()),
                            "pred_mean_original": pred_orig_mean,
                            "true_mean_original": true_orig_mean,
                        }
                    )
                except Exception as e:
                    print(
                        f"[AnalysisLogger Warning] Failed during validation batch logging: {e}",
                        file=sys.stderr,
                    )

            if getattr(data, "use_log1p", False):
                if getattr(data, "rolling", False) and idxs is not None:
                    lower_o = data.inv_transform_with_stats(
                        mean_z - half_z,
                        data.per_window_mu["valid"][idxs],
                        data.per_window_std["valid"][idxs],
                    )
                    upper_o = data.inv_transform_with_stats(
                        mean_z + half_z,
                        data.per_window_mu["valid"][idxs],
                        data.per_window_std["valid"][idxs],
                    )
                    mean_o = data.inv_transform_with_stats(
                        mean_z,
                        data.per_window_mu["valid"][idxs],
                        data.per_window_std["valid"][idxs],
                    )
                else:
                    lower_o = data.inv_transform_like(mean_z - half_z)
                    upper_o = data.inv_transform_like(mean_z + half_z)
                    mean_o = data.inv_transform_like(mean_z)
                conf_o = 0.5 * (upper_o - lower_o)
            else:
                if getattr(data, "rolling", False) and idxs is not None:
                    mean_o = data.inv_transform_with_stats(
                        mean_z,
                        data.per_window_mu["valid"][idxs],
                        data.per_window_std["valid"][idxs],
                    )
                    conf_o = half_z * data.std_expand_like(half_z, idx=idxs, split="valid")
                else:
                    mean_o = data.inv_transform_like(mean_z)
                    conf_o = half_z * data.std_expand_like(half_z)
            y_true_o = (
                data.inv_transform_like(Yb, idx=idxs, split="valid")
                if (getattr(data, "rolling", False) and idxs is not None)
                else data.inv_transform_like(Yb)
            )

            if debug_eval and b_idx == 0 and kind == "Validation":
                mean_o_finite = torch.isfinite(mean_o).sum().item()
                y_true_o_finite = torch.isfinite(y_true_o).sum().item()
                print(
                    f"[InvTransformDiag] batch=0 mean_o: size={mean_o.numel()} finite={mean_o_finite} y_true_o: size={y_true_o.numel()} finite={y_true_o_finite}"
                )
                if mean_o_finite == 0 or y_true_o_finite == 0:
                    print(
                        f"[InvTransformDiag] mean_z stats: mean={mean_z.mean():.4f} std={mean_z.std():.4f} min={mean_z.min():.4f} max={mean_z.max():.4f}"
                    )
                    print(
                        f"[InvTransformDiag] Yb stats: mean={Yb.mean():.4f} std={Yb.std():.4f} min={Yb.min():.4f} max={Yb.max():.4f}"
                    )

            if getattr(args, "residual_head", False):
                try:
                    baseline_z = last_level_baseline_expand(Xb_raw, Yb.shape[1]).detach()
                    if getattr(data, "rolling", False) and idxs is not None:
                        mu_valid = getattr(data, "per_window_mu", {}).get("valid")
                        std_valid = getattr(data, "per_window_std", {}).get("valid")
                        if mu_valid is not None and std_valid is not None:
                            baseline_o = data.inv_transform_with_stats(baseline_z, mu_valid[idxs], std_valid[idxs])
                        else:
                            baseline_o = maybe_inv_scale(baseline_z, data.scaler)
                    else:
                        baseline_o = maybe_inv_scale(baseline_z, data.scaler)
                    mean_o = mean_o + baseline_o
                except Exception:
                    pass

            if debug_eval and kind == "Validation":
                if not torch.isfinite(mean_o).all() or not torch.isfinite(y_true_o).all():
                    mean_bad = int((~torch.isfinite(mean_o)).sum().item())
                    true_bad = int((~torch.isfinite(y_true_o)).sum().item())
                    mean_finite = int(torch.isfinite(mean_o).sum().item())
                    true_finite = int(torch.isfinite(y_true_o).sum().item())
                    yb_bad = int((~torch.isfinite(Yb)).sum().item())
                    mz_bad = int((~torch.isfinite(mean_z)).sum().item())
                    print(
                        f"[InvTransformDiag] batch={b_idx} mean_o bad={mean_bad} finite={mean_finite} "
                        f"y_true_o bad={true_bad} finite={true_finite} Yb bad={yb_bad} mean_z bad={mz_bad}"
                    )

            if (
                (not torch.isfinite(mean_o).all())
                or (not torch.isfinite(y_true_o).all())
                or (not torch.isfinite(conf_o).all())
            ):
                if getattr(args, "nan_debug", False) or kind == "Validation":
                    print(f"[NanGuard] batch={b_idx} sanitizing non-finite outputs for metrics")
                mean_o = torch.nan_to_num(mean_o, nan=0.0, posinf=0.0, neginf=0.0)
                y_true_o = torch.nan_to_num(y_true_o, nan=0.0, posinf=0.0, neginf=0.0)
                conf_o = torch.nan_to_num(conf_o, nan=0.0, posinf=0.0, neginf=0.0)

            predict.append(mean_o.cpu())
            target.append(y_true_o.cpu())
            conf95.append(conf_o.cpu())
    predict = torch.cat(predict, dim=0)
    target = torch.cat(target, dim=0)
    conf95 = torch.cat(conf95, dim=0)

    conformal_q = state.get("conformal_q")
    try:
        if getattr(args, "conformal", False):
            split_lower = kind.lower()
            if split_lower in ("validation", "valid"):
                abs_res = torch.abs(target - predict)
                flat = abs_res.reshape(-1, abs_res.shape[-1])
                alpha = float(max(1e-4, min(0.5, getattr(args, "conf_alpha", 0.05))))
                qhat = torch.quantile(flat.to(torch.float32), 1.0 - alpha, dim=0)
                conformal_q = qhat.detach().cpu()
                state["conformal_q"] = conformal_q
            elif split_lower in ("testing", "test") and conformal_q is not None:
                qhat = conformal_q.to(conf95.device, conf95.dtype).view(1, 1, -1)
                conf95 = conf95 + qhat
    except Exception as _e:
        print(f"[conformal] warning: {_e}")

    if getattr(args, "robust_metrics", False):
        _EPS = 1e-12
        flat_std = torch.std(target.reshape(-1, target.shape[-1]), dim=0)
        var_mask = flat_std > 0
        if not var_mask.any():
            rrse = float("nan")
            rae = float("nan")
            correlation = float("nan")
            sm = float("nan")
        else:
            tgt_m = target[:, :, var_mask]
            pred_m = predict[:, :, var_mask]
            mean_all = tgt_m.mean(dim=(0, 1), keepdim=True)
            diff_r = tgt_m - mean_all
            if getattr(args, "weight_nodes_in_metrics", False):
                node_std = torch.std(tgt_m, dim=(0, 1)) + 1e-6
                med_std = torch.median(node_std)
                std_floor = torch.clamp(med_std * 0.1, min=1e-3)
                node_std = torch.clamp(node_std, min=std_floor)
                weights = (node_std.mean() / node_std).view(1, 1, -1)
                sq_err = ((tgt_m - pred_m) ** 2) * weights
                abs_err = torch.abs(tgt_m - pred_m) * weights
                diff_r_w = (diff_r**2) * weights
                rrse = math.sqrt(float(torch.sum(sq_err))) / (math.sqrt(float(torch.sum(diff_r_w))) + eps)
                rae = float((torch.sum(abs_err) / (torch.sum(torch.abs(diff_r) * weights) + eps)).item())
            else:
                rrse = math.sqrt(float(torch.sum((tgt_m - pred_m) ** 2))) / (
                    math.sqrt(float(torch.sum(diff_r**2))) + eps
                )
                rae = float((torch.sum(torch.abs(tgt_m - pred_m)) / (torch.sum(torch.abs(diff_r)) + eps)).item())
            p = pred_m.numpy()
            y = tgt_m.numpy()
            correlation = _safe_corr_np(p, y, eps=_EPS)
            sm_den = (np.abs(y) + np.abs(p)) + _EPS
            sm_num = np.abs(y - p)
            with np.errstate(divide="ignore", invalid="ignore"):
                sm_ratio = np.divide(sm_num, sm_den, out=np.zeros_like(sm_num), where=(sm_den > _EPS))
            sm = float(np.mean(sm_ratio))
    else:
        mean_all = torch.mean(target, dim=(0, 1))
        diff_r = target - mean_all.view(1, 1, -1)
        if getattr(args, "weight_nodes_in_metrics", False):
            node_std = torch.std(target, dim=(0, 1)) + 1e-6
            med_std = torch.median(node_std)
            std_floor = torch.clamp(med_std * 0.1, min=1e-3)
            node_std = torch.clamp(node_std, min=std_floor)
            weights = (node_std.mean() / node_std).view(1, 1, -1)
            sq_err = ((target - predict) ** 2) * weights
            abs_err = torch.abs(target - predict) * weights
            diff_r_w = (diff_r**2) * weights
            rrse = math.sqrt(torch.sum(sq_err)) / (math.sqrt(torch.sum(diff_r_w)) + eps)
            rae = float((torch.sum(abs_err) / (torch.sum(torch.abs(diff_r) * weights) + eps)).item())
        else:
            rrse = math.sqrt(torch.sum((target - predict) ** 2)) / (math.sqrt(torch.sum(diff_r**2)) + eps)
            rae = float((torch.sum(torch.abs(target - predict)) / (torch.sum(torch.abs(diff_r)) + eps)).item())
        p = predict.numpy()
        y = target.numpy()
        correlation = _safe_corr_np(p, y, eps=eps)
        sm = 0.0
        B, L, N = predict.shape
        for b in range(B):
            for n in range(N):
                yt, yp = y[b, :, n], p[b, :, n]
                den = np.abs(yt) + np.abs(yp)
                den[den == 0] = eps
                sm += float(np.mean(np.abs(yt - yp) / den))
        sm /= max(1, B * N)

    if is_plot:
        B, L, N = predict.shape
        full_len = B + L - 1
        pred_full = torch.zeros(full_len, N)
        true_full = torch.zeros(full_len, N)
        ci_full = torch.zeros(full_len, N)
        count = torch.zeros(full_len, 1)

        for s in range(B):
            sl = slice(s, s + L)
            pred_full[sl] += predict[s]
            true_full[sl] += target[s]
            ci_full[sl] += conf95[s]
            count[sl] += 1.0

        count_safe = torch.clamp(count, min=1.0)
        pred_full = pred_full / count_safe
        true_full = true_full / count_safe
        ci_full = ci_full / count_safe

        is_chrono = bool(getattr(args, "use_chronological_split", False)) or bool(
            getattr(args, "chronological_split", False)
        )
        steps_per_year = int(getattr(args, "steps_per_year", 12))
        base_year = None
        base_month = 1

        if is_chrono:
            base_year = (args.train_end_year + 1) if kind == "Validation" else (args.valid_end_year + 1)
            base_month = 1
        else:
            data_start_year = getattr(data, "start_year", None)
            data_start_month = getattr(data, "start_month", None)

            if data_start_year is not None and data_start_month is not None:
                idx_set = (
                    getattr(data, "_valid_idx_set", None)
                    if kind == "Validation"
                    else getattr(data, "_test_idx_set", None)
                )

                if idx_set and len(idx_set) > 0:
                    first_window_idx = int(idx_set[0])
                    total_months_from_start = first_window_idx
                    total_months = (data_start_month - 1) + total_months_from_start
                    base_year = data_start_year + (total_months // steps_per_year)
                    base_month = (total_months % steps_per_year) + 1

                    print(
                        f"[DateMapping] {kind} split starts at {base_year:04d}-{base_month:02d} (window index {first_window_idx})"
                    )
                else:
                    print(f"[DateMapping] Warning: Could not find index set for {kind} split")
            else:
                print("[DateMapping] Warning: No date information found in data object")

        if base_year is not None and is_chrono:
            try:
                idx_set = (
                    getattr(data, "_valid_idx_set", None)
                    if kind == "Validation"
                    else getattr(data, "_test_idx_set", None)
                )
                if idx_set:
                    base_year, base_month = _offset_year_month(
                        base_year, base_month, int(idx_set[0]), steps_per_year
                    )
            except Exception:
                pass

        try:
            if kind == "Validation":
                if args.calibration in ("val", "both"):
                    lam = 1e-6
                    X1 = torch.stack([pred_full, torch.ones_like(pred_full)], dim=-1)
                    XtX = torch.einsum("tnk,tnj->k j n", X1, X1) + lam * torch.eye(2, device=X1.device).unsqueeze(-1)
                    Xty = torch.einsum("tnk,tn->k n", X1, true_full)
                    ab = torch.linalg.solve(XtX.permute(2, 0, 1), Xty.permute(1, 0)).permute(1, 0)
                    a, b = ab[0], ab[1]
                    pred_full = a.unsqueeze(0) * pred_full + b.unsqueeze(0)
                    state["calib_ab"] = (a.detach().cpu(), b.detach().cpu())
                    jlog(
                        "calibration_linear",
                        a_med=float(torch.median(a).cpu()),
                        b_med=float(torch.median(b).cpu()),
                    )
                else:
                    state["calib_ab"] = None
            elif kind == "Testing":
                calib_ab = state.get("calib_ab")
                if args.calibration in ("test", "both") and isinstance(calib_ab, tuple) and len(calib_ab) == 2:
                    a_t, b_t = calib_ab
                    a = a_t.to(pred_full.device)
                    b = b_t.to(pred_full.device)
                    pred_full = a.unsqueeze(0) * pred_full + b.unsqueeze(0)
        except Exception as _e:
            jlog("warn_calibration_linear", split=kind, error=str(_e)[:160])

        if is_plot:
            try:
                n_cols = pred_full.shape[-1]
                for col in range(n_cols):
                    node_name = str(DataLoaderS.col[col])
                    plot_node_name = DataLoaderS.get_plot_node_name(node_name)
                    pred_col = pred_full[:, col]
                    true_col = true_full[:, col]
                    ci_col = ci_full[:, col] if ci_full is not None else None
                    plot_predicted_actual(
                        pred_col,
                        true_col,
                        plot_node_name,
                        kind,
                        ci=ci_col,
                        base_year=getattr(data, "start_year", None),
                        steps_per_year=getattr(data, "steps_per_year", None),
                        base_month=getattr(data, "base_month", 1),
                    )
            except Exception as plot_err:
                print(f"[Plot] Warning: plotting failed for {kind}: {plot_err}")

        if save_metrics_fn is not None:
            try:
                _metrics = _compute_metrics(true_full.numpy(), pred_full.numpy())
                _extras = {
                    "calibration": getattr(args, "calibration", "none"),
                    "series_len": int(pred_full.shape[0]),
                }
                if kind == "Validation":
                    _extras["val_ci_ratio"] = state.get("last_val_ci_ratio")
                save_metrics_fn(args, kind, _metrics, _extras)
                try:
                    rse = _metrics.get("RSE", None)
                    rae = _metrics.get("RAE", None)
                    print(f"[Metrics] {kind}: RSE={rse:.6f} RAE={rae:.6f}")
                except Exception:
                    pass
            except Exception as _e:
                jlog("warn_save_metrics", split=kind, error=str(_e)[:160])

        if getattr(args, "plot_norm_space", False):
            try:
                mu = data.mu.detach().cpu()
                std = torch.clamp(data.std.detach().cpu(), min=1e-6)

                def _to_z(arr):
                    view = [1] * arr.dim()
                    view[-1] = -1
                    mu_v = mu.view(*view).expand_as(arr)
                    std_v = std.view(*view).expand_as(arr)
                    return (arr - mu_v) / std_v

                pred_full_z = _to_z(pred_full)
                true_full_z = _to_z(true_full)
                n_cols = pred_full.shape[-1]
                for col in range(n_cols):
                    base_name = DataLoaderS.get_plot_node_name(str(DataLoaderS.col[col]))
                    node_name = f"{base_name}_Norm"
                    pred_col_z = pred_full_z[:, col]
                    true_col_z = true_full_z[:, col]
                    plot_predicted_actual(
                        pred_col_z,
                        true_col_z,
                        node_name,
                        f"{kind}_NormSpace",
                        ci=None,
                        base_year=getattr(data, "start_year", None),
                        steps_per_year=getattr(data, "steps_per_year", None),
                        base_month=getattr(data, "base_month", 1),
                    )
            except Exception as _e:
                print(f"[plot_norm_space] error: {_e}")

        try:
            assert isinstance(pred_full, torch.Tensor) and isinstance(true_full, torch.Tensor)
            p = pred_full.numpy()
            y = true_full.numpy()
            _m_full = _compute_metrics(y, p)
            rrse = float(_m_full.get("RSE", float("nan")))
            rae = float(_m_full.get("RAE", float("nan")))
            sm = float(_m_full.get("sMAPE", float("nan")))
            correlation = _safe_corr_np(p, y, eps=eps)
        except Exception as _e:
            print(f"[metrics_recompute] warning: {_e}")

    if getattr(args, "runlog", False) and is_plot and pred_full is not None and true_full is not None:
        try:
            err = pred_full - true_full
            bias_node = err.mean(dim=0).abs()
            std_p = pred_full.std(dim=0)
            std_t = true_full.std(dim=0) + 1e-12
            var_ratio = std_p / std_t

            def _q(t, q):
                return float(torch.quantile(t, q).cpu().item())

            jlog(
                "calibration",
                split=kind,
                bias_med=_q(bias_node, 0.5),
                bias_p95=_q(bias_node, 0.95),
                var_ratio_med=_q(var_ratio, 0.5),
                var_ratio_p95=_q(var_ratio, 0.95),
                corr_mean=float(correlation),
            )
        except Exception as _e:
            jlog("warn_calibration", split=kind, error=str(_e)[:160])

    if kind == "Validation" and r_collect:
        try:
            r_all = torch.cat([torch.as_tensor(x) for x in r_collect], dim=0).numpy()
            z_new = float(np.quantile(r_all, 0.95))
            state["conf_z"] = max(0.5, min(3.5, z_new))
            if getattr(args, "runlog", False):
                jlog("conf_calibrate", z_new=z_new, z_final=state["conf_z"])

            try:
                os.makedirs("model/Bayesian", exist_ok=True)
                np.save("model/Bayesian/residuals.npy", r_all)
                if getattr(args, "runlog", False):
                    jlog(
                        "residuals_saved",
                        path="model/Bayesian/residuals.npy",
                        count=len(r_all),
                    )
            except Exception as _save_err:
                if getattr(args, "runlog", False):
                    jlog("warn_residuals_save", err=str(_save_err)[:160])
        except Exception as _e:
            jlog("warn_conf_calibrate", err=str(_e)[:160])

    if kind == "Validation":
        try:
            targ_flat = target.reshape(-1, target.shape[-1])
            q80 = torch.quantile(targ_flat, 0.80, dim=0)
            q20 = torch.quantile(targ_flat, 0.20, dim=0)
            robust_amp = torch.clamp(q80 - q20, min=1e-6)
            ci_half_per_node = conf95.mean(dim=(0, 1))
            ratio_nodes = ci_half_per_node / robust_amp
            median_val = torch.median(ratio_nodes).item()
            state["last_val_ci_ratio"] = float(median_val) if median_val is not None else None
            if getattr(args, "runlog", False):
                jlog("val_ci_ratio", ci_ratio=state["last_val_ci_ratio"])
        except Exception:
            state["last_val_ci_ratio"] = None

    def _to_safe_float(v):
        try:
            return float(v) if v is not None else float("nan")
        except Exception:
            try:
                return float(str(v))
            except Exception:
                return float("nan")

    return (
        _to_safe_float(rrse),
        _to_safe_float(rae),
        _to_safe_float(correlation),
        _to_safe_float(sm),
    )


def generate_epoch_validation_test_plots_impl(
    Data,
    model,
    evaluateL2,
    evaluateL1,
    *,
    args,
    state,
    evaluate_fn,
    evaluate_sliding_window_fn,
):
    if getattr(args, "no_plots", False):
        return
    try:
        if Data.valid[0] is not None and Data.valid[1] is not None:
            _ = evaluate_fn(
                Data,
                Data.valid[0],
                Data.valid[1],
                model,
                evaluateL2,
                evaluateL1,
                args.batch_size,
                True,
                mc_runs=args.mc_runs,
                kind="Validation",
            )
    except Exception as _e:
        print(f"[Plot] Validation epoch plots failed: {_e}")

    try:
        if Data.test_window is not None:
            _ = evaluate_sliding_window_fn(
                Data,
                Data.test_window,
                model,
                evaluateL2,
                evaluateL1,
                args.seq_in_len,
                True,
                mc_runs=args.mc_runs,
            )
    except Exception as _e:
        print(f"[Plot] Testing epoch plots failed: {_e}")
