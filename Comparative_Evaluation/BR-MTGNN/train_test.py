"""
BR-MTGNN hyperparameter search and validation.

Workflow:
  - Splits data: 70% train, 20% validation, 10% test
  - Trains model on train set, evaluates on validation set
  - Saves best_model.pt (best validation epoch) and model.pt
  - Saves hp.txt with optimal hyperparameters
  - Generates validation/test plots and metrics

Output:
  - model/Bayesian/model.pt          (best validation epoch, ready for train.py)
  - model/Bayesian/model.pt     (same as model.pt, kept for reference)
  - model/Bayesian/hp.txt            (hyperparameters for train.py)
  - model/Bayesian/metadata.json     (data schema, transforms, conformal quantiles for forecast.py)
  - model/Bayesian/training_history.csv
  - model/Bayesian/Validation/ and Testing/ (per-RMD plots)

Usage:
  python train_test.py --device cuda:0 --output_dir model/Bayesian
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d

from net import gtnet
from util import (
    RMDPTData,
    build_gap_forecast,
    conformal_q,
    ensure_dir,
    interval_from_mc_group,
    multi_target_loss,
    nodewise_metrics,
    pick_device,
    predict_loader_mc,
    predict_loader_point,
    predict_mc,
    reshape_model_output,
    save_gap_forecast,
    save_json,
    save_metrics_table,
    save_node_metrics_csv,
    save_required_forecast_files,
    set_random_seed,
    evaluate_all_metrics,
)

# Standard colors for validation/test plots
COLOR_ACTUAL = "#0072B2"      # Blue
COLOR_PREDICTED = "#D55E00"   # Orange
_COLORS = [
    COLOR_ACTUAL,        # Blue for actual
    COLOR_PREDICTED,     # Orange for predicted
    "#D55E00",  # Red-orange
    "#009E73",  # Teal
    "#F0E442",  # Yellow
    "#CC79A7",  # Pink
    "#56B4E9",  # Light blue
    "#999999",  # Gray
]

def load_checkpoint_compat(path: Path, device: torch.device):
    """
    Load checkpoints safely across PyTorch versions.
    Newer PyTorch may default to weights_only=True in some setups.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _apply_style(ax, fig):
    """Apply reference-style formatting to validation/test plots."""
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    # Light gray gridlines, visible but subtle
    ax.grid(True, which="major", axis="both", color="#B3B3B3", linewidth=0.7, alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("Trend", fontsize=8)
    ax.set_xlabel("Year", fontsize=8)
    ax.tick_params(axis='both', labelsize=7)


def plot_per_rmd_fit(
    split_name: str,
    data: "RMDPTData",
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    out_dir: Path,
):
    """Save Nature-style PDF and PNG per RMD showing actual vs predicted (1-step-ahead).

    y_true / y_pred / y_lower / y_upper shape: [B, H, C, N]
    Uses step 0 of each window → 1-step-ahead time series per node.

    Outputs:
      - Fig_<RMD>_<split>.pdf (vector format, primary)
      - Fig_<RMD>_<split>.png (raster format, preview)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    B, H, C, N = y_true.shape
    # Step 0 of each window gives a 1-step-ahead prediction series
    true_1step = np.clip(y_true[:, 0, 0, :], 0.0, None)  # [B, N]
    pred_1step = np.clip(y_pred[:, 0, 0, :], 0.0, None)  # [B, N]
    lower_1step = np.clip(y_lower[:, 0, 0, :], 0.0, None)  # [B, N]
    upper_1step = np.clip(y_upper[:, 0, 0, :], 0.0, None)  # [B, N]

    # Recover dates aligned to step-0 targets
    seq_in = data.seq_in_len
    train_n = data.split.train_samples
    valid_n = data.split.valid_samples
    start_row = train_n if split_name == "validation" else train_n + valid_n
    target_rows = [min(start_row + s + seq_in, len(data.dates) - 1) for s in range(B)]
    split_dates = pd.DatetimeIndex([data.dates[r] for r in target_rows])

    rmd_indices = data.schema.rmd_nom_indices
    sigma = 2.0  # gaussian smoothing for cleaner lines

    split_abbr = "Val" if split_name == "validation" else "Test"

    for ni in rmd_indices:
        name = data.schema.entity_names[ni]
        # Remove "RMD_" prefix if present for display
        display_name = name.replace("RMD_", "") if name.startswith("RMD_") else name
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:100]

        true_sm = gaussian_filter1d(true_1step[:, ni].astype(float), sigma=sigma)
        pred_sm = gaussian_filter1d(pred_1step[:, ni].astype(float), sigma=sigma)
        lower_sm = gaussian_filter1d(lower_1step[:, ni].astype(float), sigma=sigma)
        upper_sm = gaussian_filter1d(upper_1step[:, ni].astype(float), sigma=sigma)

        # Reference-style figure: compact, readable, clean
        fig, ax = plt.subplots(figsize=(5.5, 3.5))

        # Plot actual and predicted lines
        ax.plot(split_dates, true_sm, color="#0072B2", linewidth=1.5,
                label="Actual", zorder=4)
        ax.plot(split_dates, pred_sm, color="#D55E00", linewidth=1.5, linestyle="--",
                label="Predicted", zorder=3)

        ax.set_title(display_name, fontsize=10, pad=8, fontweight='normal')
        ax.set_ylim(bottom=0)
        _apply_style(ax, fig)
        ax.legend(loc="upper left", fontsize=7, frameon=True, edgecolor="black",
                  fancybox=False, framealpha=0.9)
        fig.autofmt_xdate(rotation=90, ha="right")
        fig.tight_layout()

        # Save as vector PDF (primary) and PNG preview
        output_stem = f"Fig_{split_abbr}_{safe_name}"
        fig.savefig(out_dir / f"{output_stem}.pdf", format="pdf", bbox_inches="tight", dpi=300)
        fig.savefig(out_dir / f"{output_stem}.png", format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)

    print(f"  {len(rmd_indices)} per-RMD {split_name} plots → {out_dir}/ (PDF + PNG)")


def parse_args():
    p = argparse.ArgumentParser(description="BR-MTGNN training/testing for RMD NoM, RMD NoP, and PT NoM forecasting")
    p.add_argument("--data_csv", type=str, default="data/sm_data.csv")
    p.add_argument("--nodes_csv", type=str, default="data/data.csv")
    p.add_argument("--graph_csv", type=str, default="data/graph_sparse.csv")
    p.add_argument("--output_dir", type=str, default="model/Bayesian")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--trial", type=int, default=1, help="Trial id for progress logging.")

    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num_threads", type=int, default=1)
    p.add_argument("--global_regex", type=str, default="")
    p.add_argument("--no_log1p", action="store_true")
    p.add_argument("--no_clip_nonnegative", action="store_true")

    p.add_argument("--train_ratio", type=float, default=0.70)
    p.add_argument("--valid_ratio", type=float, default=0.20)
    p.add_argument("--seq_in_len", type=int, default=10)
    p.add_argument("--seq_out_len", type=int, default=36)
    p.add_argument("--batch_size", type=int, default=16)

    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--loss", type=str, default="smoothl1", choices=["mae", "mse", "smoothl1"])
    p.add_argument("--w_NoM", type=float, default=1.0)
    p.add_argument("--w_NoP", type=float, default=0.0)
    p.add_argument("--w_PT", type=float, default=1.0)

    p.add_argument("--gcn_true", action="store_true", default=True)
    p.add_argument("--no_gcn", dest="gcn_true", action="store_false")
    p.add_argument("--buildA_true", action="store_true", default=True)
    p.add_argument("--static_graph_only", dest="buildA_true", action="store_false")
    p.add_argument("--gcn_depth", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.02)
    p.add_argument("--subgraph_size", type=int, default=10)
    p.add_argument("--node_dim", type=int, default=40)
    p.add_argument("--dilation_exponential", type=int, default=2)
    p.add_argument("--conv_channels", type=int, default=32)
    p.add_argument("--residual_channels", type=int, default=32)
    p.add_argument("--skip_channels", type=int, default=64)
    p.add_argument("--end_channels", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--propalpha", type=float, default=0.05)
    p.add_argument("--tanhalpha", type=float, default=3.0)
    p.add_argument("--graph_prior_weight", type=float, default=1.0)

    p.add_argument("--lr_schedule", type=str, default="plateau", choices=["none", "plateau", "cosine"])
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--lr_decay_factor", type=float, default=0.7)
    p.add_argument("--lr_patience_sched", type=int, default=15)

    p.add_argument("--mc_runs", type=int, default=100)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--z_sigma", type=float, default=1.96,
                   help="Multiplier on MC-dropout std in CI: mean ± z_sigma*std. "
                        "Default 1.96 is normal 95%% coverage.")

    p.add_argument("--use_bai_input", action="store_true", default=False,
                   help="Add Burden-Attention Index as a 3rd input channel (scaled NoM + scaled NoP fusion).")
    p.add_argument("--bai_nom_weight", type=float, default=0.7,
                   help="Weight of scaled NoM in BAI (default 0.7).")
    p.add_argument("--bai_nop_weight", type=float, default=0.3,
                   help="Weight of scaled NoP in BAI (default 0.3).")
    p.add_argument("--use_tdb_input", action="store_true", default=True,
                   help="Add Total Disease Burden channel (NoM_scaled + NoP_scaled) as an extra input.")
    p.add_argument("--lambda_delta", type=float, default=0.02,
                   help="Regularization weight for persistence-delta penalty (default 0.02).")
    return p.parse_args()


def make_model(args, data: RMDPTData, device: torch.device):
    out_channels = 1 if args.use_tdb_input else 2
    out_dim = args.seq_out_len * out_channels
    return gtnet(
        gcn_true=args.gcn_true,
        buildA_true=args.buildA_true,
        gcn_depth=args.gcn_depth,
        num_nodes=data.schema.node_count,
        device=device,
        predefined_A=data.adj,
        dropout=args.dropout,
        subgraph_size=min(args.subgraph_size, data.schema.node_count),
        node_dim=args.node_dim,
        dilation_exponential=args.dilation_exponential,
        conv_channels=args.conv_channels,
        residual_channels=args.residual_channels,
        skip_channels=args.skip_channels,
        end_channels=args.end_channels,
        seq_length=args.seq_in_len,
        in_dim=len(data.schema.input_channels),
        out_dim=out_dim,
        out_channels=out_channels,
        layers=args.layers,
        propalpha=args.propalpha,
        tanhalpha=args.tanhalpha,
        graph_prior_weight=args.graph_prior_weight,
    ).to(device)


def train_one_epoch(model, loader, optimizer, data: RMDPTData, args, device):
    model.train()
    total_loss = 0.0
    total_batches = 0

    for X, Y, M_nom, M_nop, M_pt in loader:
        X = X.to(device)
        Y = Y.to(device)
        M_nom = M_nom.to(device)
        M_nop = M_nop.to(device)
        M_pt = M_pt.to(device)

        out_channels = 1 if args.use_tdb_input else 2
        optimizer.zero_grad(set_to_none=True)
        raw = model(X)
        pred = reshape_model_output(raw, horizon=args.seq_out_len, out_channels=out_channels, node_count=data.schema.node_count)
        loss, _ = multi_target_loss(
            pred=pred,
            true=Y,
            mask_nom=M_nom,
            mask_nop=M_nop,
            mask_pt=M_pt,
            loss_name=args.loss,
            w_nom=args.w_NoM,
            w_nop=args.w_NoP,
            w_pt=args.w_PT,
        )
        # Delta regularizer: penalize large deviations from persistence baseline
        if args.use_tdb_input:
            last_tdb = (X[:, 0, :, -1] + X[:, 1, :, -1]).unsqueeze(1).expand(-1, pred.shape[1], -1)
            base = last_tdb.unsqueeze(2)
        else:
            last_nom = X[:, 0, :, -1].unsqueeze(1).expand(-1, pred.shape[1], -1)
            last_nop = X[:, 1, :, -1].unsqueeze(1).expand(-1, pred.shape[1], -1)
            base = torch.stack([last_nom, last_nop], dim=2)
        delta_penalty = torch.mean(torch.abs(pred - base))
        loss = loss + args.lambda_delta * delta_penalty
        loss.backward()
        if args.clip and args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def eval_loss(model, loader, data: RMDPTData, args, device):
    # Keep model.train() so dropout stays active — the model is designed for
    # MC-dropout and produces exploding outputs in eval() mode with untrained weights.
    # predict_loader_mc also uses model.train(), so this keeps early-stopping
    # consistent with the actual evaluation criterion.
    model.train()
    total_loss = 0.0
    total_batches = 0
    se_sum = 0.0
    n_sum = 0.0
    true_sum = 0.0
    true_sq_sum = 0.0

    for X, Y, M_nom, M_nop, M_pt in loader:
        X = X.to(device)
        Y = Y.to(device)
        M_nom = M_nom.to(device)
        M_nop = M_nop.to(device)
        M_pt = M_pt.to(device)

        out_channels = 1 if args.use_tdb_input else 2
        raw = model(X)
        pred = reshape_model_output(raw, horizon=args.seq_out_len, out_channels=out_channels, node_count=data.schema.node_count)
        loss, _ = multi_target_loss(
            pred=pred,
            true=Y,
            mask_nom=M_nom,
            mask_nop=M_nop,
            mask_pt=M_pt,
            loss_name=args.loss,
            w_nom=args.w_NoM,
            w_nop=args.w_NoP,
            w_pt=args.w_PT,
        )
        total_loss += float(loss.item())
        total_batches += 1

        err = pred - Y
        if args.use_tdb_input:
            mask = M_nom[:, :, None, :]
        else:
            mask_nom_or_pt = torch.clamp(M_nom + M_pt, min=0.0, max=1.0)
            mask = torch.stack([mask_nom_or_pt, M_nop], dim=2)
        err2 = (err * err) * mask
        y_masked = Y * mask
        se_sum += float(err2.sum().item())
        n_sum += float(mask.sum().item())
        true_sum += float(y_masked.sum().item())
        true_sq_sum += float(((Y * Y) * mask).sum().item())

    valid_loss = total_loss / max(total_batches, 1)
    if n_sum <= 0.0:
        return valid_loss, float("nan"), float("nan")
    mse = se_sum / n_sum
    rme = math.sqrt(max(mse, 0.0))
    true_mean = true_sum / n_sum
    denom = true_sq_sum - (n_sum * true_mean * true_mean)
    rse = math.sqrt(se_sum / denom) if denom > 1e-12 else float("nan")
    return valid_loss, rme, rse


def main():
    args = parse_args()
    set_random_seed(args.seed)

    if args.num_threads and args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    device = pick_device(args.device)
    outdir = Path(args.output_dir)
    ensure_dir(outdir)
    ensure_dir(outdir / "metrics")
    ensure_dir(outdir / "forecast" / "data")
    ensure_dir(outdir / "forecast" / "gap")
    ensure_dir(outdir / "forecast" / "plots")

    data = RMDPTData(
        data_csv=args.data_csv,
        graph_csv=args.graph_csv,
        nodes_csv=args.nodes_csv,
        seq_in_len=args.seq_in_len,
        seq_out_len=args.seq_out_len,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        batch_size=args.batch_size,
        device=device,
        global_regex=args.global_regex,
        log1p=not args.no_log1p,
        clip_nonnegative=not args.no_clip_nonnegative,
        use_bai=args.use_bai_input,
        bai_nom_weight=args.bai_nom_weight,
        bai_nop_weight=args.bai_nop_weight,
        use_tdb=args.use_tdb_input,
    )

    print(f"Device: {device}")
    print(f"Rows: {len(data.raw_df)} | nodes: {data.schema.node_count} | input channels: {len(data.schema.input_channels)}")
    print(f"RMD NoM targets: {len(data.schema.rmd_nom_indices)} | NoP targets: {len(data.schema.nop_indices)} | PT NoM targets: {len(data.schema.pt_nom_indices)}")
    print(f"Global covariates: {len(data.schema.global_cols)}")
    print(f"Graph prior links (RMD->PT): {sum(len(v) for v in data.graph_links.values())}")
    print(f"Split samples: train={data.split.train_samples}, valid={data.split.valid_samples}, test={data.split.test_samples}")

    model = make_model(args, data, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_schedule == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.lr_decay_factor,
            patience=args.lr_patience_sched, min_lr=args.lr_min,
        )
    elif args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr_min,
        )
    else:
        scheduler = None

    best_valid = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []
    best_path = outdir / "model.pt"
    start = time.time()

    train_loader = data.loader("train", shuffle=True)
    valid_loader = data.loader("valid", shuffle=False)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, data, args, device)
        valid_loss, val_rme, val_rse = eval_loss(model, valid_loader, data, args, device)

        if scheduler is not None:
            if args.lr_schedule == "plateau":
                scheduler.step(valid_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "valid_rme": val_rme,
                "valid_rse": val_rse,
                "lr": current_lr,
            }
        )

        print(f"{args.trial}\t{epoch}\t{val_rme:.6f}\t{val_rse:.6f}\t{valid_loss:.6f}")
        if valid_loss < best_valid - 1e-8:
            best_valid = valid_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}, best valid loss={best_valid:.6f}")
            break

    pd.DataFrame(history).to_csv(outdir / "training_history.csv", index=False)
    model.load_state_dict(load_checkpoint_compat(best_path, device))
    torch.save(model.state_dict(), outdir / "model.pt")

    hp = {
        "gcn_depth": args.gcn_depth,
        "lr": args.lr,
        "conv_channels": args.conv_channels,
        "residual_channels": args.residual_channels,
        "skip_channels": args.skip_channels,
        "end_channels": args.end_channels,
        "subgraph_size": min(args.subgraph_size, data.schema.node_count),
        "dropout": args.dropout,
        "dilation_exponential": args.dilation_exponential,
        "node_dim": args.node_dim,
        "propalpha": args.propalpha,
        "tanhalpha": args.tanhalpha,
        "layers": args.layers,
        "seq_in_len": args.seq_in_len,
        "seq_out_len": args.seq_out_len,
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "input_channels": data.schema.input_channels,
        "num_nodes": data.schema.node_count,
        "output_channels": 1 if args.use_tdb_input else 2,
        "use_bai_input": args.use_bai_input,
        "bai_nom_weight": args.bai_nom_weight,
        "bai_nop_weight": args.bai_nop_weight,
        "use_tdb_input": args.use_tdb_input,
        "lambda_delta": args.lambda_delta,
        "w_NoM": args.w_NoM,
        "w_NoP": args.w_NoP,
        "w_PT": args.w_PT,
        "loss": args.loss,
    }
    with open(outdir / "hp.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(hp, indent=2))

    out_channels = 1 if args.use_tdb_input else 2

    # MC prediction for uncertainty intervals only
    val_pred_mc = predict_loader_mc(
        model=model,
        loader=valid_loader,
        data=data,
        mc_runs=args.mc_runs,
        device=device,
        horizon=args.seq_out_len,
        out_channels=out_channels,
    )
    # Deterministic prediction for point metrics (no dropout noise)
    val_pred = predict_loader_point(
        model=model,
        loader=valid_loader,
        data=data,
        device=device,
        horizon=args.seq_out_len,
        out_channels=out_channels,
    )
    val_pred["std_raw"] = val_pred_mc["std_raw"]

    q_nom = conformal_q(
        y_true=val_pred["true_raw"][:, :, 0, :],
        y_pred=val_pred["mean_raw"][:, :, 0, :],
        mask=val_pred["mask_nom"],
        alpha=args.alpha,
    )
    # NoP: use channel 0 in TDB mode (single combined channel)
    q_nop = conformal_q(
        y_true=val_pred["true_raw"][:, :, 0 if args.use_tdb_input else 1, :],
        y_pred=val_pred["mean_raw"][:, :, 0 if args.use_tdb_input else 1, :],
        mask=val_pred["mask_nop"],
        alpha=args.alpha,
    )
    q_pt = conformal_q(
        y_true=val_pred["true_raw"][:, :, 0, :],
        y_pred=val_pred["mean_raw"][:, :, 0, :],
        mask=val_pred["mask_pt"],
        alpha=args.alpha,
    )

    v_lower, v_upper = interval_from_mc_group(
        mean_raw=val_pred["mean_raw"],
        std_raw=val_pred["std_raw"],
        schema=data.schema,
        q_nom=q_nom,
        q_nop=q_nop,
        q_pt=q_pt,
        z=args.z_sigma,
    )

    val_metrics = evaluate_all_metrics(
        y_true=val_pred["true_raw"],
        y_pred=val_pred["mean_raw"],
        y_lower=v_lower,
        y_upper=v_upper,
        mask_nom=val_pred["mask_nom"],
        mask_nop=val_pred["mask_nop"],
        mask_pt=val_pred["mask_pt"],
    )
    save_metrics_table(outdir / "metrics" / "validation_metrics.csv", "validation", val_metrics)
    save_node_metrics_csv(
        outdir / "metrics" / "validation_node_metrics.csv",
        nodewise_metrics(
            schema=data.schema,
            y_true=val_pred["true_raw"],
            y_pred=val_pred["mean_raw"],
            y_lower=v_lower,
            y_upper=v_upper,
            mask_nom=val_pred["mask_nom"],
            mask_nop=val_pred["mask_nop"],
        ),
    )

    test_loader = data.loader("test", shuffle=False)
    # MC prediction for uncertainty intervals only
    test_pred_mc = predict_loader_mc(
        model=model,
        loader=test_loader,
        data=data,
        mc_runs=args.mc_runs,
        device=device,
        horizon=args.seq_out_len,
        out_channels=out_channels,
    )
    # Deterministic prediction for point metrics
    test_pred = predict_loader_point(
        model=model,
        loader=test_loader,
        data=data,
        device=device,
        horizon=args.seq_out_len,
        out_channels=out_channels,
    )
    test_pred["std_raw"] = test_pred_mc["std_raw"]
    t_lower, t_upper = interval_from_mc_group(
        mean_raw=test_pred["mean_raw"],
        std_raw=test_pred["std_raw"],
        schema=data.schema,
        q_nom=q_nom,
        q_nop=q_nop,
        q_pt=q_pt,
        z=args.z_sigma,
    )

    test_metrics = evaluate_all_metrics(
        y_true=test_pred["true_raw"],
        y_pred=test_pred["mean_raw"],
        y_lower=t_lower,
        y_upper=t_upper,
        mask_nom=test_pred["mask_nom"],
        mask_nop=test_pred["mask_nop"],
        mask_pt=test_pred["mask_pt"],
    )
    save_metrics_table(outdir / "metrics" / "test_metrics.csv", "test", test_metrics)
    save_node_metrics_csv(
        outdir / "metrics" / "test_node_metrics.csv",
        nodewise_metrics(
            schema=data.schema,
            y_true=test_pred["true_raw"],
            y_pred=test_pred["mean_raw"],
            y_lower=t_lower,
            y_upper=t_upper,
            mask_nom=test_pred["mask_nom"],
            mask_nop=test_pred["mask_nop"],
        ),
    )

    # Single 36-month forward forecast from latest history window.
    f_mean_b, f_std_b, _, _, _ = predict_mc(
        model=model,
        X=data.last_input_window(),
        data=data,
        mc_runs=args.mc_runs,
        device=device,
        horizon=args.seq_out_len,
        out_channels=out_channels,
    )
    f_mean = f_mean_b[0]
    f_std = f_std_b[0]

    f_lower_b, f_upper_b = interval_from_mc_group(
        mean_raw=f_mean_b,
        std_raw=f_std_b,
        schema=data.schema,
        q_nom=q_nom,
        q_nop=q_nop,
        q_pt=q_pt,
        z=args.z_sigma,
    )
    f_lower = f_lower_b[0]
    f_upper = f_upper_b[0]

    future_dates = data.future_dates()
    save_required_forecast_files(
        output_data_dir=outdir / "forecast" / "data",
        dates=future_dates,
        mean=f_mean,
        lower=f_lower,
        upper=f_upper,
        std=f_std,
        schema=data.schema,
    )

    gap_df = build_gap_forecast(
        dates=future_dates,
        mean=f_mean,
        lower=f_lower,
        upper=f_upper,
        schema=data.schema,
        graph_links=data.graph_links,
    )
    save_gap_forecast(outdir / "forecast" / "gap" / "rmd_pt_gap_forecast.csv", gap_df)

    metadata = data.metadata()
    metadata["model_args"] = vars(args)
    metadata["model_hyperparameters"] = hp
    # Conformal quantiles for residual-based calibration in forecast.py.
    # Used to extend MC-dropout intervals: lower = mean - z*std - q, upper = mean + z*std + q
    metadata["conformal_quantiles"] = {
        "NoM": q_nom,
        "NoP": q_nop,
        "PT_NoM": q_pt,
        "alpha": args.alpha,
    }
    metadata["elapsed_seconds"] = round(time.time() - start, 3)
    save_json(metadata, outdir / "metadata.json")

    # Format metrics output with recommended columns in standard order
    metrics_cols = ["group", "RAE", "RSE", "Corr", "Coverage"]

    print("Validation metrics:")
    val_df = pd.DataFrame([{"group": k, **v} for k, v in val_metrics.items()])
    cols_to_show = [c for c in metrics_cols if c in val_df.columns]
    print(val_df[cols_to_show].to_string(index=False))

    print("Test metrics:")
    test_df = pd.DataFrame([{"group": k, **v} for k, v in test_metrics.items()])
    cols_to_show = [c for c in metrics_cols if c in test_df.columns]
    print(test_df[cols_to_show].to_string(index=False))

    print(f"Saved model, metrics, forecast CSVs, and gap file to {outdir}")

    # Per-RMD fit plots for validation and test splits
    print("\nGenerating per-RMD validation fit plots...")
    plot_per_rmd_fit(
        split_name="validation",
        data=data,
        y_true=val_pred["true_raw"],
        y_pred=val_pred["mean_raw"],
        y_lower=v_lower,
        y_upper=v_upper,
        out_dir=outdir / "validation",
    )
    print("Generating per-RMD test fit plots...")
    plot_per_rmd_fit(
        split_name="test",
        data=data,
        y_true=test_pred["true_raw"],
        y_pred=test_pred["mean_raw"],
        y_lower=t_lower,
        y_upper=t_upper,
        out_dir=outdir / "test",
    )


if __name__ == "__main__":
    main()
