"""
=============================================================================
PIPELINE STEP 4 of 5 — train.py
=============================================================================
Purpose : Production training. Warm-starts from the best checkpoint saved
          by train_test.py and trains on 100% of the data for 500 epochs
          with forecast-continuity regularisation to prevent unrealistic
          spikes or jumps in the 36-month forecast horizon.

Run AFTER: train_test.py  (Step 3) — requires:
               model/Bayesian/model.pt
               model/Bayesian/hp.txt
               model/Bayesian/metadata.json
Run BEFORE: forecast.py  (Step 5)

HOW TO RUN
----------
    cd /home/sahsan03/B-MTGNN/BR_MTGNN_main

    python train.py \\
        --device cuda:1 \\
        --output_dir model/Bayesian \\
        --seq_in_len 10 --seq_out_len 36 \\
        --use_tdb_input \\
        --epochs 500 --lr 1e-5 \\
        --lambda_delta 0.02 --lambda_start 0.20 --lambda_horizon 0.05

Key options:
    --device          cuda:1 (GPU) or cpu
    --epochs          Training epochs on full data (default 500)
    --lr              Adam learning rate (default 1e-5)
    --lambda_delta    Weak global anchoring penalty (default 0.02)
    --lambda_start    First-step smoothness penalty (default 0.20)
    --lambda_horizon  Month-to-month smoothness penalty (default 0.05)

Continuity regularisation (prevents forecast divergence):
    lambda_start   : first forecast month must not jump far from last input
    lambda_horizon : penalises large month-to-month changes across horizon
    lambda_delta   : weak anchoring to recent observed level

Outputs:
    model/Bayesian/o_model.pt                  Final production model weights
    model/Bayesian/production_train_metadata.json  Training config + metrics

Expected time: ~3 min on GPU (cuda:1)
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from net import gtnet
from util import (
    RMDPTData,
    ensure_dir,
    multi_target_loss,
    pick_device,
    reshape_model_output,
    set_random_seed,
    predict_loader_point,
    predict_loader_mc,
    interval_from_mc_group,
    conformal_q,
    evaluate_all_metrics,
    save_metrics_table,
)


def parse_args():
    p = argparse.ArgumentParser(description="BC-MTGNN full-data final training")

    # Data / output
    p.add_argument("--data_csv", type=str, default="data/sm_data.csv")
    p.add_argument("--nodes_csv", type=str, default="data/data.csv")
    p.add_argument("--graph_csv", type=str, default="data/graph_sparse.csv")
    p.add_argument("--output_dir", type=str, default="model/Bayesian")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num_threads", type=int, default=1)

    p.add_argument(
        "--use_best_epoch_from_hp",
        action="store_true",
        default=True,
        help="Use best_epoch from hp.txt as training epoch budget when present",
    )
    p.add_argument(
        "--no_use_best_epoch_from_hp",
        dest="use_best_epoch_from_hp",
        action="store_false",
        help="Do not use best_epoch from hp.txt even if present",
    )
    p.add_argument(
        "--use_best_model_init",
        action="store_true",
        default=True,
        help="If model/Bayesian/model.pt exists (saved by train_test.py), initialize weights from it before training",
    )
    p.add_argument(
        "--no_use_best_model_init",
        dest="use_best_model_init",
        action="store_false",
        help="Do not initialize from model/Bayesian/model.pt even if present",
    )

    # Training hypers
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--loss", type=str, default="smoothl1", choices=["mae", "mse", "smoothl1"])

    # Loss weights
    p.add_argument("--w_NoM", type=float, default=1.0)
    p.add_argument("--w_NoP", type=float, default=0.0)
    p.add_argument("--w_PT", type=float, default=1.0)

    # LR schedule
    p.add_argument("--lr_decay_factor", type=float, default=0.7)
    p.add_argument("--lr_patience_sched", type=int, default=15)
    p.add_argument("--lr_min", type=float, default=1e-6)

    # Data args
    p.add_argument("--seq_in_len", type=int, default=36)
    p.add_argument("--seq_out_len", type=int, default=36)
    p.add_argument("--global_regex", type=str, default="")
    p.add_argument("--no_log1p", action="store_true")
    p.add_argument("--no_clip_nonnegative", action="store_true")

    p.add_argument("--use_tdb_input", action="store_true", default=True)
    p.add_argument("--no_use_tdb_input", dest="use_tdb_input", action="store_false")

    p.add_argument("--use_bai_input", action="store_true", default=False)
    p.add_argument("--bai_nom_weight", type=float, default=0.7)
    p.add_argument("--bai_nop_weight", type=float, default=0.3)

    # Forecast-continuity regularization
    p.add_argument(
        "--lambda_delta",
        type=float,
        default=0.02,
        help="Weak global anchoring penalty between forecast horizon and last observed input",
    )
    p.add_argument(
        "--lambda_start",
        type=float,
        default=0.20,
        help="Penalty for first forecast step jumping away from last observed input",
    )
    p.add_argument(
        "--lambda_horizon",
        type=float,
        default=0.05,
        help="Penalty for abrupt month-to-month changes inside forecast horizon",
    )

    # Model architecture
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

    return p.parse_args()


def _load_hp(hp_path: Path, args) -> None:
    """Override args in-place with values from hp.txt if present."""
    if not hp_path.exists():
        print(f"  hp.txt not found at {hp_path} — using CLI defaults.")
        return

    with open(hp_path, encoding="utf-8") as f:
        hp = json.load(f)

    # Architecture
    for key in (
        "gcn_true",
        "buildA_true",
        "gcn_depth",
        "conv_channels",
        "residual_channels",
        "skip_channels",
        "end_channels",
        "subgraph_size",
        "dropout",
        "dilation_exponential",
        "node_dim",
        "propalpha",
        "tanhalpha",
        "layers",
        "graph_prior_weight",
    ):
        if key in hp:
            setattr(args, key, hp[key])

    # Data window sizes
    if "seq_in_len" in hp:
        args.seq_in_len = int(hp["seq_in_len"])
    if "seq_out_len" in hp:
        args.seq_out_len = int(hp["seq_out_len"])

    # TDB / BAI mode
    if "use_tdb_input" in hp:
        args.use_tdb_input = bool(hp["use_tdb_input"])
    if "use_bai_input" in hp:
        args.use_bai_input = bool(hp["use_bai_input"])
    if "bai_nom_weight" in hp:
        args.bai_nom_weight = float(hp["bai_nom_weight"])
    if "bai_nop_weight" in hp:
        args.bai_nop_weight = float(hp["bai_nop_weight"])

    # Transforms
    if "global_regex" in hp:
        args.global_regex = str(hp["global_regex"])
    if "no_log1p" in hp:
        args.no_log1p = bool(hp["no_log1p"])
    if "no_clip_nonnegative" in hp:
        args.no_clip_nonnegative = bool(hp["no_clip_nonnegative"])

    # Loss weights and function
    if "w_NoM" in hp:
        args.w_NoM = float(hp["w_NoM"])
    if "w_NoP" in hp:
        args.w_NoP = float(hp["w_NoP"])
    if "w_PT" in hp:
        args.w_PT = float(hp["w_PT"])
    if "loss" in hp:
        args.loss = str(hp["loss"])

    # Forecast-continuity regularization
    if "lambda_delta" in hp:
        args.lambda_delta = float(hp["lambda_delta"])
    if "lambda_start" in hp:
        args.lambda_start = float(hp["lambda_start"])
    if "lambda_horizon" in hp:
        args.lambda_horizon = float(hp["lambda_horizon"])

    # Training hypers
    if "lr" in hp:
        args.lr = float(hp["lr"])
    if "batch_size" in hp:
        args.batch_size = int(hp["batch_size"])
    if "weight_decay" in hp:
        args.weight_decay = float(hp["weight_decay"])
    if "clip" in hp:
        args.clip = float(hp["clip"])

    # LR scheduler
    if "lr_decay_factor" in hp:
        args.lr_decay_factor = float(hp["lr_decay_factor"])
    if "lr_patience_sched" in hp:
        args.lr_patience_sched = int(hp["lr_patience_sched"])
    if "lr_min" in hp:
        args.lr_min = float(hp["lr_min"])

    # Use best_epoch from search as epoch budget only when allowed.
    if getattr(args, "use_best_epoch_from_hp", True) and "best_epoch" in hp:
        args.epochs = max(int(hp["best_epoch"]), 50)

    print(f"  Loaded hyperparameters from {hp_path}")


def _save_hp(hp_path: Path, args, data: RMDPTData | None = None) -> None:
    """Save the exact hyperparameters/architecture used for this production run."""
    hp = {
        # Architecture
        "gcn_true": bool(args.gcn_true),
        "buildA_true": bool(args.buildA_true),
        "gcn_depth": int(args.gcn_depth),
        "dropout": float(args.dropout),
        "subgraph_size": int(args.subgraph_size),
        "node_dim": int(args.node_dim),
        "dilation_exponential": int(args.dilation_exponential),
        "conv_channels": int(args.conv_channels),
        "residual_channels": int(args.residual_channels),
        "skip_channels": int(args.skip_channels),
        "end_channels": int(args.end_channels),
        "layers": int(args.layers),
        "propalpha": float(args.propalpha),
        "tanhalpha": float(args.tanhalpha),
        "graph_prior_weight": float(args.graph_prior_weight),

        # Forecast window
        "seq_in_len": int(args.seq_in_len),
        "seq_out_len": int(args.seq_out_len),

        # Output mode
        "use_tdb_input": bool(args.use_tdb_input),
        "use_bai_input": bool(args.use_bai_input),
        "bai_nom_weight": float(args.bai_nom_weight),
        "bai_nop_weight": float(args.bai_nop_weight),
        "output_channels": 1 if bool(args.use_tdb_input) else 2,

        # Data transform
        "global_regex": str(args.global_regex),
        "no_log1p": bool(args.no_log1p),
        "no_clip_nonnegative": bool(args.no_clip_nonnegative),

        # Training
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "loss": str(args.loss),
        "clip": float(args.clip),

        # Forecast-continuity regularization
        "lambda_delta": float(args.lambda_delta),
        "lambda_start": float(args.lambda_start),
        "lambda_horizon": float(args.lambda_horizon),

        # Loss weights
        "w_NoM": float(args.w_NoM),
        "w_NoP": float(args.w_NoP),
        "w_PT": float(args.w_PT),

        # LR scheduler
        "lr_decay_factor": float(args.lr_decay_factor),
        "lr_patience_sched": int(args.lr_patience_sched),
        "lr_min": float(args.lr_min),

        # Run info
        "seed": int(args.seed),
        "train_ratio": 0.98,
        "valid_ratio": 0.01,
    }

    if data is not None:
        hp["node_count"] = int(data.schema.node_count)
        hp["input_channels"] = int(len(data.schema.input_channels))

    hp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hp_path, "w", encoding="utf-8") as fh:
        json.dump(hp, fh, indent=2)

    print(f"Hyperparameters saved to {hp_path}")


def make_model(args, data: RMDPTData, device: torch.device):
    out_channels = 1 if args.use_tdb_input else 2

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
        out_dim=args.seq_out_len * out_channels,
        out_channels=out_channels,
        layers=args.layers,
        propalpha=args.propalpha,
        tanhalpha=args.tanhalpha,
        graph_prior_weight=args.graph_prior_weight,
    ).to(device)


def train_one_epoch(model, loader, optimizer, data: RMDPTData, args, device):
    model.train()

    out_channels = 1 if args.use_tdb_input else 2
    total_loss = 0.0
    total_batches = 0

    for X, Y, M_nom, M_nop, M_pt in loader:
        X = X.to(device)
        Y = Y.to(device)
        M_nom = M_nom.to(device)
        M_nop = M_nop.to(device)
        M_pt = M_pt.to(device)

        optimizer.zero_grad(set_to_none=True)

        raw = model(X)
        pred = reshape_model_output(
            raw,
            horizon=args.seq_out_len,
            out_channels=out_channels,
            node_count=data.schema.node_count,
        )

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

        # Build the last-observed baseline in the same channel structure as pred.
        if args.use_tdb_input:
            # TDB mode output has one channel. The last TDB input is NoM + NoP.
            last_tdb = (X[:, 0, :, -1] + X[:, 1, :, -1]).unsqueeze(1).expand(-1, pred.shape[1], -1)
            base = last_tdb.unsqueeze(2)
        else:
            # Two-channel mode output has NoM and NoP.
            last_nom = X[:, 0, :, -1].unsqueeze(1).expand(-1, pred.shape[1], -1)
            last_nop = X[:, 1, :, -1].unsqueeze(1).expand(-1, pred.shape[1], -1)
            base = torch.stack([last_nom, last_nop], dim=2)

        # Forecast-continuity regularization
        # 1) First forecast month should connect smoothly to the last observed input.
        first_step_penalty = torch.mean(torch.abs(pred[:, 0:1, :, :] - base[:, 0:1, :, :]))

        # 2) Forecast horizon should avoid artificial month-to-month step jumps.
        if pred.shape[1] > 1:
            horizon_smooth_penalty = torch.mean(torch.abs(pred[:, 1:, :, :] - pred[:, :-1, :, :]))
        else:
            horizon_smooth_penalty = torch.tensor(0.0, device=device)

        # 3) Weak global anchoring to recent level.
        delta_penalty = torch.mean(torch.abs(pred - base))

        loss = (
            loss
            + args.lambda_delta * delta_penalty
            + args.lambda_start * first_step_penalty
            + args.lambda_horizon * horizon_smooth_penalty
        )

        loss.backward()

        if args.clip and args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

    return total_loss / max(total_batches, 1)


def main():
    args = parse_args()

    set_random_seed(args.seed)

    if args.num_threads and args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    device = pick_device(args.device)
    outdir = Path(args.output_dir)
    ensure_dir(outdir)

    _load_hp(outdir / "hp.txt", args)

    print(
        f"Device: {device} | epochs: {args.epochs} | lr: {args.lr} | "
        f"tdb={args.use_tdb_input} | seq_in={args.seq_in_len} | seq_out={args.seq_out_len}"
    )
    print(
        f"Continuity penalties: lambda_delta={args.lambda_delta}, "
        f"lambda_start={args.lambda_start}, lambda_horizon={args.lambda_horizon}"
    )

    # Train on full data — 0.98 train ratio keeps a minimal valid/test split
    # required internally without wasting significant training data.
    data = RMDPTData(
        data_csv=args.data_csv,
        graph_csv=args.graph_csv,
        nodes_csv=args.nodes_csv,
        seq_in_len=args.seq_in_len,
        seq_out_len=args.seq_out_len,
        train_ratio=0.98,
        valid_ratio=0.01,
        batch_size=args.batch_size,
        device=device,
        global_regex=args.global_regex,
        log1p=not args.no_log1p,
        clip_nonnegative=not args.no_clip_nonnegative,
        use_tdb=args.use_tdb_input,
        use_bai=args.use_bai_input,
        bai_nom_weight=args.bai_nom_weight,
        bai_nop_weight=args.bai_nop_weight,
    )

    print(
        f"Nodes: {data.schema.node_count} | rows: {len(data.raw_df)} | "
        f"train windows: {data.split.train_samples} | "
        f"input channels: {len(data.schema.input_channels)}"
    )

    model = make_model(args, data, device)

    # Optionally initialize weights from model.pt (or best_model.pt for backward compatibility) saved by train_test.py.
    try:
        model_path = outdir / "model.pt"
        best_model_path = outdir / "best_model.pt"

        # Try model.pt first (new workflow), then best_model.pt (backward compatibility)
        init_path = None
        if model_path.exists():
            init_path = model_path
        elif best_model_path.exists():
            init_path = best_model_path

        if getattr(args, "use_best_model_init", True) and init_path:
            print(f"Found {init_path.name} at {init_path}; attempting to load as init weights...")
            state = torch.load(init_path, map_location=device)
            loaded = False

            if isinstance(state, dict) and "state_dict" in state:
                try:
                    model.load_state_dict(state["state_dict"])
                    loaded = True
                except Exception:
                    pass

            if not loaded:
                try:
                    model.load_state_dict(state)
                    loaded = True
                except RuntimeError:
                    try:
                        model.load_state_dict(state, strict=False)
                        loaded = True
                        print(f"Loaded {init_path.name} with strict=False partial match.")
                    except Exception as e:
                        print(f"Warning: failed to load {init_path.name}: {e}")

            if loaded:
                print(f"Initialized model weights from {init_path.name}")

    except Exception as e:
        print(f"Warning while attempting to initialize from model.pt/best_model.pt: {e}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} | receptive field: {model.receptive_field}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_decay_factor,
        patience=args.lr_patience_sched,
        min_lr=args.lr_min,
    )

    train_loader = data.loader("train", shuffle=True)
    start = time.time()

    try:
        print("Begin training")
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()

            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                data=data,
                args=args,
                device=device,
            )

            scheduler.step(train_loss)
            lr_now = optimizer.param_groups[0]["lr"]

            print(
                f"epoch:{epoch:3d} | train_loss: {train_loss:.6f} | "
                f"lr: {lr_now:.2e} | elapsed: {time.time() - epoch_start:.1f}s"
            )

    except KeyboardInterrupt:
        print("-" * 60)
        print("Exiting from training early")

    save_path = outdir / "o_model.pt"
    torch.save(model.state_dict(), save_path)

    print(f"\n{'='*60}")
    print(f"  o_model.pt saved → {save_path.resolve()}")
    print(f"  Total elapsed: {time.time() - start:.1f}s")
    print(f"{'='*60}")

    # Save production metadata alongside the model.
    meta = {
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "loss": str(args.loss),
        "w_NoM": float(args.w_NoM),
        "w_NoP": float(args.w_NoP),
        "w_PT": float(args.w_PT),
        "use_tdb_input": bool(args.use_tdb_input),
        "use_bai_input": bool(args.use_bai_input),
        "bai_nom_weight": float(args.bai_nom_weight),
        "bai_nop_weight": float(args.bai_nop_weight),
        "lambda_delta": float(args.lambda_delta),
        "lambda_start": float(args.lambda_start),
        "lambda_horizon": float(args.lambda_horizon),
        "seq_in_len": int(args.seq_in_len),
        "seq_out_len": int(args.seq_out_len),
        "train_ratio": 0.98,
        "valid_ratio": 0.01,
        "input_channels": int(len(data.schema.input_channels)),
        "node_count": int(data.schema.node_count),
        "model_parameters": int(n_params),
        "save_path": str(save_path),
    }

    meta_path = outdir / "production_train_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"Metadata saved to {meta_path}")

    # Save hp.txt in output_dir so forecast.py can rebuild the exact architecture.
    _save_hp(outdir / "hp.txt", args, data)

    # Calculate metrics on training data (full dataset)
    print(f"\n{'='*60}")
    print("Calculating training metrics...")
    print(f"{'='*60}")

    train_loader = data.loader("train", shuffle=False)
    out_channels = 1 if args.use_tdb_input else 2

    # Point predictions for metrics
    train_pred = predict_loader_point(
        model=model,
        loader=train_loader,
        data=data,
        device=device,
        horizon=args.seq_out_len,
        out_channels=out_channels,
    )

    # MC predictions for uncertainty intervals
    train_pred_mc = predict_loader_mc(
        model=model,
        loader=train_loader,
        data=data,
        mc_runs=100,
        device=device,
        horizon=args.seq_out_len,
        out_channels=out_channels,
    )
    train_pred["std_raw"] = train_pred_mc["std_raw"]

    # Compute conformal quantiles from validation data (if hp.txt has conformal_quantiles)
    try:
        hp_path = outdir / "hp.txt"
        if hp_path.exists():
            with open(hp_path, "r", encoding="utf-8") as f:
                hp_dict = json.load(f)
                if "conformal_quantiles" in hp_dict:
                    q_nom = hp_dict["conformal_quantiles"].get("NoM", 0.0)
                    q_nop = hp_dict["conformal_quantiles"].get("NoP", 0.0)
                    q_pt = hp_dict["conformal_quantiles"].get("PT_NoM", 0.0)
                else:
                    q_nom = q_nop = q_pt = 0.0
        else:
            q_nom = q_nop = q_pt = 0.0
    except Exception:
        q_nom = q_nop = q_pt = 0.0

    # Build uncertainty intervals
    train_lower, train_upper = interval_from_mc_group(
        mean_raw=train_pred["mean_raw"],
        std_raw=train_pred["std_raw"],
        schema=data.schema,
        q_nom=q_nom,
        q_nop=q_nop,
        q_pt=q_pt,
        z=1.96,  # 95% coverage
    )

    # Evaluate metrics
    train_metrics = evaluate_all_metrics(
        y_true=train_pred["true_raw"],
        y_pred=train_pred["mean_raw"],
        y_lower=train_lower,
        y_upper=train_upper,
        mask_nom=train_pred["mask_nom"],
        mask_nop=train_pred["mask_nop"],
        mask_pt=train_pred["mask_pt"],
    )

    # Display and save metrics
    metrics_cols = ["group", "RAE", "RSE", "Corr", "Coverage"]

    print("\nTraining data metrics (final production model):")
    train_df = pd.DataFrame([{"group": k, **v} for k, v in train_metrics.items()])
    cols_to_show = [c for c in metrics_cols if c in train_df.columns]
    print(train_df[cols_to_show].to_string(index=False))

    # Save metrics to CSV
    metrics_dir = outdir / "metrics"
    ensure_dir(metrics_dir)
    production_metrics = {
        group: {k: v for k, v in vals.items() if k != "Coverage"}
        for group, vals in train_metrics.items()
    }
    save_metrics_table(metrics_dir / "production_metrics.csv", "production", production_metrics)
    print(f"\nMetrics saved to {metrics_dir / 'production_metrics.csv'}")


if __name__ == "__main__":
    main()