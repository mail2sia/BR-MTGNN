import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    scale: torch.Tensor
    adj: torch.Tensor
    num_nodes: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_csv_values(data_path: Path) -> np.ndarray:
    arr = np.genfromtxt(str(data_path), delimiter=",", skip_header=1)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if np.isnan(arr).any():
        col_mean = np.nanmean(arr, axis=0)
        inds = np.where(np.isnan(arr))
        arr[inds] = np.take(col_mean, inds[1])
    return arr.astype(np.float32)


def build_adj(data_dir: Path, num_nodes: int, device: torch.device) -> torch.Tensor:
    graph_csv = data_dir / "graph.csv"
    header_csv = data_dir / "data.csv"
    if not graph_csv.exists() or not header_csv.exists():
        return torch.eye(num_nodes, device=device)

    with open(header_csv, "r", newline="") as f:
        cols = next(csv.reader(f))
    if cols and cols[0].lower().startswith("date"):
        cols = cols[1:]
    node_to_idx = {c: i for i, c in enumerate(cols[:num_nodes])}

    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    with open(graph_csv, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            src = row[0]
            if src not in node_to_idx:
                continue
            i = node_to_idx[src]
            for dst in row[1:]:
                if dst and dst in node_to_idx:
                    j = node_to_idx[dst]
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0

    adj.fill_diagonal_(1.0)
    deg = adj.sum(dim=1, keepdim=True).clamp_min(1.0)
    adj = adj / deg
    return adj.to(device)


def make_windows(data: np.ndarray, seq_in: int, seq_out: int) -> Tuple[np.ndarray, np.ndarray]:
    n = data.shape[0]
    samples = n - seq_in - seq_out + 1
    x = np.zeros((samples, seq_in, data.shape[1]), dtype=np.float32)
    y = np.zeros((samples, seq_out, data.shape[1]), dtype=np.float32)
    for i in range(samples):
        x[i] = data[i : i + seq_in]
        y[i] = data[i + seq_in : i + seq_in + seq_out]
    return x, y


def load_data(args, device: torch.device) -> DatasetBundle:
    data_path = Path(args.data)
    data_dir = data_path.parent
    raw = load_csv_values(data_path)
    num_nodes = raw.shape[1]

    n_train = int(raw.shape[0] * 0.6)
    n_val = int(raw.shape[0] * 0.2)

    train_raw = raw[:n_train]
    val_raw = raw[n_train - args.seq_in_len - args.seq_out_len + 1 : n_train + n_val]
    test_raw = raw[n_train + n_val - args.seq_in_len - args.seq_out_len + 1 :]

    scale = np.max(np.abs(train_raw), axis=0)
    scale[scale == 0] = 1.0

    train_norm = train_raw / scale
    val_norm = val_raw / scale
    test_norm = test_raw / scale

    train_x, train_y = make_windows(train_norm, args.seq_in_len, args.seq_out_len)
    val_x, val_y = make_windows(val_norm, args.seq_in_len, args.seq_out_len)
    test_x, test_y = make_windows(test_norm, args.seq_in_len, args.seq_out_len)

    train_x_t = torch.tensor(train_x, dtype=torch.float32, device=device)
    train_y_t = torch.tensor(train_y, dtype=torch.float32, device=device)
    val_x_t = torch.tensor(val_x, dtype=torch.float32, device=device)
    val_y_t = torch.tensor(val_y, dtype=torch.float32, device=device)
    test_x_t = torch.tensor(test_x, dtype=torch.float32, device=device)
    test_y_t = torch.tensor(test_y, dtype=torch.float32, device=device)

    train_loader = DataLoader(TensorDataset(train_x_t, train_y_t), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x_t, val_y_t), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_x_t, test_y_t), batch_size=args.batch_size, shuffle=False)

    adj = build_adj(data_dir, num_nodes, device)
    scale_t = torch.tensor(scale, dtype=torch.float32, device=device)

    return DatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_x=train_x_t,
        train_y=train_y_t,
        val_x=val_x_t,
        val_y=val_y_t,
        test_x=test_x_t,
        test_y=test_y_t,
        scale=scale_t,
        adj=adj,
        num_nodes=num_nodes,
    )


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    diff = pred - target
    denom_rse = torch.sqrt(torch.sum((target - target.mean()) ** 2)).clamp_min(1e-8)
    denom_rae = torch.sum(torch.abs(target - target.mean())).clamp_min(1e-8)

    rse = (torch.sqrt(torch.sum(diff**2)) / denom_rse).item()
    rae = (torch.sum(torch.abs(diff)) / denom_rae).item()

    pred_f = pred.reshape(-1, pred.shape[-1])
    target_f = target.reshape(-1, target.shape[-1])
    pred_c = pred_f - pred_f.mean(dim=0, keepdim=True)
    target_c = target_f - target_f.mean(dim=0, keepdim=True)
    cov = (pred_c * target_c).mean(dim=0)
    corr = cov / (pred_c.std(dim=0) * target_c.std(dim=0) + 1e-8)
    corr = corr[torch.isfinite(corr)]
    corr_v = corr.mean().item() if corr.numel() > 0 else 0.0
    corr_v = max(0.0, corr_v)

    smape = (2.0 * torch.abs(diff) / (torch.abs(pred) + torch.abs(target) + 1e-8)).mean().item()

    return {"RSE": rse, "RAE": rae, "Corr": corr_v, "sMAPE": smape}


def corr_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = pred.reshape(-1, pred.shape[-1])
    t = target.reshape(-1, target.shape[-1])
    p_c = p - p.mean(dim=0, keepdim=True)
    t_c = t - t.mean(dim=0, keepdim=True)
    denom = torch.sqrt(torch.sum(p_c * p_c, dim=0) * torch.sum(t_c * t_c, dim=0) + 1e-8)
    corr = torch.sum(p_c * t_c, dim=0) / denom
    corr = corr[torch.isfinite(corr)]
    if corr.numel() == 0:
        return pred.new_tensor(1.0)
    return 1.0 - corr.mean()


def evaluate(model: nn.Module, loader: DataLoader, scale: torch.Tensor, blend: float = 0.0) -> Tuple[float, Dict[str, float]]:
    model.eval()
    loss_fn = nn.L1Loss()
    total_loss = 0.0
    n = 0
    preds = []
    targs = []

    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            if blend > 0.0:
                base = x[:, -1:, :].repeat(1, out.shape[1], 1)
                out = (1.0 - blend) * out + blend * base
            out_s = out * scale.view(1, 1, -1)
            y_s = y * scale.view(1, 1, -1)
            loss = loss_fn(out_s, y_s)
            b = x.shape[0]
            total_loss += loss.item() * b
            n += b
            preds.append(out_s)
            targs.append(y_s)

    pred = torch.cat(preds, dim=0)
    targ = torch.cat(targs, dim=0)
    metrics = compute_metrics(pred, targ)
    return total_loss / max(n, 1), metrics


def train_model(args, model: nn.Module, data: DatasetBundle, device: torch.device) -> Dict[str, float]:
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=args.lr_decay_factor,
        patience=max(1, args.lr_patience_sched),
        min_lr=args.lr_min,
    ) if args.lr_schedule == "plateau" else None

    loss_fn = nn.L1Loss()
    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in data.train_loader:
            opt.zero_grad()
            out = model(x)
            out_s = out * data.scale.view(1, 1, -1)
            y_s = y * data.scale.view(1, 1, -1)
            loss = loss_fn(out_s, y_s)
            if args.corr_lambda > 0:
                loss = loss + args.corr_lambda * corr_loss(out_s, y_s)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        val_loss, val_metrics = evaluate(model, data.val_loader, data.scale, blend=0.0)
        if scheduler is not None:
            scheduler.step(val_loss)

        print(
            f"| end of epoch {epoch:3d} | valid rse {val_metrics['RSE']:.4f} | "
            f"valid rae {val_metrics['RAE']:.4f} | valid corr {val_metrics['Corr']:.4f} | "
            f"valid smape {val_metrics['sMAPE']:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    best_blend = 0.0
    best_blend_score = -1e9
    for b in [0.0, 0.2, 0.4, 0.6]:
        _, vm = evaluate(model, data.val_loader, data.scale, blend=b)
        score = vm["Corr"] - 0.02 * vm["RAE"]
        if score > best_blend_score:
            best_blend_score = score
            best_blend = b

    test_loss, test_metrics = evaluate(model, data.test_loader, data.scale, blend=best_blend)
    print(
        f"final test rse {test_metrics['RSE']:.4f} | test rae {test_metrics['RAE']:.4f} | "
        f"test corr {test_metrics['Corr']:.4f} | test smape {test_metrics['sMAPE']:.4f}"
    )
    print(f"selected blend {best_blend:.2f}")
    print("test\trse\trae\tcorr\ts-mape")
    print(
        f"mean\t{test_metrics['RSE']:.4f}\t{test_metrics['RAE']:.4f}\t"
        f"{test_metrics['Corr']:.4f}\t{test_metrics['sMAPE']:.4f}"
    )
    return {**test_metrics, "loss": test_loss}


def base_arg_parser(name: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"{name} baseline")
    p.add_argument("--data", type=str, default="./data/sm_data.csv")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seq_in_len", type=int, default=10)
    p.add_argument("--seq_out_len", type=int, default=36)
    p.add_argument("--num_nodes", type=int, default=190)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr_schedule", type=str, default="none", choices=["none", "plateau"])
    p.add_argument("--lr_patience_sched", type=int, default=10)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--lr_decay_factor", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--corr_lambda", type=float, default=0.2)
    return p


def resolve_device(device_arg: str) -> torch.device:
    d = device_arg.lower()
    if d.startswith("cuda") and torch.cuda.is_available():
        return torch.device(d)
    return torch.device("cpu")
