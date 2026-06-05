import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    scale: torch.Tensor
    num_nodes: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_csv(path: Path) -> np.ndarray:
    arr = np.genfromtxt(str(path), delimiter=",", skip_header=1)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if np.isnan(arr).any():
        m = np.nanmean(arr, axis=0)
        inds = np.where(np.isnan(arr))
        arr[inds] = np.take(m, inds[1])
    return arr.astype(np.float32)


def _make_windows(data: np.ndarray, seq_in: int, seq_out: int):
    n = data.shape[0]
    samples = n - seq_in - seq_out + 1
    x = np.zeros((samples, seq_in, data.shape[1]), dtype=np.float32)
    y = np.zeros((samples, seq_out, data.shape[1]), dtype=np.float32)
    for i in range(samples):
        x[i] = data[i : i + seq_in]
        y[i] = data[i + seq_in : i + seq_in + seq_out]
    return x, y


def load_data(args, device: torch.device) -> DataBundle:
    raw = _load_csv(Path(args.data))
    n = raw.shape[0]
    num_nodes = raw.shape[1]

    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    n_train = max(n_train, args.seq_in_len + args.seq_out_len + 1)
    n_val = max(n_val, args.seq_in_len + args.seq_out_len + 1)
    n_train = min(n_train, n - 2)
    n_val = min(n_val, n - n_train - 1)

    train_raw = raw[:n_train]
    val_raw = raw[n_train - args.seq_in_len - args.seq_out_len + 1 : n_train + n_val]
    test_raw = raw[n_train + n_val - args.seq_in_len - args.seq_out_len + 1 :]

    scale = np.max(np.abs(train_raw), axis=0)
    scale[scale == 0] = 1.0

    train_x, train_y = _make_windows(train_raw / scale, args.seq_in_len, args.seq_out_len)
    val_x, val_y = _make_windows(val_raw / scale, args.seq_in_len, args.seq_out_len)
    test_x, test_y = _make_windows(test_raw / scale, args.seq_in_len, args.seq_out_len)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(train_x, dtype=torch.float32, device=device),
            torch.tensor(train_y, dtype=torch.float32, device=device),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(val_x, dtype=torch.float32, device=device),
            torch.tensor(val_y, dtype=torch.float32, device=device),
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(test_x, dtype=torch.float32, device=device),
            torch.tensor(test_y, dtype=torch.float32, device=device),
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scale=torch.tensor(scale, dtype=torch.float32, device=device),
        num_nodes=num_nodes,
    )


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    diff = pred - target

    denom_rse = torch.sqrt(torch.sum((target - target.mean()) ** 2)).clamp_min(1e-8)
    denom_rae = torch.sum(torch.abs(target - target.mean())).clamp_min(1e-8)

    rse = (torch.sqrt(torch.sum(diff ** 2)) / denom_rse).item()
    rae = (torch.sum(torch.abs(diff)) / denom_rae).item()

    p = pred.reshape(-1, pred.shape[-1]).detach().cpu().numpy()
    t = target.reshape(-1, target.shape[-1]).detach().cpu().numpy()
    p_std = p.std(axis=0)
    t_std = t.std(axis=0)
    mask = (p_std > 0) & (t_std > 0)
    if np.any(mask):
        p_sel = p[:, mask]
        t_sel = t[:, mask]
        p_c = p_sel - p_sel.mean(axis=0, keepdims=True)
        t_c = t_sel - t_sel.mean(axis=0, keepdims=True)
        corr_vec = (p_c * t_c).mean(axis=0) / (p_std[mask] * t_std[mask] + 1e-8)
        corr_vec = corr_vec[np.isfinite(corr_vec)]
        corr = float(corr_vec.mean()) if corr_vec.size > 0 else 0.0
    else:
        corr = 0.0
    corr = max(0.0, corr)

    return {"RSE": rse, "RAE": rae, "Corr": corr}


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


class LSTMForecaster(nn.Module):
    def __init__(self, num_nodes: int, seq_out: int, hidden_dim: int, layers: int, dropout: float, univariate: bool):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_out = seq_out
        self.univariate = univariate

        in_dim = 1 if univariate else num_nodes
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers=layers, dropout=dropout if layers > 1 else 0.0, batch_first=True)
        self.head = nn.Linear(hidden_dim, seq_out if univariate else seq_out * num_nodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,N]
        if self.univariate:
            b, t, n = x.shape
            xx = x.permute(0, 2, 1).reshape(b * n, t, 1)
            z, _ = self.lstm(xx)
            delta = self.head(z[:, -1, :]).reshape(b, n, self.seq_out).transpose(1, 2)
            base = x[:, -1:, :].repeat(1, self.seq_out, 1)
            return base + delta

        z, _ = self.lstm(x)
        delta = self.head(z[:, -1, :]).reshape(x.shape[0], self.seq_out, self.num_nodes)
        base = x[:, -1:, :].repeat(1, self.seq_out, 1)
        return base + delta


class TransformerForecaster(nn.Module):
    def __init__(self, num_nodes: int, seq_out: int, d_model: int, nhead: int, layers: int, dropout: float, univariate: bool):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_out = seq_out
        self.univariate = univariate

        in_dim = 1 if univariate else num_nodes
        self.in_proj = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Linear(d_model, seq_out if univariate else seq_out * num_nodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.univariate:
            b, t, n = x.shape
            xx = x.permute(0, 2, 1).reshape(b * n, t, 1)
            z = self.encoder(self.in_proj(xx))
            delta = self.head(z[:, -1, :]).reshape(b, n, self.seq_out).transpose(1, 2)
            base = x[:, -1:, :].repeat(1, self.seq_out, 1)
            return base + delta

        z = self.encoder(self.in_proj(x))
        delta = self.head(z[:, -1, :]).reshape(x.shape[0], self.seq_out, self.num_nodes)
        base = x[:, -1:, :].repeat(1, self.seq_out, 1)
        return base + delta


def evaluate_model(model: nn.Module, loader: DataLoader, scale: torch.Tensor, blend: float = 0.0):
    model.eval()
    loss_fn = nn.L1Loss()
    total = 0.0
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
            total += loss.item() * b
            n += b
            preds.append(out_s)
            targs.append(y_s)

    pred = torch.cat(preds, dim=0)
    targ = torch.cat(targs, dim=0)
    metrics = compute_metrics(pred, targ)
    return total / max(n, 1), metrics


def fit_once(args, model: nn.Module, data: DataBundle):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = None
    if args.lr_schedule == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=args.lr_decay_factor, patience=max(1, args.lr_patience_sched), min_lr=args.lr_min
        )

    loss_fn = nn.L1Loss()
    best_state = None
    best_val = float("inf")
    bad = 0

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

        val_loss, val_metrics = evaluate_model(model, data.val_loader, data.scale, blend=0.0)
        print(
            f"| end of epoch {epoch:3d} | valid rse {val_metrics['RSE']:.4f} | "
            f"valid rae {val_metrics['RAE']:.4f} | valid corr {val_metrics['Corr']:.4f}"
        )

        if sched is not None:
            sched.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    best_blend = 0.0
    best_blend_score = -1e9
    for b in [0.0, 0.2, 0.4, 0.6]:
        _, vm = evaluate_model(model, data.val_loader, data.scale, blend=b)
        score = vm["Corr"] - 0.02 * vm["RAE"]
        if score > best_blend_score:
            best_blend_score = score
            best_blend = b

    _, test_metrics = evaluate_model(model, data.test_loader, data.scale, blend=best_blend)
    print(
        f"final test rse {test_metrics['RSE']:.4f} | test rae {test_metrics['RAE']:.4f} | "
        f"test corr {test_metrics['Corr']:.4f}"
    )
    print(f"selected blend {best_blend:.2f}")
    print("test\trse\trae\tcorr")
    print(f"mean\t{test_metrics['RSE']:.4f}\t{test_metrics['RAE']:.4f}\t{test_metrics['Corr']:.4f}")
    return test_metrics, best_val


def build_model(args, num_nodes: int, family: str, univariate: bool):
    if family == "lstm":
        return LSTMForecaster(
            num_nodes=num_nodes,
            seq_out=args.seq_out_len,
            hidden_dim=args.hidden_dim,
            layers=args.layers,
            dropout=args.dropout,
            univariate=univariate,
        )
    return TransformerForecaster(
        num_nodes=num_nodes,
        seq_out=args.seq_out_len,
        d_model=args.d_model,
        nhead=args.nhead,
        layers=args.layers,
        dropout=args.dropout,
        univariate=univariate,
    )


def run_baseline(family: str, univariate: bool):
    p = argparse.ArgumentParser(description=f"{family} {'U' if univariate else 'M'} baseline")
    p.add_argument("--data", type=str, default="./data/sm_data.csv")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seq_in_len", type=int, default=10)
    p.add_argument("--seq_out_len", type=int, default=36)
    p.add_argument("--num_nodes", type=int, default=190)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr_schedule", type=str, default="none", choices=["none", "plateau"])
    p.add_argument("--lr_patience_sched", type=int, default=5)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--lr_decay_factor", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_ratio", type=float, default=0.75)
    p.add_argument("--val_ratio", type=float, default=0.10)
    p.add_argument("--trials", type=int, default=4)

    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--corr_lambda", type=float, default=0.2)

    args = p.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")

    data = load_data(args, device)
    if data.num_nodes != args.num_nodes:
        print(f"Overriding --num_nodes={args.num_nodes} with dataset node count={data.num_nodes}")
        args.num_nodes = data.num_nodes

    best_metrics = None
    best_val = float("inf")

    if family == "lstm":
        hidden_space = [64, 128, 256]
        layer_space = [1, 2, 3]
        drop_space = [0.0, 0.1, 0.2]
    else:
        d_space = [64, 128, 192]
        head_space = [4, 8]
        layer_space = [1, 2, 3]
        drop_space = [0.0, 0.1, 0.2]

    for trial in range(1, args.trials + 1):
        print(f"trial: {trial}")
        if family == "lstm":
            args.hidden_dim = random.choice(hidden_space)
            args.layers = random.choice(layer_space)
            args.dropout = random.choice(drop_space)
        else:
            args.d_model = random.choice(d_space)
            args.nhead = random.choice(head_space)
            if args.d_model % args.nhead != 0:
                args.nhead = 4
            args.layers = random.choice(layer_space)
            args.dropout = random.choice(drop_space)

        model = build_model(args, data.num_nodes, family, univariate).to(device)
        metrics, val = fit_once(args, model, data)
        if val < best_val:
            best_val = val
            best_metrics = metrics

    if best_metrics is None:
        raise RuntimeError("No successful trial")

    print("\n\n1 run average\n")
    print("test\trse\trae\tcorr")
    print(f"mean\t{best_metrics['RSE']:.4f}\t{best_metrics['RAE']:.4f}\t{best_metrics['Corr']:.4f}")
