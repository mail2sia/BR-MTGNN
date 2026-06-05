from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class nconv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, nodes, time], A: [nodes, nodes]
        x = torch.einsum("bcnt,nm->bcmt", (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in: int, c_out: int, bias: bool = True):
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class mixprop(nn.Module):
    def __init__(self, c_in: int, c_out: int, gdep: int, dropout: float, alpha: float):
        super().__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        adj = adj + torch.eye(adj.size(0), device=x.device, dtype=adj.dtype)
        d = adj.sum(1).clamp_min(1e-12)
        a = adj / d.view(-1, 1)
        h = x
        out = [h]
        for _ in range(self.gdep):
            h = self.alpha * x + (1.0 - self.alpha) * self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = F.dropout(ho, self.dropout, training=self.training)
        return self.mlp(ho)


class dilated_inception(nn.Module):
    def __init__(self, cin: int, cout: int, dilation_factor: int = 2):
        super().__init__()
        self.kernel_set = [2, 3, 6, 7]
        cout_each = max(1, int(cout / len(self.kernel_set)))
        self.tconv = nn.ModuleList([
            nn.Conv2d(cin, cout_each, (1, kern), dilation=(1, dilation_factor))
            for kern in self.kernel_set
        ])
        self.out_channels = cout_each * len(self.kernel_set)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = [conv(input) for conv in self.tconv]
        min_time = min(t.size(3) for t in x)
        x = [t[..., -min_time:] for t in x]
        return torch.cat(x, dim=1)


class graph_constructor(nn.Module):
    def __init__(self, nnodes: int, k: int, dim: int, device: torch.device, alpha: float = 3.0, static_feat: torch.Tensor | None = None):
        super().__init__()
        self.nnodes = nnodes
        self.k = min(k, nnodes)
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat
        self.device = device

        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))
        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))

        if self.k < self.nnodes:
            mask = torch.zeros(idx.size(0), idx.size(0), device=idx.device)
            _, topk_indices = adj.topk(self.k, dim=1)
            mask.scatter_(1, topk_indices, 1.0)
            adj = adj * mask
        return adj


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(*normalized_shape))
            self.bias = nn.Parameter(torch.zeros(*normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
        if self.elementwise_affine and idx is not None and self.weight.dim() >= 2:
            weight = self.weight[:, idx, :]
            bias = self.bias[:, idx, :]
            return F.layer_norm(input, weight.shape, weight, bias, self.eps)
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
