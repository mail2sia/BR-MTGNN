from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import graph_constructor, dilated_inception, mixprop, LayerNorm


class gtnet(nn.Module):
    def __init__(
        self,
        gcn_true: bool,
        buildA_true: bool,
        gcn_depth: int,
        num_nodes: int,
        device: torch.device,
        predefined_A: torch.Tensor | None = None,
        static_feat: torch.Tensor | None = None,
        dropout: float = 0.3,
        subgraph_size: int = 20,
        node_dim: int = 40,
        dilation_exponential: int = 2,
        conv_channels: int = 32,
        residual_channels: int = 32,
        skip_channels: int = 64,
        end_channels: int = 128,
        seq_length: int = 12,
        in_dim: int = 1,
        out_dim: int = 36,
        out_channels: int = 2,
        layers: int = 3,
        propalpha: float = 0.05,
        tanhalpha: float = 3.0,
        layer_norm_affline: bool = True,
        graph_prior_weight: float = 1.0,
    ):
        super().__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.graph_prior_weight = graph_prior_weight
        self.seq_length = seq_length
        self.layers = layers

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)

        kernel_size = 7
        if dilation_exponential > 1:
            self.receptive_field = int(1 + (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        new_dilation = 1
        for j in range(1, layers + 1):
            if dilation_exponential > 1:
                rf_size_j = int(1 + (kernel_size - 1) * (dilation_exponential ** j - 1) / (dilation_exponential - 1))
            else:
                rf_size_j = j * (kernel_size - 1) + 1

            filt = dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation)
            gate = dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation)
            actual_conv_channels = filt.out_channels
            self.filter_convs.append(filt)
            self.gate_convs.append(gate)
            self.residual_convs.append(nn.Conv2d(actual_conv_channels, residual_channels, kernel_size=(1, 1)))

            if self.seq_length > self.receptive_field:
                skip_kernel = self.seq_length - rf_size_j + 1
                norm_time = self.seq_length - rf_size_j + 1
            else:
                skip_kernel = self.receptive_field - rf_size_j + 1
                norm_time = self.receptive_field - rf_size_j + 1
            self.skip_convs.append(nn.Conv2d(actual_conv_channels, skip_channels, kernel_size=(1, skip_kernel)))

            if self.gcn_true:
                self.gconv1.append(mixprop(actual_conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                self.gconv2.append(mixprop(actual_conv_channels, residual_channels, gcn_depth, dropout, propalpha))

            self.norm.append(LayerNorm((residual_channels, num_nodes, norm_time), elementwise_affine=layer_norm_affline))
            new_dilation *= dilation_exponential

        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_dim, skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(residual_channels, skip_channels, kernel_size=(1, self.seq_length - self.receptive_field + 1), bias=True)
        else:
            self.skip0 = nn.Conv2d(in_dim, skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(residual_channels, skip_channels, kernel_size=(1, 1), bias=True)

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, out_dim, kernel_size=(1, 1), bias=True)
        # Zero init — epoch 0 is exactly persistence baseline
        nn.init.zeros_(self.end_conv_2.weight)
        nn.init.zeros_(self.end_conv_2.bias)
        self.out_dim = out_dim
        self.out_channels = out_channels
        self.idx = torch.arange(self.num_nodes, device=device)

    def _build_adj(self, idx: torch.Tensor | None, x_device: torch.device) -> torch.Tensor | None:
        if not self.gcn_true:
            return None
        if self.buildA_true:
            node_idx = self.idx if idx is None else idx
            adp = self.gc(node_idx)
            if self.predefined_A is not None:
                prior = self.predefined_A.to(x_device)
                if idx is not None:
                    prior = prior[idx][:, idx]
                adp = adp + self.graph_prior_weight * prior
            return adp
        if self.predefined_A is not None:
            A = self.predefined_A.to(x_device)
            if idx is not None:
                A = A[idx][:, idx]
            return A
        return None

    def forward(self, input: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
        # input: [batch, input_channels, nodes, seq_length]
        seq_len = input.size(3)
        if seq_len != self.seq_length:
            raise ValueError(f"input sequence length {seq_len} does not match model seq_length {self.seq_length}")
        if self.seq_length < self.receptive_field:
            input = F.pad(input, (self.receptive_field - self.seq_length, 0, 0, 0))

        adp = self._build_adj(idx, input.device)
        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))

        for i in range(self.layers):
            residual = x
            filt = torch.tanh(self.filter_convs[i](x))
            gate = torch.sigmoid(self.gate_convs[i](x))
            x = filt * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = self.skip_convs[i](x)
            skip = skip[..., -s.size(3):] + s

            if self.gcn_true and adp is not None:
                x = self.gconv1[i](x, adp) + self.gconv2[i](x, adp.transpose(1, 0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[..., -x.size(3):]
            x = self.norm[i](x, idx)

        skip = self.skipE(x) + skip[..., -self.skipE(x).size(3):]
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        delta = self.end_conv_2(x).squeeze(-1)  # [B, H*C, N]

        # Persistence residual: predict delta from last observed value per channel.
        # input shape: [B, in_channels, N, seq_len]
        # out_dim = H * C where C = self.out_channels (1 for TDB, 2 for NoM+NoP)
        B, HC, N = delta.shape
        C = self.out_channels
        H = HC // C
        delta_4d = delta.view(B, C, H, N)  # [B, C, H, N]

        # Last observed value for each channel: [B, N] → expand to [B, C, H, N]
        last_obs = input[:, :C, :, -1]           # [B, C, N]
        base = last_obs.unsqueeze(2).expand(-1, -1, H, -1)  # [B, C, H, N]

        out = base + delta_4d                     # [B, C, H, N]
        return out.view(B, HC, N)                 # [B, H*C, N]
