import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        x = torch.einsum("ij,bjc->bic", adj, x)
        return self.lin(x)


class DCRNNCell(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate = GraphConv(in_dim + hidden_dim, 2 * hidden_dim)
        self.cand = GraphConv(in_dim + hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, h], dim=-1)
        z_r = torch.sigmoid(self.gate(inp, adj))
        z, r = torch.chunk(z_r, 2, dim=-1)
        hc = torch.tanh(self.cand(torch.cat([x, r * h], dim=-1), adj))
        return (1 - z) * h + z * hc


class DCRNNModel(nn.Module):
    def __init__(self, num_nodes: int, seq_in: int, seq_out: int, adj: torch.Tensor, hidden: int = 64):
        super().__init__()
        self.seq_out = seq_out
        self.hidden = hidden
        self.adj = adj
        self.encoder = DCRNNCell(1, hidden)
        self.proj = nn.Linear(hidden, seq_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,N]
        b, t, n = x.shape
        h = torch.zeros((b, n, self.hidden), device=x.device)
        for i in range(t):
            xt = x[:, i, :].unsqueeze(-1)
            h = self.encoder(xt, h, self.adj)
        out = self.proj(h)  # [B,N,H]
        out = out.transpose(1, 2)  # [B,seq_out,N]
        base = x[:, -1:, :].repeat(1, self.seq_out, 1)
        return out + base


class AGCRNCell(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_nodes: int, emb_dim: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_emb = nn.Parameter(torch.randn(num_nodes, emb_dim))
        self.gate = nn.Linear(in_dim + hidden_dim, 2 * hidden_dim)
        self.cand = nn.Linear(in_dim + hidden_dim, hidden_dim)

    def adaptive_adj(self) -> torch.Tensor:
        a = torch.relu(torch.mm(self.node_emb, self.node_emb.t()))
        return torch.softmax(a, dim=1)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        adj = self.adaptive_adj()
        xg = torch.einsum("ij,bjc->bic", adj, x)
        hg = torch.einsum("ij,bjc->bic", adj, h)
        inp = torch.cat([xg, hg], dim=-1)
        z_r = torch.sigmoid(self.gate(inp))
        z, r = torch.chunk(z_r, 2, dim=-1)
        hc = torch.tanh(self.cand(torch.cat([xg, r * hg], dim=-1)))
        return (1 - z) * h + z * hc


class AGCRNModel(nn.Module):
    def __init__(self, num_nodes: int, seq_in: int, seq_out: int, hidden: int = 64):
        super().__init__()
        self.seq_out = seq_out
        self.hidden = hidden
        self.cell = AGCRNCell(1, hidden, num_nodes)
        self.proj = nn.Linear(hidden, seq_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, n = x.shape
        h = torch.zeros((b, n, self.hidden), device=x.device)
        for i in range(t):
            xt = x[:, i, :].unsqueeze(-1)
            h = self.cell(xt, h)
        out = self.proj(h)
        return out.transpose(1, 2)


class PatchTSTModel(nn.Module):
    def __init__(self, num_nodes: int, seq_in: int, seq_out: int, d_model: int = 128, patch_len: int = 4):
        super().__init__()
        self.seq_out = seq_out
        self.patch_len = patch_len
        self.patch_stride = patch_len
        self.num_patches = max(1, (seq_in - patch_len) // self.patch_stride + 1)
        self.embed = nn.Linear(patch_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=256, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.head = nn.Linear(d_model * self.num_patches, seq_out)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        b, t, n = x.shape
        patches = []
        for s in range(0, t - self.patch_len + 1, self.patch_stride):
            patches.append(x[:, s : s + self.patch_len, :])
        if not patches:
            pad = self.patch_len - t
            xx = F.pad(x, (0, 0, pad, 0))
            patches = [xx]
        p = torch.stack(patches, dim=1)  # [B,P,L,N]
        return p.permute(0, 3, 1, 2)  # [B,N,P,L]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        p = self._patchify(x)
        b, n, pn, pl = p.shape
        p = p.reshape(b * n, pn, pl)
        tok = self.embed(p)
        z = self.encoder(tok)
        z = z.reshape(b * n, -1)
        out = self.head(z).reshape(b, n, self.seq_out)
        out = out.transpose(1, 2)  # [B,seq_out,N]
        base = x_in[:, -1:, :].repeat(1, self.seq_out, 1)
        return out + base


# ---------------------------------------------------------------------------
# TimesFM 1.0 — real pretrained backbone + fine-tuned linear head
# Checkpoint: google/timesfm-1.0-200m-pytorch  (torch_model.ckpt)
# Architecture reconstructed from the published state_dict keys/shapes.
# ---------------------------------------------------------------------------

class _RMSNorm(nn.Module):
    """RMSNorm as used in the TimesFM checkpoint (weight-only, no bias)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class _ResidualBlock(nn.Module):
    """Two-layer MLP with a linear residual skip connection (matches TimesFM tokeniser/horizon layers)."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.hidden_layer = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU())
        self.output_layer = nn.Linear(hidden_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.hidden_layer(x)) + self.residual_layer(x)


class _TimesFMAttention(nn.Module):
    """Multi-head attention matching the checkpoint layout exactly."""
    def __init__(self, d_model: int = 1280, num_heads: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scaling = nn.Parameter(torch.ones(self.head_dim))
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.o_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        scale = self.scaling.abs().unsqueeze(0).unsqueeze(0)  # [1,1,head_dim]

        qkv = self.qkv_proj(x)                               # [B, T, 3C]
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                          # each [B,T,H,D]
        q = q * scale
        q = q.transpose(1, 2)                                 # [B,H,T,D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)                           # [B,H,T,D]
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.o_proj(out)


class _TimesFMLayer(nn.Module):
    """Single transformer layer matching the checkpoint's per-layer keys."""
    def __init__(self, d_model: int = 1280):
        super().__init__()
        self.input_layernorm = _RMSNorm(d_model)
        self.self_attn = _TimesFMAttention(d_model)
        self.mlp = _TimesFMMLP(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(x)
        return x


class _TimesFMMLP(nn.Module):
    """Gated MLP matching checkpoint keys gate_proj / down_proj / layer_norm."""
    def __init__(self, d_model: int = 1280):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)
        self.down_proj = nn.Linear(d_model, d_model, bias=True)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(self.layer_norm(x)))
        return self.down_proj(gate)


class _TimesFMBackbone(nn.Module):
    """
    Reconstruction of the TimesFM 1.0 backbone whose weights are loaded
    from google/timesfm-1.0-200m-pytorch  (torch_model.ckpt).

    Hyper-parameters fixed by the checkpoint:
      patch_len  = 32   (input patch)
      d_model    = 1280
      num_heads  = 16   (head_dim = 80)
      num_layers = 20
    """
    PATCH_LEN = 32
    D_MODEL = 1280
    CKPT_PATH = (
        "/var/tmp/sahsan03/huggingface/hub/"
        "models--google--timesfm-1.0-200m-pytorch/snapshots/"
        "0581e2c56cb06feb51cfd98fc2b4005b74f7187b/torch_model.ckpt"
    )

    def __init__(self):
        super().__init__()
        # input_ff_layer: [patch_val, patch_mask] concatenated → 64 dims
        self.input_ff_layer = _ResidualBlock(64, self.D_MODEL, self.D_MODEL)
        self.freq_emb = nn.Embedding(3, self.D_MODEL)
        self.stacked_transformer = nn.ModuleDict({
            "layers": nn.ModuleList([_TimesFMLayer(self.D_MODEL) for _ in range(20)])
        })
        # horizon_ff_layer is part of the original decoding head; we do not
        # use it — our task-specific linear head replaces it.

    def _patchify(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split [B*N, T] into non-overlapping patches of length 32, with mask."""
        BN, T = x.shape
        pad = (self.PATCH_LEN - T % self.PATCH_LEN) % self.PATCH_LEN
        if pad:
            x = F.pad(x, (pad, 0))          # left-pad to multiple of PATCH_LEN
        num_patches = x.shape[1] // self.PATCH_LEN
        patches = x.reshape(BN, num_patches, self.PATCH_LEN)
        # Mask: 0 = real, 1 = padding (only the first patch may be padded)
        mask = torch.zeros_like(patches)
        if pad:
            mask[:, 0, :pad] = 1.0
        return patches, mask

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B*N, T]  (univariate series per variable)
        Returns: [B*N, num_patches, D_MODEL]
        """
        patches, mask = self._patchify(x)         # [BN, P, 32] each
        tokens = torch.cat([patches, mask], dim=-1)  # [BN, P, 64]
        tokens = self.input_ff_layer(tokens)          # [BN, P, 1280]
        # freq_emb index 0 = low freq (default); added to every patch token
        freq = self.freq_emb(torch.zeros(1, dtype=torch.long, device=x.device))
        tokens = tokens + freq.unsqueeze(0)           # broadcast over BN and P
        for layer in self.stacked_transformer["layers"]:
            tokens = layer(tokens)
        return tokens                                 # [BN, P, 1280]

    @classmethod
    def from_pretrained(cls, device: torch.device) -> "_TimesFMBackbone":
        model = cls()
        ckpt = torch.load(cls.CKPT_PATH, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        # horizon_ff_layer keys will be in `unexpected` — that is expected.
        backbone_missing = [k for k in missing if not k.startswith("horizon_ff_layer")]
        if backbone_missing:
            raise RuntimeError(f"Missing backbone keys: {backbone_missing}")
        return model.to(device)


class TimesFMModel(nn.Module):
    """
    TimesFM 1.0 pretrained backbone (frozen) + fine-tuned linear head.

    The backbone encodes each univariate channel independently into patch
    embeddings (D_MODEL=1280) using the pretrained TimesFM weights.  A
    task-specific linear head is then trained from scratch to map the
    pooled patch representation to the forecast horizon for all channels.

    Only the head is updated during training; the backbone stays frozen.
    """

    CKPT_PATH = _TimesFMBackbone.CKPT_PATH

    def __init__(self, num_nodes: int, seq_in: int, seq_out: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_out = seq_out
        self.seq_in = seq_in

        self.backbone = _TimesFMBackbone()

        # Compute how many patches the backbone will produce for seq_in steps.
        patch_len = _TimesFMBackbone.PATCH_LEN
        num_patches = math.ceil(seq_in / patch_len)
        d_model = _TimesFMBackbone.D_MODEL

        # Fine-tuning head: mean-pool patch dim → linear → seq_out
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, seq_out),
        )

    def load_pretrained(self, device: torch.device) -> None:
        ckpt = torch.load(self.CKPT_PATH, map_location="cpu", weights_only=False)
        missing, _ = self.backbone.load_state_dict(ckpt, strict=False)
        backbone_missing = [k for k in missing if not k.startswith("horizon_ff_layer")]
        if backbone_missing:
            raise RuntimeError(f"Missing backbone keys: {backbone_missing}")
        # Freeze backbone — only the head is updated.
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N]
        b, t, n = x.shape
        # Treat each variable independently.
        xbn = x.permute(0, 2, 1).reshape(b * n, t)       # [B*N, T]
        tokens = self.backbone.encode(xbn)                 # [B*N, P, D]
        pooled = tokens.mean(dim=1)                        # [B*N, D]
        out = self.head(pooled)                            # [B*N, seq_out]
        out = out.reshape(b, n, self.seq_out).transpose(1, 2)  # [B, seq_out, N]
        base = x[:, -1:, :].repeat(1, self.seq_out, 1)
        return out + base


# ---------------------------------------------------------------------------
# Temporal Fusion Transformer — full architecture (Lim et al., 2021)
# https://arxiv.org/abs/1912.09363
# Adapted to the multivariate setting of this benchmark:
#   - No categorical or static inputs (all N channels are treated as
#     observed time-varying continuous inputs, matching the data format
#     used by every other baseline in this suite).
#   - seq_in  = encoder steps (historical context)
#   - seq_out = decoder steps (forecast horizon, fed as known-future zeros)
# ---------------------------------------------------------------------------

class _GRN(nn.Module):
    """Gated Residual Network (Section 3.2 of the TFT paper).

    η₁ = LayerNorm(skip(x) + GLU(η₂))
    η₂ = Dense(ELU(Dense(x) [+ Dense(context)]))
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int | None = None,
                 context_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        out_dim = out_dim or hidden
        self.skip = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.ctx = nn.Linear(context_dim, hidden, bias=False) if context_dim else None
        self.fc2 = nn.Linear(hidden, hidden)
        # GLU: two parallel linears on η₂ → one is sigmoid gate
        self.gate_fc = nn.Linear(hidden, out_dim)
        self.gate_sig = nn.Linear(hidden, out_dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor,
                context: torch.Tensor | None = None) -> torch.Tensor:
        skip = self.skip(x)
        h = self.fc1(x)
        if self.ctx is not None and context is not None:
            h = h + self.ctx(context)
        h = F.elu(h)
        h = self.drop(self.fc2(h))
        gate = torch.sigmoid(self.gate_sig(h))
        h = self.gate_fc(h) * gate          # GLU
        return self.norm(skip + h)


class _VSN(nn.Module):
    """Variable Selection Network (Section 4.1 of the TFT paper).

    Produces a soft weighted combination of per-variable GRN embeddings.
    All N input channels are treated as continuous time-varying inputs.
    """

    def __init__(self, num_vars: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        # One GRN per variable to produce per-variable representation
        self.var_grns = nn.ModuleList([_GRN(hidden, hidden, dropout=dropout)
                                       for _ in range(num_vars)])
        # Flat GRN to produce variable-selection weights from all embeddings
        self.weight_grn = _GRN(num_vars * hidden, hidden, out_dim=num_vars,
                                context_dim=hidden, dropout=dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor,
                context: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B, T, num_vars, hidden]  →  [B, T, hidden]"""
        B, T, V, H = x.shape
        # Per-variable transformation
        var_outs = torch.stack([self.var_grns[i](x[:, :, i, :])
                                for i in range(V)], dim=2)   # [B,T,V,H]
        # Selection weights
        flat = x.reshape(B, T, V * H)
        ctx = context.unsqueeze(1).expand(-1, T, -1) if context is not None else None
        weights = self.softmax(self.weight_grn(flat, ctx))    # [B,T,V]
        # Weighted sum
        out = (var_outs * weights.unsqueeze(-1)).sum(dim=2)   # [B,T,H]
        return out


class _InterpretableMHA(nn.Module):
    """Interpretable Multi-Head Attention (Section 3.3 of the TFT paper).

    Separate Q/K projections per head, but a SINGLE shared V projection
    across all heads — this is what makes it interpretable.
    Outputs are averaged across heads before the final linear projection.
    """

    def __init__(self, hidden: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden % num_heads == 0
        self.num_heads = num_heads
        self.d_k = hidden // num_heads
        self.q_projs = nn.ModuleList([nn.Linear(hidden, self.d_k, bias=False)
                                      for _ in range(num_heads)])
        self.k_projs = nn.ModuleList([nn.Linear(hidden, self.d_k, bias=False)
                                      for _ in range(num_heads)])
        # Single shared value projection across all heads
        self.v_proj = nn.Linear(hidden, self.d_k, bias=False)
        self.out_proj = nn.Linear(hidden, hidden, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B, T, H],  mask: [T, T] causal mask (True = masked)"""
        heads = []
        for q_proj, k_proj in zip(self.q_projs, self.k_projs):
            q = q_proj(x)                                         # [B,T,d_k]
            k = k_proj(x)
            v = self.v_proj(x)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
            attn = F.softmax(scores, dim=-1)
            attn = self.drop(attn)
            heads.append(torch.matmul(attn, v))                   # [B,T,d_k]
        # Average across heads → [B, T, num_heads*d_k] = [B, T, hidden]
        out = torch.stack(heads, dim=0).mean(dim=0)
        # Repeat d_k num_heads times to recover full hidden dim before proj
        out = out.unsqueeze(2).expand(-1, -1, self.num_heads, -1).reshape(
            out.shape[0], out.shape[1], self.num_heads * self.d_k)
        return self.out_proj(out)


def _gate_and_norm(x: torch.Tensor, residual: torch.Tensor,
                   gate: nn.Linear, sig: nn.Linear,
                   norm: nn.LayerNorm, drop: nn.Dropout) -> torch.Tensor:
    """Gated skip connection: LayerNorm(residual + GLU(x))."""
    g = torch.sigmoid(sig(drop(x))) * gate(drop(x))
    return norm(residual + g)


class TFTModel(nn.Module):
    """
    Temporal Fusion Transformer — full architecture (Lim et al., 2021).

    Encoder-decoder structure:
      1. Input projection: each of N continuous channels → hidden dim
      2. Variable Selection Network on encoder inputs (with static context)
      3. LSTM encoder + LSTM decoder (decoder receives zero future inputs,
         matching the unknown-future setting of this benchmark)
      4. Gated skip connection + static enrichment via GRN
      5. Interpretable Multi-Head Attention with causal mask
      6. Position-wise GRN + final gated skip connection
      7. Linear output head → [B, seq_out, N]
    """

    def __init__(self, num_nodes: int, seq_in: int, seq_out: int,
                 hidden: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_in = seq_in
        self.seq_out = seq_out
        self.hidden = hidden

        # 1. Per-channel input projection (continuous embedding)
        self.input_proj = nn.Linear(1, hidden)

        # 2. Variable Selection Networks
        self.enc_vsn = _VSN(num_nodes, hidden, dropout)
        self.dec_vsn = _VSN(num_nodes, hidden, dropout)

        # 3. LSTM encoder / decoder
        self.enc_lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.dec_lstm = nn.LSTM(hidden, hidden, batch_first=True)

        # Gated skip after LSTM
        self.lstm_gate = nn.Linear(hidden, hidden)
        self.lstm_sig  = nn.Linear(hidden, hidden)
        self.lstm_norm = nn.LayerNorm(hidden)
        self.lstm_drop = nn.Dropout(dropout)

        # 4. Static enrichment GRN (no static inputs → context = None)
        self.enrich_grn = _GRN(hidden, hidden, dropout=dropout)

        # 5. Interpretable MHA
        self.mha = _InterpretableMHA(hidden, num_heads, dropout)
        self.mha_gate = nn.Linear(hidden, hidden)
        self.mha_sig  = nn.Linear(hidden, hidden)
        self.mha_norm = nn.LayerNorm(hidden)
        self.mha_drop = nn.Dropout(dropout)

        # 6. Position-wise GRN
        self.pw_grn = _GRN(hidden, hidden, dropout=dropout)
        self.out_gate = nn.Linear(hidden, hidden)
        self.out_sig  = nn.Linear(hidden, hidden)
        self.out_norm = nn.LayerNorm(hidden)
        self.out_drop = nn.Dropout(dropout)

        # 7. Output projection
        self.head = nn.Linear(hidden, num_nodes)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask (True = positions to block)."""
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_in, N]
        B, T_enc, N = x.shape
        T_dec = self.seq_out

        # --- 1. Project each channel to hidden dim ---
        # enc: [B, T_enc, N, hidden]
        enc_emb = self.input_proj(x.unsqueeze(-1))
        # decoder receives zeros (unknown future)
        dec_inp = torch.zeros(B, T_dec, N, device=x.device)
        dec_emb = self.input_proj(dec_inp.unsqueeze(-1))

        # --- 2. Variable Selection ---
        enc_vsn = self.enc_vsn(enc_emb)          # [B, T_enc, hidden]
        dec_vsn = self.dec_vsn(dec_emb)          # [B, T_dec, hidden]

        # --- 3. LSTM sequence-to-sequence ---
        enc_out, (h, c) = self.enc_lstm(enc_vsn)
        dec_out, _ = self.dec_lstm(dec_vsn, (h, c))

        lstm_out = torch.cat([enc_out, dec_out], dim=1)        # [B, T_enc+T_dec, hidden]
        vsn_all  = torch.cat([enc_vsn, dec_vsn], dim=1)

        # Gated skip connection
        lstm_out = _gate_and_norm(lstm_out, vsn_all,
                                  self.lstm_gate, self.lstm_sig,
                                  self.lstm_norm, self.lstm_drop)

        # --- 4. Static enrichment (no static covariates → plain GRN) ---
        enriched = self.enrich_grn(lstm_out)                   # [B, T_total, hidden]

        # --- 5. Interpretable Multi-Head Self-Attention with causal mask ---
        T_total = T_enc + T_dec
        mask = self._causal_mask(T_total, x.device)
        attn_out = self.mha(enriched, mask)

        attn_out = _gate_and_norm(attn_out, enriched,
                                  self.mha_gate, self.mha_sig,
                                  self.mha_norm, self.mha_drop)

        # --- 6. Position-wise GRN + final gated skip ---
        pw = self.pw_grn(attn_out)
        out = _gate_and_norm(pw, lstm_out,
                             self.out_gate, self.out_sig,
                             self.out_norm, self.out_drop)

        # --- 7. Output: take decoder steps only, project to N channels ---
        out = out[:, T_enc:, :]                                # [B, T_dec, hidden]
        return self.head(out)                                  # [B, seq_out, N]
