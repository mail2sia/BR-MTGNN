# Network architecture for B-MTGNN
from src.layer import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time
from src.util import DataLoaderS
import random
try:
    import torch.compiler as _tc
    def _compiler_disable_wrapper(fn):
        try:
            return _tc.disable(fn)
        except Exception:
            return fn
    _DISABLE = _compiler_disable_wrapper
except Exception:  # PyTorch <2.1 or missing component
    def _DISABLE(fn):
        return fn
from typing import List, Optional
try:
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
except Exception:
    TransformerEncoder = None
    TransformerEncoderLayer = None

fixed_seed=123

@_DISABLE
class TemporalMHSA(nn.Module):
    """Lightweight temporal Multi-Head Self Attention applied per node.

    Input shape expected: [B, C, N, T] (same as internal representation in gtnet).
    We internally transpose to [B, T, N, C] to attend over time for each node
    independently (batch*nodes treated as batch for MHA).
    """
    def __init__(self, c_in: int, attn_dim: int = 64, nhead: int = 2, pdrop: float = 0.1,
                 bn_chunk: int = 0):
        super().__init__()
        self.norm = nn.LayerNorm(c_in)
        self.qkv = nn.Linear(c_in, attn_dim, bias=False)
        self.mha = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=nhead, dropout=pdrop, batch_first=False)
        self.proj = nn.Linear(attn_dim, c_in, bias=False)
        self.dropout = nn.Dropout(pdrop)
        self.bn_chunk = int(bn_chunk) if bn_chunk is not None else 0

    def forward(self, x: torch.Tensor, window: int = 0) -> torch.Tensor:
        # x: [B, C, N, T]
        B, C, N, T = x.shape
        # Fast guards: skip attention if any dim is zero or malformed
        if B == 0 or N == 0 or T == 0:
            return x
        # Select temporal slice (local window) if requested
        if window and window > 0 and window < T:
            # select most recent 'window' timesteps
            x_head = x[..., -window:]  # [B, C, N, W]
            base = x_head
        else:
            x_head = x
            base = x

        # Move to [B, T, N, C]
        xt = x_head.permute(0, 3, 2, 1).contiguous()
        # LayerNorm over channel dim
        xt = self.norm(xt)
        # Pack (B*N, T, C)
        BN = B * N
        xt = xt.reshape(BN, xt.shape[1], xt.shape[3])  # [BN, T, C]
        try:
            qkv_full = self.qkv(xt)  # [BN, T, attn_dim]
            if qkv_full.size(-1) % self.mha.num_heads != 0:
                raise RuntimeError(f"attn_dim {qkv_full.size(-1)} not divisible by heads {self.mha.num_heads}")
            chunk = max(0, self.bn_chunk)
            if chunk and chunk < BN:
                outs = []
                for s in range(0, BN, chunk):
                    qkv = qkv_full[s:s+chunk].transpose(0, 1).contiguous()  # [T, chunk, attn_dim]
                    o, _ = self.mha(qkv, qkv, qkv, need_weights=False)
                    outs.append(o.transpose(0, 1))  # [chunk, T, attn_dim]
                attn_out = torch.cat(outs, dim=0)
            else:
                qkv = qkv_full.transpose(0, 1).contiguous()  # [T, BN, attn_dim]
                attn_out, _ = self.mha(qkv, qkv, qkv, need_weights=False)  # [T, BN, attn_dim]
                attn_out = attn_out.transpose(0, 1)  # [BN, T, attn_dim]
        except Exception as e:
            if not hasattr(self, '_failed_once'):
                print(f"[TemporalAttn] disabling after failure: {e}")
                self._failed_once = True
            return x
        attn_out = self.proj(attn_out)           # [BN, T, C]
        attn_out = self.dropout(attn_out)
        # Reshape back: [B, T, N, C]
        attn_out = attn_out.reshape(B, -1, N, C)
        # Back to [B, C, N, T]
        attn_out = attn_out.permute(0, 3, 2, 1).contiguous()
        out = base + attn_out
        if out.size(-1) != x.size(-1):  # stitched window
            x = torch.cat([x[..., : x.size(-1) - out.size(-1)], out], dim=-1)
        else:
            x = out
        return x

class TemporalTransformerEncoder(nn.Module):
    """
    Lightweight per-node Transformer encoder over time.
    Expects input [B, C, N, T] and returns same shape.
    Uses channels (C) as model dim (d_model) to avoid extra projections.
    """
    def __init__(self, c_in: int, n_layers: int = 2, nhead: int = 4, pdrop: float = 0.1):
        super().__init__()
        if TransformerEncoderLayer is None:
            raise RuntimeError("TransformerEncoder not available in this PyTorch build")
        layer = TransformerEncoderLayer(
            d_model=c_in,
            nhead=nhead,
            dim_feedforward=max(128, 4*c_in),
            dropout=pdrop,
            batch_first=True,
            activation="gelu"
        )
        if TransformerEncoder is None:
            raise RuntimeError("TransformerEncoder not available in this PyTorch build")
        self.enc = TransformerEncoder(layer, num_layers=int(n_layers))
        self.norm = nn.LayerNorm(c_in)
        
        # Conservative weight initialization to prevent gradient explosion
        self._init_weights()
    
    def _init_weights(self):
        """Apply conservative Xavier/Kaiming initialization with reduced variance."""
        for module in self.enc.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier uniform with reduced gain for stability
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N, T] -> [B*N, T, C] -> enc -> back to [B, C, N, T]
        B, C, N, T = x.shape
        if B == 0 or N == 0 or T == 0:
            return x
        xx = x.permute(0, 2, 3, 1).contiguous().view(B*N, T, C)  # [BN, T, C]
        xx = self.norm(xx)
        xx = self.enc(xx)                                        # [BN, T, C]
        xx = xx.view(B, N, T, C).permute(0, 3, 1, 2).contiguous()# [B, C, N, T]
        return x + xx  # residual

class gtnet(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_dim=12, layers=3, propalpha=0.05, tanhalpha:float=3.0, layer_norm_affline=True, temporal_attn: bool=False, attn_dim: int=64, attn_heads: int=2, attn_dropout: float=0.1, attn_window: int=0, attn_math_mode: bool=False,
                 attn_bn_chunk: int = 0, attn_gate_threshold: int = 0,
                 temporal_transformer: bool = False,
                 tt_layers: int = 0,
                 graph_mix: float = 0.0,
                 dropedge_p: float = 0.0,
                 quantiles: Optional[List[float]] = None,
                 nb_head: bool = False,
                 zinb: bool = False,
                 gauss_head: bool = False):
        super(gtnet, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.graph_mix = float(graph_mix)
        self.dropedge_p = float(dropedge_p)
        self.attn_window = int(attn_window) if attn_window is not None else 0
        # Register a static / user-provided adjacency (if any) as a buffer so
        # that DataParallel / .to(device) moves it with the module and avoids
        # cross-device addition errors when mixing with the learned graph.
        if predefined_A is not None:
            try:
                self.register_buffer('predefined_A', predefined_A.float())
            except Exception:
                # Fallback: keep attribute (will be moved manually in forward)
                self.predefined_A = predefined_A
        else:
            self.predefined_A = None
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)

        self.seq_length = seq_length
        # Register immutable metadata buffers for robust access under wrappers (DP/DDP)
        try:
            self.register_buffer('_seq_length', torch.tensor([int(seq_length)], dtype=torch.int32))
        except Exception:
            # Fallback (should not happen normally)
            self._seq_length = torch.tensor([int(seq_length)], dtype=torch.int32)
        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=residual_channels,
                                                 kernel_size=(1, 1)))
                # compute safe kernel width (>=1) and clamp to reasonable bounds
                # Kernel widths should never be larger than the input sequence
                # length or the receptive field. Also cap to a modest maximum to
                # avoid accidental huge allocations caused by bad hyperparams.
                def _safe_kw(k):
                    try:
                        k = int(k)
                    except Exception:
                        k = 1
                    # upper bound: don't exceed seq_length or receptive_field
                    ub = max(1, min(self.seq_length, max(1, int(self.receptive_field))))
                    # hard cap to avoid pathological sizes from bad math
                    HARD_CAP = 4096
                    return max(1, min(k, ub, HARD_CAP))

                kw1 = _safe_kw(self.seq_length - rf_size_j + 1)
                kw2 = _safe_kw(self.receptive_field - rf_size_j + 1)

                if self.seq_length > self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                     out_channels=skip_channels,
                                                     kernel_size=(1, kw1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                     out_channels=skip_channels,
                                                     kernel_size=(1, kw2)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                     out_channels=end_channels,
                                     kernel_size=(1,1),
                                     bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                     out_channels=out_dim,
                                     kernel_size=(1,1),
                                     bias=True)
        self.gauss_head = bool(gauss_head)
        self.end_conv_gauss = nn.Conv2d(in_channels=end_channels,
                        out_channels=out_dim,
                        kernel_size=(1,1),
                        bias=True) if self.gauss_head else None
        # Ensure kernel widths are valid (>=1) and clamped to safe bounds
        def _clamp_kw_top(k):
            try:
                k = int(k)
            except Exception:
                k = 1
            ub = max(1, min(self.seq_length, max(1, int(self.receptive_field))))
            HARD_CAP = 4096
            return max(1, min(k, ub, HARD_CAP))

        kw_skip0 = _clamp_kw_top(self.seq_length)
        kw_skipE = _clamp_kw_top(self.seq_length - self.receptive_field + 1) if self.seq_length > self.receptive_field else 1
        kw_skip0_alt = _clamp_kw_top(self.receptive_field)

        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, kw_skip0), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, kw_skipE), bias=True)
        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, kw_skip0_alt), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)


        self.idx = torch.arange(self.num_nodes).to(device)

        # Optional temporal attention block (after temporal conv stack, before skipE)
        self.temporal_attn = None
        self.attn_gate_threshold = int(attn_gate_threshold) if attn_gate_threshold is not None else 0
        if temporal_attn:
            try:
                self.temporal_attn = TemporalMHSA(
                    c_in=residual_channels,
                    attn_dim=int(attn_dim),
                    nhead=int(attn_heads),
                    pdrop=float(attn_dropout),
                    bn_chunk=int(attn_bn_chunk)
                )
                if attn_math_mode:
                    # Hint for PyTorch 2.1+ scaled_dot_product_attention backend selection
                    try:
                        torch.backends.cuda.enable_flash_sdp(False)
                        torch.backends.cuda.enable_math_sdp(True)
                        torch.backends.cuda.enable_mem_efficient_sdp(False)
                        print('[TemporalAttn] forced math SDPA kernels')
                    except Exception:
                        pass
            except Exception as e:
                print(f"[TemporalAttn] init failed, disabling attention: {e}")
                self.temporal_attn = None
        # --- NEW: optional temporal transformer encoder stack ---
        self.temporal_transformer = bool(temporal_transformer) and int(tt_layers) > 0
        if self.temporal_transformer:
            try:
                self.temporal_enc = TemporalTransformerEncoder(
                    c_in=residual_channels,
                    n_layers=int(tt_layers),
                    nhead=int(attn_heads),
                    pdrop=float(attn_dropout)
                )
                print(f'[TemporalTransformer] enabled: layers={tt_layers}, heads={attn_heads}')
            except Exception as _e:
                print(f'[TemporalTransformer] disabled: {_e}')
                self.temporal_transformer = False
                self.temporal_enc = None
        else:
            self.temporal_enc = None

        # --- NEW: optional quantile head (multi-quantile outputs) ---
        self.quantiles = list(quantiles) if quantiles is not None and len(quantiles) > 0 else []
        if len(self.quantiles) > 0:
            self.end_conv_q = nn.Conv2d(in_channels=end_channels,
                                        out_channels=out_dim * len(self.quantiles),
                                        kernel_size=(1,1), bias=True)
            # Store taus as buffer for checkpoint/export friendliness
            try:
                self.register_buffer('_taus', torch.tensor(self.quantiles, dtype=torch.float32))
            except Exception:
                self._taus = torch.tensor(self.quantiles, dtype=torch.float32)
        else:
            self.end_conv_q = None

        # --- NEW: optional NB/ZINB heads ---
        self.nb_head = bool(nb_head)
        self.zinb = bool(zinb)
        if self.nb_head:
            # Dispersion/log-alpha head (1x1 conv)
            self.end_conv_nb = nn.Conv2d(in_channels=end_channels,
                                         out_channels=1, kernel_size=(1,1), bias=True)
            self.end_conv_zi = nn.Conv2d(in_channels=end_channels,
                                         out_channels=1, kernel_size=(1,1), bias=True) if self.zinb else None
        else:
            self.end_conv_nb = None
            self.end_conv_zi = None
        


    def forward(self, input, idx=None):
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))

        # Guard: ensure input dtype matches module parameter dtype to avoid
        # conv2d half/float mismatch when using AMP + torch.compile.
        # (Autocast should handle this, but Dynamo fake tensor path can surface
        # a mismatch for the initial 1x1 conv.)
        try:
            bias_dtype = self.start_conv.bias.dtype if self.start_conv.bias is not None else self.start_conv.weight.dtype
            if input.dtype != bias_dtype:
                input = input.to(bias_dtype)
        except Exception:
            pass



        adp = None
        adp_t = None
        if self.gcn_true:
            # learned adjacency
            if self.buildA_true:
                A_learned = self.gc(self.idx if idx is None else idx)
            else:
                A_learned = None
            # user / predefined adjacency
            A_user = getattr(self, 'predefined_A', None)
            # Ensure user adjacency is on same device as learned adjacency (if any) or input
            if A_user is not None:
                target_dev = input.device
                if 'A_learned' in locals() and A_learned is not None:
                    target_dev = A_learned.device
                if A_user.device != target_dev:
                    A_user = A_user.to(target_dev)
                    # If it was a plain attribute (not registered buffer), update it
                    if 'predefined_A' in self.__dict__:
                        self.predefined_A = A_user

            # fusion: alpha * A_user + (1 - alpha) * A_learned (with safe fallbacks)
            if A_user is not None and A_learned is not None:
                adp = self.graph_mix * A_user + (1.0 - self.graph_mix) * A_learned
            elif A_user is not None:
                adp = A_user
            elif A_learned is not None:
                adp = A_learned
            else:
                raise RuntimeError('gcn_true is set but adjacency matrix is None')

            # DropEdge for robustness (train-time only)
            if self.dropedge_p > 0.0 and self.training:
                keep = (torch.rand_like(adp) > float(self.dropedge_p)).to(adp.dtype)
                adp = adp * keep

            adp_t = adp.transpose(1, 0)

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))

        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training) 
                    
            s = x
            s = self.skip_convs[i](s)
            # Crop both tensors to the common tail length before adding to avoid
            # small mismatches introduced by varying inception/dilation paths.
            minT = min(s.size(3), skip.size(3))
            if s.size(3) != minT:
                s = s[..., -minT:]
            if skip.size(3) != minT:
                skip = skip[..., -minT:]
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp) + self.gconv2[i](x, adp_t)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)

        # Optional temporal attention (operate on x before skipE)
        if self.temporal_attn is not None:
            # Auto-gate attention based on (mc_runs * batch * nodes) threshold
            allow = True
            try:
                import os as _os
                mc = int(_os.environ.get("BMTGNN_MC_RUNS", "0"))
                B = int(x.size(0))
                N = int(self.num_nodes)
                gate_th = max(0, getattr(self, 'attn_gate_threshold', 0))
                if gate_th and mc and mc * B * N > gate_th:
                    allow = False
            except Exception:
                pass
            if allow:
                try:
                    x = self.temporal_attn(x, window=self.attn_window)
                except Exception as e:
                    print(f"[TemporalAttn] forward failed, disabling: {e}")
                    self.temporal_attn = None

        # --- NEW: optional Transformer encoder over time (per-node) ---
        if self.temporal_transformer and self.temporal_enc is not None:
            try:
                x = self.temporal_enc(x)
            except Exception as _e:
                print(f"[TemporalTransformer] forward failed, disabling: {_e}")
                self.temporal_transformer = False
                self.temporal_enc = None

        sE = self.skipE(x)
        # Crop to common tail length before final merge
        minT = min(sE.size(3), skip.size(3))
        if sE.size(3) != minT:
            sE = sE[..., -minT:]
        if skip.size(3) != minT:
            skip = skip[..., -minT:]
        skip = sE + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        mean_out = self.end_conv_2(x)
        logvar_out = self.end_conv_gauss(x) if self.end_conv_gauss is not None else None

        q_out = None
        if self.end_conv_q is not None:
            q_out = self.end_conv_q(x)  # [B, K*out_dim, N, T']

        # Ensure final time dimension is singleton (last step)
        if mean_out.size(3) != 1:
            mean_out = mean_out[..., -1:].contiguous()
        if q_out is not None and q_out.size(3) != 1:
            q_out = q_out[..., -1:].contiguous()
        if logvar_out is not None and logvar_out.size(3) != 1:
            logvar_out = logvar_out[..., -1:].contiguous()

        disp_out = None
        pi_out = None
        if self.nb_head and (self.end_conv_nb is not None):
            disp_out = self.end_conv_nb(x)
            if disp_out.size(3) != 1:
                disp_out = disp_out[..., -1:].contiguous()
        if self.nb_head and (self.end_conv_zi is not None):
            pi_out = self.end_conv_zi(x)
            if pi_out.size(3) != 1:
                pi_out = pi_out[..., -1:].contiguous()

        if q_out is not None:
            # Use cached buffer for taus if available
            taus_tensor = getattr(self, '_taus', None)
            if taus_tensor is None:
                taus_tensor = torch.tensor(self.quantiles, device=mean_out.device)
            else:
                taus_tensor = taus_tensor.to(mean_out.device)
            out = {'mean': mean_out, 'quantiles': q_out, 'taus': taus_tensor}
            if logvar_out is not None:
                out['logvar'] = logvar_out
            if disp_out is not None:
                out['dispersion'] = disp_out
            if pi_out is not None:
                out['pi'] = pi_out
            return out
        else:
            if (disp_out is not None) or (pi_out is not None):
                out = {'mean': mean_out}
                if logvar_out is not None:
                    out['logvar'] = logvar_out
                if disp_out is not None: out['dispersion'] = disp_out
                if pi_out is not None:   out['pi'] = pi_out
                return out
            if logvar_out is not None:
                return {'mean': mean_out, 'logvar': logvar_out}
            return mean_out