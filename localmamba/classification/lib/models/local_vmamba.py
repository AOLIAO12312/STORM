import math
import copy
import warnings
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from .mamba.multi_mamba import MultiScan


DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


try:
    "sscore acts the same as mamba_ssm"
    SSMODE = "sscore"
    import selective_scan_cuda_core
    print("Using \"selective_scan_cuda_core\"")
except Exception as e:
    warnings.warn(f"{e}\n\"selective_scan_cuda_core\" not found, use default \"selective_scan_cuda\" instead.")
    # print(e, flush=True)
    SSMODE = "mamba_ssm"
    import selective_scan_cuda


# fvcore flops =======================================

def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    assert not with_complex 
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L    
    return flops

def selective_scan_flop_jit(inputs, outputs):
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False, with_Group=True)
    return flops


class SelectiveScan(torch.autograd.Function):
    
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
        assert nrows in [1, 2, 3, 4], f"{nrows}" # 8+ is too slow to compile
        assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows
        # all in float
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True
        
        if SSMODE == "mamba_ssm":
            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        else:
            out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
        
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        
        if SSMODE == "mamba_ssm":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
                False  # option to recompute out_z, not used here
            )
        else:
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
                # u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.nrows,
            )
        
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)


"""
Local Mamba
"""
class MultiScanVSSM(MultiScan):

    ALL_CHOICES = MultiScan.ALL_CHOICES

    def __init__(self, dim, choices=None):
        super().__init__(dim, choices=choices, token_size=None)
        self.attn = BiAttn(dim)

    def merge(self, xs):
        # xs: [B, K, D, L]
        # return: [B, D, L]

        # remove the padded tokens
        xs = [xs[:, i, :, :l] for i, l in enumerate(self.scan_lengths)]
        xs = super().multi_reverse(xs)
        xs = [self.attn(x.transpose(-2, -1)) for x in xs]
        x = super().forward(xs)
        return x

    
    def multi_scan(self, x):
        # x: [B, C, H, W]
        # return: [B, K, C, H * W]
        B, C, H, W = x.shape
        self.token_size = (H, W)

        xs = super().multi_scan(x)  # [[B, C, H, W], ...]

        self.scan_lengths = [x.shape[2] for x in xs]
        max_length = max(self.scan_lengths)

        # pad the tokens into the same length as VMamba compute all directions together
        new_xs = []
        for x in xs:
            if x.shape[2] < max_length:
                x = F.pad(x, (0, max_length - x.shape[2]))
            new_xs.append(x)
        return torch.stack(new_xs, 1)

    def __repr__(self):
        scans = ', '.join(self.choices)
        return super().__repr__().replace('MultiScanVSSM', f'MultiScanVSSM[{scans}]')


class BiAttn(nn.Module):
    def __init__(self, in_channels, act_ratio=0.125, act_fn=nn.GELU, gate_fn=nn.Sigmoid):
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.norm = nn.LayerNorm(in_channels)
        self.global_reduce = nn.Linear(in_channels, reduce_channels)
        # self.local_reduce = nn.Linear(in_channels, reduce_channels)
        self.act_fn = act_fn()
        self.channel_select = nn.Linear(reduce_channels, in_channels)
        # self.spatial_select = nn.Linear(reduce_channels * 2, 1)
        self.gate_fn = gate_fn()

    def forward(self, x):
        ori_x = x
        x = self.norm(x)
        x_global = x.mean(1, keepdim=True)
        x_global = self.act_fn(self.global_reduce(x_global))
        # x_local = self.act_fn(self.local_reduce(x))

        c_attn = self.channel_select(x_global)
        c_attn = self.gate_fn(c_attn)  # [B, 1, C]
        # s_attn = self.spatial_select(torch.cat([x_local, x_global.expand(-1, x.shape[1], -1)], dim=-1))
        # s_attn = self.gate_fn(s_attn)  # [B, N, 1]

        attn = c_attn #* s_attn  # [B, N, C]
        out = ori_x * attn
        return out


def multi_selective_scan(
    x: torch.Tensor=None, 
    x_proj_weight: torch.Tensor=None,
    x_proj_bias: torch.Tensor=None,
    dt_projs_weight: torch.Tensor=None,
    dt_projs_bias: torch.Tensor=None,
    A_logs: torch.Tensor=None,
    Ds: torch.Tensor=None,
    out_norm: torch.nn.Module=None,
    nrows = -1,
    delta_softplus = True,
    to_dtype=True,
    multi_scan=None,
):
    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    if nrows < 1:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1

    xs = multi_scan.multi_scan(x)

    L = xs.shape[-1]
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight) # l fixed

    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    
    xs = xs.view(B, -1, L).to(torch.float)
    dts = dts.contiguous().view(B, -1, L).to(torch.float)
    As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
    Bs = Bs.contiguous().to(torch.float)
    Cs = Cs.contiguous().to(torch.float)
    Ds = Ds.to(torch.float) # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)
    
    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
    
    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
    ).view(B, K, -1, L)
    
    y = multi_scan.merge(ys)
    
    y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)


class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class SS2D(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        simple_init=False,
        directions=None,
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state # 20240109
        self.d_conv = d_conv

        self.out_norm = nn.LayerNorm(d_inner)

        self.K = len(MultiScanVSSM.ALL_CHOICES) if directions is None else len(directions)
        self.K2 = self.K

        # in proj =======================================
        self.in_proj = nn.Linear(d_model, d_expand * 2, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
        
        # A, D =======================================
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True) # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True) # (K * D)

        # out proj =======================================
        self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # Local Mamba
        self.multi_scan = MultiScanVSSM(d_expand, choices=directions)

        if simple_init:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.K2 * d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner))) 

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, nrows=-1, channel_first=False):
        nrows = 1
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
        x = multi_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, self.out_norm,
            nrows=nrows, delta_softplus=True, multi_scan=self.multi_scan,
        )
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    def forward(self, x: torch.Tensor):
        xz = self.in_proj(x)
        if self.d_conv > 1:
            x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
            z = self.act(z)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.act(self.conv2d(x)) # (b, d, h, w)
        else:
            xz = self.act(xz)
            x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
        y = self.forward_core(x, channel_first=(self.d_conv > 1))
        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

from .token_reduction.FixedWindowToMe2D import FixedWindowToMe2Dv2
from .token_reduction.TokenMerge2Dv4 import TokenMerge2Dv4
from .token_reduction.EViT import EViTTokenPruning
from .token_reduction.HSA import HSA
from .token_reduction.EViT2D import EViT2DStructuredPruning
from .token_reduction.FixedWindowEViT2D import FixedWindowEViT2D
from .token_reduction.RandomHardPruneFixedWindowToMe2D import RandomHardPruneFixedWindowToMe2D

class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_simple_init=False,
        # =============================
        use_checkpoint: bool = False,
        directions=None,
        layer_idx:int = -1,
        stage_idx:int = -1,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm = norm_layer(hidden_dim)
        self.op = SS2D(
            d_model=hidden_dim, 
            d_state=ssm_d_state, 
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            # ==========================
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            # ==========================
            dropout=ssm_drop_rate,
            # bias=False,
            # ==========================
            # dt_min=0.001,
            # dt_max=0.1,
            # dt_init="random",
            # dt_scale="random",
            # dt_init_floor=1e-4,
            simple_init=ssm_simple_init,
            # ==========================
            directions=directions
        )
        self.drop_path = DropPath(drop_path)
        self.stage_idx = stage_idx
        self.layer_idx = layer_idx

        prune_strategy_hybrid_1p1 = [[220, 0], [0, 53], [0, 25, 0, 23, 0, 21, 0, 19], [0, 0]]
        prune_strategy_hybrid_2p1 = [[432, 0], [0, 100], [0, 23, 0, 21, 0, 19, 0, 17], [0, 0]]
        prune_strategy_hybrid_3p1 = [[636, 0], [0, 141], [0, 21, 0, 19, 0, 17, 0, 15], [0, 0]]
        prune_strategy_hybrid_4p1 = [[832, 0], [0, 176], [0, 19, 0, 17, 0, 15, 0, 13], [0, 0]]
        prune_strategy_hybrid_5p1 = [[1200, 0], [0, 160], [0, 17, 0, 15, 0, 13, 0, 11], [0, 0]]

        prune_strategy_hybrid_1p1_sb = [[220, 0], [0, 53], [25, 0, 0, 0, 0, 0, 0, 0, 0, 23, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 19], [0, 0]]
        prune_strategy_hybrid_2p1_sb = [[432, 0], [0, 100], [23, 0, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 0, 0, 0, 0, 0, 17], [0, 0]]
        prune_strategy_hybrid_3p1_sb = [[636, 0], [0, 141], [21, 0, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 15], [0, 0]]
        prune_strategy_hybrid_4p1_sb = [[832, 0], [0, 176], [19, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 13], [0, 0]]
        prune_strategy_hybrid_5p1_sb = [[1200, 0], [0, 160], [17, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 11], [0, 0]]

        prune_strategy_window_tome1 = [[0, 0], [0, 384], [0, 0, 0, 36, 0, 0, 0, 0], [0, 0]]
        prune_strategy_window_tome2 = [[0, 0], [384, 0], [0, 0, 0, 36, 0, 0, 0, 0], [0, 0]]
        prune_strategy_window_tome3 = [[0, 1372], [0, 0], [0, 0, 0, 57, 0, 0, 0, 0], [0, 0]]
        prune_strategy_window_tome4 = [[1372, 0], [0, 0], [0, 0, 0, 57, 0, 0, 0, 0], [0, 0]]
        self.prune_strategy = prune_strategy_hybrid_5p1_sb
        self.prune_ratio = 0.1

        # VMambaPruner
        self.fixed_window_tome2dv2 =  FixedWindowToMe2Dv2(
            if_prune=False,
            distance='l1',
            merge_mode='sum',
            window_size=5,
        )

        self.merger = TokenMerge2Dv4(
            num_prune=0,
            if_prune=True,
            if_order=True,
            distance='cosine',
            merge_mode='sum',
            choose='max'
        )

        self.HSA_pruner = HSA()
        self.EViT_pruner = EViTTokenPruning()
        self.fixed_window_evit2d = FixedWindowEViT2D(
            window_size=5,
            score_mode='absmean'
        )

        self.RandomHardPruneSTORM_pruner = RandomHardPruneFixedWindowToMe2D(
            window_size=5,
            if_order=True
        )

        self.method = "randomToMe2D" # tome/fixed_window_tome2dv2/HSA/EViT/EViT2D/fixed_window_evit2d/fixed_window_evit2d/randomToMe2D

        self.EViT2D_pruner = EViT2DStructuredPruning(score_mode="absmean", if_order=True)

    def get_prune_num(self, mode, size: int = None, ratio: float = None):
        if mode == "manual":
            assert hasattr(self, 'prune_strategy'), "prune_strategy attribute is required for manual mode"
            assert hasattr(self, 'stage_idx') and hasattr(self,
                                                          'layer_idx'), "stage_idx and layer_idx attributes are required for manual mode"
            return self.prune_strategy[self.stage_idx][self.layer_idx]

        elif mode == "auto":
            assert size is not None and ratio is not None and 0 <= ratio <= 1, \
                "size and ratio must be provided and ratio must be between 0 and 1 for auto mode"

            if self.layer_idx % 2 == 1:
                size_new = int(size * (1 - ratio))

                return size ** 2 - size_new ** 2
            else:
                return 0

        else:
            raise ValueError(f"Unsupported mode: {mode}. Supported modes are 'manual' and 'auto'.")

    def _forward(self, input: torch.Tensor):
        def prune_tokens_by_index(x: torch.Tensor, sorted_idx: torch.Tensor) -> torch.Tensor:
            """
            根据 sorted_idx 从原始 x 中保留指定 token。

            Args:
                x: torch.Tensor, shape [B, D, L_orig]
                sorted_idx: torch.Tensor, shape [B, L_keep, 1] 或 [B, L_keep]
            Returns:
                pruned_x: torch.Tensor, shape [B, D, L_keep]
            """
            B, D, L_orig = x.shape

            # 若 sorted_idx 是 [B, L_keep, 1]，去掉多余维度
            if sorted_idx.dim() == 3 and sorted_idx.shape[-1] == 1:
                sorted_idx = sorted_idx.squeeze(-1)  # [B, L_keep]

            # 检查维度匹配
            assert sorted_idx.shape[0] == B, f"Batch size 不匹配: {sorted_idx.shape[0]} vs {B}"

            # 扩展索引到 [B, D, L_keep]
            sorted_idx = sorted_idx.unsqueeze(1).expand(-1, D, -1)  # [B, D, L_keep]

            # 按索引取出需要保留的 token
            pruned_x = torch.gather(x, dim=2, index=sorted_idx)

            return pruned_x
        def random_prune_tokens(x: torch.Tensor, num_prune: int, seed: int = None):
            """
            在 [B, C, L] (L=H*W) 中随机删除 num_prune 个 token。
            所有 batch 共用相同的随机删除模式。
            返回剪枝后的张量，以及保留的 sorted_idx（升序），包装为 [B, L_keep, 1]。

            Args:
                x (torch.Tensor): 输入张量 [B, C, L]
                num_prune (int): 要删除的 token 数
                seed (int, optional): 随机种子，保证可复现

            Returns:
                x_pruned (torch.Tensor): 删除后的张量 [B, C, L - num_prune]
                sorted_idx (torch.Tensor): 保留索引 [B, L - num_prune, 1]，升序排列
            """
            assert x.dim() == 3, f"Input must be [B, C, L], got {x.shape}"
            B, C, L = x.shape
            assert num_prune < L, "num_prune must be smaller than sequence length"

            device = x.device
            if seed is not None:
                torch.manual_seed(seed)

            # 随机生成要删除的 token 索引
            prune_idx = torch.randperm(L, device=device)[:num_prune]
            mask = torch.ones(L, dtype=torch.bool, device=device)
            mask[prune_idx] = False

            # 获取保留索引（升序）
            sorted_idx = torch.nonzero(mask, as_tuple=True)[0]  # [L_keep]

            # 剪枝
            x_pruned = x.index_select(dim=2, index=sorted_idx)

            # 包装 sorted_idx 为 [B, L_keep, 1]（每个 batch 共用）
            sorted_idx = sorted_idx.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1).contiguous()

            return x_pruned, sorted_idx
        def pool_downsample(x, mode='avg'):
            """
            将 F.interpolate(x, size=[H//2, W//2], mode="nearest")
            替换为池化操作版本。

            支持 mode:
                - 'avg' : AvgPool2d
                - 'max' : MaxPool2d
                - 'nearest' : 用 nearest 插值保持一致行为
                - 'lp' : L2 pooling (可选扩展)
            """
            B, C, H, W = x.shape
            if mode == 'avg':
                pool = nn.AvgPool2d(kernel_size=2, stride=2)
                return pool(x)
            elif mode == 'max':
                pool = nn.MaxPool2d(kernel_size=2, stride=2)
                return pool(x)
            elif mode == 'nearest':
                # 如果希望和 interpolate 一样，就保留插值写法
                return F.interpolate(x, size=[H // 2, W // 2], mode="nearest")
            elif mode == 'lp':
                # L2 Pooling (等价于对平方求平均再开根号)
                pool = nn.LPPool2d(norm_type=2, kernel_size=2, stride=2)
                return pool(x)
            else:
                raise ValueError("mode must be 'avg', 'max', 'nearest', or 'lp'")
        def pad_zeros(x: torch.Tensor) -> torch.Tensor:
            assert x.dim() == 3, "x must be [B, D, L]"
            B, D, L = x.shape
            H = math.isqrt(L)
            L_orig = H * H if H * H == L else (H + 1) * (H + 1)
            assert L <= L_orig, "L must not exceed L_orig"
            # pad 在最后一个维度补 (0, L_orig - L)
            return F.pad(x, (0, L_orig - L))
        x = input
        B,H,W,D = x.shape # TODO: LocalVMamba中使BHWD
        if self.method == "none" or self.method is None:
            x = x + self.drop_path(self.op(self.norm(x)))
        elif self.method == "scale":
            # pre_stage像素级进行nearest剪枝
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio)
            L = H * W
            H_new = math.isqrt(L - num_prune)
            if self.stage_idx == 0:
                if x.shape[-2:] != (H_new, H_new):
                    x = F.interpolate(x, size=[H_new, H_new], mode="nearest")  # nearest/bilinear/area
                x_op = self.op(self.norm(x))
            else:
                x_op = self.op(self.norm(x))
                if x.shape[-2:] != (H_new, H_new):
                    x = F.interpolate(x, size=[H_new, H_new], mode="nearest")  # nearest/bilinear/area
                    x_op = F.interpolate(x_op, size=[H_new, H_new], mode="nearest")
            x = x + self.drop_path(x_op)
        elif self.method == "random":
            x_op = self.op(self.norm(x))
            num_prune = self.prune_strategy[self.stage_idx][self.layer_idx]
            x_op, sorted_idx = random_prune_tokens(x_op.view(B, D, H * W), num_prune)
            if sorted_idx is not None:
                x = prune_tokens_by_index(x.flatten(2), sorted_idx)
                x = pad_zeros(x)
                L = x.shape[-1]
                H = math.isqrt(L)
                x = x.view(B, D, H, H)
            x = x + self.drop_path(x_op.view(B, D, H, H))
        elif self.method == "tome2d":
            x_op = self.op(self.norm(x))
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio)
            L = H * W
            H_new = math.isqrt(L - num_prune)
            if x.shape[-2:] != (H_new, H_new):
                prune_fn = self.merge2d(x, num_prune_w=H - H_new, num_prune_h=H - H_new)
                x_op = prune_fn(x_op)
                x = prune_fn(x)
            x = x + self.drop_path(x_op)
        elif self.method == "fixed_window_tome2dv2":
            x_op = self.op(self.norm(x))
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio)
            L = H * W
            H_new = math.isqrt(L - num_prune)
            if x.shape[-2:] != (H_new, H_new):
                prune_fn = self.fixed_window_tome2dv2(x.permute(0, 3, 1, 2), num_prune_w=H - H_new, num_prune_h=H - H_new)
                x_op = prune_fn(x_op.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                x = prune_fn(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            x = x + self.drop_path(x_op)
        elif self.method == "tome":
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio)
            L = H * W
            H_new = W_new = math.isqrt(L - num_prune)  # classfication
            x_op = self.op(self.norm(x))
            if num_prune != 0:
                x_op_merged, x_merged = self.merger(x_op.permute(0, 3, 1, 2).view(B, D, H * W), x.permute(0, 3, 1, 2).view(B, D, H * W), None,
                                                    num_prune)  # [B, D, L_kept]
                x = x_merged.permute(0, 2, 1).view(B, H_new, W_new, D)
                x_op = x_op_merged.permute(0, 2, 1).view(B, H_new, W_new, D)
            x = x + self.drop_path(x_op)
        elif self.method == "HSA":
            x_op = self.op(self.norm(x))
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio)
            L = H * W
            H_new = W_new = math.isqrt(L - num_prune)  # classfication
            if x.shape[-2:] != (H_new, W_new):
                x_op_merged, x_merged = self.HSA_pruner(x_op.permute(0, 3, 1, 2).view(B, D, H * W), x.permute(0, 3, 1, 2).view(B, D, H * W), num_prune)
                x = x_merged.permute(0, 2, 1).view(B, H_new, W_new, D)
                x_op = x_op_merged.permute(0, 2, 1).view(B, H_new, W_new, D)
            x = x + self.drop_path(x_op)
        elif self.method == "EViT":
            x_op = self.op(self.norm(x))
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio)
            L = H * W
            H_new = W_new = math.isqrt(L - num_prune)  # classfication
            if x.shape[-2:] != (H_new, W_new):
                x_op_merged, x_merged = self.EViT_pruner(x_op.permute(0, 3, 1, 2).view(B, D, H * W), x.permute(0, 3, 1, 2).view(B, D, H * W), num_prune)
                x = x_merged.permute(0, 2, 1).view(B, H_new, W_new, D)
                x_op = x_op_merged.permute(0, 2, 1).view(B, H_new, W_new, D)
            x = x + self.drop_path(x_op)
        elif self.method == "EViT2D":
            x_op = self.op(self.norm(x))
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio)
            L = H * W
            H_new = math.isqrt(L - num_prune)
            if x.shape[-2:] != (H_new, H_new):
                prune_fn = self.EViT2D_pruner(x.permute(0, 3, 1, 2), num_prune_w=H - H_new, num_prune_h=H - H_new)
                x_op = prune_fn(x_op.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                x = prune_fn(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            x = x + self.drop_path(x_op)
        elif self.method == "fixed_window_evit2d":
            x_op = self.op(self.norm(x))
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio)
            L = H * W
            H_new = math.isqrt(L - num_prune)
            if x.shape[-2:] != (H_new, H_new):
                prune_fn = self.fixed_window_evit2d(x.permute(0, 3, 1, 2), num_prune_w=H - H_new, num_prune_h=H - H_new)
                x_op = prune_fn(x_op.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                x = prune_fn(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            x = x + self.drop_path(x_op)
        elif self.method == "randomToMe2D":
            x_op = self.op(self.norm(x))
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio)
            L = H * W
            H_new = math.isqrt(L - num_prune)
            if x.shape[-2:] != (H_new, H_new):
                prune_fn = self.RandomHardPruneSTORM_pruner(x.permute(0, 3, 1, 2), num_prune_w=H - H_new, num_prune_h=H - H_new)
                x_op = prune_fn(x_op.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                x = prune_fn(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            x = x + self.drop_path(x_op)

        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


class VSSM(nn.Module):
    def __init__(
        self, 
        patch_size=4, 
        in_chans=3, 
        num_classes=1000, 
        depths=[2, 2, 9, 2], 
        dims=[96, 192, 384, 768], 
        # =========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",        
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_simple_init=False,
        # =========================
        drop_path_rate=0.1, 
        patch_norm=True, 
        norm_layer="LN",
        use_checkpoint=False, 
        directions=None,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        if norm_layer.lower() in ["ln"]:
            norm_layer: nn.Module = _NORMLAYERS[norm_layer.lower()]

        if ssm_act_layer.lower() in ["silu", "gelu", "relu"]:
            ssm_act_layer: nn.Module = _ACTLAYERS[ssm_act_layer.lower()]

        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=patch_size, stride=patch_size, bias=True),
            Permute(0, 2, 3, 1),
            (norm_layer(dims[0]) if patch_norm else nn.Identity()), 
        )

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = PatchMerging2D(
                self.dims[i_layer], 
                self.dims[i_layer + 1], 
                norm_layer=norm_layer,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim = self.dims[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_simple_init=ssm_simple_init,
                # =================
                directions=None if directions is None else directions[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                stage_idx=i_layer,
            ))

        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features), # B,H,W,C
            permute=Permute(0, 3, 1, 2),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay(self):
        return {}
    
    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            Permute(0, 2, 3, 1),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
        dim=96, 
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_simple_init=False,
        # ===========================
        directions=None,
        stage_idx:int = -1,
        **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_simple_init=ssm_simple_init,
                use_checkpoint=use_checkpoint,
                directions=directions[d] if directions is not None else None,
                layer_idx=d,
                stage_idx=stage_idx
            ))
        
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks,),
            downsample=downsample,
        ))

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        return x

    def flops(self, shape=(3, 224, 224)):
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            "prim::PythonOp.SelectiveScan": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit,
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9

# compatible with openmmlab
class Backbone_LocalVSSM(VSSM):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer=nn.LayerNorm, **kwargs):
        print(kwargs['directions'])
        super().__init__(**kwargs)
        
        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.classifier
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt):
        if ckpt is None:
            return
        print(f'Load backbone state dict from {ckpt}')
        if ckpt.startswith('http'):
            from mmengine.utils.dl_utils import load_url
            state_dict = load_url(ckpt, map_location='cpu')['state_dict']
        else:
            state_dict = torch.load(ckpt, map_location='cpu')['state_dict']
        res = self.load_state_dict(state_dict, strict=False)
        print(res)

    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x) # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                out = out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        if len(self.out_indices) == 0:
            return x
        
        return outs


@register_model
def local_vssm_tiny_search(*args, drop_path_rate=0.1, **kwargs):
    return VSSM(dims=[32, 64, 128, 256], depths=[2, 2, 9, 2], d_state=16, drop_path_rate=drop_path_rate)
    
@register_model
def local_vssm_tiny(*args, drop_path_rate=0.2, **kwargs):
    directions = [
        ['h', 'h_flip', 'w7', 'w7_flip'],
        ['h_flip', 'v_flip', 'w2', 'w2_flip'],
        ['h_flip', 'v_flip', 'w2_flip', 'w7'],
        ['h_flip', 'v', 'v_flip', 'w2'],
        ['h', 'h_flip', 'v_flip', 'w2_flip'],
        ['h_flip', 'v_flip', 'w2', 'w2_flip'],
        ['h', 'w2_flip', 'w7', 'w7_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'v_flip', 'w7', 'w7_flip'],
        ['h_flip', 'v', 'w2', 'w7_flip'],
        ['v', 'v_flip', 'w2', 'w7_flip'],
        ['h', 'h_flip', 'v_flip', 'w2_flip'],
        ['v_flip', 'w2_flip', 'w7', 'w7_flip'],
        ['h_flip', 'v_flip', 'w2_flip', 'w7_flip'],
        ['h_flip', 'v', 'w7', 'w7_flip'],
    ]
    return VSSM(dims=[96, 192, 384, 768], depths=[2, 2, 9, 2], d_state=16, drop_path_rate=drop_path_rate, directions=directions)

@register_model
def local_vssm_small_search(*args, drop_path_rate=0.1, **kwargs):
    return VSSM(dims=[32, 64, 128, 256], depths=[2, 2, 27, 2], d_state=16, drop_path_rate=drop_path_rate)


@register_model
def local_vssm_small(*args, drop_path_rate=0.2, **kwargs):
    directions = [
        ['h', 'v', 'v_flip', 'w7_flip'],
        ['h', 'h_flip', 'v', 'w7'],
        ['h_flip', 'v', 'v_flip', 'w7'],
        ['h_flip', 'v', 'w2_flip', 'w7'],
        ['h_flip', 'v', 'w2_flip', 'w7'],
        ['h_flip', 'v_flip', 'w2', 'w7_flip'],
        ['h_flip', 'v', 'v_flip', 'w7'],
        ['h', 'v', 'v_flip', 'w7'],
        ['h', 'v', 'v_flip', 'w7'],
        ['h', 'v', 'v_flip', 'w7_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'v', 'v_flip', 'w2'],
        ['v', 'v_flip', 'w2_flip', 'w7'],
        ['h', 'h_flip', 'v', 'w2'],
        ['h_flip', 'v', 'v_flip', 'w7_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'v', 'v_flip', 'w7_flip'],
        ['h', 'v', 'v_flip', 'w7_flip'],
        ['h', 'h_flip', 'v_flip', 'w7'],
        ['h_flip', 'v_flip', 'w2_flip', 'w7'],
        ['h_flip', 'v', 'v_flip', 'w7_flip'],
        ['v', 'v_flip', 'w7', 'w7_flip'],
        ['h', 'v', 'v_flip', 'w7_flip'],
        ['h_flip', 'v', 'v_flip', 'w2'],
        ['h', 'v', 'v_flip', 'w2_flip'],
        ['h', 'h_flip', 'v', 'w7'],
        ['h', 'h_flip', 'w7', 'w7_flip'],
        ['h', 'v_flip', 'w2', 'w2_flip'],
        ['h', 'v_flip', 'w2', 'w7'],
        ['h', 'v', 'v_flip', 'w7_flip'],
        ['h_flip', 'v', 'w2_flip', 'w7'],
        ['h_flip', 'v_flip', 'w7', 'w7_flip'],
        ['h', 'v', 'w7', 'w7_flip']
    ]
    return VSSM(dims=[96, 192, 384, 768], depths=[2, 2, 27, 2], d_state=16, drop_path_rate=drop_path_rate, directions=directions)

