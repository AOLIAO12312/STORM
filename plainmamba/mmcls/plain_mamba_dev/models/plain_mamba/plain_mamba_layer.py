'''
Author: Chenhongyi Yang
'''

import math
from einops import repeat

import torch
import torch.nn as nn

from mmcv.cnn.bricks.transformer import build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.layernorm import RMSNorm
import torch.nn.functional as F
from sympy import floor


class PlainMamba2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_size=7,
        conv_bias=True,
        bias=False,
        init_layer_scale=None,
        default_hw_shape=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.default_hw_shape = default_hw_shape
        self.default_permute_order = None
        self.default_permute_order_inverse = None
        self.n_directions = 4
        if default_hw_shape is not None:
            orders, inverse_orders, directions = self.get_permute_order(default_hw_shape)
            (
                self.default_permute_order,
                self.default_permute_order_inverse,
                self.default_direction
            ) = orders, inverse_orders, directions

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        assert conv_size % 2 == 1
        padding = int(conv_size // 2)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=(conv_size, conv_size),
            stride=(1, 1),
            padding=(padding, padding),
            groups=self.d_inner
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False,
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True
        )

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        self.direction_Bs = nn.Parameter(torch.zeros(self.n_directions+1, self.d_state))
        trunc_normal_(self.direction_Bs, std=0.02)


    def get_permute_order(self, hw_shape):
        if self.default_permute_order is not None:
            if hw_shape[0] == self.default_hw_shape[0] and hw_shape[1] == self.default_hw_shape[1]:
                return self.default_permute_order, self.default_permute_order_inverse, self.default_direction
        H, W = hw_shape
        L = H * W

        # [start, right, left, up, down] [0, 1, 2, 3, 4]

        o1 = []
        d1 = []
        o1_inverse = [-1 for _ in range(L)]
        i, j = 0, 0
        j_d = "right"
        while i < H:
            assert j_d in ["right", "left"]
            idx = i * W + j
            o1_inverse[idx] = len(o1)
            o1.append(idx)
            if j_d == "right":
                if j < W-1:
                    j = j + 1
                    d1.append(1)
                else:
                    i = i + 1
                    d1.append(4)
                    j_d = "left"

            else:
                if j > 0:
                    j = j - 1
                    d1.append(2)
                else:
                    i = i + 1
                    d1.append(4)
                    j_d = "right"
        d1 = [0] + d1[:-1]

        o2 = []
        d2 = []
        o2_inverse = [-1 for _ in range(L)]

        if H % 2 == 1:
            i, j = H-1, W-1
            j_d = "left"
        else:
            i, j = H-1, 0
            j_d = "right"

        while i > -1:
            assert j_d in ["right", "left"]
            idx = i * W + j
            o2_inverse[idx] = len(o2)
            o2.append(idx)
            if j_d == "right":
                if j < W - 1:
                    j = j + 1
                    d2.append(1)
                else:
                    i = i - 1
                    d2.append(3)
                    j_d = "left"
            else:
                if j > 0:
                    j = j - 1
                    d2.append(2)
                else:
                    i = i - 1
                    d2.append(3)
                    j_d = "right"
        d2 = [0] + d2[:-1]

        o3 = []
        d3 = []
        o3_inverse = [-1 for _ in range(L)]
        i, j = 0, 0
        i_d = "down"
        while j < W:
            assert i_d in ["down", "up"]
            idx = i * W + j
            o3_inverse[idx] = len(o3)
            o3.append(idx)
            if i_d == "down":
                if i < H - 1:
                    i = i + 1
                    d3.append(4)
                else:
                    j = j + 1
                    d3.append(1)
                    i_d = "up"
            else:
                if i > 0:
                    i = i - 1
                    d3.append(3)
                else:
                    j = j + 1
                    d3.append(1)
                    i_d = "down"
        d3 = [0] + d3[:-1]

        o4 = []
        d4 = []
        o4_inverse = [-1 for _ in range(L)]

        if W % 2 == 1:
            i, j = H - 1, W - 1
            i_d = "up"
        else:
            i, j = 0, W - 1
            i_d = "down"
        while j > -1:
            assert i_d in ["down", "up"]
            idx = i * W + j
            o4_inverse[idx] = len(o4)
            o4.append(idx)
            if i_d == "down":
                if i < H - 1:
                    i = i + 1
                    d4.append(4)
                else:
                    j = j - 1
                    d4.append(2)
                    i_d = "up"
            else:
                if i > 0:
                    i = i - 1
                    d4.append(3)
                else:
                    j = j - 1
                    d4.append(2)
                    i_d = "down"
        d4 = [0] + d4[:-1]

        o1 = tuple(o1)
        d1 = tuple(d1)
        o1_inverse = tuple(o1_inverse)

        o2 = tuple(o2)
        d2 = tuple(d2)
        o2_inverse = tuple(o2_inverse)

        o3 = tuple(o3)
        d3 = tuple(d3)
        o3_inverse = tuple(o3_inverse)

        o4 = tuple(o4)
        d4 = tuple(d4)
        o4_inverse = tuple(o4_inverse)

        return (o1, o2, o3, o4), (o1_inverse, o2_inverse, o3_inverse, o4_inverse), (d1, d2, d3, d4)

    def forward(self, x, hw_shape):
        batch_size, L, _ = x.shape
        H, W = hw_shape
        E = self.d_inner

        conv_state, ssm_state = None, None

        xz = self.in_proj(x)  # [B, L, 2 * E]
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        x, z = xz.chunk(2, dim=-1)
        x_2d = x.reshape(batch_size, H, W, E).permute(0, 3, 1, 2)
        x_2d = self.act(self.conv2d(x_2d)) # Conv2D op
        x_conv = x_2d.permute(0, 2, 3, 1).reshape(batch_size, L, E)

        x_dbl = self.x_proj(x_conv)  # (B, L, dt_rank + d_state * 2)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)

        dt = dt.permute(0, 2, 1).contiguous()  # [B, d_innter, L]
        B = B.permute(0, 2, 1).contiguous()  # [B, d_state, L]
        C = C.permute(0, 2, 1).contiguous()  # [B, d_state, L]

        assert self.activation in ["silu", "swish"]

        orders, inverse_orders, directions = self.get_permute_order(hw_shape)
        direction_Bs = [self.direction_Bs[d, :] for d in directions]  # each [L, d_state]
        direction_Bs = [dB[None, :, :].expand(batch_size, -1, -1).permute(0, 2, 1).to(dtype=B.dtype) for dB in direction_Bs]
        ys = [
            selective_scan_fn(
                x_conv[:, o, :].permute(0, 2, 1).contiguous(),
                dt,
                A,
                (B + dB).contiguous(),
                C,
                self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            ).permute(0, 2, 1)[:, inv_o, :]
            for o, inv_o, dB in zip(orders, inverse_orders, direction_Bs)
        ]
        y = sum(ys) * self.act(z)
        out = self.out_proj(y)

        if self.init_layer_scale is not None:
            out = out * self.gamma
        return out

from mmcls.plain_mamba_dev.models.plain_mamba.token_reduction.RandomPrune import RandomPrune
from mmcls.plain_mamba_dev.models.plain_mamba.token_reduction.InterpolatePrune import InterpolatePrune
from mmcls.plain_mamba_dev.models.plain_mamba.token_reduction.ToMe1D import TokenMerge2Dv4
from mmcls.plain_mamba_dev.models.plain_mamba.token_reduction.ToMe2D import ToMe2D
from mmcls.plain_mamba_dev.models.plain_mamba.token_reduction.ConvToMe2D import ConvToMe2D
from mmcls.plain_mamba_dev.models.plain_mamba.token_reduction.FixedWindowToMe2Dv2 import FixedWindowToMe2Dv2
from mmcls.plain_mamba_dev.models.plain_mamba.token_reduction.HSA import HSA
from mmcls.plain_mamba_dev.models.plain_mamba.token_reduction.EViT import EViTTokenPruning
from mmcls.plain_mamba_dev.models.plain_mamba.token_reduction.EViT2D import EViT2DStructuredPruning
from mmcls.plain_mamba_dev.models.plain_mamba.token_reduction.FixedWindowEViT2D import FixedWindowEViT2D
from mmcls.plain_mamba_dev.models.plain_mamba.token_reduction.RandomHardPruneFixedWindowToMe2D import RandomHardPruneFixedWindowToMe2D

class PlainMambaLayer(nn.Module):
    def __init__(
        self,
        embed_dims,
        use_rms_norm,
        with_dwconv,
        drop_path_rate,
        mamba_cfg,
        layer_idx
    ):
        super(PlainMambaLayer, self).__init__()
        mamba_cfg.update({'d_model': embed_dims})

        if use_rms_norm:
            self.norm = RMSNorm(embed_dims)
        else:
            self.norm = nn.LayerNorm(embed_dims)

        self.with_dwconv = with_dwconv
        if self.with_dwconv:
            self.dw = nn.Sequential(
                nn.Conv2d(
                    embed_dims,
                    embed_dims,
                    kernel_size=(3, 3),
                    padding=(1, 1),
                    bias=False,
                    groups=embed_dims
                ),
                nn.BatchNorm2d(embed_dims),
                nn.GELU(),
            )
        self.mamba = PlainMamba2D(**mamba_cfg)
        self.drop_path = build_dropout(dict(type='DropPath', drop_prob=drop_path_rate))
        self.layer_idx = layer_idx
        schedual_0 = {}

        schedual_1 = {
            4: 169,
            10: 144,
            16: 100,
            22: 81
        }

        schedual_2 = {
            4: 144,
            10: 100,
            16: 64,
            22: 36
        }

        schedual_3 = {
            4: 100,
            10: 64,
            16: 36,
            22: 16
        }

        schedual_4 = {
            4: 100,
            10: 64,
            16: 25,
            22: 9
        }

        schedual_5 = {
            4: 64,
            10: 36,
            16: 16,
            22: 9
        }

        self.schedual = schedual_1
        self.prune_ratio = 0.1
        self.random_prune = RandomPrune(share_across_batch=True)
        self.interpolate_prune = InterpolatePrune(assume_square=True)
        self.merger = TokenMerge2Dv4(
            num_prune=0,  # 每轮合并 r 对（等价减少 r 个 token）
            if_prune=False,  # True=纯删；False=合并到 dst（推荐）
            if_order=True,  # True=保持原顺序；False=不保持
            distance='cosine',  # 'cosine' / 'l1' / 'l2'
            merge_mode='sum',  # 权重聚合方式（传给 scatter_reduce）
            choose='max'  # 选择策略
        )
        self.merge2d = ToMe2D(if_order=True, distance='cosine', merge_mode='sum')
        self.conv_tome_2d = ConvToMe2D(
            imp_mode="l2",
            alpha=3.0,
            beta=15.0,
        )
        self.fixed_window_tome2dv2 = FixedWindowToMe2Dv2(
            if_prune=False,
            distance='l1',
            merge_mode='sum',
            window_size=5,  # 或单独给 window_size_w / window_size_h
        )
        self.HSA_pruner = HSA()
        self.EViT_pruner = EViTTokenPruning()
        self.EViT2D_pruner = EViT2DStructuredPruning(score_mode="absmean", if_order=True)
        self.fixed_window_evit2d = FixedWindowEViT2D(
            window_size=20,
            score_mode='absmean'
        )
        self.RandomHardPruneSTORM_pruner = RandomHardPruneFixedWindowToMe2D(
            window_size=5,
            if_order=True
        )

        self.method = "fixed_window_tome2dv2" # none/random/tome/tome2d/interpolate/conv_tome2d/fixed_window_tome2dv2/HSA/EViT/EViT2D/fixed_window_evit2d/randomToMe2D

    def get_num_prune_for_coco(self, hw_shape):
        H,W = hw_shape
        if (self.layer_idx + 1)%4 == 0:
            H_new = math.floor(H * (1-self.prune_ratio))
            W_new = math.floor(W * (1-self.prune_ratio))
            new_hw_shape = (H_new, W_new)
            return new_hw_shape
        else:
            return hw_shape

    def guess_shape(self,hw_shape,L):
        """
        For coco eval
        """
        H,W = hw_shape
        for i in range(24):
            if H * W == L:
                return (H,W)
            else:
                H = math.floor(H * (1 - self.prune_ratio))
                W = math.floor(W * (1 - self.prune_ratio))
        return hw_shape

    def forward(self, x, hw_shape): # for classification
        B,L,D = x.shape
        # hw_shape = self.guess_shape(hw_shape,L) # coco
        H,W = hw_shape
        mixed_x = self.drop_path(self.mamba(self.norm(x), hw_shape)) # [B,L,C]
        # TODO:Insert Pruning/Merging method
        if self.method == "none" or self.method is None:
            pass
        elif self.method == "random":
            L_target = self.schedual.get(self.layer_idx,H*W)
            num_prune = H*W - L_target
            if num_prune > 0:
                prune_fn = self.random_prune(mixed_x, num_prune)
                mixed_x = prune_fn(mixed_x)
                x = prune_fn(x)
                hw_shape = (int(L_target**0.5), int(L_target**0.5))
        elif self.method == "tome":
            # L_target = self.schedual.get(self.layer_idx, H * W)
            # num_prune = H * W - L_target

            new_hw_shape = self.get_num_prune_for_coco(hw_shape)  # coco
            H_new, W_new = new_hw_shape  # coco
            num_prune = H * W - H_new * W_new
            if num_prune > 0:
                mixed_x, x = self.merger(mixed_x, x, None,num_prune)
                # hw_shape = (int(L_target ** 0.5), int(L_target ** 0.5))
                hw_shape = new_hw_shape
        elif self.method == "tome2d":
            L_target = self.schedual.get(self.layer_idx, H * W)
            num_prune = H * W - L_target
            if num_prune > 0:
                # L = H * W，这里也可以直接用 H_new = math.isqrt(L_target)
                H_new = math.isqrt(L - num_prune)  # 或 math.isqrt(L_target)

                # [B, L, D] -> [B, D, H, W]
                mixed_x_bchw = mixed_x.transpose(1, 2).reshape(B, D, H, W)
                x_bchw = x.transpose(1, 2).reshape(B, D, H, W)

                if mixed_x_bchw.shape[-2:] != (H_new, H_new):
                    prune_fn = self.merge2d(
                        x_bchw,
                        num_prune_w=H - H_new,
                        num_prune_h=H - H_new,
                    )
                    mixed_x_bchw = prune_fn(mixed_x_bchw)
                    x_bchw = prune_fn(x_bchw)

                # [B, D, H_new, H_new] -> [B, L_new, D]
                mixed_x = mixed_x_bchw.flatten(2).transpose(1, 2)
                x = x_bchw.flatten(2).transpose(1, 2)
                hw_shape = (int(L_target ** 0.5), int(L_target ** 0.5))
        elif self.method == "interpolate":
            L_target = self.schedual.get(self.layer_idx,H*W)
            num_prune = H*W - L_target
            if num_prune > 0:
                mixed_x = self.interpolate_prune(mixed_x, num_prune)
                x = self.interpolate_prune(x,num_prune)
                hw_shape = (int(L_target ** 0.5), int(L_target ** 0.5))
        elif self.method == "conv_tome2d":
            L_target = self.schedual.get(self.layer_idx, H * W)
            num_prune = H * W - L_target
            if num_prune > 0:
                # L = H * W，这里也可以直接用 H_new = math.isqrt(L_target)
                H_new = math.isqrt(L - num_prune)  # 或 math.isqrt(L_target)

                # [B, L, D] -> [B, D, H, W]
                mixed_x_bchw = mixed_x.transpose(1, 2).reshape(B, D, H, W)
                x_bchw = x.transpose(1, 2).reshape(B, D, H, W)

                if mixed_x_bchw.shape[-2:] != (H_new, H_new):
                    prune_fn = self.conv_tome_2d(
                        x_bchw,
                        target_hw=(H_new, H_new)
                    )
                    mixed_x_bchw = prune_fn(mixed_x_bchw)
                    x_bchw = prune_fn(x_bchw)

                # [B, D, H_new, H_new] -> [B, L_new, D]
                mixed_x = mixed_x_bchw.flatten(2).transpose(1, 2)
                x = x_bchw.flatten(2).transpose(1, 2)
                hw_shape = (int(L_target ** 0.5), int(L_target ** 0.5))
        elif self.method == "fixed_window_tome2dv2":
            L_target = self.schedual.get(self.layer_idx, H * W) # classification
            num_prune = H * W - L_target # classification
            H_new, W_new = (int(L_target ** 0.5), int(L_target ** 0.5))
            new_hw_shape = (H_new, W_new)

            # new_hw_shape = self.get_num_prune_for_coco(hw_shape) # coco
            # H_new, W_new = new_hw_shape # coco

            # num_prune = H * W - H_new*W_new
            if num_prune > 0:
                # L = H * W，这里也可以直接用 H_new = math.isqrt(L_target)
                # H_new = math.isqrt(L - num_prune)  # 或 math.isqrt(L_target)

                # [B, L, D] -> [B, D, H, W]
                mixed_x_bchw = mixed_x.transpose(1, 2).reshape(B, D, H, W)
                x_bchw = x.transpose(1, 2).reshape(B, D, H, W)

                if mixed_x_bchw.shape[-2:] != (H_new, W_new):
                    prune_fn = self.fixed_window_tome2dv2(
                        x_bchw,
                        num_prune_w=W - W_new,
                        num_prune_h=H - H_new,
                    )
                    mixed_x_bchw = prune_fn(mixed_x_bchw)
                    x_bchw = prune_fn(x_bchw)

                # [B, D, H_new, H_new] -> [B, L_new, D]
                mixed_x = mixed_x_bchw.flatten(2).transpose(1, 2)
                x = x_bchw.flatten(2).transpose(1, 2)
                # hw_shape = (int(L_target ** 0.5), int(L_target ** 0.5)) # classification
                hw_shape = new_hw_shape
        elif self.method == "HSA":
            L_target = self.schedual.get(self.layer_idx, H * W)
            num_prune = H * W - L_target
            if num_prune > 0:
                # [B, L, D] -> [B, D, H, W]
                mixed_x_bchw = mixed_x.transpose(1, 2)
                x_bchw = x.transpose(1, 2)

                mixed_x_bchw, x_bchw = self.HSA_pruner(mixed_x_bchw, x_bchw,num_prune)

                mixed_x = mixed_x_bchw.transpose(1, 2)
                x = x_bchw.transpose(1, 2)
                hw_shape = (int(L_target ** 0.5), int(L_target ** 0.5))
        elif self.method == "EViT":
            # L_target = self.schedual.get(self.layer_idx, H * W)
            # num_prune = H * W - L_target
            new_hw_shape = self.get_num_prune_for_coco(hw_shape)  # coco
            H_new, W_new = new_hw_shape  # coco
            num_prune = H * W - H_new * W_new
            if num_prune > 0:
                # [B, L, D] -> [B, D, H, W]
                mixed_x_bchw = mixed_x.transpose(1, 2)
                x_bchw = x.transpose(1, 2)

                mixed_x_bchw, x_bchw = self.EViT_pruner(mixed_x_bchw, x_bchw,num_prune)

                mixed_x = mixed_x_bchw.transpose(1, 2)
                x = x_bchw.transpose(1, 2)
            # hw_shape = (int(L_target ** 0.5), int(L_target ** 0.5))
            hw_shape = new_hw_shape
        elif self.method == "EViT2D":
            L_target = self.schedual.get(self.layer_idx, H * W)
            num_prune = H * W - L_target
            if num_prune > 0:
                # L = H * W，这里也可以直接用 H_new = math.isqrt(L_target)
                H_new = math.isqrt(L - num_prune)  # 或 math.isqrt(L_target)

                # [B, L, D] -> [B, D, H, W]
                mixed_x_bchw = mixed_x.transpose(1, 2).reshape(B, D, H, W)
                x_bchw = x.transpose(1, 2).reshape(B, D, H, W)

                if mixed_x_bchw.shape[-2:] != (H_new, H_new):
                    prune_fn = self.EViT2D_pruner(
                        x_bchw,
                        num_prune_w=H - H_new,
                        num_prune_h=H - H_new,
                    )
                    mixed_x_bchw = prune_fn(mixed_x_bchw)
                    x_bchw = prune_fn(x_bchw)

                # [B, D, H_new, H_new] -> [B, L_new, D]
                mixed_x = mixed_x_bchw.flatten(2).transpose(1, 2)
                x = x_bchw.flatten(2).transpose(1, 2)
                hw_shape = (int(L_target ** 0.5), int(L_target ** 0.5))
        elif self.method == "fixed_window_evit2d":
            # L_target = self.schedual.get(self.layer_idx, H * W) # classification
            new_hw_shape = self.get_num_prune_for_coco(hw_shape) # coco
            H_new, W_new = new_hw_shape # coco
            # num_prune = H * W - L_target # classification
            num_prune = H * W - H_new*W_new
            if num_prune > 0:
                # L = H * W，这里也可以直接用 H_new = math.isqrt(L_target)
                # H_new = math.isqrt(L - num_prune)  # 或 math.isqrt(L_target)

                # [B, L, D] -> [B, D, H, W]
                mixed_x_bchw = mixed_x.transpose(1, 2).reshape(B, D, H, W)
                x_bchw = x.transpose(1, 2).reshape(B, D, H, W)

                if mixed_x_bchw.shape[-2:] != (H_new, W_new):
                    prune_fn = self.fixed_window_evit2d(
                        x_bchw,
                        num_prune_w=W - W_new,
                        num_prune_h=H - H_new,
                    )
                    mixed_x_bchw = prune_fn(mixed_x_bchw)
                    x_bchw = prune_fn(x_bchw)

                # [B, D, H_new, H_new] -> [B, L_new, D]
                mixed_x = mixed_x_bchw.flatten(2).transpose(1, 2)
                x = x_bchw.flatten(2).transpose(1, 2)
                # hw_shape = (int(L_target ** 0.5), int(L_target ** 0.5)) # classification
                hw_shape = new_hw_shape
        elif self.method == "randomToMe2D":
            L_target = self.schedual.get(self.layer_idx, H * W)
            num_prune = H * W - L_target
            if num_prune > 0:
                # L = H * W，这里也可以直接用 H_new = math.isqrt(L_target)
                H_new = math.isqrt(L - num_prune)  # 或 math.isqrt(L_target)

                # [B, L, D] -> [B, D, H, W]
                mixed_x_bchw = mixed_x.transpose(1, 2).reshape(B, D, H, W)
                x_bchw = x.transpose(1, 2).reshape(B, D, H, W)

                if mixed_x_bchw.shape[-2:] != (H_new, H_new):
                    prune_fn = self.RandomHardPruneSTORM_pruner(
                        x_bchw,
                        num_prune_w=H - H_new,
                        num_prune_h=H - H_new,
                    )
                    mixed_x_bchw = prune_fn(mixed_x_bchw)
                    x_bchw = prune_fn(x_bchw)

                # [B, D, H_new, H_new] -> [B, L_new, D]
                mixed_x = mixed_x_bchw.flatten(2).transpose(1, 2)
                x = x_bchw.flatten(2).transpose(1, 2)
                hw_shape = (int(L_target ** 0.5), int(L_target ** 0.5))
        mixed_x = mixed_x + x
        if self.with_dwconv:
            b, l, c = mixed_x.shape
            h, w = hw_shape
            mixed_x = mixed_x.reshape(b, h, w, c).permute(0, 3, 1, 2)
            mixed_x = self.dw(mixed_x)
            mixed_x = mixed_x.reshape(b, c, h * w).permute(0, 2, 1)
        return mixed_x, hw_shape

    # def forward(self, x, hw_shape): # for coco
    #     B,L,D = x.shape
    #     hw_shape = self.guess_shape(hw_shape,L) # coco
    #     H,W = hw_shape
    #     x_orig = x.clone()
    #     if self.method == "none" or self.method is None:
    #         new_hw_shape = hw_shape
    #     elif self.method == "random":
    #         pass
    #         # L_target = self.schedual.get(self.layer_idx,H*W)
    #         # num_prune = H*W - L_target
    #         # if num_prune > 0:
    #         #     prune_fn = self.random_prune(mixed_x, num_prune)
    #         #     mixed_x = prune_fn(mixed_x)
    #         #     x = prune_fn(x)
    #         #     hw_shape = (int(L_target**0.5), int(L_target**0.5))
    #     elif self.method == "tome":
    #         # L_target = self.schedual.get(self.layer_idx, H * W)
    #         # num_prune = H * W - L_target
    #         new_hw_shape = self.get_num_prune_for_coco(hw_shape)  # coco
    #         H_new, W_new = new_hw_shape  # coco
    #         num_prune = H * W - H_new * W_new
    #         if num_prune > 0:
    #             _, x = self.merger(x, x, None,num_prune)
    #             # hw_shape = (int(L_target ** 0.5), int(L_target ** 0.5))
    #             # hw_shape = new_hw_shape
    #     elif self.method == "tome2d":
    #         pass
    #         # L_target = self.schedual.get(self.layer_idx, H * W)
    #         # num_prune = H * W - L_target
    #         # if num_prune > 0:
    #         #     # L = H * W，这里也可以直接用 H_new = math.isqrt(L_target)
    #         #     H_new = math.isqrt(L - num_prune)  # 或 math.isqrt(L_target)
    #         #
    #         #     # [B, L, D] -> [B, D, H, W]
    #         #     mixed_x_bchw = mixed_x.transpose(1, 2).reshape(B, D, H, W)
    #         #     x_bchw = x.transpose(1, 2).reshape(B, D, H, W)
    #         #
    #         #     if mixed_x_bchw.shape[-2:] != (H_new, H_new):
    #         #         prune_fn = self.merge2d(
    #         #             x_bchw,
    #         #             num_prune_w=H - H_new,
    #         #             num_prune_h=H - H_new,
    #         #         )
    #         #         mixed_x_bchw = prune_fn(mixed_x_bchw)
    #         #         x_bchw = prune_fn(x_bchw)
    #         #
    #         #     # [B, D, H_new, H_new] -> [B, L_new, D]
    #         #     mixed_x = mixed_x_bchw.flatten(2).transpose(1, 2)
    #         #     x = x_bchw.flatten(2).transpose(1, 2)
    #         #     hw_shape = (int(L_target ** 0.5), int(L_target ** 0.5))
    #     elif self.method == "interpolate":
    #         pass
    #         # L_target = self.schedual.get(self.layer_idx,H*W)
    #         # num_prune = H*W - L_target
    #         # if num_prune > 0:
    #         #     mixed_x = self.interpolate_prune(mixed_x, num_prune)
    #         #     x = self.interpolate_prune(x,num_prune)
    #         #     hw_shape = (int(L_target ** 0.5), int(L_target ** 0.5))
    #     elif self.method == "conv_tome2d":
    #         pass
    #         # L_target = self.schedual.get(self.layer_idx, H * W)
    #         # num_prune = H * W - L_target
    #         # if num_prune > 0:
    #         #     # L = H * W，这里也可以直接用 H_new = math.isqrt(L_target)
    #         #     H_new = math.isqrt(L - num_prune)  # 或 math.isqrt(L_target)
    #         #
    #         #     # [B, L, D] -> [B, D, H, W]
    #         #     mixed_x_bchw = x.transpose(1, 2).reshape(B, D, H, W)
    #         #     x_bchw = x.transpose(1, 2).reshape(B, D, H, W)
    #         #
    #         #     if mixed_x_bchw.shape[-2:] != (H_new, H_new):
    #         #         prune_fn = self.conv_tome_2d(
    #         #             x_bchw,
    #         #             target_hw=(H_new, H_new)
    #         #         )
    #         #         mixed_x_bchw = prune_fn(mixed_x_bchw)
    #         #         x_bchw = prune_fn(x_bchw)
    #         #
    #         #     # [B, D, H_new, H_new] -> [B, L_new, D]
    #         #     mixed_x = mixed_x_bchw.flatten(2).transpose(1, 2)
    #         #     x = x_bchw.flatten(2).transpose(1, 2)
    #         #     hw_shape = (int(L_target ** 0.5), int(L_target ** 0.5))
    #     elif self.method == "fixed_window_tome2dv2":
    #         # L_target = self.schedual.get(self.layer_idx, H * W) # classification
    #         new_hw_shape = self.get_num_prune_for_coco(hw_shape) # coco
    #         H_new, W_new = new_hw_shape # coco
    #         # num_prune = H * W - L_target # classification
    #         num_prune = H * W - H_new*W_new
    #         if num_prune > 0:
    #             # L = H * W，这里也可以直接用 H_new = math.isqrt(L_target)
    #             # H_new = math.isqrt(L - num_prune)  # 或 math.isqrt(L_target)
    #
    #             # [B, L, D] -> [B, D, H, W]
    #             mixed_x_bchw = x.transpose(1, 2).reshape(B, D, H, W)
    #             x_bchw = x.transpose(1, 2).reshape(B, D, H, W)
    #
    #             if mixed_x_bchw.shape[-2:] != (H_new, W_new):
    #                 prune_fn = self.fixed_window_tome2dv2(
    #                     x_bchw,
    #                     num_prune_w=W - W_new,
    #                     num_prune_h=H - H_new,
    #                 )
    #                 mixed_x_bchw = prune_fn(mixed_x_bchw)
    #                 x_bchw = prune_fn(x_bchw)
    #
    #             # [B, D, H_new, H_new] -> [B, L_new, D]
    #             mixed_x = mixed_x_bchw.flatten(2).transpose(1, 2)
    #             x = x_bchw.flatten(2).transpose(1, 2)
    #             # hw_shape = (int(L_target ** 0.5), int(L_target ** 0.5)) # classification
    #     elif self.method == "HSA":
    #         pass
    #         # L_target = self.schedual.get(self.layer_idx, H * W)
    #         # num_prune = H * W - L_target
    #         # if num_prune > 0:
    #         #     # [B, L, D] -> [B, D, H, W]
    #         #     mixed_x_bchw = mixed_x.transpose(1, 2)
    #         #     x_bchw = x.transpose(1, 2)
    #         #
    #         #     mixed_x_bchw, x_bchw = self.HSA_pruner(mixed_x_bchw, x_bchw,num_prune)
    #         #
    #         #     mixed_x = mixed_x_bchw.transpose(1, 2)
    #         #     x = x_bchw.transpose(1, 2)
    #         #     hw_shape = (int(L_target ** 0.5), int(L_target ** 0.5))
    #     elif self.method == "EViT":
    #         # L_target = self.schedual.get(self.layer_idx, H * W)
    #         # num_prune = H * W - L_target
    #         new_hw_shape = self.get_num_prune_for_coco(hw_shape)  # coco
    #         H_new, W_new = new_hw_shape  # coco
    #         num_prune = H * W - H_new * W_new
    #         if num_prune > 0:
    #             # [B, L, D] -> [B, D, H, W]
    #             mixed_x_bchw = x.transpose(1, 2)
    #             x_bchw = x.transpose(1, 2)
    #
    #             mixed_x_bchw, x_bchw = self.EViT_pruner(mixed_x_bchw, x_bchw,num_prune)
    #
    #             mixed_x = mixed_x_bchw.transpose(1, 2)
    #             x = x_bchw.transpose(1, 2)
    #         # hw_shape = (int(L_target ** 0.5), int(L_target ** 0.5))
    #         # hw_shape = new_hw_shape
    #     elif self.method == "EViT2D":
    #         # L_target = self.schedual.get(self.layer_idx, H * W) # classification
    #         new_hw_shape = self.get_num_prune_for_coco(hw_shape) # coco
    #         H_new, W_new = new_hw_shape # coco
    #         # num_prune = H * W - L_target # classification
    #         num_prune = H * W - H_new*W_new
    #         if num_prune > 0:
    #             # L = H * W，这里也可以直接用 H_new = math.isqrt(L_target)
    #             # H_new = math.isqrt(L - num_prune)  # 或 math.isqrt(L_target)
    #
    #             # [B, L, D] -> [B, D, H, W]
    #             mixed_x_bchw = x.transpose(1, 2).reshape(B, D, H, W)
    #             x_bchw = x.transpose(1, 2).reshape(B, D, H, W)
    #
    #             if mixed_x_bchw.shape[-2:] != (H_new, W_new):
    #                 prune_fn = self.EViT2D_pruner(
    #                     x_bchw,
    #                     num_prune_w=W - W_new,
    #                     num_prune_h=H - H_new,
    #                 )
    #                 mixed_x_bchw = prune_fn(mixed_x_bchw)
    #                 x_bchw = prune_fn(x_bchw)
    #
    #             # [B, D, H_new, H_new] -> [B, L_new, D]
    #             mixed_x = mixed_x_bchw.flatten(2).transpose(1, 2)
    #             x = x_bchw.flatten(2).transpose(1, 2)
    #     elif self.method == "fixed_window_evit2d":
    #         # L_target = self.schedual.get(self.layer_idx, H * W) # classification
    #         new_hw_shape = self.get_num_prune_for_coco(hw_shape) # coco
    #         H_new, W_new = new_hw_shape # coco
    #         # num_prune = H * W - L_target # classification
    #         num_prune = H * W - H_new*W_new
    #         if num_prune > 0:
    #             # L = H * W，这里也可以直接用 H_new = math.isqrt(L_target)
    #             # H_new = math.isqrt(L - num_prune)  # 或 math.isqrt(L_target)
    #
    #             # [B, L, D] -> [B, D, H, W]
    #             mixed_x_bchw = x.transpose(1, 2).reshape(B, D, H, W)
    #             x_bchw = x.transpose(1, 2).reshape(B, D, H, W)
    #
    #             if mixed_x_bchw.shape[-2:] != (H_new, W_new):
    #                 prune_fn = self.fixed_window_evit2d(
    #                     x_bchw,
    #                     num_prune_w=W - W_new,
    #                     num_prune_h=H - H_new,
    #                 )
    #                 mixed_x_bchw = prune_fn(mixed_x_bchw)
    #                 x_bchw = prune_fn(x_bchw)
    #
    #             # [B, D, H_new, H_new] -> [B, L_new, D]
    #             mixed_x = mixed_x_bchw.flatten(2).transpose(1, 2)
    #             x = x_bchw.flatten(2).transpose(1, 2)
    #     elif self.method == "randomToMe2D":
    #         L_target = self.schedual.get(self.layer_idx, H * W)
    #         num_prune = H * W - L_target
    #         if num_prune > 0:
    #             # L = H * W，这里也可以直接用 H_new = math.isqrt(L_target)
    #             H_new = math.isqrt(L - num_prune)  # 或 math.isqrt(L_target)
    #
    #             # [B, L, D] -> [B, D, H, W]
    #             mixed_x_bchw = x.transpose(1, 2).reshape(B, D, H, W)
    #             x_bchw = x.transpose(1, 2).reshape(B, D, H, W)
    #
    #             if mixed_x_bchw.shape[-2:] != (H_new, H_new):
    #                 prune_fn = self.RandomHardPruneSTORM_pruner(
    #                     x_bchw,
    #                     num_prune_w=H - H_new,
    #                     num_prune_h=H - H_new,
    #                 )
    #                 mixed_x_bchw = prune_fn(mixed_x_bchw)
    #                 x_bchw = prune_fn(x_bchw)
    #
    #             # [B, D, H_new, H_new] -> [B, L_new, D]
    #             mixed_x = mixed_x_bchw.flatten(2).transpose(1, 2)
    #             x = x_bchw.flatten(2).transpose(1, 2)
    #             # hw_shape = (int(L_target ** 0.5), int(L_target ** 0.5))
    #
    #
    #     mixed_x = self.drop_path(self.mamba(self.norm(x), new_hw_shape)) # [B,L,C]
    #     # TODO:Insert Pruning/Merging method
    #
    #     mixed_x = F.interpolate(mixed_x.transpose(1, 2).reshape(B,D,new_hw_shape[0],new_hw_shape[1]), size=hw_shape, mode="nearest").flatten(2).transpose(1, 2)
    #     mixed_x = mixed_x + x_orig
    #     if self.with_dwconv:
    #         b, l, c = mixed_x.shape
    #         h, w = hw_shape
    #         mixed_x = mixed_x.reshape(b, h, w, c).permute(0, 3, 1, 2)
    #         mixed_x = self.dw(mixed_x)
    #         mixed_x = mixed_x.reshape(b, c, h * w).permute(0, 2, 1)
    #     return mixed_x