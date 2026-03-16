import yaml
from pathlib import Path
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from .token_reduction.GridToMeDownSample import GridToMePlanBCHW
from .token_reduction.FlexToMePlanBCHWv2 import FlexiToMePlanBCHWv2
from .token_reduction.ToMe2D import ToMe2D
from .token_reduction.ToMe2Dv2 import ToMe2Dv2
from .token_reduction.WToMe2D import AnyScaleToMe2D
from .token_reduction.HybridScaleToMe2D import HybridScaleToMe2D
from .token_reduction.ConvToMe2D import ConvToMe2D
from .token_reduction.TopoToMe2D import TopoToMe2D
from .token_reduction.AdptivaPooling2D import AdaptivePool2D
from .token_reduction.WindowToMe2D import AdaptiveWindowToMe2D
from .token_reduction.FixedWindowToMe2D import FixedWindowToMe2D
from .token_reduction.FixedWindowToMe1D import FixedWindowToMe1D
from .token_reduction.FixedWindowToMe2Dv2 import FixedWindowToMe2Dv2
from .token_reduction.FixedWindowToMe2Dv3 import FixedWindowToMe2Dv3
from .token_reduction.HSA import HSA
from .token_reduction.EViT import EViTTokenPruning
from .token_reduction.EViT2D import EViT2DStructuredPruning
from .token_reduction.RandomHardPruneFixedWindowToMe2D import RandomHardPruneFixedWindowToMe2D
from .token_reduction.FixedWindowEViT2D import FixedWindowEViT2D
from .token_reduction.TokenMerge2Dv4 import TokenMerge2Dv4,pad_zeros
from .utils import pool_downsample, random_prune_tokens, prune_tokens_by_index
current_script_path = Path(__file__).resolve()
config_path = current_script_path.parent / "token_reduction" / "strategies.yml"
if not config_path.exists():
    raise FileNotFoundError(f"config file not found: {config_path}")
with open(config_path, "r", encoding="utf-8") as f:
    strategies = yaml.safe_load(f)

class TokenReduction(nn.Module):
    def __init__(self, method: str = None, prune_strategy: str = None, prune_ratio: float = .0):
        super(TokenReduction, self).__init__()
        self.prune_ratio = prune_ratio
        if method is None:
            raise ValueError("Parameter 'method' cannot be empty. Please specify a reduction method.")
        if prune_strategy is None:
            raise ValueError("Parameter 'prune_strategy' cannot be empty. Please specify a reduction strategy.")
        self.merger = TokenMerge2Dv4(
            num_prune=0,
            if_prune=True,
            if_order=True,
            distance='cosine',
            merge_mode='sum',
            choose='max'
        )
        self.grid_tome_down = GridToMePlanBCHW(win=2, distance='cosine', pos_lambda=0.1, weighting='softmax')
        # self.flex_tome = FlexiToMePlanBCHW(
        #     kernel=2,
        #     distance='cosine',
        #     weighting='softmax',
        #     alpha=1.0,
        #     beta=1,
        #     pos_lambda=0,
        #     gate_tau=None,
        # )
        self.merge2d = ToMe2D(if_order=True, distance='l1', merge_mode='sum')
        self.merge2dv2 = ToMe2Dv2(
            if_order=True,
            distance='cosine',
            merge_mode='sum',
            use_importance=True,
            imp_lambda=0.5,
        )
        self.WToMe2D = AnyScaleToMe2D(imp_mode="ones")
        self.flex_tome = FlexiToMePlanBCHWv2(
            kernel=2,
            distance='cosine',
            weighting='softmax',
            alpha=25.0,
            beta=1.0,
            pos_lambda=0.0,
            gate_tau=None,
            align_corners=False,
            requires_grad_plan=False,
            strategy="flex",
        )
        self.conv_tome_2d = ConvToMe2D(
            imp_mode="l2",
            alpha=6.0,
            beta=1.0,
        )
        self.hybrid_scale_tome2d = HybridScaleToMe2D(
            rr_tome_max=0.20,
            tome_distance="cosine",
            imp_mode="l2"
        )
        self.topo_tome2d = TopoToMe2D(
            imp_mode="l2",
            alpha=1.0,
            beta=1.0,
            local_window=(3, 3),
            radius_factor=2.0,
        )
        self.adaptive_pool2d = AdaptivePool2D()
        self.window_tome2d = AdaptiveWindowToMe2D(if_order=True, distance='cosine', merge_mode='sum',
                                                  max_pad_h=2, max_pad_w=2)
        self.fixed_window_tome2d = FixedWindowToMe2D(
            window_size=9,
            distance='l1',
            merge_mode='sum',
            if_prune=False
        )
        # storm
        self.fixed_window_tome2dv2 = FixedWindowToMe2Dv2(
            if_prune=False,
            distance='l1',
            merge_mode='sum',
            window_size=5,
        )
        self.fixed_window_tome2dv3 = FixedWindowToMe2Dv3(
            if_prune=False,
            distance='l1',
            merge_mode='sum',
            window_size=5,
        )
        self.fixed_window_tome1d = FixedWindowToMe1D(
            window_size=5,
            distance="cosine",
            merge_mode="sum",
            if_prune=False,
        )
        self.HSA_pruner = HSA()
        self.EViT_pruner = EViTTokenPruning()
        self.EViT2D_pruner = EViT2DStructuredPruning(score_mode="absmean", if_order=True)
        self.RandomHardPruneSTORM_pruner = RandomHardPruneFixedWindowToMe2D(
            window_size=6,
            if_order=True
        )
        self.fixed_window_evit2d = FixedWindowEViT2D(
            window_size=5,
            score_mode='absmean'
        )
        self.prune_strategy = strategies[prune_strategy]
        self.quater_prune_strategy = strategies["quater_prune_strategy_0"]
        self.method = method  # scale/quater/tome/pooling/random/none/grid_tome/flex_tome/tome2d/hybrid/tome2dv2/hybrid_scale_tome2d/conv_tome2d/topo_tome2d/a_pooling/
        # adaptive_window_tome2d/fixed_window_tome2d/fixed_window_tome1d/fixed_window_tome2dv2/fixed_window_tome2dv3/HSA/EViT/EViT2D/fixedWindowEViT2D
        # fixed_window_tome2dv2 is vmambapruner

    def get_prune_num(self, mode, size: int = None, ratio: float = None, layer_idx: int = None, stage_idx: int = None):
        if layer_idx is None or stage_idx is None:
            raise ValueError("Parameter 'layer_idx' and 'stage_idx' cannot be empty.")
        if mode == "manual":
            return self.prune_strategy[stage_idx][layer_idx]
        elif mode == "auto":
            assert size is not None and ratio is not None and 0 <= ratio <= 1, \
                "size and ratio must be provided and ratio must be between 0 and 1 for auto mode"
            if layer_idx % 2 == 1 and stage_idx != 3:
                size_new = int(size * (1 - ratio))
                return size ** 2 - size_new ** 2
            else:
                return 0
        else:
            raise ValueError(f"Unsupported mode: {mode}. Supported modes are 'manual' and 'auto'.")

    def forward(self, x: torch.Tensor, layer_idx: int = None, stage_idx: int = None, drop_path: nn.Module=None, op: nn.Module=None, norm: nn.Module=None) -> torch.Tensor:
        if layer_idx is None or stage_idx is None:
            raise ValueError("Parameter 'layer_idx' and 'stage_idx' cannot be empty.")
        if drop_path is None or op is None or norm is None:
            raise ValueError("Parameter 'drop_path' and 'norm' and 'op' cannot be empty.")
        B, D, H, W = x.shape
        if self.method == "none" or self.method is None:
            x = x + drop_path(op(norm(x)))
        elif self.method == "scale":
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio)  # classfication
            L = H * W
            H_new = W_new = math.isqrt(L - num_prune)
            # H_new, W_new = self.get_prune_HW(H,W) # coco
            if self.stage_idx == 0 and False:
                if x.shape[-2:] != (H_new, W_new):
                    x = F.interpolate(x, size=[H_new, W_new], mode="area")  # nearest/bilinear/area
                x_op = op(norm(x))
            else:
                x_op = op(norm(x))
                if x.shape[-2:] != (H_new, W_new):
                    x = F.interpolate(x, size=[H_new, W_new], mode="nearest")  # nearest/bilinear/area
                    x_op = F.interpolate(x_op, size=[H_new, W_new], mode="nearest")
            x = x + drop_path(x_op)
        elif self.method == "tome":
            # num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio)
            L = H * W
            # H_new = W_new = math.isqrt(L - num_prune)  # classfication
            H_new, W_new = self.get_prune_HW(H, W)  # coco
            x_op = op(norm(x))
            if H * W - H_new * W_new != 0:
                # x_op_merged, x_merged = self.merger(x_op.view(B, D, H * W), x.view(B, D, H * W), None,
                #                                     num_prune)  # [B, D, L_kept]
                x_op_merged, x_merged = self.merger(x_op.view(B, D, H * W), x.view(B, D, H * W), None,
                                                    H * W - H_new * W_new)  # [B, D, L_kept]
                x = x_merged.view(B, D, H_new, W_new)
                x_op = x_op_merged.view(B, D, H_new, W_new)
            x = x + drop_path(x_op)
        elif self.method == "random":
            x_op = op(norm(x))
            num_prune = self.prune_strategy[self.stage_idx][self.layer_idx]
            x_op, sorted_idx = random_prune_tokens(x_op.view(B, D, H * W), num_prune)
            if sorted_idx is not None:
                x = prune_tokens_by_index(x.flatten(2), sorted_idx)
                x = pad_zeros(x)
                L = x.shape[-1]
                H = math.isqrt(L)
                x = x.view(B, D, H, H)
            x = x + drop_path(x_op.view(B, D, H, H))
        elif self.method == "quater":
            # if_prune = self.quater_prune_strategy[self.stage_idx][self.layer_idx]
            if_prune = True
            if if_prune:
                x_prune = F.interpolate(x, size=[H // 2, W // 2], mode="nearest")
                x_op = op(norm(x_prune))
                x_op = F.interpolate(x_op, size=[H, W], mode="nearest")
            else:
                x_op = op(norm(x))
            x = x + drop_path(x_op)
        elif self.method == "pooling":
            if_prune = self.quater_prune_strategy[self.stage_idx][self.layer_idx]
            if if_prune == 1:
                x_prune = pool_downsample(x, mode='max')  # 或 'max' / 'nearest' / 'lp'
                x_op = op(norm(x_prune))
                x_op = F.interpolate(x_op, size=[H, W], mode="nearest")
            else:
                x_op = op(norm(x))
            x = x + drop_path(x_op)
        elif self.method == "grid_tome":
            x_op = op(norm(x))
            if self.stage_idx == 0 and self.layer_idx == 0:
                prune = self.grid_tome_down(metric=x_op)
                x_op = prune(x_op, mode='wmean')
                x = prune(x, mode='wmean')
            x = x + drop_path(x_op)
        elif self.method == "flex_tome":
            x_op = op(norm(x))
            # num_prune需要自动生成，生成的大小必须符合正方形要求
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio)
            L = H * W
            H = math.isqrt(L - num_prune)
            if x.shape[-2:] != [H, H]:
                prune_fn = self.flex_tome(metric=x_op, target_hw=(H, H))
                x_op = prune_fn(x_op, mode='wsum')
                x = prune_fn(x, mode='wsum')
            x = x + drop_path(x_op)
        elif self.method == "tome2d":
            x_op = op(norm(x))
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio)
            L = H * W
            H_new = math.isqrt(L - num_prune)
            if x.shape[-2:] != (H_new, H_new):
                prune_fn = self.merge2d(x, num_prune_w=H - H_new, num_prune_h=H - H_new)
                x_op = prune_fn(x_op)
                x = prune_fn(x)
            x = x + drop_path(x_op)
        elif self.method == "hybrid":
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio)
            L = H * W
            H_new = math.isqrt(L - num_prune)
            if self.stage_idx == 0:
                if x.shape[-2:] != (H_new, H_new):
                    x = F.interpolate(x, size=[H_new, H_new], mode="nearest")  # nearest/bilinear/area
                x_op = op(norm(x))
            else:
                x_op = op(norm(x))
                if x.shape[-2:] != (H_new, H_new):
                    prune_fn = self.merge2d(x, num_prune_w=H - H_new, num_prune_h=H - H_new)
                    x_op = prune_fn(x_op)
                    x = prune_fn(x)
            x = x + drop_path(x_op)
        elif self.method == "tome2dv2":
            x_op = op(norm(x))
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio)
            L = H * W
            H_new = math.isqrt(L - num_prune)
            if x.shape[-2:] != (H_new, H_new):
                prune_fn = self.merge2dv2(x, num_prune_w=H - H_new, num_prune_h=H - H_new)
                x_op = prune_fn(x_op)
                x = prune_fn(x)
            x = x + drop_path(x_op)
        elif self.method == "anytome2d":
            x_op = op(norm(x))
            # num_prune must be cut to square
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio)
            L = H * W
            H = math.isqrt(L - num_prune)
            if x.shape[-2:] != [H, H]:
                prune_fn = self.WToMe2D(metric=x, target_hw=(H, H))
                x_op = prune_fn(x_op)
                x = prune_fn(x)
            x = x + drop_path(x_op)
        elif self.method == "hybrid_scale_tome2d":
            x_op = op(norm(x))
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio)
            L = H * W
            H_new = math.isqrt(L - num_prune)
            if x.shape[-2:] != (H_new, H_new):
                prune_fn = self.hybrid_scale_tome2d(x, target_hw=(H_new, H_new))
                x_op = prune_fn(x_op)
                x = prune_fn(x)
            x = x + drop_path(x_op)
        elif self.method == "conv_tome2d":
            x_op = op(norm(x))
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio)
            L = H * W
            H_new = math.isqrt(L - num_prune)
            if x.shape[-2:] != (H_new, H_new):
                prune_fn = self.conv_tome_2d(x, target_hw=(H_new, H_new))
                x_op = prune_fn(x_op)
                x = prune_fn(x)
            x = x + drop_path(x_op)
        elif self.method == "topo_tome2d":
            x_op = op(norm(x))
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio)
            L = H * W
            H_new = math.isqrt(L - num_prune)
            if x.shape[-2:] != (H_new, H_new):
                prune_fn = self.topo_tome2d(x, target_hw=(H_new, H_new))
                x_op = prune_fn(x_op)
                x = prune_fn(x)
            x = x + drop_path(x_op)
        elif self.method == "a_pooling":
            x_op = op(norm(x))
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio)
            L = H * W
            H_new = math.isqrt(L - num_prune)
            if x.shape[-2:] != (H_new, H_new):
                prune_fn = self.adaptive_pool2d(x, target_hw=(H_new, H_new))
                x_op = prune_fn(x_op)
                x = prune_fn(x)
            x = x + drop_path(x_op)
        elif self.method == "adaptive_window_tome2d":
            x_op = op(norm(x))
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio)
            L = H * W
            H_new = math.isqrt(L - num_prune)
            if x.shape[-2:] != (H_new, H_new):
                prune_fn = self.window_tome2d(x, num_prune_w=H - H_new, num_prune_h=H - H_new)
                x_op = prune_fn(x_op)
                x = prune_fn(x)
            x = x + drop_path(x_op)
        elif self.method == "fixed_window_tome2d":
            x_op = op(norm(x))
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio)
            L = H * W
            H_new = math.isqrt(L - num_prune)
            if x.shape[-2:] != (H_new, H_new):
                prune_fn = self.fixed_window_tome2d(x, num_prune_w=H - H_new, num_prune_h=H - H_new)
                x_op = prune_fn(x_op)
                x = prune_fn(x)
            x = x + drop_path(x_op)
        elif self.method == "fixed_window_tome1d":
            x_op = op(norm(x))
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio)
            L = H * W
            H_new = math.isqrt(L - num_prune)
            if x.shape[-2:] != (H_new, H_new):
                prune_fn = self.fixed_window_tome1d(x, num_prune_w=H - H_new, num_prune_h=H - H_new)
                x_op = prune_fn(x_op)
                x = prune_fn(x)
            x = x + drop_path(x_op)
        elif self.method == "fixed_window_tome2dv2":
            x_op = op(norm(x))
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio, layer_idx=layer_idx,stage_idx=stage_idx)
            L = H * W
            H_new = W_new = math.isqrt(L - num_prune)  # classfication
            # H_new, W_new = self.get_prune_HW(H, W) # coco
            if x.shape[-2:] != (H_new, W_new):
                prune_fn = self.fixed_window_tome2dv2(x, num_prune_w=W - W_new, num_prune_h=H - H_new)
                x_op = prune_fn(x_op)
                x = prune_fn(x)
            x = x + drop_path(x_op)
        elif self.method == "fixed_window_tome2dv3":
            x_op = op(norm(x))
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio)
            L = H * W
            H_new = math.isqrt(L - num_prune)
            if x.shape[-2:] != (H_new, H_new):
                prune_fn = self.fixed_window_tome2dv3(x, num_prune_w=H - H_new, num_prune_h=H - H_new)
                x_op = prune_fn(x_op)
                x = prune_fn(x)
            x = x + drop_path(x_op)
        elif self.method == "HSA":
            x_op = op(norm(x))
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio)
            L = H * W
            H_new = W_new = math.isqrt(L - num_prune)  # classfication
            if x.shape[-2:] != (H_new, W_new):
                x_op_merged, x_merged = self.HSA_pruner(x_op.view(B, D, H * W), x.view(B, D, H * W), num_prune)
                x = x_merged.view(B, D, H_new, W_new)
                x_op = x_op_merged.view(B, D, H_new, W_new)
            x = x + drop_path(x_op)
        elif self.method == "EViT":
            x_op = op(norm(x))
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio)
            L = H * W
            H_new = W_new = math.isqrt(L - num_prune)  # classfication
            # H_new, W_new = self.get_prune_HW(H, W)
            if x.shape[-2:] != (H_new, W_new):
                x_op_merged, x_merged = self.EViT_pruner(x_op.view(B, D, H * W), x.view(B, D, H * W),
                                                         H * W - H_new * W_new)
                x = x_merged.view(B, D, H_new, W_new)
                x_op = x_op_merged.view(B, D, H_new, W_new)
            x = x + drop_path(x_op)
        elif self.method == "EViT2D":
            x_op = op(norm(x))
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio)
            L = H * W
            # H_new = W_new = math.isqrt(L - num_prune) # classfication
            H_new, W_new = self.get_prune_HW(H, W)
            if x.shape[-2:] != (H_new, W_new):
                prune_fn = self.EViT2D_pruner(x, num_prune_w=H - H_new, num_prune_h=W - W_new)
                x_op = prune_fn(x_op)
                x = prune_fn(x)
            x = x + drop_path(x_op)
        elif self.method == "randomToMe2D":
            x_op = op(norm(x))
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio)
            L = H * W
            H_new = W_new = math.isqrt(L - num_prune)  # classfication
            # H_new, W_new = self.get_prune_HW(H, W)
            if x.shape[-2:] != (H_new, W_new):
                prune_fn = self.RandomHardPruneSTORM_pruner(x, num_prune_w=H - H_new, num_prune_h=W - W_new)
                x_op = prune_fn(x_op)
                x = prune_fn(x)
            x = x + drop_path(x_op)
        elif self.method == "fixedWindowEViT2D":
            x_op = op(norm(x))
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio)
            L = H * W
            H_new = W_new = math.isqrt(L - num_prune)  # classfication
            # H_new, W_new = self.get_prune_HW(H, W) # coco
            if x.shape[-2:] != (H_new, W_new):
                prune_fn = self.fixed_window_evit2d(x, num_prune_w=W - W_new, num_prune_h=H - H_new)
                x_op = prune_fn(x_op)
                x = prune_fn(x)
            x = x + drop_path(x_op)
        elif self.method == "adaptive_max_pooling":
            x_op = op(norm(x))
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio)
            L = H * W
            H_new = W_new = math.isqrt(L - num_prune)  # classfication
            # H_new, W_new = self.get_prune_HW(H, W) # coco
            if x.shape[-2:] != (H_new, W_new):
                x_op = F.adaptive_avg_pool2d(x_op, (H_new, W_new))
                x = F.adaptive_avg_pool2d(x, (H_new, W_new))
            x = x + drop_path(x_op)
        return x