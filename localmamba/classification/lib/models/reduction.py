import yaml
from pathlib import Path
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# TODO: Translate to English

from .token_reduction.FixedWindowToMe2D import FixedWindowToMe2Dv2
from .token_reduction.TokenMerge2Dv4 import TokenMerge2Dv4
from .token_reduction.EViT import EViTTokenPruning
from .token_reduction.HSA import HSA
from .token_reduction.EViT2D import EViT2DStructuredPruning
from .token_reduction.FixedWindowEViT2D import FixedWindowEViT2D
from .token_reduction.RandomHardPruneFixedWindowToMe2D import RandomHardPruneFixedWindowToMe2D
from .util import pool_downsample, random_prune_tokens, prune_tokens_by_index, pad_zeros
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
        # VMambaPruner
        self.fixed_window_tome2dv2 = FixedWindowToMe2Dv2(
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
        self.EViT2D_pruner = EViT2DStructuredPruning(score_mode="absmean", if_order=True)
        self.prune_strategy = strategies[prune_strategy]
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
        B, H, W, D = x.shape
        if self.method == "none" or self.method is None:
            x = x + drop_path(op(norm(x)))
        elif self.method == "scale":
            # pre_stage像素级进行nearest剪枝
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio, layer_idx=layer_idx, stage_idx=stage_idx)
            L = H * W
            H_new = math.isqrt(L - num_prune)
            if self.stage_idx == 0:
                if x.shape[-2:] != (H_new, H_new):
                    x = F.interpolate(x, size=[H_new, H_new], mode="nearest")  # nearest/bilinear/area
                x_op = op(norm(x))
            else:
                x_op = op(norm(x))
                if x.shape[-2:] != (H_new, H_new):
                    x = F.interpolate(x, size=[H_new, H_new], mode="nearest")  # nearest/bilinear/area
                    x_op = F.interpolate(x_op, size=[H_new, H_new], mode="nearest")
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
        elif self.method == "tome2d":
            x_op = op(norm(x))
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio, layer_idx=layer_idx, stage_idx=stage_idx)
            L = H * W
            H_new = math.isqrt(L - num_prune)
            if x.shape[-2:] != (H_new, H_new):
                prune_fn = self.merge2d(x, num_prune_w=H - H_new, num_prune_h=H - H_new)
                x_op = prune_fn(x_op)
                x = prune_fn(x)
            x = x + drop_path(x_op)
        elif self.method == "fixed_window_tome2dv2":
            x_op = op(norm(x))
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio, layer_idx=layer_idx, stage_idx=stage_idx)
            L = H * W
            H_new = math.isqrt(L - num_prune)
            if x.shape[-2:] != (H_new, H_new):
                prune_fn = self.fixed_window_tome2dv2(x.permute(0, 3, 1, 2), num_prune_w=H - H_new,
                                                      num_prune_h=H - H_new)
                x_op = prune_fn(x_op.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                x = prune_fn(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            x = x + drop_path(x_op)
        elif self.method == "tome":
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio, layer_idx=layer_idx, stage_idx=stage_idx)
            L = H * W
            H_new = W_new = math.isqrt(L - num_prune)  # classfication
            x_op = op(norm(x))
            if num_prune != 0:
                x_op_merged, x_merged = self.merger(x_op.permute(0, 3, 1, 2).view(B, D, H * W),
                                                    x.permute(0, 3, 1, 2).view(B, D, H * W), None,
                                                    num_prune)  # [B, D, L_kept]
                x = x_merged.permute(0, 2, 1).view(B, H_new, W_new, D)
                x_op = x_op_merged.permute(0, 2, 1).view(B, H_new, W_new, D)
            x = x + drop_path(x_op)
        elif self.method == "HSA":
            x_op = op(norm(x))
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio, layer_idx=layer_idx, stage_idx=stage_idx)
            L = H * W
            H_new = W_new = math.isqrt(L - num_prune)  # classfication
            if x.shape[-2:] != (H_new, W_new):
                x_op_merged, x_merged = self.HSA_pruner(x_op.permute(0, 3, 1, 2).view(B, D, H * W),
                                                        x.permute(0, 3, 1, 2).view(B, D, H * W), num_prune)
                x = x_merged.permute(0, 2, 1).view(B, H_new, W_new, D)
                x_op = x_op_merged.permute(0, 2, 1).view(B, H_new, W_new, D)
            x = x + drop_path(x_op)
        elif self.method == "EViT":
            x_op = op(norm(x))
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio, layer_idx=layer_idx, stage_idx=stage_idx)
            L = H * W
            H_new = W_new = math.isqrt(L - num_prune)  # classfication
            if x.shape[-2:] != (H_new, W_new):
                x_op_merged, x_merged = self.EViT_pruner(x_op.permute(0, 3, 1, 2).view(B, D, H * W),
                                                         x.permute(0, 3, 1, 2).view(B, D, H * W), num_prune)
                x = x_merged.permute(0, 2, 1).view(B, H_new, W_new, D)
                x_op = x_op_merged.permute(0, 2, 1).view(B, H_new, W_new, D)
            x = x + drop_path(x_op)
        elif self.method == "EViT2D":
            x_op = op(norm(x))
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio, layer_idx=layer_idx, stage_idx=stage_idx)
            L = H * W
            H_new = math.isqrt(L - num_prune)
            if x.shape[-2:] != (H_new, H_new):
                prune_fn = self.EViT2D_pruner(x.permute(0, 3, 1, 2), num_prune_w=H - H_new, num_prune_h=H - H_new)
                x_op = prune_fn(x_op.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                x = prune_fn(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            x = x + drop_path(x_op)
        elif self.method == "fixed_window_evit2d":
            x_op = op(norm(x))
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio, layer_idx=layer_idx, stage_idx=stage_idx)
            L = H * W
            H_new = math.isqrt(L - num_prune)
            if x.shape[-2:] != (H_new, H_new):
                prune_fn = self.fixed_window_evit2d(x.permute(0, 3, 1, 2), num_prune_w=H - H_new, num_prune_h=H - H_new)
                x_op = prune_fn(x_op.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                x = prune_fn(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            x = x + drop_path(x_op)
        elif self.method == "randomToMe2D":
            x_op = op(norm(x))
            num_prune = self.get_prune_num(mode="manual", size=H, ratio=self.prune_ratio, layer_idx=layer_idx, stage_idx=stage_idx)
            L = H * W
            H_new = math.isqrt(L - num_prune)
            if x.shape[-2:] != (H_new, H_new):
                prune_fn = self.RandomHardPruneSTORM_pruner(x.permute(0, 3, 1, 2), num_prune_w=H - H_new,
                                                            num_prune_h=H - H_new)
                x_op = prune_fn(x_op.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                x = prune_fn(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            x = x + drop_path(x_op)
        return x