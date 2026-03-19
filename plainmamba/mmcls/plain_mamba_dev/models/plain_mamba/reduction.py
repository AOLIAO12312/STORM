import yaml
from pathlib import Path
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

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

current_script_path = Path(__file__).resolve()
config_path = current_script_path.parent / "token_reduction" / "scheduals.yml"
if not config_path.exists():
    raise FileNotFoundError(f"config file not found: {config_path}")
with open(config_path, "r", encoding="utf-8") as f:
    scheduals = yaml.safe_load(f)


class TokenReduction(nn.Module):
    def __init__(self,method: str,schedual: str):
        super(TokenReduction, self).__init__()
        self.method = method
        self.schedual = scheduals[schedual]
        self.random_prune = RandomPrune(share_across_batch=True)
        self.interpolate_prune = InterpolatePrune(assume_square=True)
        self.merger = TokenMerge2Dv4(
            num_prune=0,
            if_prune=False,
            if_order=True,
            distance='cosine',
            merge_mode='sum',
            choose='max'
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
            window_size=5,
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

    def forward(self, x: torch.Tensor, mixed_x: torch.Tensor, hw_shape: tuple, layer_idx):
        B, L, D = x.shape
        H, W = hw_shape
        if self.method == "none" or self.method is None:
            pass
        elif self.method == "random":
            L_target = self.schedual.get(layer_idx, H * W)
            num_prune = H * W - L_target
            if num_prune > 0:
                prune_fn = self.random_prune(mixed_x, num_prune)
                mixed_x = prune_fn(mixed_x)
                x = prune_fn(x)
                hw_shape = (int(L_target ** 0.5), int(L_target ** 0.5))
        elif self.method == "tome":
            # L_target = self.schedual.get(layer_idx, H * W)
            # num_prune = H * W - L_target

            new_hw_shape = self.get_num_prune_for_coco(hw_shape)  # coco
            H_new, W_new = new_hw_shape  # coco
            num_prune = H * W - H_new * W_new
            if num_prune > 0:
                mixed_x, x = self.merger(mixed_x, x, None, num_prune)
                # hw_shape = (int(L_target ** 0.5), int(L_target ** 0.5))
                hw_shape = new_hw_shape
        elif self.method == "tome2d":
            L_target = self.schedual.get(layer_idx, H * W)
            num_prune = H * W - L_target
            if num_prune > 0:
                H_new = math.isqrt(L - num_prune)

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
            L_target = self.schedual.get(layer_idx, H * W)
            num_prune = H * W - L_target
            if num_prune > 0:
                mixed_x = self.interpolate_prune(mixed_x, num_prune)
                x = self.interpolate_prune(x, num_prune)
                hw_shape = (int(L_target ** 0.5), int(L_target ** 0.5))
        elif self.method == "conv_tome2d":
            L_target = self.schedual.get(layer_idx, H * W)
            num_prune = H * W - L_target
            if num_prune > 0:
                H_new = math.isqrt(L - num_prune)

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
            L_target = self.schedual.get(layer_idx, H * W)  # classification
            num_prune = H * W - L_target  # classification
            H_new, W_new = (int(L_target ** 0.5), int(L_target ** 0.5))
            new_hw_shape = (H_new, W_new)

            # new_hw_shape = self.get_num_prune_for_coco(hw_shape) # coco
            # H_new, W_new = new_hw_shape # coco

            # num_prune = H * W - H_new*W_new
            if num_prune > 0:

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
            L_target = self.schedual.get(layer_idx, H * W)
            num_prune = H * W - L_target
            if num_prune > 0:
                # [B, L, D] -> [B, D, H, W]
                mixed_x_bchw = mixed_x.transpose(1, 2)
                x_bchw = x.transpose(1, 2)

                mixed_x_bchw, x_bchw = self.HSA_pruner(mixed_x_bchw, x_bchw, num_prune)

                mixed_x = mixed_x_bchw.transpose(1, 2)
                x = x_bchw.transpose(1, 2)
                hw_shape = (int(L_target ** 0.5), int(L_target ** 0.5))
        elif self.method == "EViT":
            # L_target = self.schedual.get(layer_idx, H * W)
            # num_prune = H * W - L_target
            new_hw_shape = self.get_num_prune_for_coco(hw_shape)  # coco
            H_new, W_new = new_hw_shape  # coco
            num_prune = H * W - H_new * W_new
            if num_prune > 0:
                # [B, L, D] -> [B, D, H, W]
                mixed_x_bchw = mixed_x.transpose(1, 2)
                x_bchw = x.transpose(1, 2)

                mixed_x_bchw, x_bchw = self.EViT_pruner(mixed_x_bchw, x_bchw, num_prune)

                mixed_x = mixed_x_bchw.transpose(1, 2)
                x = x_bchw.transpose(1, 2)
            # hw_shape = (int(L_target ** 0.5), int(L_target ** 0.5))
            hw_shape = new_hw_shape
        elif self.method == "EViT2D":
            L_target = self.schedual.get(layer_idx, H * W)
            num_prune = H * W - L_target
            if num_prune > 0:
                H_new = math.isqrt(L - num_prune)

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
            # L_target = self.schedual.get(layer_idx, H * W) # classification
            new_hw_shape = self.get_num_prune_for_coco(hw_shape)  # coco
            H_new, W_new = new_hw_shape  # coco
            # num_prune = H * W - L_target # classification
            num_prune = H * W - H_new * W_new
            if num_prune > 0:

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
            L_target = self.schedual.get(layer_idx, H * W)
            num_prune = H * W - L_target
            if num_prune > 0:
                H_new = math.isqrt(L - num_prune)

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
        return x, mixed_x, hw_shape