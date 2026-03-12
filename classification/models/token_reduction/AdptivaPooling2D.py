import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AdaptivePool2D(nn.Module):
    """
    纯 AdaptiveAvgPool2d 的下采样方案，接口与 ConvToMe2D / TopoToMe2D 保持一致：

    - forward(metric, target_hw) -> prune_fn
    - prune_fn(x): [B, Cx, H, W] -> [B, Cx, H_out, W_out]
    - merge_wavg2d(metric, x, size=None, target_hw)
    """

    def __init__(self):
        super().__init__()

    def forward(self, metric: torch.Tensor, target_hw: Tuple[int, int]):
        """
        Args:
            metric: [B, C, H, W]，这里只用于获取 B,H,W 信息
            target_hw: (H_out, W_out)

        Returns:
            prune_fn(x): x -> [B, Cx, H_out, W_out]
        """
        assert metric.dim() == 4, "metric 必须是 BCHW"
        B, C, H, W = metric.shape
        H_out, W_out = target_hw
        assert 1 <= H_out <= H and 1 <= W_out <= W, "target_hw 必须 <= 原始尺寸"

        # 不剪枝：直接 identity
        if H_out == H and W_out == W:
            def identity_fn(x: torch.Tensor) -> torch.Tensor:
                return x
            return identity_fn

        # 这里 metric 其实不参与下采样，只是接口对齐
        def prune_fn(x: torch.Tensor) -> torch.Tensor:
            """
            x: [B, Cx, H, W] -> [B, Cx, H_out, W_out]
            """
            assert x.dim() == 4, "x 必须是 BCHW"
            Bx, Cx, Hx, Wx = x.shape
            assert Bx == B and Hx == H and Wx == W, \
                f"x 形状 {x.shape} 必须与 metric 的 B,H,W 一致"

            # 直接用 AdaptiveAvgPool2d 下采样到目标大小
            y = F.adaptive_avg_pool2d(x, output_size=(H_out, W_out))
            return y

        return prune_fn

    def merge_wavg2d(
        self,
        metric: torch.Tensor,
        x: torch.Tensor,
        size: torch.Tensor = None,
        target_hw: Tuple[int, int] = None,
    ):
        """
        与 ToMe2D 风格一致的接口，只是内部用 avg_pool 实现。

        Args:
            metric: [B, C, H, W]，仅用于构建 prune_fn（获取 B,H,W）
            x:      [B, C, H, W]，要被合并的特征
            size:   [B, 1, H, W] 或 [B, C, H, W] 的 size map (可选)，若为 None 则用全 1
            target_hw: (H_out, W_out)

        Returns:
            x_merged:   [B, C, H_out, W_out]
            size_merged:[B, 1, H_out, W_out]
        """
        assert target_hw is not None, "必须提供 target_hw=(H_out, W_out)"
        prune_fn = self.forward(metric, target_hw=target_hw)

        if size is None:
            size = torch.ones_like(x[:, :1])  # [B, 1, H, W]

        # 带 size 权重的 avg_pool：先加权，再 pool，再除以 pooled size
        x_weighted = x * size
        x_merged = prune_fn(x_weighted)
        size_merged = prune_fn(size)

        x_merged = x_merged / (size_merged + (size_merged == 0).to(x_merged.dtype))
        return x_merged, size_merged


if __name__ == "__main__":
    B, C, H, W = 2, 64, 14, 14
    x = torch.randn(B, C, H, W)

    pruner = AdaptivePool2D()

    target_hw = (10, 10)
    prune_fn = pruner(metric=x, target_hw=target_hw)

    y = prune_fn(x)
    print("input shape :", x.shape)  # [2, 64, 14, 14]
    print("output shape:", y.shape)  # [2, 64, 10, 10]

    # merge_wavg2d 用法
    metric2 = torch.randn_like(x)
    size = torch.ones(B, 1, H, W)
    y2, s2 = pruner.merge_wavg2d(metric2, x, size=size, target_hw=target_hw)
    print("merge_wavg2d:", y2.shape, s2.shape)
