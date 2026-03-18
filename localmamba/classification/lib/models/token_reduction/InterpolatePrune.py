import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class InterpolatePrune(nn.Module):
    """
    使用 F.interpolate(nearest) 的“插值剪枝”：
    - 输入:  x  [B, L, D]
    - 给定: num_prune
    - 要求: L_new = L - num_prune 为完全平方数
    - 输出: x_pruned [B, L_new, D]
    """

    def __init__(self, assume_square: bool = True):
        """
        Args:
            assume_square: True 时假设 L = H * W 且 H == W
        """
        super().__init__()
        self.assume_square = assume_square

    def forward(self, x: torch.Tensor, num_prune: int) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
            num_prune: 要剪掉的 token 数 (沿 L 维)

        Returns:
            x_pruned: [B, L_new, D], 其中 L_new = L - num_prune 且为平方数
        """
        assert x.dim() == 3, f"Expected x shape [B, L, D], got {x.shape}"
        B, L, D = x.shape

        # 计算新的长度，并保证是平方数
        L_new = L - num_prune
        assert L_new > 0, f"L_new = L - num_prune 必须 > 0, got {L_new}"

        side_new = int(math.sqrt(L_new))
        assert side_new * side_new == L_new, (
            f"L_new = {L_new} 不是完全平方数，请调整 num_prune"
        )
        H_new = W_new = side_new

        # 原始 H, W
        if self.assume_square:
            side = int(math.sqrt(L))
            assert side * side == L, (
                f"当前 L = {L} 不是完全平方数，无法假设为 H=W 的方阵"
            )
            H = W = side
        else:
            # 如果未来你有非方阵的情况，可以在这里改成显式传入 H, W
            raise NotImplementedError(
                "当前实现仅支持 L = H*W 且 H=W 的情况，如需非方阵请显式提供 H, W"
            )

        # [B, L, D] -> [B, H, W, D] -> [B, D, H, W]
        x_2d = x.view(B, H, W, D).permute(0, 3, 1, 2)  # [B, D, H, W]

        # 使用最近邻插值到 [H_new, W_new]
        x_down = F.interpolate(
            x_2d,
            size=(H_new, W_new),
            mode="nearest",
        )  # [B, D, H_new, W_new]

        # [B, D, H_new, W_new] -> [B, H_new, W_new, D] -> [B, L_new, D]
        x_pruned = x_down.permute(0, 2, 3, 1).reshape(B, L_new, D)

        return x_pruned
