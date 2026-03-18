import torch
import torch.nn as nn
from typing import Callable, List, Optional


class RandomPrune(nn.Module):
    """
    随机剪枝模块：

    forward(x, num_prune) 返回一个 prune_fn，
    之后调用 prune_fn(x, residual, ...) 即可在 L 维上同步剪枝。

    - x: [B, L, D]，也可以视作 metric
    - num_prune: 需要剪掉的 token 数（沿 L 维）
    """

    def __init__(self, share_across_batch: bool = True):
        """
        Args:
            share_across_batch: 若为 True，同一批样本使用相同的随机索引；
                                若为 False，则每个样本独立随机。
        """
        super().__init__()
        self.share_across_batch = share_across_batch

    def forward(self, x: torch.Tensor, num_prune: int) -> Callable:
        """
        仅生成剪枝计划，返回 prune_fn，不直接对 x 做剪枝。

        Args:
            x: [B, L, D]
            num_prune: 需要剪掉的 token 数

        Returns:
            prune_fn: 一个函数，调用方式：
                      x_pruned, residual_pruned = prune_fn(x, residual)
        """
        assert x.dim() == 3, f"Expected x shape [B, L, D], got {x.shape}"
        B, L, D = x.shape
        assert 0 <= num_prune < L, f"num_prune 必须在 [0, {L})，当前为 {num_prune}"

        device = x.device
        keep_len = L - num_prune

        if self.share_across_batch:
            # 整个 batch 共享一套随机索引
            perm = torch.randperm(L, device=device)  # [L]
            idx_keep = perm[num_prune:]             # [keep_len]
            # 为了保持原始顺序，可按升序排序
            idx_keep, _ = torch.sort(idx_keep)      # [keep_len]
            # idx_prune 可选：如果你后面想统计或可视化
            idx_prune = perm[:num_prune]
        else:
            # 每个样本独立随机一套索引，shape: [B, keep_len]
            perms = []
            keep_list = []
            prune_list = []
            for _ in range(B):
                p = torch.randperm(L, device=device)
                perms.append(p)
                keep_i = p[num_prune:]
                keep_i, _ = torch.sort(keep_i)
                keep_list.append(keep_i)
                prune_list.append(p[:num_prune])

            idx_keep = torch.stack(keep_list, dim=0)   # [B, keep_len]
            idx_prune = torch.stack(prune_list, dim=0) # [B, num_prune]

        def prune_fn(*tensors: Optional[torch.Tensor]):
            """
            对传入的所有张量在 L 维上同步剪枝。

            每个张量要求：
            - shape 为 [B, L, D]
            - 或为 None（会原样返回 None）

            Returns:
                与输入数量相同的剪枝后张量（或 None）。
                若只传入一个 tensor，则直接返回该 tensor，而不是 tuple。
            """
            pruned: List[Optional[torch.Tensor]] = []

            for t in tensors:
                if t is None:
                    pruned.append(None)
                    continue

                assert t.dim() == 3, f"Expected tensor shape [B, L, D], got {t.shape}"
                assert t.size(1) == L, (
                    f"L 维不匹配：计划使用 L={L}，但 tensor.size(1)={t.size(1)}"
                )

                if self.share_across_batch:
                    # 共享索引：[B, L, D] -> [B, keep_len, D]
                    t_pruned = t[:, idx_keep, :]
                else:
                    # 每个样本一套索引：
                    # idx_keep: [B, keep_len]
                    # 需要做 batch gather
                    # 扩展 idx_keep 为 [B, keep_len, D] 用于 gather
                    idx_expand = idx_keep.unsqueeze(-1).expand(-1, -1, t.size(-1))
                    t_pruned = torch.gather(t, dim=1, index_idx=idx_expand)

                pruned.append(t_pruned)

            if len(pruned) == 1:
                return pruned[0]
            return tuple(pruned)

        # 如果你后面还想用到 idx_keep / idx_prune，可以一起返回：
        # return prune_fn, idx_keep, idx_prune
        return prune_fn

if __name__ == "__main__":
    B, L, D = 2, 16, 64
    x = torch.randn(B, L, D).cuda()
    residual = torch.randn(B, L, D).cuda()

    pruner = RandomPrune(share_across_batch=True).cuda()
    num_prune = 4

    # 1. 生成剪枝函数（不修改任何张量）
    prune_fn = pruner(x, num_prune)

    # 2. 在需要的地方实际剪枝
    x_pruned, residual_pruned = prune_fn(x, residual)

    print(x_pruned.shape)  # torch.Size([2, 12, 64])
    print(residual_pruned.shape)  # torch.Size([2, 12, 64])