import torch
import torch.nn as nn
from typing import Tuple


class AnyScaleToMe2D(nn.Module):
    """
    AnyScaleToMe2D: 任意 HxW -> H_out x W_out 的内容感知 2D token merging。

    - 用类似 AdaptiveAvgPool2d 的 2D bin 作为窗口，每个输出位置对应一个局部矩形窗口。
    - 每个窗口内的 token 用 importance 加权平均做 merge（importance 从 metric 计算）。
    - forward(metric, target_hw) 返回 prune_fn(x)，可对任意同形状 BCHW 特征做同样剪枝。
    - merge_wavg2d 提供和 ToMe merge_wavg 类似的 weighted-average 接口。

    完全向量化：没有对 H/W 或 window 的 Python for-loop。
    """

    def __init__(self, imp_mode: str = "l2", eps: float = 1e-6):
        """
        imp_mode:
            - 'l2':  importance = ||x||_2^2
            - 'l1':  importance = ||x||_1
            - 'mean_abs': importance = mean(|x|)
            - 'ones': 全 1（退化为 AdaptiveAvgPool 风格）
        """
        super().__init__()
        assert imp_mode in ("l2", "l1", "mean_abs", "ones")
        self.imp_mode = imp_mode
        self.eps = eps

    # ---------- 辅助：计算 input -> output bin 映射 ----------
    @staticmethod
    def _compute_bin_indices(H: int, W: int, H_out: int, W_out: int, device):
        """
        仿照 AdaptiveAvgPool2d 的 index 生成方式：
        每个输入空间位置 (i,j) 映射到一个输出 bin index in [0, H_out*W_out)
        """
        i = torch.arange(H, device=device)        # [H]
        j = torch.arange(W, device=device)        # [W]

        oi = torch.floor(i.float() * H_out / H).long()   # [H]
        oj = torch.floor(j.float() * W_out / W).long()   # [W]

        oi = torch.clamp(oi, 0, H_out - 1)
        oj = torch.clamp(oj, 0, W_out - 1)

        oi_grid = oi[:, None].expand(H, W)        # [H,W]
        oj_grid = oj[None, :].expand(H, W)        # [H,W]

        bin_idx = oi_grid * W_out + oj_grid       # [H,W] in [0, H_out*W_out)
        return bin_idx

    # ---------- 辅助：importance 计算 ----------
    def _importance(self, m: torch.Tensor):
        """
        m: [B, HW, C] 展开的 metric
        return: [B, HW] importance
        """
        if self.imp_mode == "l2":
            imp = m.pow(2).sum(dim=-1)
        elif self.imp_mode == "l1":
            imp = m.abs().sum(dim=-1)
        elif self.imp_mode == "mean_abs":
            imp = m.abs().mean(dim=-1)
        elif self.imp_mode == "ones":
            imp = torch.ones(m.shape[:-1], device=m.device, dtype=m.dtype)
        else:
            raise ValueError(f"Unsupported imp_mode {self.imp_mode}")
        return imp

    # ---------- 主接口：构建剪枝 plan ----------
    def forward(self, metric: torch.Tensor, target_hw: Tuple[int, int]):
        """
        根据 metric 和目标大小构建剪枝 plan，并返回 prune_fn(x)。

        Args:
            metric: [B, C, H, W]
            target_hw: (H_out, W_out)

        Returns:
            prune_fn(x): x -> [B, C, H_out, W_out]
        """
        assert metric.dim() == 4, "metric 必须是 BCHW"
        B, C, H, W = metric.shape
        H_out, W_out = target_hw
        assert 1 <= H_out <= H and 1 <= W_out <= W, "target_hw 必须 <= 原始尺寸"

        device = metric.device
        dtype = metric.dtype

        HW = H * W
        HW_out = H_out * W_out

        # 1) 计算每个 (i,j) 属于哪个输出 bin（窗口），完全 2D、本地化
        bin_idx = self._compute_bin_indices(H, W, H_out, W_out, device=device)  # [H,W]

        # 2) 展平 metric: [B, H, W, C] -> [B, HW, C]
        m = metric.permute(0, 2, 3, 1).contiguous().view(B, HW, C)

        # 3) 计算每个 token 的 importance: [B, HW]
        imp = self._importance(m)                     # [B, HW]
        imp_flat = imp.reshape(B * HW)                # [B*HW]

        # 4) 准备 scatter_add 用的全局 index
        #    bin_flat: [B, HW]，每个 batch 拥有各自的 [0, HW_out) bin
        bin_flat = bin_idx.view(1, HW).expand(B, HW)  # [B, HW]
        global_index = (
            torch.arange(B, device=device).unsqueeze(1) * HW_out + bin_flat
        )                                             # [B, HW]
        global_index_flat = global_index.reshape(B * HW)  # [B*HW]

        # 5) 预计算每个 bin 的 importance 总和，用作归一化因子
        total_bins = B * HW_out
        w_sum = torch.zeros(total_bins, device=device, dtype=dtype)
        w_sum.index_add_(0, global_index_flat, imp_flat)  # sum importance in each bin
        w_sum = w_sum.clamp_min(self.eps)

        # ---------- 返回剪枝函数 prune_fn ----------
        def prune_fn(x: torch.Tensor) -> torch.Tensor:
            """
            对 x 做 AnyScaleToMe2D 剪枝：
            x: [B, Cx, H, W]，B/H/W 必须和 metric 一致，Cx 可不同
            返回: [B, Cx, H_out, W_out]
            """
            assert x.dim() == 4, "x 必须是 BCHW"
            Bx, Cx, Hx, Wx = x.shape
            assert Bx == B and Hx == H and Wx == W, \
                f"x 形状 {x.shape} 必须与 metric 的 B,H,W 一致 {metric.shape}"

            # [B, H, W, Cx] -> [B*HW, Cx]
            x_perm = x.permute(0, 2, 3, 1).contiguous()
            x_flat = x_perm.view(B * HW, Cx)  # [B*HW, Cx]

            # 用 metric 的 importance 作为固定权重：imp_flat [B*HW]
            weighted_x_flat = x_flat * imp_flat.unsqueeze(-1)  # [B*HW, Cx]

            # 聚合到每个 bin 上
            y_num = torch.zeros(total_bins, Cx, device=device, dtype=dtype)
            y_num.index_add_(0, global_index_flat, weighted_x_flat)  # sum(weight * x)

            # 归一化
            y_flat = y_num / w_sum.unsqueeze(-1)  # [B*HW_out, Cx]

            # [B, H_out, W_out, Cx] -> [B, Cx, H_out, W_out]
            y = y_flat.view(B, H_out, W_out, Cx).permute(0, 3, 1, 2).contiguous()
            return y

        return prune_fn

    # ---------- 可选：类似 ToMe 的 merge_wavg2d ----------
    def merge_wavg2d(
        self,
        metric: torch.Tensor,
        x: torch.Tensor,
        size: torch.Tensor = None,
        target_hw: Tuple[int, int] = None,
    ):
        """
        Weighted-average merging 接口，和你原来的 ToMe2D.merge_wavg2d 类似。

        Args:
            metric: [B, C, H, W]，用于构建剪枝 plan
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
            size = torch.ones_like(x[:, :1])  # BCH1

        # x 按 size 加权后再剪枝
        x_weighted = x * size
        x_merged = prune_fn(x_weighted)
        size_merged = prune_fn(size)

        x_merged = x_merged / (size_merged + (size_merged == 0).to(x_merged.dtype))
        return x_merged, size_merged

if __name__ == "__main__":
    B, C, H, W = 2, 64, 14, 14
    x = torch.randn(B, C, H, W)

    pruner = AnyScaleToMe2D(imp_mode="l2")  # 也可以试试 'l1', 'mean_abs', 'ones'
    target_hw = (10, 10)

    # 用 x 自身当 metric（也可以用 backbone 某一层的 feature 当 metric）
    prune_fn = pruner(metric=x, target_hw=target_hw)

    y = prune_fn(x)
    print("input shape :", x.shape)  # [2, 64, 14, 14]
    print("output shape:", y.shape)  # [2, 64, 10, 10]
