import torch
import torch.nn as nn
from typing import Tuple


class ConvToMe2D(nn.Module):
    """
    Convolutional Token Merging 2D (ConvToMe2D)

    统一的任意尺度 2D token 合并模块：
      - 任意 [B, C, H, W] -> [B, C, H_out, W_out]
      - 输出网格严格 2D，使用类似 AdaptiveAvgPool2d 的局部窗口划分
      - 每个窗口内使用 ToMe 风格的内容权重：
            score_i = alpha * log( importance(x_i) ) + beta * log( centrality(x_i) )
        在每个窗口内做 softmax(score_i) 得到 w_i，
        输出 y_bin = sum_i w_i * x_i

    特性：
      - 小剪枝时，多数 bin 只有 1 个 token 或极少量 token，近似 identity / almost lossless
      - 大剪枝时，在局部窗口内做内容感知聚合，行为类似卷积式 pooling，鲁棒性好
      - 全流程向量化，无 Python 级 for-loop
    """

    def __init__(
        self,
        imp_mode: str = "l2",   # 'l2' | 'l1' | 'mean_abs' | 'ones'
        alpha: float = 1.0,     # importance 权重系数
        beta: float = 1.0,      # centrality (cosine) 权重系数
        eps: float = 1e-6,
    ):
        super().__init__()
        assert imp_mode in ("l2", "l1", "mean_abs", "ones")
        self.imp_mode = imp_mode
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    # ---------- 2D adaptive bin 映射 ----------
    @staticmethod
    def _compute_bin_indices(H: int, W: int, H_out: int, W_out: int, device: torch.device):
        """
        仿照 AdaptiveAvgPool2d 的 index 生成方式：
        每个输入空间位置 (h, w) 映射到一个输出 bin index in [0, H_out * W_out)
        """
        i = torch.arange(H, device=device)   # [H]
        j = torch.arange(W, device=device)   # [W]

        oi = torch.floor(i.float() * H_out / H).long()   # [H]
        oj = torch.floor(j.float() * W_out / W).long()   # [W]

        oi = torch.clamp(oi, 0, H_out - 1)
        oj = torch.clamp(oj, 0, W_out - 1)

        oi_grid = oi[:, None].expand(H, W)   # [H, W]
        oj_grid = oj[None, :].expand(H, W)   # [H, W]

        bin_idx = oi_grid * W_out + oj_grid  # [H, W]
        return bin_idx  # [H, W]

    # ---------- importance ----------
    def _importance(self, m: torch.Tensor) -> torch.Tensor:
        """
        m: [B, HW, C] 展开的 metric
        返回: [B, HW] importance
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
        return imp  # [B, HW]

    @staticmethod
    def _normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    # ---------- 构建剪枝 plan ----------
    def forward(self, metric: torch.Tensor, target_hw: Tuple[int, int]):
        """
        根据 metric 和目标大小 target_hw 构建 ConvToMe2D 剪枝 plan，并返回 prune_fn(x)。

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

        # 不剪就直接 identity
        if H_out == H and W_out == W:
            def identity_fn(x: torch.Tensor) -> torch.Tensor:
                return x
            return identity_fn

        HW = H * W
        HW_out = H_out * W_out

        # 1) 计算 input -> output bin 映射
        bin_idx = self._compute_bin_indices(H, W, H_out, W_out, device=device)  # [H, W]
        bin_flat = bin_idx.view(1, HW).expand(B, HW)                             # [B, HW]

        # 不同 batch 占用不同 bin 段： [0..HW_out-1], [HW_out..2HW_out-1], ...
        global_index = (
            torch.arange(B, device=device).unsqueeze(1) * HW_out + bin_flat
        )  # [B, HW]
        global_index_flat = global_index.reshape(B * HW)  # [B*HW]
        total_bins = B * HW_out

        # 2) 展平 metric: [B, C, H, W] -> [B, H, W, C] -> [B, HW, C] -> [B*HW, C]
        m = metric.permute(0, 2, 3, 1).contiguous().view(B, HW, C)  # [B, HW, C]
        m_flat = m.view(B * HW, C)                                  # [B*HW, C]

        # 3) importance: [B, HW] -> [B*HW]
        imp = self._importance(m)          # [B, HW]
        imp_flat = imp.reshape(B * HW)     # [B*HW]
        imp_flat_clamped = imp_flat.clamp_min(self.eps)

        # 4) 计算每个 bin 的中心特征 mu_bin（简单平均）
        sum_x = torch.zeros(total_bins, C, device=device, dtype=dtype)
        ones = torch.ones(B * HW, device=device, dtype=dtype)

        sum_x.index_add_(0, global_index_flat, m_flat)   # [total_bins, C]
        count = torch.zeros(total_bins, device=device, dtype=dtype)
        count.index_add_(0, global_index_flat, ones)     # [total_bins]
        count = count.clamp_min(self.eps)

        mu_flat = sum_x / count.unsqueeze(-1)            # [total_bins, C]

        # 为每个 token 取所属 bin 的中心特征 μ_bin
        mu_token_flat = mu_flat.index_select(0, global_index_flat)  # [B*HW, C]

        # 5) ToMe 风格 centrality：cos(x_i, mu_bin)
        m_norm = self._normalize(m_flat, self.eps)            # [B*HW, C]
        mu_norm = self._normalize(mu_token_flat, self.eps)    # [B*HW, C]
        cent_flat = (m_norm * mu_norm).sum(dim=-1)            # [B*HW], in [-1, 1]

        # 映射到正数区间 [0, 1]，避免负权重
        cent_pos_flat = (cent_flat + 1.0) * 0.5
        cent_pos_flat = cent_pos_flat.clamp_min(1e-3)

        # 6) 组合 importance + centrality -> score，再在每个 bin 内做 softmax
        #    用 log 形式避免 importance 尺度差异过大
        log_imp = imp_flat_clamped.log()           # log importance
        log_cent = cent_pos_flat.log()             # log centrality

        score = self.alpha * log_imp + self.beta * log_cent   # [B*HW]

        # per-bin softmax: w_i = exp(score_i) / sum_j exp(score_j in same bin)
        exp_score = torch.exp(score)  # [B*HW]

        sum_exp = torch.zeros(total_bins, device=device, dtype=dtype)
        sum_exp.index_add_(0, global_index_flat, exp_score)  # [total_bins]
        sum_exp = sum_exp.clamp_min(self.eps)

        w_flat = exp_score / sum_exp.index_select(0, global_index_flat)  # [B*HW]

        # 把所有 plan 存下来，只存跟 dtype 无关的内容 & float32 权重即可
        plan = dict(
            B=B, H=H, W=W,
            H_out=H_out, W_out=W_out,
            HW=HW, HW_out=HW_out,
            total_bins=total_bins,
            global_index_flat=global_index_flat,  # Long
            w_flat=w_flat,                        # float (metric 的 dtype)
        )

        # ---------- 返回 prune_fn ----------
        def prune_fn(x: torch.Tensor) -> torch.Tensor:
            """
            对 x 做 ConvToMe2D 剪枝：
            x: [B, Cx, H, W]，B/H/W 必须和 metric 一致，Cx 可不同
            返回: [B, Cx, H_out, W_out]
            """
            assert x.dim() == 4, "x 必须是 BCHW"
            Bx, Cx, Hx, Wx = x.shape
            assert Bx == plan["B"] and Hx == plan["H"] and Wx == plan["W"], \
                f"x 形状 {x.shape} 必须与 metric 的 B,H,W 一致"

            B = plan["B"]
            H = plan["H"]
            W = plan["W"]
            H_out = plan["H_out"]
            W_out = plan["W_out"]
            HW = plan["HW"]
            total_bins = plan["total_bins"]
            global_index_flat = plan["global_index_flat"]
            # 关键：把 w_flat cast 成 x 的 dtype，避免 Half/Float 冲突
            w_flat = plan["w_flat"].to(x.dtype)    # [B*HW]，与 x_flat 同 dtype

            # [B, H, W, Cx] -> [B*HW, Cx]
            x_perm = x.permute(0, 2, 3, 1).contiguous()
            x_flat = x_perm.view(B * HW, Cx)       # [B*HW, Cx]

            weighted_x_flat = x_flat * w_flat.unsqueeze(-1)  # [B*HW, Cx]

            # y_num / y_flat 也都用 x.dtype，保证 index_add_ 不会 dtype 冲突
            y_num = torch.zeros(total_bins, Cx, device=x.device, dtype=x.dtype)
            y_num.index_add_(0, global_index_flat, weighted_x_flat)  # [total_bins, Cx]

            y_flat = y_num  # 已经是聚合后的值

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
        Weighted-average merging 接口，和 ToMe2D.merge_wavg2d 风格一致。

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

        x_weighted = x * size
        x_merged = prune_fn(x_weighted)
        size_merged = prune_fn(size)

        x_merged = x_merged / (size_merged + (size_merged == 0).to(x_merged.dtype))
        return x_merged, size_merged

if __name__ == "__main__":
    B, C, H, W = 2, 64, 14, 14
    x = torch.randn(B, C, H, W)

    pruner = ConvToMe2D(
        imp_mode="l2",
        alpha=1.0,
        beta=1.0,
    )

    target_hw = (10, 10)  # 任意目标尺度
    prune_fn = pruner(metric=x, target_hw=target_hw)

    y = prune_fn(x)
    print("input shape :", x.shape)  # [2, 64, 14, 14]
    print("output shape:", y.shape)  # [2, 64, 10, 10]
