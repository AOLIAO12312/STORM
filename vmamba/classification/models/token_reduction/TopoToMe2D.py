import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopoToMe2D(nn.Module):
    """
    Topology-preserving, window-aware ToMe-style token merging 2D.

    与 ConvToMe2D 的接口保持一致：
      - forward(metric, target_hw) 返回 prune_fn(x)
      - prune_fn: [B, Cx, H, W] -> [B, Cx, H_out, W_out]

    核心思想：
      1. 基于 metric 计算每个 token 的 importance + 局部 centrality。
      2. 在整个 HW 上选出 H_out * W_out 个 sink tokens（ToMe 风格的“中心”）。
      3. 所有 token 按 content 相似度 + 2D 距离，window-aware 地分配到这些 sinks 上，
         每个 token 只归属一个 cluster，cluster 内再做 ToMe 风格的加权平均。
      4. 根据 sink 的 2D 坐标排序（y 优先，x 次之），把 cluster 排成 H_out×W_out 网格，
         维持 2D 拓扑。
    """

    def __init__(
        self,
        imp_mode: str = "l2",          # 'l2' | 'l1' | 'mean_abs' | 'ones'
        alpha: float = 1.0,            # importance 权重系数
        beta: float = 1.0,             # centrality 权重系数
        local_window: Tuple[int, int] = (3, 3),  # 计算 centrality 的局部窗口 (Wh, Ww)
        radius_factor: float = 2.0,    # 控制 window-aware 合并半径
        eps: float = 1e-6,
    ):
        super().__init__()
        assert imp_mode in ("l2", "l1", "mean_abs", "ones")
        self.imp_mode = imp_mode
        self.alpha = alpha
        self.beta = beta
        self.local_window = local_window
        self.radius_factor = radius_factor
        self.eps = eps

    # ---------- importance ----------
    def _importance(self, m: torch.Tensor) -> torch.Tensor:
        """
        m: [B, HW, C]
        返回: [B, HW]
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

    @staticmethod
    def _normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    # ---------- 核心：根据 metric 构建 Topo-ToMe plan ----------
    def forward(self, metric: torch.Tensor, target_hw: Tuple[int, int]):
        """
        根据 metric 和目标大小 target_hw 构建 Topo-ToMe2D 剪枝 plan，并返回 prune_fn(x)。

        Args:
            metric: [B, C, H, W]
            target_hw: (H_out, W_out)

        Returns:
            prune_fn(x): x -> [B, Cx, H_out, W_out]
        """
        assert metric.dim() == 4, "metric 必须是 BCHW"
        B, C, H, W = metric.shape
        H_out, W_out = target_hw
        assert 1 <= H_out <= H and 1 <= W_out <= W, "target_hw 必须 <= 原始尺寸"

        HW = H * W
        HW_out = H_out * W_out
        assert 1 <= HW_out <= HW, "H_out * W_out 必须在 [1, H*W] 范围内"

        device = metric.device
        dtype = metric.dtype

        # 不剪枝：直接 identity
        if HW_out == HW:
            def identity_fn(x: torch.Tensor) -> torch.Tensor:
                return x
            return identity_fn

        # 1) 展平 metric 为 tokens: [B, HW, C]
        x_tokens = metric.view(B, C, HW).transpose(1, 2).contiguous()  # [B, HW, C]

        # 2) 构造归一化 2D 坐标: [HW, 2] in [0,1]
        h_coords = torch.arange(H, device=device)
        w_coords = torch.arange(W, device=device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")
        coords = torch.stack([grid_h, grid_w], dim=-1).view(HW, 2).float()  # [HW, 2]

        if H > 1:
            coords_y = coords[:, 0] / (H - 1)
        else:
            coords_y = torch.zeros_like(coords[:, 0])
        if W > 1:
            coords_x = coords[:, 1] / (W - 1)
        else:
            coords_x = torch.zeros_like(coords[:, 1])
        coords_norm = torch.stack([coords_y, coords_x], dim=-1)  # [HW, 2]

        # 3) importance: [B, HW]
        imp = self._importance(x_tokens)          # [B, HW]
        imp_clamped = imp.clamp_min(self.eps)
        log_imp = imp_clamped.log()

        # 4) centrality：token 与其局部窗口平均特征的 cos 相似度
        Wh, Ww = self.local_window
        pad_h, pad_w = Wh // 2, Ww // 2
        patches = F.unfold(
            metric, kernel_size=(Wh, Ww), padding=(pad_h, pad_w)
        )                                          # [B, C*Wh*Ww, HW]
        patches = patches.transpose(1, 2).contiguous()
        patches = patches.view(B, HW, C, Wh * Ww)
        patches = patches.permute(0, 1, 3, 2).contiguous()   # [B, HW, Nw, C]
        local_mean = patches.mean(dim=2)                     # [B, HW, C]

        x_norm = self._normalize(x_tokens, self.eps)         # [B, HW, C]
        mean_norm = self._normalize(local_mean, self.eps)    # [B, HW, C]
        cent = (x_norm * mean_norm).sum(dim=-1)              # [B, HW] in [-1,1]

        cent_pos = (cent + 1.0) * 0.5
        cent_pos = cent_pos.clamp_min(1e-3)
        log_cent = cent_pos.log()

        # 5) ToMe-style score: importance + centrality
        score = self.alpha * log_imp + self.beta * log_cent  # [B, HW]

        # 6) 选出 HW_out 个 sink tokens (Top-K)
        _, sink_idx_unsorted = torch.topk(
            score, k=HW_out, dim=1, largest=True, sorted=False
        )                                                     # [B, HW_out]

        # 7) 按 2D 坐标排序 sink，确定 cluster -> grid 顺序
        sink_pos_unsorted = coords_norm[sink_idx_unsorted]    # [B, HW_out, 2]
        # 行主序：先 y 后 x
        sort_key = sink_pos_unsorted[..., 0] * float(H_out) + sink_pos_unsorted[..., 1]
        perm = sort_key.argsort(dim=1)                        # [B, HW_out]
        sink_idx = sink_idx_unsorted.gather(1, perm)          # [B, HW_out]
        sink_pos = sink_pos_unsorted.gather(
            1, perm.unsqueeze(-1).expand(-1, -1, 2)
        )                                                     # [B, HW_out, 2]

        # 8) 所有 token 与 sink 的相似度 (cosine)
        sinks_feat = x_tokens.gather(
            1, sink_idx.unsqueeze(-1).expand(-1, -1, C)
        )                                                     # [B, HW_out, C]
        sinks_feat_norm = self._normalize(sinks_feat, self.eps)

        # [B, HW, C] @ [B, C, HW_out] -> [B, HW, HW_out]
        sim = torch.matmul(
            x_norm, sinks_feat_norm.transpose(1, 2)
        )                                                     # [B, HW, HW_out]

        # 9) window-aware：基于 2D 距离构造局部 mask
        pos_tokens = coords_norm.unsqueeze(0).expand(B, -1, -1)  # [B, HW, 2]
        # [B, HW, 1, 2] - [B, 1, HW_out, 2] -> [B, HW, HW_out, 2]
        diff = pos_tokens.unsqueeze(2) - sink_pos.unsqueeze(1)
        dist2 = (diff ** 2).sum(dim=-1)                         # [B, HW, HW_out]

        K = HW_out
        base_radius = math.sqrt(1.0 / max(K, 1))                # 期望 cluster 尺度 ~ sqrt(1/K)
        R = self.radius_factor * base_radius
        R2 = R * R

        mask_far = dist2 > R2  # True 表示距离太远

        # 用一个在 float16 / bfloat16 / float32 都数值安全的负数
        if sim.dtype in (torch.float16, torch.bfloat16):
            large_neg_value = -1e4
        else:
            large_neg_value = -1e9

        # 用 sim 的 dtype 和 device 构造标量，避免类型转换问题
        large_neg = torch.full((), large_neg_value, dtype=sim.dtype, device=sim.device)

        sim_masked = sim.masked_fill(mask_far, large_neg)

        # 对于被所有 sink mask 掉的 token，强制连接最近的一个 sink
        all_masked = mask_far.all(dim=-1)                       # [B, HW]
        if all_masked.any():
            nearest_sink = dist2.argmin(dim=-1)                 # [B, HW]
            b_idx, t_idx = all_masked.nonzero(as_tuple=True)
            k_idx = nearest_sink[b_idx, t_idx]
            sim_masked[b_idx, t_idx, k_idx] = sim[b_idx, t_idx, k_idx]

        # 10) 每个 token 选一个 sink cluster_id
        assign_idx = sim_masked.argmax(dim=-1)                  # [B, HW] in [0, HW_out)

        # 选中的相似度，用于 cluster 内 softmax 权重（ToMe-style merge_wavg）
        sim_chosen = sim.gather(2, assign_idx.unsqueeze(-1)).squeeze(-1)  # [B, HW]

        # 11) 计算每个 token 在其 cluster 内的归一化权重
        total_clusters = B * HW_out
        batch_offsets = torch.arange(B, device=device).view(B, 1) * HW_out
        global_index = batch_offsets + assign_idx               # [B, HW]
        global_index_flat = global_index.reshape(B * HW)        # [B*HW]

        sim_chosen_flat = sim_chosen.reshape(B * HW).to(torch.float32)
        sim_chosen_flat = sim_chosen_flat - sim_chosen_flat.max()  # 数值稳定
        exp_sim32 = sim_chosen_flat.exp()                       # float32

        sum_exp32 = torch.zeros(total_clusters, device=device, dtype=torch.float32)
        sum_exp32.index_add_(0, global_index_flat, exp_sim32)
        sum_exp32 = sum_exp32.clamp_min(self.eps)

        w_flat = (exp_sim32 / sum_exp32.index_select(0, global_index_flat)).to(dtype)

        # 把所有信息打包成 plan，之后对任意 x 复用
        plan = dict(
            B=B,
            H=H,
            W=W,
            H_out=H_out,
            W_out=W_out,
            HW=HW,
            HW_out=HW_out,
            total_clusters=total_clusters,
            global_index_flat=global_index_flat,  # Long
            w_flat=w_flat,                        # float (metric 的 dtype)
        )

        # ---------- 返回 prune_fn ----------
        def prune_fn(x: torch.Tensor) -> torch.Tensor:
            """
            对 x 做 Topo-ToMe2D 剪枝：
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
            HW_out = plan["HW_out"]
            total_clusters = plan["total_clusters"]
            global_index_flat = plan["global_index_flat"]
            # 保证与 x 的 dtype 一致
            w_flat = plan["w_flat"].to(x.dtype)                  # [B*HW]

            # [B, H, W, Cx] -> [B*HW, Cx]
            x_perm = x.permute(0, 2, 3, 1).contiguous()
            x_flat = x_perm.view(B * HW, Cx)

            weighted = x_flat * w_flat.unsqueeze(-1)             # [B*HW, Cx]

            y_flat = torch.zeros(total_clusters, Cx, device=x.device, dtype=x.dtype)
            y_flat.index_add_(0, global_index_flat, weighted)    # [B*HW_out, Cx]

            # [B, HW_out, Cx] -> [B, Cx, H_out, W_out]
            y = y_flat.view(B, HW_out, Cx).view(B, H_out, W_out, Cx)
            y = y.permute(0, 3, 1, 2).contiguous()
            return y

        return prune_fn

    # ---------- ToMe 风格：merge_wavg2d ----------
    def merge_wavg2d(
        self,
        metric: torch.Tensor,
        x: torch.Tensor,
        size: torch.Tensor = None,
        target_hw: Tuple[int, int] = None,
    ):
        """
        Weighted-average merging 接口，和你之前 ConvToMe2D.merge_wavg2d 风格一致。

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
            size = torch.ones_like(x[:, :1])  # [B, 1, H, W]

        x_weighted = x * size
        x_merged = prune_fn(x_weighted)
        size_merged = prune_fn(size)

        x_merged = x_merged / (size_merged + (size_merged == 0).to(x_merged.dtype))
        return x_merged, size_merged


if __name__ == "__main__":
    # 简单自测
    B, C, H, W = 2, 64, 14, 14
    x = torch.randn(B, C, H, W)

    pruner = TopoToMe2D(
        imp_mode="l2",
        alpha=1.0,
        beta=1.0,
        local_window=(3, 3),
        radius_factor=2.0,
    )

    target_hw = (10, 10)
    prune_fn = pruner(metric=x, target_hw=target_hw)
    y = prune_fn(x)
    print("input shape :", x.shape)   # [2, 64, 14, 14]
    print("output shape:", y.shape)   # [2, 64, 10, 10]

    # merge_wavg2d 用法
    metric2 = torch.randn_like(x)
    size = torch.ones(B, 1, H, W)
    y2, s2 = pruner.merge_wavg2d(metric2, x, size=size, target_hw=target_hw)
    print("merge_wavg2d:", y2.shape, s2.shape)
