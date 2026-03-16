import torch
import torch.nn as nn
from typing import Tuple, Optional, Callable

import torch
import torch.nn as nn
from typing import Callable, Tuple, Optional


class ToMe2D(nn.Module):
    """
    A 2D-friendly ToMe-style token merging module for VMamba-like backbones.

    - Works on BCHW tensors.
    - Preserves 2D adjacency: merges only local even/odd neighbors along W then H.
    - forward() returns a prune_fn that can be applied to both main path and residual path
      to keep them synchronized.

    Design:
      1) Horizontal stage: for each row, pair tokens at (2j, 2j+1); compute content
         similarity per-pair and select r_w pairs to merge (per row).
         Unselected even positions stay (unm), all odd positions remain as receivers (dst).
         For odd W, the last even (no pair) is kept as a tail and concatenated back.
         Result width = W - r_w.
      2) Vertical stage: on the horizontally merged feature, for each column, pair
         (2i, 2i+1); select r_h pairs to merge (per column). Similarly keep possible tail.
         Result height = H - r_h.

    The indices (which pairs to merge, which to keep) are computed once in forward(metric, ...),
    then the returned prune_fn(x) reuses those indices to prune ANY BCHW tensor of the same
    spatial size.
    """

    def __init__(
        self,
        if_prune: bool = False,         # if True: drop src instead of merging into dst (hard prune)
        if_order: bool = True,          # keep spatial order in output
        distance: str = 'cosine',       # 'cosine' | 'l1' | 'l2'
        merge_mode: str = 'sum',        # 'sum' | 'mean' | 'amax' (reduce op for scatter)
        eps: float = 1e-6
    ):
        super().__init__()
        self.if_prune = if_prune
        self.if_order = if_order
        self.distance = distance
        self.merge_mode = merge_mode
        self.eps = eps

    # ---------- Utilities ----------
    @staticmethod
    def _normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def _pair_scores(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity score for LOCAL pairs: a[:, j, :] with b[:, j, :].
        a: [N, T_pair, C]  (even positions up to T_pair)
        b: [N, T_pair, C]  (odd positions)
        return: [N, T_pair] pairwise scores
        """
        if self.distance == 'cosine':
            a_n = self._normalize(a, self.eps)
            b_n = self._normalize(b, self.eps)
            return (a_n * b_n).sum(dim=-1)
        elif self.distance == 'l1':
            return - (a - b).abs().sum(dim=-1)
        elif self.distance == 'l2':
            return - ((a - b) ** 2).sum(dim=-1).sqrt()
        else:
            raise ValueError(f"Unsupported distance {self.distance}")

    @staticmethod
    def _safe_scatter_reduce(dst: torch.Tensor, index: torch.Tensor,
                             src: torch.Tensor, reduce: str) -> torch.Tensor:
        """
        torch.scatter_reduce is available in recent PyTorch.
        This wrapper supports 'sum' | 'mean' | 'amax'.
        """
        if reduce == 'sum':
            return dst.scatter_reduce(-2, index, src, reduce='sum')
        elif reduce == 'amax':
            return dst.scatter_reduce(-2, index, src, reduce='amax')
        elif reduce == 'mean':
            ones = torch.ones_like(src)
            count = dst.scatter_reduce(-2, index, ones, reduce='sum')
            summed = dst.scatter_reduce(-2, index, src, reduce='sum')
            return summed / (count + (count == 0).to(count.dtype))
        else:
            raise ValueError(f"Unsupported reduce {reduce}")

    # ---------- Horizontal planning (per row) ----------
    def _plan_along_width(
        self,
        feat: torch.Tensor,
        num_prune_w: Optional[int]
    ):
        """
        Compute merge plan along width (per row).
        feat: [B, C, H, W]
        Returns a dict with indices and shapes needed by the horizontal merge.
        """
        B, C, H, W = feat.shape
        device = feat.device

        # Layout: [N=B*H, W, C]
        x = feat.permute(0, 2, 3, 1).contiguous().view(B * H, W, C)

        # Split into even(src) and odd(dst). For odd W, src has one more than dst.
        src = x[:, 0::2, :]                         # [N, ceil(W/2), C]
        dst = x[:, 1::2, :]                         # [N, floor(W/2), C]
        N, T_src, _ = src.shape
        T_dst = dst.shape[1]

        if T_dst == 0:
            # width==1，无可合并，但仍可能有 1 个 src 作为 tail
            return dict(
                N=N, W=W, W_out=W,
                T_src=T_src, T_dst=T_dst,
                T_pair=0, tail_len=T_src,
                unm_idx=None, src_idx=None, dst_idx=None,
                src_orig=None, dst_orig=None, tail_orig=None
            )

        # 只对前 T_pair = T_dst 个 even/odd 成对计算相似度
        T_pair = T_dst
        tail_len = T_src - T_pair    # 0 or 1，对应奇数 W 时最后一个 even

        # 决定每行合并多少对
        r_w = T_pair if (num_prune_w is None) else min(num_prune_w, T_pair)
        if r_w < 0:
            r_w = 0

        # Compute local pair scores: [N, T_pair]
        scores = self._pair_scores(src[:, :T_pair, :], dst)

        # Sort pairs by score (descending): edge_idx shape [N, T_pair, 1]
        edge_idx = scores.argsort(dim=-1, descending=True)[..., None]  # [N, T_pair, 1]

        # Indices for src_main that will remain (unmerged) vs be merged
        unm_idx = edge_idx[..., r_w:, :]       # [N, T_pair - r_w, 1]  -- refer to src positions (0..T_pair-1)
        src_idx = edge_idx[..., :r_w, :]       # [N, r_w, 1]
        # dst partner is the same index j for local pairs:
        dst_idx = src_idx.clone()              # [N, r_w, 1]

        # Original column indices for ordering
        idx_origin = torch.arange(W, device=device).view(1, W, 1).expand(N, W, 1)
        src_orig = idx_origin[:, 0::2, :]                      # [N, T_src, 1]
        dst_orig = idx_origin[:, 1::2, :]                      # [N, T_dst, 1]
        tail_orig = src_orig[:, T_pair:, :]                    # [N, tail_len, 1] (0 or 1)

        # output width after horizontal stage:
        # true T_out = (T_pair - r_w) + T_dst + tail_len = W - r_w
        W_out = W - r_w

        plan = dict(
            N=N, W=W, W_out=W_out,
            T_src=T_src, T_dst=T_dst,
            T_pair=T_pair, tail_len=tail_len,
            unm_idx=unm_idx, src_idx=src_idx, dst_idx=dst_idx,
            src_orig=src_orig, dst_orig=dst_orig, tail_orig=tail_orig
        )
        return plan

    # ---------- Vertical planning (per column) ----------
    def _plan_along_height(
        self,
        feat_after_w: torch.Tensor,
        num_prune_h: Optional[int]
    ):
        """
        Compute merge plan along height (per column), on already horizontally merged features.
        feat_after_w: [B, C, H, Ww]  (Ww is width after horizontal stage)
        """
        B, C, H, Ww = feat_after_w.shape
        device = feat_after_w.device

        # Layout: [N=B*Ww, H, C]  (plan per column)
        x = feat_after_w.permute(0, 3, 2, 1).contiguous().view(B * Ww, H, C)

        src = x[:, 0::2, :]                         # [N, ceil(H/2), C]
        dst = x[:, 1::2, :]                         # [N, floor(H/2), C]
        N, T_src, _ = src.shape
        T_dst = dst.shape[1]

        if T_dst == 0:
            return dict(
                N=N, H=H, H_out=H,
                T_src=T_src, T_dst=T_dst,
                T_pair=0, tail_len=T_src,
                unm_idx=None, src_idx=None, dst_idx=None,
                src_orig=None, dst_orig=None, tail_orig=None
            )

        T_pair = T_dst
        tail_len = T_src - T_pair

        r_h = T_pair if (num_prune_h is None) else min(num_prune_h, T_pair)
        if r_h < 0:
            r_h = 0

        scores = self._pair_scores(src[:, :T_pair, :], dst)   # [N, T_pair]
        edge_idx = scores.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r_h:, :]                      # [N, T_pair - r_h, 1]
        src_idx = edge_idx[..., :r_h, :]                      # [N, r_h, 1]
        dst_idx = src_idx.clone()

        idx_origin = torch.arange(H, device=device).view(1, H, 1).expand(N, H, 1)
        src_orig = idx_origin[:, 0::2, :]                     # [N, T_src, 1]
        dst_orig = idx_origin[:, 1::2, :]                     # [N, T_dst, 1]
        tail_orig = src_orig[:, T_pair:, :]                   # [N, tail_len, 1]

        # output height after vertical stage: H_out = H - r_h
        H_out = H - r_h

        plan = dict(
            N=N, H=H, H_out=H_out,
            T_src=T_src, T_dst=T_dst,
            T_pair=T_pair, tail_len=tail_len,
            unm_idx=unm_idx, src_idx=src_idx, dst_idx=dst_idx,
            src_orig=src_orig, dst_orig=dst_orig, tail_orig=tail_orig
        )
        return plan

    # ---------- Apply one-axis merge ----------
    def _merge_along_width(
        self, x: torch.Tensor, plan: dict
    ) -> torch.Tensor:
        """
        Apply horizontal merge according to plan.
        x: [B, C, H, W] -> [B, C, H, W_out]
        """
        B, C, H, W = x.shape
        N = plan['N']
        T_src = plan['T_src']
        T_dst = plan['T_dst']
        T_pair = plan['T_pair']
        tail_len = plan['tail_len']

        if T_dst == 0:
            # no pairs,可能只有 1 个 tail，直接返回
            return x

        # [B, H, W, C] -> [N=B*H, W, C]
        x_seq = x.permute(0, 2, 3, 1).contiguous().view(N, W, C)
        src = x_seq[:, 0::2, :]                     # [N, T_src, C]
        dst = x_seq[:, 1::2, :]                     # [N, T_dst, C]
        assert src.shape[1] == T_src and dst.shape[1] == T_dst

        # main pairs + possible tail
        src_main = src[:, :T_pair, :]              # [N, T_pair, C]
        tail = src[:, T_pair:, :]                  # [N, tail_len, C] (0 or 1)

        unm_idx = plan['unm_idx']
        src_idx = plan['src_idx']
        dst_idx = plan['dst_idx']
        src_orig = plan['src_orig']
        dst_orig = plan['dst_orig']
        tail_orig = plan['tail_orig']

        # gather unmerged evens from src_main
        if unm_idx is not None:
            unm = src_main.gather(dim=-2, index=unm_idx.expand(N, unm_idx.shape[-2], C))  # [N, T_pair - r, C]
        else:
            unm = src_main

        # gather merged evens that will be added into dst
        if src_idx is not None and src_idx.numel() > 0:
            src_sel = src_main.gather(dim=-2, index=src_idx.expand(N, src_idx.shape[-2], C))  # [N, r, C]
        else:
            src_sel = None

        if not self.if_prune and src_sel is not None and dst_idx.numel() > 0:
            dst = self._safe_scatter_reduce(
                dst,
                dst_idx.expand(N, src_idx.shape[-2], C),
                src_sel,
                reduce=self.merge_mode
            )
        # else: hard prune, just drop src_sel, keep dst

        if self.if_order:
            # Order by original column indices
            src_orig_main = src_orig[:, :T_pair, :]
            if unm_idx is not None:
                src_idx_original = src_orig_main.gather(dim=-2, index=unm_idx)     # [N, T_pair - r, 1]
            else:
                src_idx_original = src_orig_main

            original_idx = torch.cat([src_idx_original, tail_orig, dst_orig], dim=1)  # [N, T_out, 1]
            seq = torch.cat([unm, tail, dst], dim=1)                                  # [N, T_out, C]

            sorted_idx, idx = original_idx.sort(dim=1)
            seq = seq.gather(dim=-2, index=idx.expand(N, seq.shape[1], C))            # restore left->right
        else:
            seq = torch.cat([unm, tail, dst], dim=1)

        # back to [B, C, H, W_out]
        W_out = plan['W_out']
        assert seq.shape[1] == W_out, f"Width mismatch: {seq.shape[1]} vs {W_out}"
        seq = seq.view(B, H, W_out, C).permute(0, 3, 1, 2).contiguous()
        return seq

    def _merge_along_height(
        self, x: torch.Tensor, plan: dict
    ) -> torch.Tensor:
        """
        Apply vertical merge according to plan.
        x: [B, C, H, W] -> [B, C, H_out, W]
        """
        B, C, H, W = x.shape
        N = plan['N']
        T_src = plan['T_src']
        T_dst = plan['T_dst']
        T_pair = plan['T_pair']
        tail_len = plan['tail_len']

        if T_dst == 0:
            return x

        # [B, W, H, C] -> [N=B*W, H, C]
        x_seq = x.permute(0, 3, 2, 1).contiguous().view(N, H, C)
        src = x_seq[:, 0::2, :]  # [N, T_src, C]
        dst = x_seq[:, 1::2, :]  # [N, T_dst, C]
        assert src.shape[1] == T_src and dst.shape[1] == T_dst

        src_main = src[:, :T_pair, :]
        tail = src[:, T_pair:, :]

        unm_idx = plan['unm_idx']
        src_idx = plan['src_idx']
        dst_idx = plan['dst_idx']
        src_orig = plan['src_orig']
        dst_orig = plan['dst_orig']
        tail_orig = plan['tail_orig']

        if unm_idx is not None:
            unm = src_main.gather(dim=-2, index=unm_idx.expand(N, unm_idx.shape[-2], C))
        else:
            unm = src_main

        if src_idx is not None and src_idx.numel() > 0:
            src_sel = src_main.gather(dim=-2, index=src_idx.expand(N, src_idx.shape[-2], C))
        else:
            src_sel = None

        if not self.if_prune and src_sel is not None and dst_idx.numel() > 0:
            dst = self._safe_scatter_reduce(
                dst,
                dst_idx.expand(N, src_idx.shape[-2], C),
                src_sel,
                reduce=self.merge_mode
            )

        if self.if_order:
            src_orig_main = src_orig[:, :T_pair, :]
            if unm_idx is not None:
                src_idx_original = src_orig_main.gather(dim=-2, index=unm_idx)
            else:
                src_idx_original = src_orig_main

            original_idx = torch.cat([src_idx_original, tail_orig, dst_orig], dim=1)
            seq = torch.cat([unm, tail, dst], dim=1)

            sorted_idx, idx = original_idx.sort(dim=1)
            seq = seq.gather(dim=-2, index=idx.expand(N, seq.shape[1], C))  # restore top->bottom
        else:
            seq = torch.cat([unm, tail, dst], dim=1)

        H_out = plan['H_out']
        assert seq.shape[1] == H_out, f"Height mismatch: {seq.shape[1]} vs {H_out}"
        seq = seq.view(B, W, H_out, C).permute(0, 3, 2, 1).contiguous()
        return seq

    # ---------- Public: forward returns a prune_fn ----------
    def forward(
        self,
        metric: torch.Tensor,                 # BCHW feature used to compute the merging plan
        num_prune_w: Optional[int] = None,    # per-row number of pairs to merge along width; None -> max (halve-ish)
        num_prune_h: Optional[int] = None     # per-column number of pairs to merge along height; None -> max (halve-ish)
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Plan merges along W then H using 'metric', and return a prune_fn that applies those
        merges to ANY BCHW tensor.

        - If num_prune_w is None: merge all local width pairs (W -> W - floor(W/2))
        - If num_prune_h is None: merge all local height pairs (H -> H - floor(H/2))
        """
        assert metric.dim() == 4, "metric must be BCHW"
        B, C, H, W = metric.shape

        # 1) Horizontal plan on 'metric'
        plan_w = self._plan_along_width(metric, num_prune_w=num_prune_w)
        metric_w = self._merge_along_width(metric, plan_w)

        # 2) Vertical plan on the horizontally-merged metric
        plan_h = self._plan_along_height(metric_w, num_prune_h=num_prune_h)

        # Build the callable prune_fn that applies both stages to any BCHW tensor
        def prune_fn(x: torch.Tensor) -> torch.Tensor:
            assert x.dim() == 4, "input must be BCHW"
            # same spatial size as metric used to plan
            assert x.shape[2] == H and x.shape[3] == W, \
                "input spatial size must match the metric used in planning"
            x = self._merge_along_width(x, plan_w)
            x = self._merge_along_height(x, plan_h)
            return x

        return prune_fn

    # ---------- Optional: weighted average merging (size map) ----------
    def merge_wavg2d(
        self,
        prune_fn_builder: Callable[..., Callable[[torch.Tensor], torch.Tensor]],
        x: torch.Tensor,
        size: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply prune_fn with weighted average, similar to ToMe's merge_wavg.
        - x: BCHW features
        - size: BCH1 or BCHW scalar weights; if None, use ones.
        Returns: (x_merged, size_merged)
        """
        if size is None:
            size = torch.ones_like(x[:, :1])  # BCH1

        prune_fn = prune_fn_builder(x)
        x_merged = prune_fn(x * size)
        size_merged = prune_fn(size)

        x_merged = x_merged / (size_merged + (size_merged == 0).to(x_merged.dtype))
        return x_merged, size_merged


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


class HybridScaleToMe2D(nn.Module):
    """
    HybridScaleToMe2D: 融合 ToMe2D 和 AnyScaleToMe2D 的任意尺度 2D token merging。

    设计目标：
      - 低剪枝率时，尽量完全用 ToMe2D（行/列 pair merge），保持 almost lossless 的特性；
      - 高剪枝率时，让 ToMe2D 只做一小部分“精细局部剪枝”，主要的尺度变化交给 AnyScaleToMe2D，
        利用其在大剪枝率上的鲁棒性和任意尺度能力。

    接口：
      forward(metric, target_hw) -> prune_fn(x)
        - metric: [B, C, H, W]
        - target_hw: (H_out, W_out)
        - prune_fn(x): x -> [B, C, H_out, W_out]
    """

    def __init__(
        self,
        # ToMe2D 超参
        tome_if_prune: bool = False,
        tome_if_order: bool = True,
        tome_distance: str = "cosine",
        tome_merge_mode: str = "sum",
        tome_eps: float = 1e-6,
        # AnyScaleToMe2D 超参
        imp_mode: str = "l2",
        any_eps: float = 1e-6,
        # ToMe2D 在总剪枝里的最大占比（面积比例）
        rr_tome_max: float = 0.20,
    ):
        super().__init__()

        self.rr_tome_max = rr_tome_max

        # 条带式 ToMe2D
        self.tome2d = ToMe2D(
            if_prune=tome_if_prune,
            if_order=tome_if_order,
            distance=tome_distance,
            merge_mode=tome_merge_mode,
            eps=tome_eps,
        )

        # 任意尺度的窗口聚合 AnyScaleToMe2D
        self.any_tome2d = AnyScaleToMe2D(imp_mode=imp_mode, eps=any_eps)

    def forward(
        self,
        metric: torch.Tensor,
        target_hw: Tuple[int, int],
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        根据 metric 和目标大小 target_hw 规划混合剪枝，并返回 prune_fn(x)。

        - metric: [B, C, H, W]
        - target_hw: (H_out, W_out)
        - 返回 prune_fn: 对任意同 BCHW 的 x 做同样剪枝。
        """
        assert metric.dim() == 4, "metric 必须是 BCHW"
        B, C, H, W = metric.shape
        H_out, W_out = target_hw
        assert 1 <= H_out <= H and 1 <= W_out <= W, "target_hw 必须 <= 原始尺寸"

        # 0) 不剪枝就直接 identity
        if H_out == H and W_out == W:
            def identity_fn(x: torch.Tensor) -> torch.Tensor:
                return x
            return identity_fn

        # 总的剪枝率（面积）
        total_tokens = H * W
        target_tokens = H_out * W_out
        rr_total = 1.0 - float(target_tokens) / float(total_tokens)  # in [0, 1)

        # 每个维度总共要减多少
        delta_H_total = H - H_out
        delta_W_total = W - W_out

        # ToMe2D 单次在 H/W 方向最多能减多少（因为 pair_merge 限制）
        max_dH_tome = H // 2
        max_dW_tome = W // 2

        # --------- 1) 决定 ToMe2D 在这个剪枝任务里要承担多少比例 ---------
        # 当总剪枝率 <= rr_tome_max 且目标尺寸在 ToMe2D 一次能力范围内时，直接让 ToMe2D 完成全部剪枝
        can_tome_fully_handle = (
            delta_H_total <= max_dH_tome and
            delta_W_total <= max_dW_tome
        )

        if rr_total <= self.rr_tome_max and can_tome_fully_handle:
            # 完全用 ToMe2D：保持你当前 almost lossless 的行为
            num_prune_h_stage1 = delta_H_total
            num_prune_w_stage1 = delta_W_total
            use_any_stage = False
        else:
            # 高剪枝：ToMe2D 只做一部分，比例约为 rr_tome_max / rr_total （上界 rr_tome_max）
            # 比如总剪掉 40%，rr_tome_max=20% -> ToMe2D 做约一半剪枝，其余交给 AnyToMe2D
            share = min(1.0, self.rr_tome_max / max(rr_total, 1e-6))  # in (0,1]

            # ToMe2D 在 H/W 方向上承担的剪枝量：
            #   delta_H_stage1 ~= share * delta_H_total
            #   delta_W_stage1 ~= share * delta_W_total
            delta_H_stage1 = int(round(
                float(delta_H_total.item() if isinstance(delta_H_total, torch.Tensor) else delta_H_total) * share))
            delta_W_stage1 = int(round(
                float(delta_W_total.item() if isinstance(delta_W_total, torch.Tensor) else delta_W_total) * share))

            # 限制在 ToMe2D 单次能做到的范围内
            delta_H_stage1 = max(0, min(delta_H_stage1, max_dH_tome))
            delta_W_stage1 = max(0, min(delta_W_stage1, max_dW_tome))

            num_prune_h_stage1 = delta_H_stage1
            num_prune_w_stage1 = delta_W_stage1
            use_any_stage = True  # 剩下的交给 AnyScaleToMe2D

        # --------- 2) Stage 1: 条带式 ToMe2D 规划 & 中间特征 ---------
        if num_prune_h_stage1 > 0 or num_prune_w_stage1 > 0:
            prune_fn_tome = self.tome2d(
                metric,
                num_prune_w=num_prune_w_stage1,
                num_prune_h=num_prune_h_stage1
            )
            metric_mid = prune_fn_tome(metric)  # [B, C, H_mid, W_mid]
        else:
            prune_fn_tome = None
            metric_mid = metric

        Bm, Cm, H_mid, W_mid = metric_mid.shape

        # 期望 ToMe2D 之后的中间目标尺寸
        H_mid_expected = H - num_prune_h_stage1
        W_mid_expected = W - num_prune_w_stage1
        assert H_mid == H_mid_expected and W_mid == W_mid_expected, \
            f"中间尺寸不一致: got ({H_mid}, {W_mid}) vs expected ({H_mid_expected}, {W_mid_expected})"

        # --------- 3) Stage 2: AnyScaleToMe2D 做任意尺度补齐 ---------
        # 如果中间尺寸已经等于目标，就不用 AnyScaleToMe2D
        if H_mid == H_out and W_mid == W_out:
            prune_fn_any = None
        else:
            assert H_mid >= H_out and W_mid >= W_out, \
                "中间尺寸必须 >= 目标尺寸，才能使用 AnyScaleToMe2D 继续下采样"
            prune_fn_any = self.any_tome2d(
                metric=metric_mid,
                target_hw=(H_out, W_out)
            )

        # --------- 4) 返回最终 prune_fn，串联两段剪枝 ---------
        def prune_fn(x: torch.Tensor) -> torch.Tensor:
            """
            x: [B, Cx, H, W] -> [B, Cx, H_out, W_out]
            """
            assert x.dim() == 4, "输入必须是 BCHW"
            Bx, Cx, Hx, Wx = x.shape
            assert Bx == B and Hx == H and Wx == W, \
                f"x 的空间尺寸必须与 metric 一致, x={x.shape}, metric={metric.shape}"

            if prune_fn_tome is not None:
                x = prune_fn_tome(x)   # -> [B, Cx, H_mid, W_mid]
            if prune_fn_any is not None:
                x = prune_fn_any(x)    # -> [B, Cx, H_out, W_out]
            return x

        return prune_fn

if __name__ == "__main__":
    B, C, H, W = 2, 64, 14, 14
    x = torch.randn(B, C, H, W)

    hybrid = HybridScaleToMe2D(
        rr_tome_max=0.20,      # ToMe2D 最大负责 20% 面积剪枝
        tome_distance="cosine",
        imp_mode="l2"
    )

    target_hw = (12, 12)      # 轻微剪枝
    prune_fn = hybrid(metric=x, target_hw=target_hw)
    y = prune_fn(x)

    print("input shape :", x.shape)  # [2, 64, 14, 14]
    print("output shape:", y.shape)  # [2, 64, 12, 12]，纯 ToMe2D 完成
