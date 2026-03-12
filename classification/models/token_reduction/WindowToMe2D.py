import math
import torch
import torch.nn.functional as F
from typing import Optional, Callable, Tuple
import torch.nn as nn


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



class AdaptiveWindowToMe2D(ToMe2D):
    """
    Window-aware 2D ToMe.

    - 接口与 ToMe2D 完全对齐：
        forward(metric, num_prune_w, num_prune_h) -> prune_fn
    - 区别：不再在整张图上做 ToMe，而是在 window 内做：
        1) 先根据 (H, num_prune_h), (W, num_prune_w) 得到目标 H_out, W_out（与原版完全一致）。
        2) 沿 H / W 分别选定 window 划分（可选 padding 提高 gcd）。
        3) 在每个 window 内调用你原来的 _plan_along_width/_plan_along_height。
        4) 再把所有 window 拼回整图。
    """

    def __init__(
        self,
        if_prune: bool = False,
        if_order: bool = True,
        distance: str = "cosine",
        merge_mode: str = "sum",
        eps: float = 1e-6,
        max_pad_h: int = 2,   # 默认不自动 padding；想要更 aggressive 的窗口划分可以设成 >0
        max_pad_w: int = 2,
    ):
        super().__init__(if_prune=if_prune,
                         if_order=if_order,
                         distance=distance,
                         merge_mode=merge_mode,
                         eps=eps)
        self.max_pad_h = max_pad_h
        self.max_pad_w = max_pad_w

    # ---------- 轴向辅助函数 ----------

    @staticmethod
    def _axis_target_len(L: int, num_prune: Optional[int]) -> int:
        """
        按照原版 ToMe2D 的规则，从长度 L 和 num_prune 得到输出长度 L_out。
        """
        if L <= 1:
            return L
        T_pair = L // 2
        if num_prune is None:
            r = T_pair
        else:
            r = max(0, min(num_prune, T_pair))
        return L - r

    @staticmethod
    def _divisors(n: int):
        """返回 n 的所有正因子。"""
        n = abs(int(n))
        if n == 0:
            return [1]
        small, large = [], []
        i = 1
        while i * i <= n:
            if n % i == 0:
                small.append(i)
                if i * i != n:
                    large.append(n // i)
            i += 1
        return small + large[::-1]

    def _choose_axis_partition(
        self,
        L: int,
        num_prune: Optional[int],
        max_pad: int,
    ):
        """
        对单个轴（高或宽）做 window 规划：

        输入:
          - L: 原始长度
          - num_prune: 与原 ToMe2D 一致
          - max_pad: 允许在该轴上 padding 的最大长度（只在末尾 pad）

        输出 dict:
          - L: 原始长度
          - L_out: 目标长度（与原 ToMe2D 完全一致）
          - L_pad: padding 后的长度 (>= L)
          - pad: 实际 pad 了多少（L_pad - L）
          - n_win: 该轴上的窗口个数
          - L_win_in: 每个窗口的输入长度
          - L_win_out: 每个窗口的输出长度
          - delta_win: 每个窗口要 merge 掉多少个 token（在该轴上）
        """
        L_out = self._axis_target_len(L, num_prune)

        # 不需要 merge 的情况，直接单窗口返回
        if L_out == L:
            return dict(
                L=L,
                L_out=L_out,
                L_pad=L,
                pad=0,
                n_win=1,
                L_win_in=L,
                L_win_out=L_out,
                delta_win=0,
            )

        best = None
        best_key = None  # (n_win, -pad) 先比窗口数，再比 padding 少

        for pad in range(0, max_pad + 1):
            L_pad = L + pad
            if L_pad < L_out:
                continue

            # 假设我们要把 L_pad -> L_out，总共需要 merge 掉的 token 数
            delta = L_pad - L_out
            # ToMe 的硬上限：最多只能 merge floor(L_pad/2) 对
            if delta > L_pad // 2:
                continue

            g = math.gcd(L_pad, L_out)
            if g <= 0:
                continue

            # 在 g 的因子里找一个最大的 n_win，使得每个 window 都能合法 merge
            best_n_for_this_pad = None
            best_L_win_in = None
            best_L_win_out = None
            best_delta_win = None

            for n in self._divisors(g):
                L_win_in = L_pad // n
                L_win_out = L_out // n
                delta_win = L_win_in - L_win_out

                if delta_win < 0:
                    continue
                if delta_win > L_win_in // 2:
                    # 每个 window 也不能超过自己的 pair capacity
                    continue

                if best_n_for_this_pad is None or n > best_n_for_this_pad:
                    best_n_for_this_pad = n
                    best_L_win_in = L_win_in
                    best_L_win_out = L_win_out
                    best_delta_win = delta_win

            if best_n_for_this_pad is None:
                continue

            key = (best_n_for_this_pad, -pad)
            if best is None or key > best_key:
                best_key = key
                best = dict(
                    L=L,
                    L_out=L_out,
                    L_pad=L_pad,
                    pad=pad,
                    n_win=best_n_for_this_pad,
                    L_win_in=best_L_win_in,
                    L_win_out=best_L_win_out,
                    delta_win=best_delta_win,
                )

        # 如果连 pad 也找不到合适方案，就退化成“无窗口 + 无 pad”，等价于原始 ToMe2D
        if best is None:
            L_pad = L
            pad = 0
            L_out = self._axis_target_len(L, num_prune)
            L_win_in = L_pad
            L_win_out = L_out
            delta_win = L_win_in - L_win_out
            return dict(
                L=L,
                L_out=L_out,
                L_pad=L_pad,
                pad=pad,
                n_win=1,
                L_win_in=L_win_in,
                L_win_out=L_win_out,
                delta_win=delta_win,
            )

        return best

    # ---------- public: window-aware forward ----------

    def forward(
        self,
        metric: torch.Tensor,
        num_prune_w: Optional[int] = None,
        num_prune_h: Optional[int] = None,
    ) -> Callable[[torch.Tensor], torch.Tensor]:

        assert metric.dim() == 4, "metric must be BCHW"
        B, C_metric, H, W = metric.shape

        # 高 / 宽两个轴分别选窗口划分策略
        axis_h = self._choose_axis_partition(H, num_prune_h, self.max_pad_h)
        axis_w = self._choose_axis_partition(W, num_prune_w, self.max_pad_w)

        H_out = axis_h["L_out"]
        W_out = axis_w["L_out"]
        H_pad = axis_h["L_pad"]
        W_pad = axis_w["L_pad"]
        pad_h = axis_h["pad"]
        pad_w = axis_w["pad"]
        n_h = axis_h["n_win"]
        n_w = axis_w["n_win"]
        H_win_in = axis_h["L_win_in"]
        H_win_out = axis_h["L_win_out"]
        W_win_in = axis_w["L_win_in"]
        W_win_out = axis_w["L_win_out"]
        delta_h_win = axis_h["delta_win"]
        delta_w_win = axis_w["delta_win"]

        # 一些 sanity check
        assert H_pad == H + pad_h and W_pad == W + pad_w
        assert H_pad % n_h == 0 and H_out % n_h == 0
        assert W_pad % n_w == 0 and W_out % n_w == 0
        assert H_win_in * n_h == H_pad and H_win_out * n_h == H_out
        assert W_win_in * n_w == W_pad and W_win_out * n_w == W_out

        # 1) 对 metric 做 padding
        if pad_h > 0 or pad_w > 0:
            metric_pad = F.pad(metric, (0, pad_w, 0, pad_h), mode='replicate')  # (left, right, top, bottom)
        else:
            metric_pad = metric

        # 2) 划分成 window，并把 (B, n_h, n_w) 合并到 batch 维
        metric_win = (
            metric_pad.view(B, C_metric, n_h, H_win_in, n_w, W_win_in)
            .permute(0, 2, 4, 1, 3, 5)                  # [B, n_h, n_w, C, H_win_in, W_win_in]
            .contiguous()
            .view(B * n_h * n_w, C_metric, H_win_in, W_win_in)
        )

        # 3) 在 window 内调用原来的 ToMe2D 规划：先宽后高
        plan_w = self._plan_along_width(
            metric_win,
            num_prune_w=delta_w_win if delta_w_win > 0 else 0,
        )
        metric_win_w = self._merge_along_width(
            metric_win, plan_w
        )  # [B*nh*nw, C_metric, H_win_in, W_win_out]

        plan_h = self._plan_along_height(
            metric_win_w,
            num_prune_h=delta_h_win if delta_h_win > 0 else 0,
        )

        # 4) 构建 prune_fn
        def prune_fn(x: torch.Tensor) -> torch.Tensor:
            assert x.dim() == 4, "input must be BCHW"
            B_x, C_x, H_x, W_x = x.shape
            assert B_x == B, "batch size must match the metric used to plan"
            assert H_x == H and W_x == W, \
                f"input spatial size must be {H}x{W}, got {H_x}x{W_x}"

            # pad 到 [B, C_x, H_pad, W_pad]
            if pad_h > 0 or pad_w > 0:
                x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
            else:
                x_pad = x

            # 按窗口切分
            x_win = (
                x_pad.view(B, C_x, n_h, H_win_in, n_w, W_win_in)
                .permute(0, 2, 4, 1, 3, 5)
                .contiguous()
                .view(B * n_h * n_w, C_x, H_win_in, W_win_in)
            )

            # 在窗口内应用 ToMe merge
            x_win = self._merge_along_width(x_win, plan_w)
            x_win = self._merge_along_height(x_win, plan_h)  # [B*nh*nw, C_x, H_win_out, W_win_out]

            # 再拼回整图
            x_out = (
                x_win.view(B, n_h, n_w, C_x, H_win_out, W_win_out)
                .permute(0, 3, 1, 4, 2, 5)
                .contiguous()
                .view(B, C_x, H_out, W_out)
            )
            return x_out

        return prune_fn


if __name__ == "__main__":
    # ===== 56x56 -> 54x54，gcd=2，得到 2x2 window，每个 28x28 -> 27x27 =====
    B, C, H, W = 1, 64, 56, 56
    x = torch.randn(B, C, H, W)

    merge2d_win = AdaptiveWindowToMe2D(if_order=True, distance='cosine', merge_mode='sum',
                                       max_pad_h=0, max_pad_w=0)  # 不自动 padding

    prune_fn = merge2d_win(x, num_prune_w=2, num_prune_h=2)
    y = prune_fn(x)
    print("56x56 ->", y.shape)  # torch.Size([1, 64, 54, 54])

    # 看看窗口划分
    axis_h = merge2d_win._choose_axis_partition(H, 2, merge2d_win.max_pad_h)
    axis_w = merge2d_win._choose_axis_partition(W, 2, merge2d_win.max_pad_w)
    print("H axis:", axis_h)
    print("W axis:", axis_w)
    # H axis: L_win_in=28, L_win_out=27, n_win=2  -> 2 个 28x28 window

    # ===== 13x13 -> 12x12，gcd=1，允许 padding 到 14x14 再做 2x2 window =====
    B, C, H, W = 1, 64, 13, 13
    x = torch.randn(B, C, H, W)

    merge2d_win_pad = AdaptiveWindowToMe2D(if_order=True, distance='cosine',
                                           merge_mode='sum', max_pad_h=1, max_pad_w=1)

    prune_fn = merge2d_win_pad(x, num_prune_w=1, num_prune_h=1)
    y = prune_fn(x)
    print("13x13 ->", y.shape)  # torch.Size([1, 64, 12, 12])

    axis_h = merge2d_win_pad._choose_axis_partition(H, 1, merge2d_win_pad.max_pad_h)
    axis_w = merge2d_win_pad._choose_axis_partition(W, 1, merge2d_win_pad.max_pad_w)
    print("H axis:", axis_h)
    print("W axis:", axis_w)
    # 可以看到 L_pad=14, n_win=2, L_win_in=7, L_win_out=6，即 2x2 个 7x7 -> 6x6 的窗口
