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


if __name__ == '__main__':
    B, C, H, W = 2, 64, 14, 14
    x = torch.randn(B, C, H, W)
    merge2d = ToMe2D(if_order=True, distance='cosine', merge_mode='sum')

    # default: halve width then halve height => 14x14 -> 7x7
    prune_fn = merge2d(x, num_prune_w=None, num_prune_h=None)
    y = prune_fn(x)
    print(y.shape)  # torch.Size([2, 64, 7, 7])

    # only halve width: 14x14 -> 14x7
    prune_fn_w = merge2d(x, num_prune_w=None, num_prune_h=0)
    y_w = prune_fn_w(x)
    print(y_w.shape)  # torch.Size([2, 64, 14, 7])

    # partial merges: per-row merge 3 pairs -> width becomes 14-3=11; height unchanged
    prune_fn_partial = merge2d(x, num_prune_w=3, num_prune_h=0)
    y_p = prune_fn_partial(x)
    print(y_p.shape)  # torch.Size([2, 64, 14, 11])

    # 奇数尺寸测试：27x27
    B, C, H, W = 1, 192, 27, 27
    x = torch.randn(B, C, H, W)
    merge2d = ToMe2D(if_order=True, distance='cosine', merge_mode='sum')

    prune_fn = merge2d(x, num_prune_w=2, num_prune_h=2)
    y = prune_fn(x)
    print(y.shape)  # 应该是 [1, 192, 25, 25]，不会再报 view 错误
