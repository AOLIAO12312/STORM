import torch
import torch.nn as nn
from typing import Optional, Callable, Tuple

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
            return dict(
                N=N, W=W, W_out=W,
                T_src=T_src, T_dst=T_dst,
                T_pair=0, tail_len=T_src,
                unm_idx=None, src_idx=None, dst_idx=None,
                src_orig=None, dst_orig=None, tail_orig=None
            )

        T_pair = T_dst
        tail_len = T_src - T_pair

        r_w = T_pair if (num_prune_w is None) else min(num_prune_w, T_pair)
        if r_w < 0:
            r_w = 0
        scores = self._pair_scores(src[:, :T_pair, :], dst)
        # Sort pairs by score (descending): edge_idx shape [N, T_pair, 1]
        edge_idx = scores.argsort(dim=-1, descending=True)[..., None]  # [N, T_pair, 1]
        # Indices for src_main that will remain (unmerged) vs be merged
        unm_idx = edge_idx[..., r_w:, :]
        src_idx = edge_idx[..., :r_w, :]
        # dst partner is the same index j for local pairs:
        dst_idx = src_idx.clone()
        # Original column indices for ordering
        idx_origin = torch.arange(W, device=device).view(1, W, 1).expand(N, W, 1)
        src_orig = idx_origin[:, 0::2, :]
        dst_orig = idx_origin[:, 1::2, :]
        tail_orig = src_orig[:, T_pair:, :]
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

    # Vertical planning
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

        scores = self._pair_scores(src[:, :T_pair, :], dst)
        edge_idx = scores.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r_h:, :]
        src_idx = edge_idx[..., :r_h, :]
        dst_idx = src_idx.clone()
        idx_origin = torch.arange(H, device=device).view(1, H, 1).expand(N, H, 1)
        src_orig = idx_origin[:, 0::2, :]
        dst_orig = idx_origin[:, 1::2, :]
        tail_orig = src_orig[:, T_pair:, :]
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

    # Apply one-axis merge
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
            return x
        x_seq = x.permute(0, 2, 3, 1).contiguous().view(N, W, C)
        src = x_seq[:, 0::2, :]
        dst = x_seq[:, 1::2, :]
        assert src.shape[1] == T_src and dst.shape[1] == T_dst
        # main pairs + possible tail
        src_main = src[:, :T_pair, :]
        tail = src[:, T_pair:, :]
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
        if self.if_order:
            # Order by original column indices
            src_orig_main = src_orig[:, :T_pair, :]
            if unm_idx is not None:
                src_idx_original = src_orig_main.gather(dim=-2, index=unm_idx)
            else:
                src_idx_original = src_orig_main

            original_idx = torch.cat([src_idx_original, tail_orig, dst_orig], dim=1)
            seq = torch.cat([unm, tail, dst], dim=1)

            sorted_idx, idx = original_idx.sort(dim=1)
            seq = seq.gather(dim=-2, index=idx.expand(N, seq.shape[1], C))
        else:
            seq = torch.cat([unm, tail, dst], dim=1)
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
        x_seq = x.permute(0, 3, 2, 1).contiguous().view(N, H, C)
        src = x_seq[:, 0::2, :]
        dst = x_seq[:, 1::2, :]
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

    def forward(
        self,
        metric: torch.Tensor,
        num_prune_w: Optional[int] = None,
        num_prune_h: Optional[int] = None
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Plan merges along W then H using 'metric', and return a prune_fn that applies those
        merges to ANY BCHW tensor.

        - If num_prune_w is None: merge all local width pairs (W -> W - floor(W/2))
        - If num_prune_h is None: merge all local height pairs (H -> H - floor(H/2))
        """
        assert metric.dim() == 4, "metric must be BCHW"
        B, C, H, W = metric.shape
        plan_w = self._plan_along_width(metric, num_prune_w=num_prune_w)
        metric_w = self._merge_along_width(metric, plan_w)
        plan_h = self._plan_along_height(metric_w, num_prune_h=num_prune_h)
        def prune_fn(x: torch.Tensor) -> torch.Tensor:
            assert x.dim() == 4, "input must be BCHW"
            # same spatial size as metric used to plan
            assert x.shape[2] == H and x.shape[3] == W, \
                "input spatial size must match the metric used in planning"
            x = self._merge_along_width(x, plan_w)
            x = self._merge_along_height(x, plan_h)
            return x

        return prune_fn

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
            size = torch.ones_like(x[:, :1])
        prune_fn = prune_fn_builder(x)
        x_merged = prune_fn(x * size)
        size_merged = prune_fn(size)
        x_merged = x_merged / (size_merged + (size_merged == 0).to(x_merged.dtype))
        return x_merged, size_merged


class FixedWindowToMe2Dv2(ToMe2D):
    """
    Window-aware 2D ToMe module.
    Extends ToMe2D by splitting the pair dimension into non-overlapping windows.
    Merge selection occurs independently within each window; no cross-window competition.
    Units:
      - `window_size` (w/h) refers to "pair indices" (the `j` dim in `scores[:, j]`).
      - Example: T_pair=9, window_size_w=3 → Windows: [0,1,2], [3,4,5], [6,7,8].
    Behavior:
      - If `window_size` is None or ≤ 0: Reverts to original ToMe2D (global top-r).
      - `num_prune_w/h`: Total pairs to merge per row/col, distributed proportionally across windows.
    """

    def __init__(
        self,
        if_prune: bool = False,
        if_order: bool = True,
        distance: str = "cosine",
        merge_mode: str = "sum",
        eps: float = 1e-6,
        window_size: Optional[int] = None,
        window_size_w: Optional[int] = None,
        window_size_h: Optional[int] = None,
    ):
        super().__init__(
            if_prune=if_prune,
            if_order=if_order,
            distance=distance,
            merge_mode=merge_mode,
            eps=eps,
        )
        if window_size_w is None:
            window_size_w = window_size
        if window_size_h is None:
            window_size_h = window_size
        self.window_size_w = window_size_w
        self.window_size_h = window_size_h

    @staticmethod
    def _distribute_merges(
        total_pairs: int,
        total_to_merge: int,
        window_size: int,
    ) -> Tuple[int, list, list]:

        if total_pairs == 0:
            return 0, [], []
        win = max(int(window_size), 1)
        win = min(win, total_pairs)
        windows = []
        for start in range(0, total_pairs, win):
            end = min(start + win, total_pairs)
            windows.append((start, end))
        num_win = len(windows)

        if total_to_merge <= 0:
            return 0, windows, [0] * num_win
        total_to_merge = min(total_to_merge, total_pairs)
        lengths = [e - s for (s, e) in windows]
        ratio = float(total_to_merge) / float(total_pairs)
        r_per_win = []
        frac = []
        for L in lengths:
            val = ratio * L
            base = int(val)
            if base > L:
                base = L
            r_per_win.append(base)
            frac.append(val - base if base < L else -1.0)

        assigned = sum(r_per_win)
        remain = total_to_merge - assigned

        if remain > 0:
            order = sorted(range(num_win), key=lambda i: frac[i], reverse=True)
            for idx in order:
                if remain <= 0:
                    break
                if r_per_win[idx] < lengths[idx]:
                    r_per_win[idx] += 1
                    remain -= 1

        r_total = sum(r_per_win)
        return r_total, windows, r_per_win

    # overwrite: width
    def _plan_along_width(
        self,
        feat: torch.Tensor,
        num_prune_w: Optional[int],
    ):
        # if no window, use super() method directly
        if self.window_size_w is None or self.window_size_w <= 0:
            return super()._plan_along_width(feat, num_prune_w)

        B, C, H, W = feat.shape
        device = feat.device

        x = feat.permute(0, 2, 3, 1).contiguous().view(B * H, W, C)

        src = x[:, 0::2, :]
        dst = x[:, 1::2, :]
        N, T_src, _ = src.shape
        T_dst = dst.shape[1]

        if T_dst == 0:
            return dict(
                N=N,
                W=W,
                W_out=W,
                T_src=T_src,
                T_dst=T_dst,
                T_pair=0,
                tail_len=T_src,
                unm_idx=None,
                src_idx=None,
                dst_idx=None,
                src_orig=None,
                dst_orig=None,
                tail_orig=None,
            )

        T_pair = T_dst
        tail_len = T_src - T_pair

        if num_prune_w is None:
            r_global = T_pair
        else:
            r_global = max(min(num_prune_w, T_pair), 0)

        # Similarity Calculation
        scores = self._pair_scores(src[:, :T_pair, :], dst)
        r_total, windows, r_per_win = self._distribute_merges(
            total_pairs=T_pair,
            total_to_merge=r_global,
            window_size=self.window_size_w,
        )
        unm_idx_chunks = []
        src_idx_chunks = []
        dst_idx_chunks = []
        for (start, end), r_win in zip(windows, r_per_win):
            length = end - start
            if length == 0:
                continue
            scores_w = scores[:, start:end]
            edge_idx_w = scores_w.argsort(dim=-1, descending=True)[..., None]
            r_win = max(min(r_win, length), 0)
            if r_win < length:
                unm_idx_chunks.append(edge_idx_w[..., r_win:, :] + start)
            if r_win > 0:
                sel = edge_idx_w[..., :r_win, :] + start
                src_idx_chunks.append(sel)
                dst_idx_chunks.append(sel.clone())
        unm_total = T_pair - r_total
        src_total = r_total
        if unm_total > 0:
            unm_idx = torch.cat(unm_idx_chunks, dim=-2)
        else:
            unm_idx = torch.empty(N, 0, 1, dtype=torch.long, device=device)

        if src_total > 0:
            src_idx = torch.cat(src_idx_chunks, dim=-2)
            dst_idx = torch.cat(dst_idx_chunks, dim=-2)
        else:
            src_idx = torch.empty(N, 0, 1, dtype=torch.long, device=device)
            dst_idx = torch.empty(N, 0, 1, dtype=torch.long, device=device)

        idx_origin = torch.arange(W, device=device).view(1, W, 1).expand(N, W, 1)
        src_orig = idx_origin[:, 0::2, :]
        dst_orig = idx_origin[:, 1::2, :]
        tail_orig = src_orig[:, T_pair:, :]

        W_out = W - r_total
        plan = dict(
            N=N,
            W=W,
            W_out=W_out,
            T_src=T_src,
            T_dst=T_dst,
            T_pair=T_pair,
            tail_len=tail_len,
            unm_idx=unm_idx,
            src_idx=src_idx,
            dst_idx=dst_idx,
            src_orig=src_orig,
            dst_orig=dst_orig,
            tail_orig=tail_orig,
        )
        return plan

    # override: height
    def _plan_along_height(
        self,
        feat_after_w: torch.Tensor,
        num_prune_h: Optional[int],
    ):
        if self.window_size_h is None or self.window_size_h <= 0:
            return super()._plan_along_height(feat_after_w, num_prune_h)

        B, C, H, Ww = feat_after_w.shape
        device = feat_after_w.device

        # [B, C, H, Ww] -> [N=B*Ww, H, C]
        x = feat_after_w.permute(0, 3, 2, 1).contiguous().view(B * Ww, H, C)

        src = x[:, 0::2, :]   # [N, ceil(H/2), C]
        dst = x[:, 1::2, :]   # [N, floor(H/2), C]
        N, T_src, _ = src.shape
        T_dst = dst.shape[1]

        if T_dst == 0:
            return dict(
                N=N,
                H=H,
                H_out=H,
                T_src=T_src,
                T_dst=T_dst,
                T_pair=0,
                tail_len=T_src,
                unm_idx=None,
                src_idx=None,
                dst_idx=None,
                src_orig=None,
                dst_orig=None,
                tail_orig=None,
            )

        T_pair = T_dst
        tail_len = T_src - T_pair

        if num_prune_h is None:
            r_global = T_pair
        else:
            r_global = max(min(num_prune_h, T_pair), 0)

        scores = self._pair_scores(src[:, :T_pair, :], dst)  # [N, T_pair]

        r_total, windows, r_per_win = self._distribute_merges(
            total_pairs=T_pair,
            total_to_merge=r_global,
            window_size=self.window_size_h,
        )

        unm_idx_chunks = []
        src_idx_chunks = []
        dst_idx_chunks = []

        for (start, end), r_win in zip(windows, r_per_win):
            length = end - start
            if length == 0:
                continue

            scores_w = scores[:, start:end]
            edge_idx_w = scores_w.argsort(dim=-1, descending=True)[..., None]

            r_win = max(min(r_win, length), 0)

            if r_win < length:
                unm_idx_chunks.append(edge_idx_w[..., r_win:, :] + start)

            if r_win > 0:
                sel = edge_idx_w[..., :r_win, :] + start
                src_idx_chunks.append(sel)
                dst_idx_chunks.append(sel.clone())

        unm_total = T_pair - r_total
        src_total = r_total

        if unm_total > 0:
            unm_idx = torch.cat(unm_idx_chunks, dim=-2)
        else:
            unm_idx = torch.empty(N, 0, 1, dtype=torch.long, device=device)

        if src_total > 0:
            src_idx = torch.cat(src_idx_chunks, dim=-2)
            dst_idx = torch.cat(dst_idx_chunks, dim=-2)
        else:
            src_idx = torch.empty(N, 0, 1, dtype=torch.long, device=device)
            dst_idx = torch.empty(N, 0, 1, dtype=torch.long, device=device)

        idx_origin = torch.arange(H, device=device).view(1, H, 1).expand(N, H, 1)
        src_orig = idx_origin[:, 0::2, :]
        dst_orig = idx_origin[:, 1::2, :]
        tail_orig = src_orig[:, T_pair:, :]

        H_out = H - r_total

        plan = dict(
            N=N,
            H=H,
            H_out=H_out,
            T_src=T_src,
            T_dst=T_dst,
            T_pair=T_pair,
            tail_len=tail_len,
            unm_idx=unm_idx,
            src_idx=src_idx,
            dst_idx=dst_idx,
            src_orig=src_orig,
            dst_orig=dst_orig,
            tail_orig=tail_orig,
        )
        return plan

if __name__ == '__main__':
    B, C, H, W = 2, 8, 14, 14
    metric = torch.randn(B, C, H, W)
    x = torch.randn(B, C, H, W)
    print("Input metric size:", metric.shape)
    print("Input x size     :", x.shape)
    storm = FixedWindowToMe2Dv2(
        window_size=3,
        distance='cosine',
        merge_mode='sum',
        if_prune=False
    )
    H_target, W_target = 10, 10
    num_prune_h = H - H_target
    num_prune_w = W - W_target
    print(f"\nPruning: prune_h={num_prune_h}, prune_w={num_prune_w}")
    prune_fn = storm(
        metric,
        num_prune_w=num_prune_w,
        num_prune_h=num_prune_h
    )
    y = prune_fn(x)
    print("Output y size:", y.shape)
    assert y.shape == (B, C, H_target, W_target)
    print("\nTest passed: output matches expected size 13×13")