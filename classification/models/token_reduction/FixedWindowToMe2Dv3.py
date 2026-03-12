import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Tuple, List


class ToMe2D(nn.Module):
    """
    2D-friendly ToMe-style token merging for VMamba-like backbones.

    - Works on BCHW tensors.
    - Preserves 2D adjacency: merges only local even/odd neighbors along W then H.
    - forward() returns a prune_fn that can be applied to both main/residual paths.
    """

    def __init__(
        self,
        if_prune: bool = False,         # if True: drop src instead of merging into dst
        if_order: bool = True,          # keep spatial order in output
        distance: str = "cosine",       # 'cosine' | 'l1' | 'l2'
        merge_mode: str = "sum",        # 'sum' | 'mean' | 'amax'
        eps: float = 1e-6,
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
        if self.distance == "cosine":
            a_n = self._normalize(a, self.eps)
            b_n = self._normalize(b, self.eps)
            return (a_n * b_n).sum(dim=-1)
        elif self.distance == "l1":
            return - (a - b).abs().sum(dim=-1)
        elif self.distance == "l2":
            return - ((a - b) ** 2).sum(dim=-1).sqrt()
        else:
            raise ValueError(f"Unsupported distance {self.distance}")

    @staticmethod
    def _scatter_add_sum_inplace(
        dst: torch.Tensor,  # [N, T_dst, C]
        index: torch.Tensor,  # [N, r, 1]
        src: torch.Tensor,  # [N, r, C]
    ) -> torch.Tensor:
        """
        Fast path for 'sum' reduction using in-place index_add_ on a flattened view.
        Semantics: for each batch n, add src[n, k, :] to dst[n, index[n, k, 0], :].
        """
        if src.numel() == 0:
            return dst

        N, T_dst, C = dst.shape
        _, r, _ = index.shape

        # Flatten (N, T_dst, C) -> (N*T_dst, C)
        dst_flat = dst.view(N * T_dst, C)
        idx = index.view(N, r)  # [N, r]

        # offset per batch
        offsets = torch.arange(N, device=dst.device, dtype=idx.dtype).view(N, 1) * T_dst
        idx_flat = (idx + offsets).reshape(-1)  # [N*r]

        src_flat = src.reshape(N * r, C)
        dst_flat.index_add_(0, idx_flat, src_flat)
        return dst

    @staticmethod
    def _safe_scatter_reduce(
        dst: torch.Tensor,
        index: torch.Tensor,
        src: torch.Tensor,
        reduce: str,
    ) -> torch.Tensor:
        """
        Fallback for 'mean' / 'amax' 或非 fast-path 情况。
        """
        if reduce == "sum":
            # 虽然有 fast-path，但这里作为兜底（极少触发）
            return dst.scatter_reduce(-2, index.expand_as(src), src, reduce="sum")
        elif reduce == "amax":
            return dst.scatter_reduce(-2, index.expand_as(src), src, reduce="amax")
        elif reduce == "mean":
            ones = torch.ones_like(src)
            count = dst.scatter_reduce(-2, index.expand_as(src), ones, reduce="sum")
            summed = dst.scatter_reduce(-2, index.expand_as(src), src, reduce="sum")
            return summed / (count + (count == 0).to(count.dtype))
        else:
            raise ValueError(f"Unsupported reduce {reduce}")

    # ---------- Horizontal planning (per row) ----------
    def _plan_along_width(
        self,
        feat: torch.Tensor,
        num_prune_w: Optional[int],
    ):
        """
        Compute merge plan along width (per row).
        feat: [B, C, H, W]
        """
        B, C, H, W = feat.shape
        device = feat.device

        # Layout: [N=B*H, W, C]
        x = feat.permute(0, 2, 3, 1).contiguous().view(B * H, W, C)

        src = x[:, 0::2, :]  # [N, ceil(W/2), C]
        dst = x[:, 1::2, :]  # [N, floor(W/2), C]
        N, T_src, _ = src.shape
        T_dst = dst.shape[1]

        if T_dst == 0:
            # width==1，无可合并，但仍可能有 1 个 src 作为 tail
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
        tail_len = T_src - T_pair  # 0 or 1

        r_w = T_pair if (num_prune_w is None) else min(max(num_prune_w, 0), T_pair)

        # [N, T_pair]
        scores = self._pair_scores(src[:, :T_pair, :], dst)

        # Sort pairs by score (descending): edge_idx shape [N, T_pair, 1]
        edge_idx = scores.argsort(dim=-1, descending=True)[..., None]  # [N, T_pair, 1]

        unm_idx = edge_idx[..., r_w:, :]       # [N, T_pair - r_w, 1]
        src_idx = edge_idx[..., :r_w, :]       # [N, r_w, 1]
        dst_idx = src_idx.clone()              # local pair index

        idx_origin = torch.arange(W, device=device).view(1, W, 1).expand(B * H, W, 1)
        src_orig = idx_origin[:, 0::2, :]                      # [N, T_src, 1]
        dst_orig = idx_origin[:, 1::2, :]                      # [N, T_dst, 1]
        tail_orig = src_orig[:, T_pair:, :]                    # [N, tail_len, 1]

        W_out = W - r_w

        return dict(
            N=B * H,
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

    # ---------- Vertical planning (per column) ----------
    def _plan_along_height(
        self,
        feat_after_w: torch.Tensor,
        num_prune_h: Optional[int],
    ):
        """
        Compute merge plan along height (per column), on horizontally merged features.
        feat_after_w: [B, C, H, Ww]
        """
        B, C, H, Ww = feat_after_w.shape
        device = feat_after_w.device

        # Layout: [N=B*Ww, H, C]
        x = feat_after_w.permute(0, 3, 2, 1).contiguous().view(B * Ww, H, C)

        src = x[:, 0::2, :]  # [N, ceil(H/2), C]
        dst = x[:, 1::2, :]  # [N, floor(H/2), C]
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

        r_h = T_pair if (num_prune_h is None) else min(max(num_prune_h, 0), T_pair)

        scores = self._pair_scores(src[:, :T_pair, :], dst)   # [N, T_pair]
        edge_idx = scores.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r_h:, :]                      # [N, T_pair - r_h, 1]
        src_idx = edge_idx[..., :r_h, :]                      # [N, r_h, 1]
        dst_idx = src_idx.clone()

        idx_origin = torch.arange(H, device=device).view(1, H, 1).expand(N, H, 1)
        src_orig = idx_origin[:, 0::2, :]                     # [N, T_src, 1]
        dst_orig = idx_origin[:, 1::2, :]                     # [N, T_dst, 1]
        tail_orig = src_orig[:, T_pair:, :]                   # [N, tail_len, 1]

        H_out = H - r_h

        return dict(
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

    # ---------- Apply one-axis merge ----------
    def _merge_along_width(self, x: torch.Tensor, plan: dict) -> torch.Tensor:
        """
        x: [B, C, H, W] -> [B, C, H, W_out]
        """
        B, C, H, W = x.shape
        N = plan["N"]
        T_src = plan["T_src"]
        T_dst = plan["T_dst"]
        T_pair = plan["T_pair"]
        tail_len = plan["tail_len"]

        if T_dst == 0:
            return x

        # [B, H, W, C] -> [N=B*H, W, C]
        x_seq = x.permute(0, 2, 3, 1).contiguous().view(N, W, C)
        src = x_seq[:, 0::2, :]                     # [N, T_src, C]
        dst = x_seq[:, 1::2, :]                     # [N, T_dst, C]
        assert src.shape[1] == T_src and dst.shape[1] == T_dst

        src_main = src[:, :T_pair, :]              # [N, T_pair, C]
        tail = src[:, T_pair:, :]                  # [N, tail_len, C] (0 or 1)

        unm_idx = plan["unm_idx"]
        src_idx = plan["src_idx"]
        dst_idx = plan["dst_idx"]
        src_orig = plan["src_orig"]
        dst_orig = plan["dst_orig"]
        tail_orig = plan["tail_orig"]

        # gather unmerged evens from src_main
        if unm_idx is not None and unm_idx.numel() > 0:
            # [N, T_pair - r, C]
            unm = src_main.gather(
                dim=-2, index=unm_idx.expand(unm_idx.size(0), unm_idx.size(1), C)
            )
        else:
            unm = src_main

        # gather merged evens that will be added into dst
        src_sel = None
        if src_idx is not None and src_idx.numel() > 0:
            src_sel = src_main.gather(
                dim=-2, index=src_idx.expand(src_idx.size(0), src_idx.size(1), C)
            )

        # merge into dst
        if (not self.if_prune) and (src_sel is not None) and dst_idx.numel() > 0:
            if self.merge_mode == "sum":
                dst = self._scatter_add_sum_inplace(dst, dst_idx, src_sel)
            else:
                dst = self._safe_scatter_reduce(
                dst, dst_idx, src_sel, reduce=self.merge_mode
            )

        # restore spatial order if needed
        if self.if_order:
            src_orig_main = src_orig[:, :T_pair, :]
            if unm_idx is not None and unm_idx.numel() > 0:
                src_idx_original = src_orig_main.gather(dim=-2, index=unm_idx)
            else:
                src_idx_original = src_orig_main

            original_idx = torch.cat([src_idx_original, tail_orig, dst_orig], dim=1)
            seq = torch.cat([unm, tail, dst], dim=1)

            _, idx = original_idx.sort(dim=1)
            seq = seq.gather(dim=-2, index=idx.expand(idx.size(0), idx.size(1), C))
        else:
            seq = torch.cat([unm, tail, dst], dim=1)

        W_out = plan["W_out"]
        assert seq.shape[1] == W_out, f"Width mismatch: {seq.shape[1]} vs {W_out}"
        seq = seq.view(B, H, W_out, C).permute(0, 3, 1, 2).contiguous()
        return seq

    def _merge_along_height(self, x: torch.Tensor, plan: dict) -> torch.Tensor:
        """
        x: [B, C, H, W] -> [B, C, H_out, W]
        """
        B, C, H, W = x.shape
        N = plan["N"]
        T_src = plan["T_src"]
        T_dst = plan["T_dst"]
        T_pair = plan["T_pair"]
        tail_len = plan["tail_len"]

        if T_dst == 0:
            return x

        # [B, W, H, C] -> [N=B*W, H, C]
        x_seq = x.permute(0, 3, 2, 1).contiguous().view(N, H, C)
        src = x_seq[:, 0::2, :]  # [N, T_src, C]
        dst = x_seq[:, 1::2, :]  # [N, T_dst, C]
        assert src.shape[1] == T_src and dst.shape[1] == T_dst

        src_main = src[:, :T_pair, :]
        tail = src[:, T_pair:, :]

        unm_idx = plan["unm_idx"]
        src_idx = plan["src_idx"]
        dst_idx = plan["dst_idx"]
        src_orig = plan["src_orig"]
        dst_orig = plan["dst_orig"]
        tail_orig = plan["tail_orig"]

        if unm_idx is not None and unm_idx.numel() > 0:
            unm = src_main.gather(
                dim=-2, index=unm_idx.expand(unm_idx.size(0), unm_idx.size(1), C)
            )
        else:
            unm = src_main

        src_sel = None
        if src_idx is not None and src_idx.numel() > 0:
            src_sel = src_main.gather(
                dim=-2, index=src_idx.expand(src_idx.size(0), src_idx.size(1), C)
            )

        if (not self.if_prune) and (src_sel is not None) and dst_idx.numel() > 0:
            if self.merge_mode == "sum":
                dst = self._scatter_add_sum_inplace(dst, dst_idx, src_sel)
            else:
                dst = self._safe_scatter_reduce(
                    dst, dst_idx, src_sel, reduce=self.merge_mode
                )

        if self.if_order:
            src_orig_main = src_orig[:, :T_pair, :]
            if unm_idx is not None and unm_idx.numel() > 0:
                src_idx_original = src_orig_main.gather(dim=-2, index=unm_idx)
            else:
                src_idx_original = src_orig_main

            original_idx = torch.cat([src_idx_original, tail_orig, dst_orig], dim=1)
            seq = torch.cat([unm, tail, dst], dim=1)

            _, idx = original_idx.sort(dim=1)
            seq = seq.gather(dim=-2, index=idx.expand(idx.size(0), idx.size(1), C))
        else:
            seq = torch.cat([unm, tail, dst], dim=1)

        H_out = plan["H_out"]
        assert seq.shape[1] == H_out, f"Height mismatch: {seq.shape[1]} vs {H_out}"
        seq = seq.view(B, W, H_out, C).permute(0, 3, 2, 1).contiguous()
        return seq

    # ---------- Public: forward returns a prune_fn ----------
    def forward(
        self,
        metric: torch.Tensor,
        num_prune_w: Optional[int] = None,
        num_prune_h: Optional[int] = None,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Plan merges along W then H using 'metric', and return a prune_fn that
        applies those merges to ANY BCHW tensor of the same spatial size.
        """
        assert metric.dim() == 4, "metric must be BCHW"
        B, C, H, W = metric.shape

        # 为加速/减小反向图体积，这里默认 detach 规划所用的特征
        # 如需对 metric 反向，请改为 metric_det = metric
        metric_det = metric.detach()

        plan_w = self._plan_along_width(metric_det, num_prune_w=num_prune_w)
        metric_w = self._merge_along_width(metric_det, plan_w)

        plan_h = self._plan_along_height(metric_w, num_prune_h=num_prune_h)

        def prune_fn(x: torch.Tensor) -> torch.Tensor:
            assert x.dim() == 4, "input must be BCHW"
            assert x.shape[2] == H and x.shape[3] == W, \
                "input spatial size must match the metric used in planning"
            x_ = self._merge_along_width(x, plan_w)
            x_ = self._merge_along_height(x_, plan_h)
            return x_

        return prune_fn

    # ---------- Optional: weighted average merging (size map) ----------
    def merge_wavg2d(
        self,
        prune_fn_builder: Callable[..., Callable[[torch.Tensor], torch.Tensor]],
        x: torch.Tensor,
        size: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if size is None:
            size = torch.ones_like(x[:, :1])  # BCH1

        prune_fn = prune_fn_builder(x)
        x_merged = prune_fn(x * size)
        size_merged = prune_fn(size)

        x_merged = x_merged / (size_merged + (size_merged == 0).to(x_merged.dtype))
        return x_merged, size_merged


class FixedWindowToMe2Dv3(ToMe2D):
    """
    Window-aware 2D ToMe module (执行逻辑与原版保持一致，但内部实现做了局部提速).
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

    # --------- helper: 把全局要 merge 的数量分配到每个窗口 ----------
    @staticmethod
    def _distribute_merges(
        total_pairs: int,
        total_to_merge: int,
        window_size: int,
    ) -> Tuple[int, List[Tuple[int, int]], List[int]]:
        if total_pairs == 0:
            return 0, [], []

        win = max(int(window_size), 1)
        win = min(win, total_pairs)

        windows: List[Tuple[int, int]] = []
        for start in range(0, total_pairs, win):
            end = min(start + win, total_pairs)
            windows.append((start, end))
        num_win = len(windows)

        if total_to_merge <= 0:
            return 0, windows, [0] * num_win

        total_to_merge = min(total_to_merge, total_pairs)

        lengths = [e - s for (s, e) in windows]
        ratio = float(total_to_merge) / float(total_pairs)

        r_per_win: List[int] = []
        frac: List[float] = []
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

    # ---------- override: 带 window 的宽度方向 plan ----------
    def _plan_along_width(
        self,
        feat: torch.Tensor,
        num_prune_w: Optional[int],
    ):
        if self.window_size_w is None or self.window_size_w <= 0:
            return super()._plan_along_width(feat, num_prune_w)

        B, C, H, W = feat.shape
        device = feat.device

        x = feat.permute(0, 2, 3, 1).contiguous().view(B * H, W, C)
        src = x[:, 0::2, :]  # [N, ceil(W/2), C]
        dst = x[:, 1::2, :]  # [N, floor(W/2), C]
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

        scores = self._pair_scores(src[:, :T_pair, :], dst)  # [N, T_pair]

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

            scores_w = scores[:, start:end]  # [N, length]
            edge_idx_w = scores_w.argsort(dim=-1, descending=True)[..., None]  # [N, length, 1]

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
        src_orig = idx_origin[:, 0::2, :]        # [N, T_src, 1]
        dst_orig = idx_origin[:, 1::2, :]        # [N, T_dst, 1]
        tail_orig = src_orig[:, T_pair:, :]      # [N, tail_len, 1]

        W_out = W - r_total

        return dict(
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

    # ---------- override: 带 window 的高度方向 plan ----------
    def _plan_along_height(
        self,
        feat_after_w: torch.Tensor,
        num_prune_h: Optional[int],
    ):
        if self.window_size_h is None or self.window_size_h <= 0:
            return super()._plan_along_height(feat_after_w, num_prune_h)

        B, C, H, Ww = feat_after_w.shape
        device = feat_after_w.device

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

        return dict(
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


if __name__ == "__main__":
    # 简单自测：B=2, C=8, H=W=14 -> 目标 10×10
    B, C, H, W = 2, 8, 14, 14
    metric = torch.randn(B, C, H, W)
    x = torch.randn(B, C, H, W)

    tome = FixedWindowToMe2Dv3(
        window_size=5,
        distance="cosine",
        merge_mode="sum",
        if_prune=False,
    )

    H_target, W_target = 10, 10
    num_prune_h = H - H_target
    num_prune_w = W - W_target

    prune_fn = tome(metric, num_prune_w=num_prune_w, num_prune_h=num_prune_h)
    y = prune_fn(x)
    print("Output y size:", y.shape)
    assert y.shape == (B, C, H_target, W_target)
    print("✔ Test passed.")
