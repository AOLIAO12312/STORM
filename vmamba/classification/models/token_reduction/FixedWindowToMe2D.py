# TODO: FixedWToMe2d

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple, Optional


class FixedWindowToMe2D(nn.Module):
    """
    Window-aware 2D ToMe-style token merging.

    - 接口：forward(metric, num_prune_w, num_prune_h) -> prune_fn(x)
    - metric / x: [B, C, H, W]
    - window_size: 指定窗口大小 (Wh, Ww) 或单个 int（正方形窗口）

    特性：
      * 只在固定窗口内做 ToMe2D 合并（沿 W、H 的 pair 都不跨窗口边界）。
      * H, W 任意；若无法整除窗口大小，则在 bottom / right 方向 replicate pad。
      * 输出尺寸为：
            H_out = H - num_prune_h_eff
            W_out = W - num_prune_w_eff
        （num_prune_* 会 clamp 到合法范围）
    """

    def __init__(
        self,
        if_prune: bool = False,
        if_order: bool = True,
        distance: str = 'cosine',   # 'cosine' | 'l1' | 'l2'
        merge_mode: str = 'sum',    # 'sum' | 'mean' | 'amax'
        window_size: Optional[Tuple[int, int]] = None,  # (Wh, Ww) 或 int
        eps: float = 1e-6,
    ):
        super().__init__()
        self.if_prune = if_prune
        self.if_order = if_order
        self.distance = distance
        self.merge_mode = merge_mode
        self.eps = eps

        if window_size is None:
            self.win_h = None
            self.win_w = None
        elif isinstance(window_size, int):
            self.win_h = window_size
            self.win_w = window_size
        else:
            assert len(window_size) == 2
            self.win_h, self.win_w = window_size

    # ---------- Utilities ----------
    @staticmethod
    def _normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def _pair_scores(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        a: [N, T_pair, C]
        b: [N, T_pair, C]
        return: [N, T_pair]
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

    # ---------- 1D window mask for pair scores ----------
    def _mask_cross_window_pairs(
        self,
        scores: torch.Tensor,
        axis_len: int,
        win: Optional[int],
    ) -> torch.Tensor:
        """
        scores: [N, T_pair]
        axis_len: 这一维的长度 (H_pad 或 W_pad)
        win: 对应窗口大小 (win_h 或 win_w)

        对跨窗口边界的 pair (2j, 2j+1) 赋一个很小的值，防止被选中。
        """
        if win is None or win <= 0:
            return scores

        N, T_pair = scores.shape
        if T_pair == 0:
            return scores

        device = scores.device
        # 第 j 个 pair 使用的位置是 (2j, 2j+1)
        pos_even = torch.arange(T_pair, device=device) * 2
        pos_odd = pos_even + 1
        assert pos_odd.max().item() < axis_len, \
            f"pos_odd {pos_odd.max()} >= axis_len {axis_len}, something wrong"

        win_id_even = pos_even // win
        win_id_odd = pos_odd // win
        cross = (win_id_even != win_id_odd)  # [T_pair]
        if not cross.any():
            return scores

        # 用 dtype 能表示的最小有限值，避免 fp16 溢出
        finfo = torch.finfo(scores.dtype)
        large_neg = finfo.min
        mask = cross.view(1, T_pair).expand(N, T_pair)
        scores = scores.masked_fill(mask, large_neg)
        return scores

        # 这里如果你想“只在第一列窗口剪枝”，可以进一步在 cross 上加条件：
        #   win_id_even != 0 -> 也置为 True，从而屏蔽非第 0 窗口的 pair

    # ---------- Horizontal planning (per row) ----------
    def _plan_along_width(
        self,
        feat: torch.Tensor,
        num_prune_w: Optional[int],
    ):
        """
        feat: [B, C, H_pad, W_pad]
        """
        B, C, H, W = feat.shape
        device = feat.device

        x = feat.permute(0, 2, 3, 1).contiguous().view(B * H, W, C)  # [N, W, C]
        src = x[:, 0::2, :]  # [N, ceil(W/2), C]
        dst = x[:, 1::2, :]  # [N, floor(W/2), C]
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

        # 每行最多能 merge 的 pair 数
        max_pairs = T_pair
        if num_prune_w is None:
            r_w = max_pairs   # 默认：尽可能多
        else:
            r_w = max(0, min(num_prune_w, max_pairs))

        # scores: [N, T_pair]
        scores = self._pair_scores(src[:, :T_pair, :], dst)

        # 限制：只在窗口内部允许 merge，不跨窗口边界
        scores = self._mask_cross_window_pairs(scores, axis_len=W, win=self.win_w)

        edge_idx = scores.argsort(dim=-1, descending=True)[..., None]  # [N, T_pair, 1]
        unm_idx = edge_idx[..., r_w:, :]  # [N, T_pair - r_w, 1]
        src_idx = edge_idx[..., :r_w, :]  # [N, r_w, 1]
        dst_idx = src_idx.clone()

        # 记录原始列索引
        idx_origin = torch.arange(W, device=device).view(1, W, 1).expand(N, W, 1)
        src_orig = idx_origin[:, 0::2, :]      # [N, T_src, 1]
        dst_orig = idx_origin[:, 1::2, :]      # [N, T_dst, 1]
        tail_orig = src_orig[:, T_pair:, :]    # [N, tail_len, 1]

        W_out = W - r_w  # 在 padded 空间下的输出宽度

        return dict(
            N=N, W=W, W_out=W_out,
            T_src=T_src, T_dst=T_dst,
            T_pair=T_pair, tail_len=tail_len,
            unm_idx=unm_idx, src_idx=src_idx, dst_idx=dst_idx,
            src_orig=src_orig, dst_orig=dst_orig, tail_orig=tail_orig
        )

    # ---------- Vertical planning (per column) ----------
    def _plan_along_height(
        self,
        feat_after_w: torch.Tensor,
        num_prune_h: Optional[int],
    ):
        """
        feat_after_w: [B, C, H_pad, W_out_pad]
        """
        B, C, H, Ww = feat_after_w.shape
        device = feat_after_w.device

        x = feat_after_w.permute(0, 3, 2, 1).contiguous().view(B * Ww, H, C)  # [N, H, C]
        src = x[:, 0::2, :]
        dst = x[:, 1::2, :]
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

        max_pairs = T_pair
        if num_prune_h is None:
            r_h = max_pairs
        else:
            r_h = max(0, min(num_prune_h, max_pairs))

        scores = self._pair_scores(src[:, :T_pair, :], dst)  # [N, T_pair]

        # 限制：只在窗口内部允许 merge，不跨窗口边界
        scores = self._mask_cross_window_pairs(scores, axis_len=H, win=self.win_h)

        edge_idx = scores.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r_h:, :]
        src_idx = edge_idx[..., :r_h, :]
        dst_idx = src_idx.clone()

        idx_origin = torch.arange(H, device=device).view(1, H, 1).expand(N, H, 1)
        src_orig = idx_origin[:, 0::2, :]
        dst_orig = idx_origin[:, 1::2, :]
        tail_orig = src_orig[:, T_pair:, :]

        H_out = H - r_h  # 在 padded 空间下的输出高度

        return dict(
            N=N, H=H, H_out=H_out,
            T_src=T_src, T_dst=T_dst,
            T_pair=T_pair, tail_len=tail_len,
            unm_idx=unm_idx, src_idx=src_idx, dst_idx=dst_idx,
            src_orig=src_orig, dst_orig=dst_orig, tail_orig=tail_orig
        )

    # ---------- Apply merges ----------
    def _merge_along_width(self, x: torch.Tensor, plan: dict) -> torch.Tensor:
        B, C, H, W = x.shape
        N = plan['N']
        T_src = plan['T_src']
        T_dst = plan['T_dst']
        T_pair = plan['T_pair']

        if T_dst == 0:
            return x

        x_seq = x.permute(0, 2, 3, 1).contiguous().view(N, W, C)
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

        if (not self.if_prune) and (src_sel is not None) and (dst_idx is not None) and (dst_idx.numel() > 0):
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

            _, idx = original_idx.sort(dim=1)
            seq = seq.gather(dim=-2, index=idx.expand(N, seq.shape[1], C))
        else:
            seq = torch.cat([unm, tail, dst], dim=1)

        W_out = plan['W_out']
        assert seq.shape[1] == W_out, f"Width mismatch: {seq.shape[1]} vs {W_out}"
        seq = seq.view(B, H, W_out, C).permute(0, 3, 1, 2).contiguous()
        return seq

    def _merge_along_height(self, x: torch.Tensor, plan: dict) -> torch.Tensor:
        B, C, H, W = x.shape
        N = plan['N']
        T_src = plan['T_src']
        T_dst = plan['T_dst']
        T_pair = plan['T_pair']

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

        if (not self.if_prune) and (src_sel is not None) and (dst_idx is not None) and (dst_idx.numel() > 0):
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

            _, idx = original_idx.sort(dim=1)
            seq = seq.gather(dim=-2, index=idx.expand(N, seq.shape[1], C))
        else:
            seq = torch.cat([unm, tail, dst], dim=1)

        H_out = plan['H_out']
        assert seq.shape[1] == H_out, f"Height mismatch: {seq.shape[1]} vs {H_out}"
        seq = seq.view(B, W, H_out, C).permute(0, 3, 2, 1).contiguous()
        return seq

    # ---------- Public forward ----------
    def forward(
        self,
        metric: torch.Tensor,                 # [B, C, H, W]
        num_prune_w: Optional[int] = None,    # 每行合并的 pair 数（≈剪掉的列数）
        num_prune_h: Optional[int] = None,    # 每列合并的 pair 数（≈剪掉的行数）
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        assert metric.dim() == 4, "metric must be BCHW"
        B, C, H, W = metric.shape

        # ---- step0: 计算 padding（bottom / right）----
        if self.win_h is not None:
            pad_h = (self.win_h - H % self.win_h) % self.win_h
        else:
            pad_h = 0
        if self.win_w is not None:
            pad_w = (self.win_w - W % self.win_w) % self.win_w
        else:
            pad_w = 0

        if pad_h > 0 or pad_w > 0:
            # F.pad: (left, right, top, bottom)
            # metric_p = F.pad(metric, (0, pad_w, 0, pad_h), mode='replicate')
            metric_p = F.pad(metric, (0, pad_w, 0, pad_h), mode='constant', value=0)
        else:
            metric_p = metric

        # ---- step1: 水平规划 + 应用在 metric_p 上 ----
        plan_w = self._plan_along_width(metric_p, num_prune_w=num_prune_w)
        metric_pw = self._merge_along_width(metric_p, plan_w)

        # ---- step2: 垂直规划 + 应用在 metric_pw 上 ----
        plan_h = self._plan_along_height(metric_pw, num_prune_h=num_prune_h)

        H_pad_out = plan_h['H_out']
        W_pad_out = plan_w['W_out']

        # 剪掉 padding 之后的最终输出尺寸
        H_out = H_pad_out - pad_h
        W_out = W_pad_out - pad_w
        H_out = max(H_out, 1)
        W_out = max(W_out, 1)

        def prune_fn(x: torch.Tensor) -> torch.Tensor:
            assert x.dim() == 4, "input must be BCHW"
            assert x.shape[2] == H and x.shape[3] == W, \
                "input spatial size must match the metric used in planning"

            if pad_h > 0 or pad_w > 0:
                # x_p = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
                x_p = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
            else:
                x_p = x

            x_p = self._merge_along_width(x_p, plan_w)
            x_p = self._merge_along_height(x_p, plan_h)

            # 去掉 padding 对应的 bottom/right 区域
            x_p = x_p[:, :, :H_out, :W_out].contiguous()
            return x_p

        return prune_fn

    # ---------- Weighted average merging (兼容 ToMe 接口) ----------
    def merge_wavg2d(
        self,
        prune_fn_builder: Callable[..., Callable[[torch.Tensor], torch.Tensor]],
        x: torch.Tensor,
        size: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if size is None:
            size = torch.ones_like(x[:, :1])  # [B, 1, H, W]

        prune_fn = prune_fn_builder(x)
        x_merged = prune_fn(x * size)
        size_merged = prune_fn(size)

        x_merged = x_merged / (size_merged + (size_merged == 0).to(x_merged.dtype))
        return x_merged, size_merged

if __name__ == '__main__':

    # -------------------------
    # 1. 生成测试输入：B=2, C=8, H=W=14
    # -------------------------
    B, C, H, W = 2, 8, 14, 14
    metric = torch.randn(B, C, H, W)
    x = torch.randn(B, C, H, W)

    print("Input metric size:", metric.shape)
    print("Input x size     :", x.shape)

    # -------------------------
    # 2. 初始化 WindowToMe2D
    # -------------------------
    tome = FixedWindowToMe2D(
        window_size=3,        # 固定窗口 7×7
        distance='cosine',
        merge_mode='sum',
        if_prune=False        # 使用 merge 而不是硬剪枝
    )

    # -------------------------
    # 3. 设置目标输出 13×13
    # -------------------------
    H_target, W_target = 10, 10

    num_prune_h = H - H_target   # = 1
    num_prune_w = W - W_target   # = 1

    print(f"\nPruning: prune_h={num_prune_h}, prune_w={num_prune_w}")

    # -------------------------
    # 4. 规划并执行 prune_fn
    # -------------------------
    prune_fn = tome(
        metric,
        num_prune_w=num_prune_w,
        num_prune_h=num_prune_h
    )

    y = prune_fn(x)

    # -------------------------
    # 5. 输出
    # -------------------------
    print("Output y size:", y.shape)

    # 验证是否正确剪到 13×13
    assert y.shape == (B, C, H_target, W_target)
    print("\n✔ Test passed: output matches expected size 13×13")
