import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Tuple


class FixedWindowToMe1D(nn.Module):
    """
    Window-aware 1D-ToMe on 2D feature maps.

    - 接口：
        forward(metric, num_prune_w, num_prune_h) -> prune_fn(x)
      其中 metric/x: [B, C, H, W]

    - 固定窗口大小 window_size (int or (Wh, Ww))：
        * 先在 bottom/right 方向做 replicate padding，使得 H_pad, W_pad 能整除窗口。
        * 然后把整图切成 Nh×Nw 个不重叠窗口，每个窗口大小 win_h×win_w。

    - 在每个窗口内：
        * flatten 成 [L_win, C] 序列；
        * 使用 1D ToMe 风格合并相邻 pair (2j, 2j+1)，只在窗口内部 merge；
        * 每个窗口保留 L_win_out_target 个 token，再 reshape 为
              H_win_out × W_win_out 的小 patch。

    - 全局输出：
        * 所有窗口的 patch 瓦片式拼接，得到 H_after × W_after；
        * 最后裁剪到：
              H_out = H - num_prune_h
              W_out = W - num_prune_w
          （会自动 clamp，确保 ≥1）

    性质：
      - ToMe 只在局部窗口内剪枝，不跨窗口；
      - 剩余 token 也只会留在原窗口对应的 patch 区域；
      - window-size 固定，但输出尺寸任意（靠 padding + cropping 解耦）。
    """

    def __init__(
        self,
        window_size: int | Tuple[int, int] = 7,
        if_prune: bool = False,    # True: 只丢 src，不加到 dst
        if_order: bool = True,     # True: 按原始顺序还原
        distance: str = "cosine",  # 'cosine' | 'l1' | 'l2'
        merge_mode: str = "sum",   # 'sum' | 'mean' | 'amax'
        eps: float = 1e-6,
    ):
        super().__init__()
        # 固定窗口大小
        if isinstance(window_size, int):
            self.win_h = window_size
            self.win_w = window_size
        else:
            assert len(window_size) == 2
            self.win_h, self.win_w = window_size

        self.if_prune = if_prune
        self.if_order = if_order
        self.distance = distance
        self.merge_mode = merge_mode
        self.eps = eps

    # ---------- ToMe 1D core: utilities ----------

    @staticmethod
    def _normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def _pair_scores_1d(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        a, b: [N, T_pair, C]
        return: [N, T_pair]
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
    def _safe_scatter_reduce(dst: torch.Tensor,
                             index: torch.Tensor,
                             src: torch.Tensor,
                             reduce: str) -> torch.Tensor:
        """
        wrapper of scatter_reduce for 'sum' | 'mean' | 'amax'.
        All tensors: [N, T_dst, C], index: [N, r, C] with indices along T_dst dim.
        """
        if reduce == "sum":
            return dst.scatter_reduce(-2, index, src, reduce="sum")
        elif reduce == "amax":
            return dst.scatter_reduce(-2, index, src, reduce="amax")
        elif reduce == "mean":
            ones = torch.ones_like(src)
            count = dst.scatter_reduce(-2, index, ones, reduce="sum")
            summed = dst.scatter_reduce(-2, index, src, reduce="sum")
            return summed / (count + (count == 0).to(count.dtype))
        else:
            raise ValueError(f"Unsupported reduce {reduce}")

    # ---------- ToMe 1D: plan & merge on [N, L, C] ----------

    def _plan_tome_1d(self,
                      metric_seq: torch.Tensor,
                      num_prune: int):
        """
        1D ToMe 风格规划：
          - metric_seq: [N, L, C]
          - num_prune: 想要合并的 pair 数（即剪掉多少 token）

        只考虑 local pair:
          token 0<->1, 2<->3, 4<->5, ...
        """
        N, L, C = metric_seq.shape
        device = metric_seq.device

        src = metric_seq[:, 0::2, :]   # [N, ceil(L/2), C]
        dst = metric_seq[:, 1::2, :]   # [N, floor(L/2), C]
        T_src = src.shape[1]
        T_dst = dst.shape[1]

        if T_dst == 0:
            return dict(
                N=N, L=L, L_out=L,
                T_src=T_src, T_dst=T_dst,
                T_pair=0, tail_len=T_src,
                unm_idx=None, src_idx=None, dst_idx=None,
                src_orig=None, dst_orig=None, tail_orig=None,
            )

        T_pair = T_dst
        tail_len = T_src - T_pair

        max_pairs = T_pair
        r = max(0, min(num_prune, max_pairs))

        # scores: [N, T_pair]
        scores = self._pair_scores_1d(src[:, :T_pair, :], dst)
        edge_idx = scores.argsort(dim=-1, descending=True)[..., None]   # [N, T_pair, 1]

        unm_idx = edge_idx[..., r:, :]   # [N, T_pair - r, 1]
        src_idx = edge_idx[..., :r, :]   # [N, r, 1]
        dst_idx = src_idx.clone()

        # 原始 1D 序列的 index，用于按顺序还原
        idx_origin = torch.arange(L, device=device).view(1, L, 1).expand(N, L, 1)
        src_orig = idx_origin[:, 0::2, :]      # [N, T_src, 1]
        dst_orig = idx_origin[:, 1::2, :]      # [N, T_dst, 1]
        tail_orig = src_orig[:, T_pair:, :]    # [N, tail_len, 1]

        L_out = L - r

        return dict(
            N=N, L=L, L_out=L_out,
            T_src=T_src, T_dst=T_dst,
            T_pair=T_pair, tail_len=tail_len,
            unm_idx=unm_idx, src_idx=src_idx, dst_idx=dst_idx,
            src_orig=src_orig, dst_orig=dst_orig, tail_orig=tail_orig,
        )

    def _merge_tome_1d(self,
                       x_seq: torch.Tensor,
                       plan: dict) -> torch.Tensor:
        """
        在 1D 序列上应用 ToMe merge:

          x_seq: [N, L, C]
          plan: 由 _plan_tome_1d 输出
        """
        N, L, C = x_seq.shape
        T_src = plan["T_src"]
        T_dst = plan["T_dst"]
        T_pair = plan["T_pair"]

        if T_dst == 0:
            return x_seq

        src = x_seq[:, 0::2, :]    # [N, T_src, C]
        dst = x_seq[:, 1::2, :]    # [N, T_dst, C]
        assert src.shape[1] == T_src
        assert dst.shape[1] == T_dst

        src_main = src[:, :T_pair, :]           # [N, T_pair, C]
        tail = src[:, T_pair:, :]               # [N, tail_len, C]

        unm_idx = plan["unm_idx"]
        src_idx = plan["src_idx"]
        dst_idx = plan["dst_idx"]
        src_orig = plan["src_orig"]
        dst_orig = plan["dst_orig"]
        tail_orig = plan["tail_orig"]

        if unm_idx is not None:
            unm = src_main.gather(dim=-2, index=unm_idx.expand(N, unm_idx.shape[-2], C))
        else:
            unm = src_main

        if src_idx is not None and src_idx.numel() > 0:
            src_sel = src_main.gather(dim=-2, index=src_idx.expand(N, src_idx.shape[-2], C))
        else:
            src_sel = None

        # 把 src_sel merge 到 dst 里
        if (not self.if_prune) and (src_sel is not None) and (dst_idx is not None) and (dst_idx.numel() > 0):
            dst = self._safe_scatter_reduce(
                dst,
                dst_idx.expand(N, src_idx.shape[-2], C),
                src_sel,
                reduce=self.merge_mode,
            )

        if self.if_order:
            src_orig_main = src_orig[:, :T_pair, :]
            if unm_idx is not None:
                src_idx_original = src_orig_main.gather(dim=-2, index=unm_idx)
            else:
                src_idx_original = src_orig_main

            original_idx = torch.cat([src_idx_original, tail_orig, dst_orig], dim=1)  # [N, L_out, 1]
            seq = torch.cat([unm, tail, dst], dim=1)                                  # [N, L_out, C]

            _, idx = original_idx.sort(dim=1)
            seq = seq.gather(dim=-2, index=idx.expand(N, seq.shape[1], C))
        else:
            seq = torch.cat([unm, tail, dst], dim=1)

        L_out = plan["L_out"]
        assert seq.shape[1] == L_out, f"L mismatch: {seq.shape[1]} vs {L_out}"
        return seq

    # ---------- Public: window-aware forward ----------

    def forward(
        self,
        metric: torch.Tensor,                 # [B, C, H, W]
        num_prune_w: Optional[int] = None,    # 希望剪掉的列数
        num_prune_h: Optional[int] = None,    # 希望剪掉的行数
    ) -> Callable[[torch.Tensor], torch.Tensor]:

        assert metric.dim() == 4, "metric must be BCHW"
        B, C, H, W = metric.shape

        total_prune_h = max(0, num_prune_h or 0)
        total_prune_w = max(0, num_prune_w or 0)

        H_target = max(1, H - total_prune_h)
        W_target = max(1, W - total_prune_w)

        # 1) padding 到整数个 window
        pad_h = (self.win_h - H % self.win_h) % self.win_h
        pad_w = (self.win_w - W % self.win_w) % self.win_w

        H_pad = H + pad_h
        W_pad = W + pad_w

        if pad_h > 0 or pad_w > 0:
            metric_p = F.pad(metric, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            metric_p = metric

        # window 网格
        Nh = H_pad // self.win_h
        Nw = W_pad // self.win_w
        num_windows = Nh * Nw

        # 2) 为每个 window 设计“目标 patch 形状”
        #    在 H/W 上的 per-window 剪枝量（不超过 total_prune_* / Nh 或 Nw，也不超过 win_*/2）
        if total_prune_h > 0:
            r_h_per_win = min(self.win_h // 2, total_prune_h // Nh)
        else:
            r_h_per_win = 0

        if total_prune_w > 0:
            r_w_per_win = min(self.win_w // 2, total_prune_w // Nw)
        else:
            r_w_per_win = 0

        H_win_out = self.win_h - r_h_per_win
        W_win_out = self.win_w - r_w_per_win
        H_win_out = max(1, H_win_out)
        W_win_out = max(1, W_win_out)

        L_win = self.win_h * self.win_w
        L_win_out_target = H_win_out * W_win_out

        # 对应 1D ToMe 的 per-window merge 数
        # 先尝试精确剪到 L_win_out_target 个 token，然后再 clamp 到合法范围
        r_win = L_win - L_win_out_target
        max_pairs_1d = (L_win // 2)
        r_win = max(0, min(r_win, max_pairs_1d))

        # 真实 1D ToMe 剪枝后 L_win_eff >= L_win_out_target（因为 r_win <= L_win - L_win_out_target）
        L_win_eff = L_win - r_win
        assert L_win_eff >= L_win_out_target, "Internal logic error for per-window token count."

        # 3) 构造 per-window metric 序列 [N_win, L_win, C]
        metric_win = (
            metric_p.view(B, C, Nh, self.win_h, Nw, self.win_w)
            .permute(0, 2, 4, 3, 5, 1)  # [B, Nh, Nw, win_h, win_w, C]
            .contiguous()
            .view(B * Nh * Nw, self.win_h * self.win_w, C)  # [N_win, L_win, C]
        )

        # 在所有窗口上一次性做 ToMe 规划
        plan_1d = self._plan_tome_1d(metric_win, num_prune=r_win)

        # 4) 返回 prune_fn，对任意 BCHW x 生效
        def prune_fn(x: torch.Tensor) -> torch.Tensor:
            assert x.dim() == 4, "input must be BCHW"
            Bx, Cx, Hx, Wx = x.shape
            assert Bx == B and Cx == C and Hx == H and Wx == W, \
                f"x shape {x.shape} must match metric {metric.shape}"

            # 4.1 padding
            if pad_h > 0 or pad_w > 0:
                x_p = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
            else:
                x_p = x
            # x_p: [B, C, H_pad, W_pad]

            # 4.2 切成窗口 -> flatten 成序列
            x_win = (
                x_p.view(B, C, Nh, self.win_h, Nw, self.win_w)
                .permute(0, 2, 4, 3, 5, 1)   # [B, Nh, Nw, win_h, win_w, C]
                .contiguous()
                .view(B * Nh * Nw, L_win, C)  # [N_win, L_win, C]
            )

            # 4.3 在每个窗口的 1D 序列上做 ToMe merge
            x_seq_merged = self._merge_tome_1d(x_win, plan_1d)  # [N_win, L_win_eff, C]
            assert x_seq_merged.shape[1] == L_win_eff

            # 4.4 截断到 L_win_out_target 个 token（额外的当作被裁掉）
            x_seq_final = x_seq_merged[:, :L_win_out_target, :]  # [N_win, L_win_out_target, C]

            # 4.5 每个窗口 reshape 成 [C, H_win_out, W_win_out]，再拼回整图
            x_win_out = (
                x_seq_final
                .view(B, Nh, Nw, H_win_out, W_win_out, C)   # [B, Nh, Nw, h_out, w_out, C]
                .permute(0, 5, 1, 3, 2, 4)                 # [B, C, Nh, h_out, Nw, w_out]
                .contiguous()
                .view(B, C, Nh * H_win_out, Nw * W_win_out)
            )

            H_after = Nh * H_win_out
            W_after = Nw * W_win_out

            # 4.6 最后裁剪到精确的 H_target × W_target
            assert H_after >= H_target, f"H_after={H_after} < H_target={H_target}"
            assert W_after >= W_target, f"W_after={W_after} < W_target={W_target}"

            x_out = x_win_out[:, :, :H_target, :W_target].contiguous()
            return x_out

        return prune_fn

    # ---------- Optional: weighted-average merge, ToMe-style ----------

    def merge_wavg2d(
        self,
        prune_fn_builder: Callable[..., Callable[[torch.Tensor], torch.Tensor]],
        x: torch.Tensor,
        size: Optional[torch.Tensor] = None,
    ):
        """
        用法类似于 ToMe 的 merge_wavg：

            tome = FixedWindowToMe2D(window_size=7, ...)
            def build(metric):
                return tome(metric, num_prune_w, num_prune_h)

            x_merged, size_merged = tome.merge_wavg2d(
                prune_fn_builder=build,
                x=x,
                size=size,
            )
        """
        if size is None:
            size = torch.ones_like(x[:, :1])  # [B,1,H,W]

        prune_fn = prune_fn_builder(x)
        x_merged = prune_fn(x * size)
        size_merged = prune_fn(size)

        x_merged = x_merged / (size_merged + (size_merged == 0).to(x_merged.dtype))
        return x_merged, size_merged


if __name__ == "__main__":
    B, C, H, W = 2, 8, 14, 14
    x = torch.randn(B, C, H, W)
    metric = torch.randn(B, C, H, W)

    H_target, W_target = 11, 11
    num_prune_h = H - H_target  # 3
    num_prune_w = W - W_target  # 3

    tome = FixedWindowToMe1D(
        window_size=3,
        distance="cosine",
        merge_mode="sum",
        if_prune=False,
    )

    prune_fn = tome(metric, num_prune_w=num_prune_w, num_prune_h=num_prune_h)
    y = prune_fn(x)

    print("Input:", x.shape)
    print("Output:", y.shape)
    assert y.shape == (B, C, H_target, W_target)
    print("✔ shape OK")
