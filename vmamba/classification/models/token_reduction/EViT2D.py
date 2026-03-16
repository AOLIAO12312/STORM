import torch
import torch.nn as nn
from typing import Optional, Callable


class EViT2DStructuredPruning(nn.Module):
    """
    EViT-style *importance pruning* with ToMe2D-style *structured 2D planning* (row/col units).

    Key idea:
      - Keep ToMe2D's 2-stage structure: prune along W per-row, then prune along H per-column.
      - Replace "similarity-based merge" with "importance-based prune":
          * form local pairs (even, odd)
          * pick r pairs to prune (drop exactly 1 token per chosen pair)
          * within each chosen pair, keep the more important token, drop the less important one
      - Planning is done once on `metric` (BCHW), then returns `prune_fn(x)` that can prune
        ANY BCHW tensor with the same original H,W (e.g., main path + residual).

    Shapes:
      input:  x  [B, C, H, W]
      output: x' [B, C, H_out, W_out]
        where W_out = W - r_w, H_out = H - r_h

    Note:
      - `num_prune_w` and `num_prune_h` are "pairs-to-prune per row/col" (like ToMe2D's r_w/r_h),
        not raw token count.
      - If None: prune all possible pairs => W_out ~ ceil(W/2), H_out ~ ceil(H/2).
    """

    def __init__(
        self,
        if_order: bool = True,
        score_mode: str = "absmean",   # "mean" | "absmean" | "l2"
        eps: float = 1e-6,
    ):
        super().__init__()
        self.if_order = if_order
        self.score_mode = score_mode
        self.eps = eps

    # ---------- scoring ----------
    def _token_importance(self, metric: torch.Tensor) -> torch.Tensor:
        """
        metric: [B, C, H, W]
        return: importance map [B, H, W]
        """
        if self.score_mode == "mean":
            return metric.mean(dim=1)
        elif self.score_mode == "absmean":
            return metric.abs().mean(dim=1)
        elif self.score_mode == "l2":
            # sqrt(mean(x^2)) for numerical stability
            return (metric.float().pow(2).mean(dim=1) + self.eps).sqrt().to(metric.dtype)
        else:
            raise ValueError(f"Unsupported score_mode: {self.score_mode}")

    # ---------- plan helpers ----------
    @staticmethod
    def _build_keep_indices_from_pairs(
        *,
        N: int,
        T_pair: int,
        r: int,
        keep_even: torch.Tensor,    # [N, T_pair] bool, for each pair keep even? else keep odd
        device: torch.device,
        tail_pos: Optional[int],    # for odd length, the last even has no pair, keep it
        axis_len: int,              # original length (W or H)
    ) -> torch.Tensor:
        """
        Build ordered keep indices in [0..axis_len-1] for each sequence (row/col), shape [N, axis_out].

        For pairs (2j, 2j+1), if pair is pruned -> keep one; else -> keep both.
        We prune exactly r pairs per sequence.
        """
        if T_pair == 0:
            # axis_len == 1, keep [0]
            idx = torch.arange(axis_len, device=device).view(1, axis_len).expand(N, -1)
            return idx

        # positions for each pair
        even_pos = torch.arange(0, 2 * T_pair, 2, device=device).view(1, T_pair).expand(N, -1)  # [N, T_pair]
        odd_pos = even_pos + 1                                                                     # [N, T_pair]

        # prune selection: choose r pairs with smallest "drop score"
        # (we'll build prune_mask outside; here we assume caller provides keep_even + prune_mask)
        raise RuntimeError("This helper should not be called directly.")

    def _plan_along_width(self, metric: torch.Tensor, num_prune_w: Optional[int]) -> dict:
        """
        metric: [B, C, H, W]
        return plan dict including keep_idx_w: [B, H, W_out]
        """
        B, C, H, W = metric.shape
        device = metric.device

        imp = self._token_importance(metric)  # [B, H, W]
        # per-row sequences: [N=B*H, W]
        imp_seq = imp.contiguous().view(B * H, W)
        N = imp_seq.shape[0]

        # split into pairs: even/odd
        imp_even = imp_seq[:, 0::2]   # [N, ceil(W/2)]
        imp_odd = imp_seq[:, 1::2]    # [N, floor(W/2)]
        T_src = imp_even.shape[1]
        T_dst = imp_odd.shape[1]

        if T_dst == 0:
            keep_idx = torch.arange(W, device=device).view(1, W).expand(N, -1)  # keep all
            keep_idx = keep_idx.view(B, H, W)
            return dict(H=H, W=W, W_out=W, keep_idx_w=keep_idx)

        T_pair = T_dst
        tail_len = T_src - T_pair  # 0 or 1 (odd W => 1)
        tail_pos = 2 * T_pair if tail_len == 1 else None  # last even position

        r_w = T_pair if (num_prune_w is None) else int(min(max(num_prune_w, 0), T_pair))
        W_out = W - r_w

        # pair-wise "drop score" = min(even, odd) -> we prune pairs where the *droppable* token is tiniest
        drop_score = torch.minimum(imp_even[:, :T_pair], imp_odd)  # [N, T_pair]
        # select r_w smallest per row
        prune_pairs = drop_score.argsort(dim=-1, descending=False)[:, :r_w]  # [N, r_w]
        prune_mask = torch.zeros((N, T_pair), device=device, dtype=torch.bool)
        if r_w > 0:
            prune_mask.scatter_(1, prune_pairs, True)

        # within each pair, keep the more important token
        keep_even = (imp_even[:, :T_pair] >= imp_odd)  # [N, T_pair] bool

        # Build keep indices in original order
        even_pos = torch.arange(0, 2 * T_pair, 2, device=device).view(1, T_pair).expand(N, -1)  # [N, T_pair]
        odd_pos = even_pos + 1

        # For pruned pairs:
        drop_even = prune_mask & (~keep_even)
        drop_odd = prune_mask & keep_even

        mask_even = ~drop_even  # keep even unless we decided to drop it
        mask_odd = ~drop_odd    # keep odd unless we decided to drop it

        pos_flat = torch.stack([even_pos, odd_pos], dim=2).reshape(N, 2 * T_pair)                 # [N, 2*T_pair]
        mask_flat = torch.stack([mask_even, mask_odd], dim=2).reshape(N, 2 * T_pair)              # [N, 2*T_pair]

        kept_from_pairs = torch.masked_select(pos_flat, mask_flat).view(N, 2 * T_pair - r_w)      # fixed length

        if tail_pos is not None:
            tail = torch.full((N, 1), int(tail_pos), device=device, dtype=kept_from_pairs.dtype)
            keep_idx_row = torch.cat([kept_from_pairs, tail], dim=1)  # [N, W_out]
        else:
            keep_idx_row = kept_from_pairs

        assert keep_idx_row.shape[1] == W_out, f"W_out mismatch: {keep_idx_row.shape[1]} vs {W_out}"

        if self.if_order:
            # already in left->right order by construction
            pass
        else:
            # optional: could skip ordering; but keep_idx_row is already ordered anyway
            pass

        keep_idx_w = keep_idx_row.view(B, H, W_out)
        return dict(H=H, W=W, W_out=W_out, keep_idx_w=keep_idx_w)

    def _apply_along_width(self, x: torch.Tensor, plan_w: dict) -> torch.Tensor:
        """
        x: [B, C, H, W] -> [B, C, H, W_out]
        """
        B, C, H, W = x.shape
        keep_idx_w = plan_w["keep_idx_w"]  # [B, H, W_out]
        W_out = plan_w["W_out"]
        assert keep_idx_w.shape[:2] == (B, H)

        # [B, H, W, C]
        x_bhwc = x.permute(0, 2, 3, 1).contiguous()
        # gather along W dim=2
        idx = keep_idx_w.unsqueeze(-1).expand(B, H, W_out, C)  # [B, H, W_out, C]
        x_kept = torch.gather(x_bhwc, dim=2, index=idx)        # [B, H, W_out, C]
        return x_kept.permute(0, 3, 1, 2).contiguous()

    def _plan_along_height(self, metric_w: torch.Tensor, num_prune_h: Optional[int]) -> dict:
        """
        metric_w: [B, C, H, Ww] (already width-pruned)
        return plan dict including keep_idx_h: [B, Ww, H_out] (indices along H)
        """
        B, C, H, Ww = metric_w.shape
        device = metric_w.device

        imp = self._token_importance(metric_w)  # [B, H, Ww]
        # per-column sequences: [N=B*Ww, H]
        imp_seq = imp.permute(0, 2, 1).contiguous().view(B * Ww, H)  # [N, H]
        N = imp_seq.shape[0]

        imp_even = imp_seq[:, 0::2]  # [N, ceil(H/2)]
        imp_odd = imp_seq[:, 1::2]   # [N, floor(H/2)]
        T_src = imp_even.shape[1]
        T_dst = imp_odd.shape[1]

        if T_dst == 0:
            keep_idx = torch.arange(H, device=device).view(1, H).expand(N, -1)
            keep_idx = keep_idx.view(B, Ww, H)
            return dict(H=H, Ww=Ww, H_out=H, keep_idx_h=keep_idx)

        T_pair = T_dst
        tail_len = T_src - T_pair
        tail_pos = 2 * T_pair if tail_len == 1 else None

        r_h = T_pair if (num_prune_h is None) else int(min(max(num_prune_h, 0), T_pair))
        H_out = H - r_h

        drop_score = torch.minimum(imp_even[:, :T_pair], imp_odd)  # [N, T_pair]
        prune_pairs = drop_score.argsort(dim=-1, descending=False)[:, :r_h]  # [N, r_h]
        prune_mask = torch.zeros((N, T_pair), device=device, dtype=torch.bool)
        if r_h > 0:
            prune_mask.scatter_(1, prune_pairs, True)

        keep_even = (imp_even[:, :T_pair] >= imp_odd)  # [N, T_pair]

        even_pos = torch.arange(0, 2 * T_pair, 2, device=device).view(1, T_pair).expand(N, -1)
        odd_pos = even_pos + 1

        drop_even = prune_mask & (~keep_even)
        drop_odd = prune_mask & keep_even

        mask_even = ~drop_even
        mask_odd = ~drop_odd

        pos_flat = torch.stack([even_pos, odd_pos], dim=2).reshape(N, 2 * T_pair)
        mask_flat = torch.stack([mask_even, mask_odd], dim=2).reshape(N, 2 * T_pair)

        kept_from_pairs = torch.masked_select(pos_flat, mask_flat).view(N, 2 * T_pair - r_h)

        if tail_pos is not None:
            tail = torch.full((N, 1), int(tail_pos), device=device, dtype=kept_from_pairs.dtype)
            keep_idx_col = torch.cat([kept_from_pairs, tail], dim=1)
        else:
            keep_idx_col = kept_from_pairs

        assert keep_idx_col.shape[1] == H_out, f"H_out mismatch: {keep_idx_col.shape[1]} vs {H_out}"

        keep_idx_h = keep_idx_col.view(B, Ww, H_out)  # per column
        return dict(H=H, Ww=Ww, H_out=H_out, keep_idx_h=keep_idx_h)

    def _apply_along_height(self, x_w: torch.Tensor, plan_h: dict) -> torch.Tensor:
        """
        x_w: [B, C, H, Ww] -> [B, C, H_out, Ww]
        """
        B, C, H, Ww = x_w.shape
        keep_idx_h = plan_h["keep_idx_h"]  # [B, Ww, H_out]
        H_out = plan_h["H_out"]
        assert keep_idx_h.shape[:2] == (B, Ww)

        # [B, Ww, H, C]
        x_bwhc = x_w.permute(0, 3, 2, 1).contiguous()
        idx = keep_idx_h.unsqueeze(-1).expand(B, Ww, H_out, C)  # [B, Ww, H_out, C]
        x_kept = torch.gather(x_bwhc, dim=2, index=idx)         # [B, Ww, H_out, C]
        return x_kept.permute(0, 3, 2, 1).contiguous()

    # ---------- public ----------
    def forward(
        self,
        metric: torch.Tensor,                 # BCHW feature used to compute the pruning plan (e.g., residual or hidden)
        num_prune_w: Optional[int] = None,    # per-row number of PAIRS to prune along width
        num_prune_h: Optional[int] = None,    # per-col number of PAIRS to prune along height (after width pruning)
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        assert metric.dim() == 4, "metric must be BCHW"
        B, C, H, W = metric.shape

        # 1) plan + apply along width on metric
        plan_w = self._plan_along_width(metric, num_prune_w=num_prune_w)
        metric_w = self._apply_along_width(metric, plan_w)

        # 2) plan along height using width-pruned metric
        plan_h = self._plan_along_height(metric_w, num_prune_h=num_prune_h)

        def prune_fn(x: torch.Tensor) -> torch.Tensor:
            assert x.dim() == 4, "input must be BCHW"
            assert x.shape[2] == H and x.shape[3] == W, "input spatial size must match planning metric"
            x_w = self._apply_along_width(x, plan_w)
            x_hw = self._apply_along_height(x_w, plan_h)
            return x_hw

        return prune_fn


# -------------------- Example --------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    pruner = EViT2DStructuredPruning(score_mode="absmean", if_order=True)

    B, C, H, W = 2, 128, 14, 14
    metric = torch.randn(B, C, H, W)         # e.g., residual feature used for importance planning
    x = torch.randn(B, C, H, W)              # main path feature
    res = torch.randn(B, C, H, W)            # residual path feature

    # prune r_w pairs per row, r_h pairs per column (after width pruning)
    prune_fn = pruner(metric, num_prune_w=3, num_prune_h=2)

    x2 = prune_fn(x)
    res2 = prune_fn(res)

    print("x  :", x.shape, "->", x2.shape)
    print("res:", res.shape, "->", res2.shape)
