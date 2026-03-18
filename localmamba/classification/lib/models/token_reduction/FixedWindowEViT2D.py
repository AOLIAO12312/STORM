import torch
import torch.nn as nn
from typing import Optional, Callable, Tuple


class FixedWindowEViT2D(nn.Module):
    """
    Fixed Window EViT-style Structured Pruning.

    Logic:
      1. Calculate Token Importance (e.g., Magnitude).
      2. Window Partitioning (optional):
         - Split the row/col into fixed-size windows.
         - Distribute the total `num_prune` budget across these windows.
      3. Top-K Selection:
         - Within each window (or globally if no window), select top-k tokens to KEEP.
         - Sort the indices to maintain relative spatial order (left-to-right, top-to-bottom).
      4. Prune:
         - Gather selected tokens.

    No odd/even pairing. Pure importance-based selection with window constraints.
    """

    def __init__(
            self,
            score_mode: str = "absmean",  # "mean" | "absmean"
            eps: float = 1e-6,
            window_size: Optional[int] = None,  # Default window size
            window_size_w: Optional[int] = None,  # Window size for Width
            window_size_h: Optional[int] = None,  # Window size for Height
    ):
        super().__init__()
        self.score_mode = score_mode
        self.eps = eps

        if window_size_w is None:
            window_size_w = window_size
        if window_size_h is None:
            window_size_h = window_size

        self.window_size_w = window_size_w
        self.window_size_h = window_size_h

    # ---------- Utilities ----------

    def _token_importance(self, metric: torch.Tensor) -> torch.Tensor:
        """Calculate importance score [B, H, W] from [B, C, H, W]"""
        if self.score_mode == "mean":
            return metric.mean(dim=1)
        elif self.score_mode == "absmean":
            return metric.abs().mean(dim=1)
        elif self.score_mode == "l2":
            return (metric.float().pow(2).mean(dim=1) + self.eps).sqrt().to(metric.dtype)
        else:
            raise ValueError(f"Unsupported score_mode: {self.score_mode}")

    @staticmethod
    def _distribute_pruning_counts(
            total_len: int,
            total_to_prune: int,
            window_size: int,
    ) -> Tuple[int, list, list]:
        """
        Distribute the total number of tokens to prune across windows.
        Returns:
            total_pruned: Actual total count
            windows: List of (start, end) tuples
            prune_per_win: List of counts to prune in each window
        """
        if total_len == 0:
            return 0, [], []

        win = max(int(window_size), 1)
        # Ensure window isn't larger than the sequence itself
        win = min(win, total_len)

        # 1. Define Windows [start, end)
        windows = []
        for start in range(0, total_len, win):
            end = min(start + win, total_len)
            windows.append((start, end))

        num_win = len(windows)

        if total_to_prune <= 0:
            return 0, windows, [0] * num_win

        # Clamp max prune to total length (minus 1 if we want to ensure not empty, optional)
        # Here we allow pruning everything if requested, but practically we constrain inputs.
        total_to_prune = min(total_to_prune, total_len)

        # 2. Distribute budget proportionally to window length
        lengths = [e - s for (s, e) in windows]
        ratio = float(total_to_prune) / float(total_len)

        prune_per_win = []
        frac = []
        for L in lengths:
            val = ratio * L
            base = int(val)
            prune_per_win.append(base)
            frac.append(val - base)

        assigned = sum(prune_per_win)
        remain = total_to_prune - assigned

        # 3. Distribute remaining budget to windows with largest fractional parts
        if remain > 0:
            # Sort windows by fractional part descending
            order = sorted(range(num_win), key=lambda i: frac[i], reverse=True)
            for idx in order:
                if remain <= 0: break
                # Don't prune more than the window length
                if prune_per_win[idx] < lengths[idx]:
                    prune_per_win[idx] += 1
                    remain -= 1

        return sum(prune_per_win), windows, prune_per_win

    # ---------- Core 1D Plan Logic (Reused for W and H) ----------

    def _plan_1d(
            self,
            importance: torch.Tensor,  # [N, L]
            num_prune: int,
            window_size: Optional[int]
    ) -> torch.Tensor:
        """
        Generic 1D planning.
        Args:
            importance: [N, L] scores
            num_prune: Total tokens to drop per sequence
            window_size: Size of local window (None = Global)
        Returns:
            keep_indices: [N, L_out] sorted indices to keep
        """
        N, L = importance.shape
        device = importance.device

        # Determine strict number to prune
        num_prune = max(0, min(num_prune, L))

        # --- Case A: No Window (Global Top-K) ---
        if window_size is None or window_size <= 0 or window_size >= L:
            num_keep = L - num_prune
            if num_keep == 0:
                return torch.empty(N, 0, dtype=torch.long, device=device)
            if num_keep == L:
                return torch.arange(L, device=device).unsqueeze(0).expand(N, -1)

            # Top-K Keep
            _, keep_idx = torch.topk(importance, k=num_keep, dim=1, largest=True)  # [N, num_keep]
            # Sort to preserve spatial order
            keep_idx, _ = torch.sort(keep_idx, dim=-1)
            return keep_idx

        # --- Case B: Fixed Window ---
        # 1. Distribute prune budget
        r_total, windows, prune_per_win = self._distribute_pruning_counts(
            total_len=L,
            total_to_prune=num_prune,
            window_size=window_size
        )

        all_kept_indices = []

        for (start, end), r_prune in zip(windows, prune_per_win):
            win_len = end - start
            r_keep = win_len - r_prune

            if r_keep <= 0:
                continue  # Prune entire window

            # If we keep everything in this window
            if r_keep == win_len:
                # [N, win_len]
                local_idx = torch.arange(start, end, device=device).unsqueeze(0).expand(N, -1)
                all_kept_indices.append(local_idx)
                continue

            # Slice importance
            imp_win = importance[:, start:end]  # [N, win_len]

            # Select Top-K within window
            _, local_keep_idx = torch.topk(imp_win, k=r_keep, dim=1, largest=True)  # [N, r_keep]

            # Sort local indices (to keep order 0..win_len)
            local_keep_idx, _ = torch.sort(local_keep_idx, dim=-1)

            # Convert to global indices
            global_keep_idx = local_keep_idx + start
            all_kept_indices.append(global_keep_idx)

        if len(all_kept_indices) == 0:
            return torch.empty(N, 0, dtype=torch.long, device=device)

        # Concatenate all windows
        keep_indices = torch.cat(all_kept_indices, dim=1)  # [N, L - r_total]
        return keep_indices

    # ---------- Width & Height Orchestration ----------

    def _plan_along_width(self, metric: torch.Tensor, num_prune_w: Optional[int]) -> dict:
        B, C, H, W = metric.shape

        # Calculate Importance [B, H, W]
        imp = self._token_importance(metric)
        # Flatten to [N=B*H, W]
        imp_flat = imp.view(B * H, W)

        if num_prune_w is None: num_prune_w = 0

        # Plan
        keep_idx_flat = self._plan_1d(imp_flat, num_prune_w, self.window_size_w)

        W_out = keep_idx_flat.shape[1]

        # Reshape back to [B, H, W_out]
        keep_idx_w = keep_idx_flat.view(B, H, W_out)

        return dict(W_out=W_out, keep_idx_w=keep_idx_w)

    def _apply_along_width(self, x: torch.Tensor, plan_w: dict) -> torch.Tensor:
        # x: [B, C, H, W]
        keep_idx_w = plan_w["keep_idx_w"]  # [B, H, W_out]
        W_out = plan_w["W_out"]
        B, C, H, W = x.shape

        # Permute to [B, H, W, C] for gather
        x_bhwc = x.permute(0, 2, 3, 1)

        # Expand index to [B, H, W_out, C]
        idx_expanded = keep_idx_w.unsqueeze(-1).expand(-1, -1, -1, C)

        # Gather
        x_out = torch.gather(x_bhwc, 2, idx_expanded)

        # Permute back to [B, C, H, W_out]
        return x_out.permute(0, 3, 1, 2).contiguous()

    def _plan_along_height(self, metric_w: torch.Tensor, num_prune_h: Optional[int]) -> dict:
        B, C, H, Ww = metric_w.shape

        # Importance [B, H, Ww]
        imp = self._token_importance(metric_w)
        # Permute to [B, Ww, H] then flatten to [N=B*Ww, H]
        imp_flat = imp.permute(0, 2, 1).contiguous().view(B * Ww, H)

        if num_prune_h is None: num_prune_h = 0

        # Plan
        keep_idx_flat = self._plan_1d(imp_flat, num_prune_h, self.window_size_h)
        H_out = keep_idx_flat.shape[1]

        # Reshape to [B, Ww, H_out]
        keep_idx_h = keep_idx_flat.view(B, Ww, H_out)

        return dict(H_out=H_out, keep_idx_h=keep_idx_h)

    def _apply_along_height(self, x_w: torch.Tensor, plan_h: dict) -> torch.Tensor:
        # x_w: [B, C, H, Ww]
        keep_idx_h = plan_h["keep_idx_h"]  # [B, Ww, H_out]
        H_out = plan_h["H_out"]
        B, C, H, Ww = x_w.shape

        # Permute to [B, Ww, H, C]
        x_bwhc = x_w.permute(0, 3, 2, 1)

        # Expand index
        idx_expanded = keep_idx_h.unsqueeze(-1).expand(-1, -1, -1, C)

        # Gather
        x_out = torch.gather(x_bwhc, 2, idx_expanded)

        # Permute back to [B, C, H_out, Ww]
        return x_out.permute(0, 3, 2, 1).contiguous()

    # ---------- Public Interface ----------

    def forward(
            self,
            metric: torch.Tensor,
            num_prune_w: Optional[int] = None,
            num_prune_h: Optional[int] = None,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        metric: [B, C, H, W] - Used to calculate importance.
        num_prune_w: Total number of tokens to drop per row.
        num_prune_h: Total number of tokens to drop per column (after W-pruning).
        """
        assert metric.dim() == 4

        # 1. Plan W
        plan_w = self._plan_along_width(metric, num_prune_w)
        # Apply W plan to metric so we can calculate H importance accurately
        metric_w = self._apply_along_width(metric, plan_w)

        # 2. Plan H
        plan_h = self._plan_along_height(metric_w, num_prune_h)

        def prune_fn(x: torch.Tensor) -> torch.Tensor:
            """
            Applies the pre-calculated pruning mask to input x.
            x and metric must have same spatial dims.
            """
            # Apply W
            x_w = self._apply_along_width(x, plan_w)
            # Apply H
            x_hw = self._apply_along_height(x_w, plan_h)
            return x_hw

        return prune_fn


if __name__ == '__main__':
    # Test Code
    torch.manual_seed(42)
    B, C, H, W = 2, 16, 12, 12

    # Create input
    x = torch.randn(B, C, H, W)
    metric = x.clone()

    # Initialize
    # Window size 4. With Width 12, we have 3 windows [0-4, 4-8, 8-12]
    pruner = FixedWindowEViT2D(window_size=4, score_mode='l2')

    # Prune 3 tokens per row, 3 tokens per column
    # With 3 windows, likely 1 token pruned per window
    num_prune_w = 3
    num_prune_h = 3

    # Get prune function
    prune_fn = pruner(metric, num_prune_w=num_prune_w, num_prune_h=num_prune_h)

    # Execute
    out = prune_fn(x)

    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")

    expected_H = H - num_prune_h
    expected_W = W - num_prune_w

    assert out.shape == (B, C, expected_H, expected_W)
    print("Test passed: Output shape matches expectations.")

    # 验证是否保留了原始值的子集 (而不是被修改的值)
    # 简单检查：输出的第一个元素应该能在输入里找到
    sample_val = out[0, 0, 0, 0]
    is_present = (x[0, 0] == sample_val).any()
    print(f"Value consistency check: {is_present}")