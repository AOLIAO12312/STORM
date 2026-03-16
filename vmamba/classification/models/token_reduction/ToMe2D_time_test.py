import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Tuple, Dict


class ToMe2D(nn.Module):
    """
    Optimized ToMe2D.

    Improvements:
    - Pre-calculates 'restore_indices' during planning to avoid sorting features during inference.
    - Replaces slow scatter_reduce with vectorized masked addition.
    - Uses topk instead of argsort for faster selection.
    - Minimizes contiguous() calls and memory permutations.
    """

    def __init__(
            self,
            if_prune: bool = False,
            if_order: bool = True,
            distance: str = 'cosine',
            merge_mode: str = 'sum',
            eps: float = 1e-6
    ):
        super().__init__()
        self.if_prune = if_prune
        self.if_order = if_order
        self.distance = distance
        self.merge_mode = merge_mode
        self.eps = eps

    def _pair_scores(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Optimized score calculation.
        a, b: [N, T, C]
        """
        if self.distance == 'cosine':
            # Fuse normalization and dot product slightly if possible,
            # but explicit norm is usually numerically safer for cosine.
            a_norm = a / (a.norm(dim=-1, keepdim=True) + self.eps)
            b_norm = b / (b.norm(dim=-1, keepdim=True) + self.eps)
            return (a_norm * b_norm).sum(dim=-1)
        elif self.distance == 'l1':
            return -torch.norm(a - b, p=1, dim=-1)
        elif self.distance == 'l2':
            return -torch.norm(a - b, p=2, dim=-1)
        else:
            raise ValueError(f"Unsupported distance {self.distance}")

    def _plan_1d(self, x: torch.Tensor, num_prune: Optional[int], dim_len: int):
        """
        Generic 1D planning (works for flattened W or H).
        x: [N, L, C] where L is W or H
        Returns:
            dict containing:
            - merge_mask: [N, T_pair] boolean (True where merge happens)
            - restore_inds: [N, L_out] indices to gather final result to preserve order
            - L_out: int, output length
        """
        N, L, C = x.shape
        device = x.device

        # View as even/odd pairs
        # src (even): 0, 2, 4...
        # dst (odd):  1, 3, 5...
        src = x[:, 0::2, :]
        dst = x[:, 1::2, :]

        T_src = src.shape[1]
        T_dst = dst.shape[1]

        if T_dst == 0:
            # Cannot merge anything (width/height < 2)
            # Create a dummy restore index strictly for consistency if needed,
            # but usually we just bypass.
            return dict(bypass=True, L_out=L)

        # Only pair up to the length of dst (handles odd L where src has 1 tail)
        T_pair = T_dst

        # Calculate scores for pairs
        scores = self._pair_scores(src[:, :T_pair], dst)  # [N, T_pair]

        # Determine how many to merge
        r = T_pair if (num_prune is None) else min(num_prune, T_pair)
        if r < 0: r = 0

        # --- Selection ---
        # Use topk instead of argsort for speed on large tensors
        if r < T_pair:
            # We need to find the top r scores to MERGE.
            # The indices returned by topk are the ones where merge = True.
            _, top_k_idx = scores.topk(k=r, dim=-1)

            # Create a boolean mask for merging: [N, T_pair]
            # scatter is faster than creating zeros and indexing for large batches
            merge_mask = torch.zeros_like(scores, dtype=torch.bool)
            merge_mask.scatter_(1, top_k_idx, True)
        else:
            # Merge all
            merge_mask = torch.ones_like(scores, dtype=torch.bool)

        L_out = L - r

        # --- Order Restoration Plan (Indices only) ---
        restore_inds = None
        if self.if_order:
            # We simulate the merge process on INDICES [0, 1, 2, ..., L-1] to find the final permutation.
            # This is done once per forward pass.

            # 1. Create original indices
            # shape: [1, L] -> expanded to [N, L] only if strictly needed,
            # but usually broadcast works if selection is uniform across batch.
            # However, selection is data-dependent (per row), so we need [N, L].

            # Optimization: operate on int32/int64 indices
            orig_idx = torch.arange(L, device=device, dtype=torch.long).unsqueeze(0).expand(N, L)

            idx_src = orig_idx[:, 0::2]  # [N, T_src]
            idx_dst = orig_idx[:, 1::2]  # [N, T_dst]

            idx_src_main = idx_src[:, :T_pair]
            idx_tail = idx_src[:, T_pair:]  # [N, 0 or 1]

            # Split src indices into those that merge and those that stay (unm)
            # kept_src_mask = ~merge_mask

            # Using masked_select flattens the tensor, which we don't want.
            # We want to gather the indices.

            # UNMERGED Evens: Those where merge_mask is False
            # We need to collect them. Since the count is constant per row (T_pair - r),
            # we can just sort/partition the indices.
            # But we already know merge_mask.

            # Approach:
            # 1. 'dst' indices always survive (modified or not).
            # 2. 'src' indices survive ONLY if NOT merged.
            # 3. 'tail' always survives.

            # Collect all surviving original indices:
            # surviving = concat([idx_src_main[~merge_mask], idx_tail, idx_dst])
            # BUT: boolean indexing returns 1D array. We need [N, L_out].
            # Since r is constant for all N, we can reshape.

            idx_unm = idx_src_main[~merge_mask].view(N, T_pair - r)

            # Concatenate all surviving indices [N, L_out]
            surviving_indices = torch.cat([idx_unm, idx_tail, idx_dst], dim=1)

            # Now, we need the permutation that sorts these 'surviving_indices' back to ascending order.
            # Because the features will be concatenated in the order [unm, tail, dst],
            # we want to map: Position i in (unm, tail, dst) -> Position k in Output Image
            # Wait, 'surviving_indices' tells us the Original ID of the content at that position.
            # We want the content to appear in the order of Original IDs.
            # So if we argsort 'surviving_indices', we get the gather indices for the features.

            _, restore_inds = surviving_indices.sort(dim=1)

        return dict(
            bypass=False,
            merge_mask=merge_mask,  # [N, T_pair] bool
            restore_inds=restore_inds,  # [N, L_out] or None
            L_out=L_out,
            T_pair=T_pair
        )

    def _apply_merge_1d(self, x: torch.Tensor, plan: dict) -> torch.Tensor:
        """
        Apply the planned merge to x.
        x: [N, L, C]
        """
        if plan['bypass']:
            return x

        merge_mask = plan['merge_mask']  # [N, T_pair]
        restore_inds = plan['restore_inds']  # [N, L_out]
        T_pair = plan['T_pair']

        # Split
        src = x[:, 0::2, :]
        dst = x[:, 1::2, :]

        src_main = src[:, :T_pair, :]
        tail = src[:, T_pair:, :]

        # --- 1. Merge (Update dst) ---
        # We want: dst[mask] += src_main[mask]
        # Since src and dst are aligned by pair index, we can just add directly where mask is True.
        # This assumes 'sum' mode.

        if not self.if_prune:
            # In-place add is faster
            # But we need to be careful not to modify the input tensor 'x' in place if it's used elsewhere?
            # 'dst' is a slice of 'x'. Modifying 'dst' modifies 'x'.
            # Usually safe in a sequential block, but safer to clone if unsure.
            # To be safe and functional:

            # Weighted merge support
            if self.merge_mode == 'sum':
                # Advanced indexing copy logic:
                # dst_merged = dst.clone() # Optional safety
                # dst_merged[merge_mask] += src_main[merge_mask]

                # To avoid clone, we calculate the update term
                update = torch.zeros_like(dst)
                update[:, :T_pair, :][merge_mask] = src_main[merge_mask]
                dst = dst + update  # Out-of-place addition

            elif self.merge_mode == 'mean':
                # Requires size logic usually, but simplified here:
                update = torch.zeros_like(dst)
                update[:, :T_pair, :][merge_mask] = src_main[merge_mask]
                dst = (dst + update) * 0.5  # Rough approx for mean without explicit count tracking
                # Note: Correct mean requires tracking size (see merge_wavg2d),
                # here we stick to basic 'sum' logic implied by the mask.
            elif self.merge_mode == 'amax':
                update = torch.full_like(dst, -float('inf'))
                update[:, :T_pair, :][merge_mask] = src_main[merge_mask]
                dst = torch.maximum(dst, update)

        # --- 2. Prune (Keep unmerged src) ---
        # src_main[~merge_mask] -> flattening
        # We rely on N being preserved implicitly by reshape
        N = x.shape[0]
        r = merge_mask.sum() // N  # integer division to get per-row count

        unm = src_main[~merge_mask].view(N, -1, x.shape[-1])

        # --- 3. Concatenate ---
        # Layout: [unm, tail, dst]
        out_unordered = torch.cat([unm, tail, dst], dim=1)

        # --- 4. Restore Order ---
        if self.if_order and restore_inds is not None:
            # gather(dim=1, index=...)
            # We need to expand index to C
            restore_inds_exp = restore_inds.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            out = out_unordered.gather(1, restore_inds_exp)
            return out
        else:
            return out_unordered

    def forward(
            self,
            metric: torch.Tensor,
            num_prune_w: Optional[int] = None,
            num_prune_h: Optional[int] = None
    ) -> Callable[[torch.Tensor], torch.Tensor]:

        B, C, H, W = metric.shape

        # --- Plan Width ---
        # [B, C, H, W] -> [B, H, W, C] -> [B*H, W, C]
        # Using permute().reshape() is standard.
        metric_w_in = metric.permute(0, 2, 3, 1).reshape(B * H, W, C)
        plan_w = self._plan_1d(metric_w_in, num_prune_w, W)

        # Apply merge to metric to prepare for height planning
        # We must keep the metric synced
        metric_w_out = self._apply_merge_1d(metric_w_in, plan_w)

        # Reshape for height planning: [B*H, W_out, C] -> [B, H, W_out, C] -> [B, W_out, H, C] -> [B*W_out, H, C]
        W_out = plan_w['L_out']
        metric_h_in = metric_w_out.view(B, H, W_out, C).permute(0, 2, 1, 3).reshape(B * W_out, H, C)

        # --- Plan Height ---
        plan_h = self._plan_1d(metric_h_in, num_prune_h, H)

        # Capture plans in closure
        def prune_fn(x: torch.Tensor) -> torch.Tensor:
            # x: [B, C, H, W]
            B_x, C_x, H_x, W_x = x.shape

            # --- Apply Width ---
            # [B, C, H, W] -> [B*H, W, C]
            x_w = x.permute(0, 2, 3, 1).reshape(B_x * H_x, W_x, C_x)
            x_w = self._apply_merge_1d(x_w, plan_w)

            # --- Apply Height ---
            # x_w: [B*H, W_out, C]
            # Need [B*W_out, H, C]
            x_h = x_w.view(B_x, H_x, W_out, C_x).permute(0, 2, 1, 3).reshape(B_x * W_out, H_x, C_x)
            x_h = self._apply_merge_1d(x_h, plan_h)

            # Final reshape: [B*W_out, H_out, C] -> [B, W_out, H_out, C] -> [B, C, H_out, W_out]
            H_out = plan_h['L_out']
            x_out = x_h.view(B_x, W_out, H_out, C_x).permute(0, 3, 2, 1)

            return x_out

        return prune_fn

    def merge_wavg2d(
            self,
            prune_fn_builder: Callable[..., Callable[[torch.Tensor], torch.Tensor]],
            x: torch.Tensor,
            size: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if size is None:
            size = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)

        # Generate the prune function based on x
        prune_fn = prune_fn_builder(x)

        # Apply to weighted features
        # Note: prune_fn handles 'sum' logic.
        x_weighted = x * size
        x_merged = prune_fn(x_weighted)
        size_merged = prune_fn(size)

        # Normalize
        out = x_merged / (size_merged + self.eps)
        return out, size_merged


if __name__ == '__main__':
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 配置不同的下采样场景：56x56->50x50, 14x14->8x8
    scenarios = [
        {
            "name": "56x56 -> 50x50",
            "shape": (1, 384, 56, 56),
            "num_prune_w": 6,   # W_out = 56 - 6 = 50
            "num_prune_h": 6    # H_out = 56 - 6 = 50
        },
        {
            "name": "14x14 -> 8x8",
            "shape": (1, 384, 14, 14),
            "num_prune_w": 6,   # W_out = 14 - 6 = 8
            "num_prune_h": 6    # H_out = 14 - 6 = 8
        },
    ]

    num_iters = 1000  # 每个场景重复次数

    for cfg in scenarios:
        print(f"\n===== Scenario: {cfg['name']} =====")
        B, C, H, W = cfg["shape"]
        num_prune_w = cfg["num_prune_w"]
        num_prune_h = cfg["num_prune_h"]

        H_out = H - num_prune_h
        W_out = W - num_prune_w

        # ------------ ToMe2D profiling ------------
        merge2d = ToMe2D(if_order=True, distance='cosine', merge_mode='sum').to(device)
        merge2d.reset_profile()

        for _ in range(num_iters):
            metric = torch.randn(B, C, H, W, device=device)
            x = torch.randn(B, C, H, W, device=device)

            prune_fn = merge2d(metric, num_prune_w=num_prune_w, num_prune_h=num_prune_h)
            y = prune_fn(x)

        times = merge2d._profile_times
        total_time = sum(times.values())

        print("---- ToMe2D step-wise time ----")
        for k, v in times.items():
            avg_ms = (v / num_iters) * 1000.0
            pct = (v / total_time * 100.0) if total_time > 0 else 0.0
            print(f"{k:16s}: {avg_ms:8.4f} ms/iter  ({pct:5.2f}%)")

        tome_avg_ms = (total_time / num_iters) * 1000.0 if total_time > 0 else 0.0
        print(f"ToMe2D total (sum of steps): {tome_avg_ms:8.4f} ms/iter")

        # ------------ nearest 下采样 profiling ------------
        nearest_total_time = 0.0
        for _ in range(num_iters):
            x = torch.randn(B, C, H, W, device=device)
            if x.is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            y_nearest = F.interpolate(x, size=(H_out, W_out), mode='nearest')
            if x.is_cuda:
                torch.cuda.synchronize()
            nearest_total_time += time.perf_counter() - t0

        nearest_avg_ms = (nearest_total_time / num_iters) * 1000.0
        print(f"nearest total           : {nearest_avg_ms:8.4f} ms/iter")

        if nearest_avg_ms > 0:
            print(f"ToMe2D / nearest        : {tome_avg_ms / nearest_avg_ms:5.2f}x")
        else:
            print("nearest avg time is 0, ratio undefined.")
