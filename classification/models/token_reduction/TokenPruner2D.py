import time

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

# =============== 方案 B：在 cross-scan 序列上按 mask 精确删列 ===================
def _build_index_maps(H: int, W: int, device=None):
    """
    构建四个方向的索引映射：
      dir0: 原始行优先展平       idx0 = h*W + w
      dir1: 交换 H/W 后展平       idx1 = w*H + h
      dir2: dir0 的反向           idx2 = L-1-idx0
      dir3: dir1 的反向           idx3 = L-1-idx1
    返回:
      maps: dict { 'd0': (L,), 'd1': (L,), 'd2': (L,), 'd3': (L,) }
    """
    L = H * W
    idx0 = torch.arange(L, device=device).view(H, W)          # row-major
    idx1 = idx0.transpose(0, 1).contiguous().view(-1)         # transpose then flatten
    idx0 = idx0.view(-1)
    idx2 = (L - 1) - idx0
    idx3 = (L - 1) - idx1
    return {'d0': idx0, 'd1': idx1, 'd2': idx2, 'd3': idx3}

def _gather_last_dim(x: torch.Tensor, keep_idx: torch.Tensor) -> torch.Tensor:
    """
    x: (B, D, L)
    keep_idx: (B, K)  —— 每个 batch 的要保留的列索引（最后一维）
    return: (B, D, K)
    """
    B, D, L = x.shape
    Bk, K = keep_idx.shape
    assert B == Bk
    # 扩展索引到 (B, D, K)
    idx = keep_idx.unsqueeze(1).expand(B, D, K)
    return torch.gather(x, dim=-1, index=idx)

@torch.no_grad()
def prune_crossscan_by_mask(y: torch.Tensor, prune_mask: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    从 cross-scan 输出 y 中彻底删掉被剪的 token 列。
    Args:
        y: (B, 4, D, H*W)  —— cross-scan 产生的 4 个方向的序列
        prune_mask: (B, H, W), True=要剪
        H, W: 与 y 的 L=H*W 一致
    Return:
        y_kept: (B, 4, D, L_kept)
    说明：
      - 我们不用“值为 0”来判断要删谁，而是严格依赖 prune_mask（更安全）。
      - 四个方向的列索引不同（转置/反向），用映射表统一处理，确保删除的是同一批空间位置。
    """
    assert y.dim() == 4 and y.size(1) == 4, "y must be (B,4,D,L)"
    B, _, D, L = y.shape
    assert L == H * W
    assert prune_mask.shape == (B, H, W)

    maps = _build_index_maps(H, W, device=y.device)
    keep_mask_2d = (~prune_mask).to(torch.bool)     # True=保留
    keep_flat_d0 = keep_mask_2d.view(B, -1)         # 对应 dir0 的展平顺序 (row-major)
    K_per_b = keep_flat_d0.sum(dim=1)               # 每个 batch 的保留数

    # 要求各 batch 保留的数量一致（常见做法是全局或分窗口等比例剪枝 => 一致）
    K = int(K_per_b[0].item())
    if not torch.all(K_per_b == K):
        # 若不一致，可按最小 K 对齐/或做 padding；这里我们选择报错，避免 silent shape mismatch
        raise ValueError(f"Each batch must keep same count, got {K_per_b.tolist()}")

    # dir0 的保留索引（每个 batch）
    keep_idx_d0 = torch.nonzero(keep_flat_d0, as_tuple=False).split(K, dim=0)
    keep_idx_d0 = torch.stack([t[:, 1] for t in keep_idx_d0], dim=0)  # (B, K)

    # 其他三个方向的保留索引：通过映射表从 dir0 索引转换
    # m1/m2/m3 都是 (L,)
    m1, m2, m3 = maps['d1'], maps['d2'], maps['d3']  # device 已对齐
    keep_idx_d1 = m1[keep_idx_d0]  # (B,K)
    keep_idx_d2 = m2[keep_idx_d0]
    keep_idx_d3 = m3[keep_idx_d0]

    # 逐方向 gather
    y0 = _gather_last_dim(y[:, 0], keep_idx_d0)  # (B,D,K)
    y1 = _gather_last_dim(y[:, 1], keep_idx_d1)
    y2 = _gather_last_dim(y[:, 2], keep_idx_d2)
    y3 = _gather_last_dim(y[:, 3], keep_idx_d3)

    y_kept = torch.stack([y0, y1, y2, y3], dim=1)   # (B,4,D,K)
    return y_kept

class TokenPruner2D:
    """
    实用工具：在 (B, D, H, W) 的 2D 特征图上做 token 重要性评分与剪枝位置选择。
    - 支持多种评分: 'l2', 'var', 'entropy'（熵用softmax后的熵近似）
    - 支持按全局或分窗口（window_size）选择要剪的token
    - 保存每个 batch 的被剪位置，供后续“留空/填充/恢复”使用

    说明：
    - “留空”可理解为仅保存 mask/indices，不真正改变张量；或可用 apply_holes() 把被剪位置置零/置为占位值。
    - 后续对 SS2D/下采样等模块，如果需要保持 (H, W) 不变，可借助保存的 mask 做占位填充（占位接口已留）。

    用法示例：
        pruner = TokenPruner2D(score_type='l2', prune_ratio=0.3, window_size=None)
        scores = pruner.compute_importance(x)           # x: (B, D, H, W)
        mask = pruner.build_prune_mask(scores)          # True=要剪，False=保留，形状 (B, H, W)
        pruner.save_last_prune(mask)                    # 记住这次的剪枝位置
        x_hole = pruner.apply_holes(x, hole_value=0.0)  # （可选）把被剪位置“留空”（置0）
    """

    def __init__(
        self,
        score_type: str = "l2",     # 'l2' | 'var' | 'entropy'
        prune_ratio: float = 0.3,   # 剪掉比例 [0,1)
        window_size: Optional[Tuple[int, int]] = None,  # e.g., (7,7) 按窗口内排名剪枝；None=全局
        min_keep_per_window: int = 1,                   # 分窗口时，每窗口至少保留多少个
        eps: float = 1e-6,
    ):
        assert 0.0 <= prune_ratio < 1.0, "prune_ratio must be in [0,1)."
        assert score_type in {"l2", "var", "entropy"}
        self.score_type = score_type
        self.prune_ratio = prune_ratio
        self.window_size = window_size
        self.min_keep_per_window = max(0, int(min_keep_per_window))
        self.eps = eps
        # 累计掩码（跨层传递），True=已剪
        self.prev_mask: torch.Tensor | None = None

    def reset(self):
        self.prev_mask = None

    @torch.no_grad()
    def compute_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算每个 token 的重要性分数。
        输入: x (B, D, H, W)
        输出: scores (B, H, W)  —— 值越大越“重要”（保留倾向越强）
        """
        assert x.dim() == 4, "x must be (B, D, H, W)"
        B, D, H, W = x.shape

        if self.score_type == "l2":
            # 通道维做 L2 能量：||x||_2
            scores = torch.sqrt(torch.clamp((x ** 2).sum(dim=1), min=self.eps))  # (B,H,W)

        elif self.score_type == "var":
            # 通道方差：重要 token 通常激活更“尖锐”
            mean = x.mean(dim=1, keepdim=True)
            scores = ((x - mean) ** 2).mean(dim=1)  # (B,H,W)

        elif self.score_type == "entropy":
            # 近似“类别分布”熵：先对通道做 softmax，再算熵，熵小=>更“确定”=>更重要
            # 这里将“低熵 = 重要”转成“高分 = 重要”，所以用 (max_entropy - entropy)
            p = F.softmax(x, dim=1).clamp_min(self.eps)  # (B,D,H,W)
            entropy = -(p * (p + self.eps).log()).sum(dim=1)  # (B,H,W)
            max_entropy = torch.log(torch.tensor(D, device=x.device, dtype=x.dtype))
            scores = (max_entropy - entropy)  # 熵越小分数越大

        return scores

    @torch.no_grad()
    def build_prune_mask(self, scores: torch.Tensor, *, update_state: bool = True) -> torch.Tensor:
        """
        多层（逐层累计）剪枝的核心：
        - 输入 scores: (B,H,W)（分数越“大”越重要）
        - 仅在“未被剪掉”的位置上，按 self.prune_ratio 再剪一轮
        - 与 self.prev_mask 合并，得到新的累计掩码并（可选）写回

        返回:
            prune_mask: (B,H,W) ，True=已剪（包含历史 + 本层）
        """
        assert scores.dim() == 3, "scores must be (B,H,W)"
        B, H, W = scores.shape
        device = scores.device

        # 上一层累计掩码（True=已剪）
        if self.prev_mask is None:
            prev = torch.zeros((B, H, W), dtype=torch.bool, device=device)
        else:
            assert self.prev_mask.shape == (B, H, W)
            prev = self.prev_mask

        # 候选集：上一层未被剪的位置
        cand = ~prev  # True=可参与本层选择
        # 若一整个 batch 都无候选，直接返回累计掩码
        if cand.sum() == 0:
            return prev

        # ---------- 全局策略 ----------
        # 逐 batch 独立处理（每个 batch 对“自己的剩余候选”剪固定比例）
        new_mask = prev.clone()  # 从累计开始
        flat_scores = scores.view(B, -1)
        flat_cand = cand.view(B, -1)

        for b in range(B):
            cand_idx = torch.nonzero(flat_cand[b], as_tuple=False).flatten()  # 剩余候选的线性下标
            n_cand = cand_idx.numel()
            if n_cand == 0:
                continue  # 这个 batch 已经没有可剪位置了

            # 本层要剪的数量（对“候选”按比例）
            k_prune = int(n_cand * self.prune_ratio)
            # 留至少 1 个
            k_prune = max(0, min(n_cand - 1, k_prune))
            if k_prune == 0:
                continue  # 本层不剪

            # 对候选位置按分数升序剪（低分更不重要）
            cand_scores = flat_scores[b, cand_idx]  # (n_cand,)
            # 也可以用 topk 保留：k_keep = n_cand - k_prune；这里直接选出要剪的下标
            prune_rel = torch.topk(cand_scores, k=k_prune, largest=False, sorted=False).indices
            prune_abs = cand_idx[prune_rel]  # 映射回全局线性下标

            # 写入到 new_mask（True=剪）
            new_mask.view(B, -1)[b, prune_abs] = True

        prune_mask = new_mask


        # 状态回写（使其成为“累计掩码”）
        if update_state:
            self.prev_mask = prune_mask
        return prune_mask

    @torch.no_grad()
    def save_last_prune(self, prune_mask: torch.Tensor):
        """
        保存最近一次的剪枝位置（True=剪）。
        """
        assert prune_mask.dtype == torch.bool and prune_mask.dim() == 3
        self._state["last_prune_mask"] = prune_mask.detach().clone()

        # 也保存索引（方便后续复用/调试）
        B, H, W = prune_mask.shape
        idx = torch.nonzero(prune_mask, as_tuple=False)  # (N_pruned, 3) -> (b,y,z)
        self._state["last_pruned_indices"] = idx

    @torch.no_grad()
    def apply_holes(self, x: torch.Tensor, hole_value: float = 0.0) -> torch.Tensor:
        """
        （可选）把被剪的位置“留空”，不改变形状：
        - 用保存的 last_prune_mask 将对应 (H,W) 位置置为 hole_value。
        - 如果你只是想“先空着”，也可以不调用本函数，只保存 mask 即可。
        """
        assert "last_prune_mask" in self._state, "No saved prune mask. Call save_last_prune() first."
        prune_mask = self._state["last_prune_mask"]  # (B,H,W)
        B, D, H, W = x.shape
        assert prune_mask.shape == (B, H, W)

        x_out = x.clone()
        # 扩展到通道维做掩膜
        mask = prune_mask.unsqueeze(1).expand(B, D, H, W)  # True=剪
        x_out[mask] = hole_value
        return x_out

    # ====== 可扩展的占位接口（后续你可按需要完善） ======
    @torch.no_grad()
    def fill_holes(self, x: torch.Tensor, mode: str = "zeros") -> torch.Tensor:
        """
        TODO: 根据保存的位置做“空位填充”（如插值、从原图回填、邻域均值等）。
        现在先占位：与 apply_holes 等价或直接返回 x。
        """
        return x

    @torch.no_grad()
    def restore_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: 如果后续你做了 pack/下采样前需要“还原”至 (H,W)，在这里实现。
        现在先占位：直接返回 x。
        """
        return x

def cross_scan_with_pruning(x: torch.Tensor, prune_mask: Optional[torch.Tensor] = None):
    # x: (B, C, H, W) -> y: (B,4,C,L) or (B,4,C,L_kept)
    B, C, H, W = x.shape
    y = x.new_empty((B, 4, C, H * W))
    y[:, 0, :, :] = x.flatten(2, 3)                           # 水平展开
    y[:, 1, :, :] = x.transpose(2, 3).flatten(2, 3)           # 交换H/W后展开
    y[:, 2:4, :, :] = torch.flip(y[:, 0:2, :, :], dims=[-1])  # 反向两个方向
    if prune_mask is None:
        return y
    return prune_crossscan_by_mask(y, prune_mask, H, W)

def _scatter_last_dim(y_full: torch.Tensor, src: torch.Tensor, keep_idx: torch.Tensor):
    """
    把 src (B,D,K) 按 keep_idx (B,K) 散射到 y_full (B,D,L) 的最后一维对应位置上。
    其他位置保持为 0。
    """
    B, D, L = y_full.shape
    Bk, K = keep_idx.shape
    assert src.shape == (B, D, K)
    assert B == Bk
    idx = keep_idx.unsqueeze(1).expand(B, D, K)   # (B,D,K)
    return y_full.scatter_(dim=-1, index=idx, src=src)

@torch.no_grad()
def restore_pruned_crossscan(xs: torch.Tensor, prune_mask: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    将剪枝后的 cross-scan 序列 xs (B,4,D,L_kept) 还原到同一空间网格 (B,4,D,H,W)：
    - 以 row-major 的空间索引为“锚”，四个方向在同一 (h,w) 上的值一致
    - 被剪的位置补 0
    """
    assert xs.dim() == 4 and xs.size(1) == 4, "xs must be (B,4,D,L_kept)"
    B, _, D, K = xs.shape
    assert prune_mask.shape == (B, H, W)

    # row-major 的保留位置（同一空间锚）
    keep_mask_d0 = (~prune_mask).view(B, -1)   # (B, L)
    K_per_b = keep_mask_d0.sum(dim=1)
    if not torch.all(K_per_b == K):
        raise ValueError(f"Kept length mismatch with prune_mask: mask keeps {K_per_b.tolist()}, xs has {K}")

    # (B,K) 每个 batch 保留位置的 row-major 下标 r
    keep_idx_d0 = torch.nonzero(keep_mask_d0, as_tuple=False).split(K, dim=0)
    keep_idx_d0 = torch.stack([t[:, 1] for t in keep_idx_d0], dim=0)  # (B,K)

    # 将 row-major 下标转成 (h,w)
    h_idx = (keep_idx_d0 // W).long()   # (B,K)
    w_idx = (keep_idx_d0 %  W).long()   # (B,K)

    # 目标网格，全部置 0
    y_full_2d = xs.new_zeros((B, 4, D, H, W))  # (B,4,D,H,W)

    # 把四个方向的保留列写回同一 (h,w)
    # 用“先展平成 L 再 scatter”的方式向量化写入
    L = H * W
    y_full_flat = y_full_2d.view(B, 4, D, L)   # (B,4,D,L)
    pos = keep_idx_d0                          # (B,K)，所有方向共享的空间位置

    # 展开到 (B,D,K) 做 scatter
    pos_exp = pos.unsqueeze(1).expand(B, D, K)     # (B,D,K)

    # 逐方向 scatter 到同一空间坐标
    for ddir in range(4):
        y_full_flat[:, ddir].scatter_(dim=-1, index=pos_exp, src=xs[:, ddir])

    # 回到 (B,4,D,H,W)
    return y_full_flat.view(B, 4, D, H, W)

def merge_crossscan_directions(y_full_2d: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """
    合并 cross-scan 四个方向的结果 -> (B,D,H,W)

    Args:
        y_full_2d: (B,4,D,H,W)，已经对齐好的结果
        mode: "mean" 或 "sum"
              - mean: 四方向取平均
              - sum:  四方向相加

    Return:
        y_merged: (B,D,H,W)
    """
    assert y_full_2d.dim() == 5 and y_full_2d.size(1) == 4, "y_full_2d must be (B,4,D,H,W)"
    if mode == "mean":
        return y_full_2d.mean(dim=1)  # (B,D,H,W)
    elif mode == "sum":
        return y_full_2d.sum(dim=1)   # (B,D,H,W)
    else:
        raise ValueError("mode must be 'mean' or 'sum'")

# ------------------------- 测试用例 -------------------------
# ------------------- 多层剪枝主流程 -------------------
# ================== Config ==================
if __name__ == "__main__":
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True

    B, D, H, W = 128, 384, 14, 14
    L = H * W

    # 剪枝器：每层剪 10%
    pruner = TokenPruner2D(score_type="l2", prune_ratio=0.10)

    num_layers  = 8
    num_repeats = 100
    num_warmup  = 5
    device = "cuda:0"

    # ================== Helpers ==================
    def cuda_sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def new_events():
        return (torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True))

    def elapsed_ms(ev_start, ev_end):
        return ev_start.elapsed_time(ev_end)  # ms

    # ================== Warmup (不计入统计) ==================
    for _ in range(num_warmup):
        x = torch.randn(B, D, H, W, device=device)
        scores = pruner.compute_importance(x)
        prune_mask = pruner.build_prune_mask(scores)
        xs = cross_scan_with_pruning(x, prune_mask)
        y_full_2d = restore_pruned_crossscan(xs, prune_mask, H, W)
        y_merged  = merge_crossscan_directions(y_full_2d, mode="mean")
        pruner.reset()
    cuda_sync()

    # ================== Accumulators ==================
    t_import_ms   = 0.0
    t_mask_ms     = 0.0
    t_scan_ms     = 0.0
    t_restore_ms  = 0.0
    t_merge_ms    = 0.0
    wall_ms_total = 0.0

    # ================== Main benchmark ==================
    cuda_sync()
    wall_start = time.time()

    for repeat in range(num_repeats):
        x = torch.randn(B, D, H, W, device=device)
        kept_expected = L
        x_cur = x.clone()

        for layer in range(1, num_layers + 1):
            # a) compute_importance
            e1s, e1e = new_events()
            e1s.record()
            scores = pruner.compute_importance(x_cur)  # (B,H,W)
            e1e.record(); cuda_sync()
            t_import_ms += elapsed_ms(e1s, e1e)

            # b) build_prune_mask
            e2s, e2e = new_events()
            e2s.record()
            prune_mask = pruner.build_prune_mask(scores)  # (B,H,W)
            e2e.record(); cuda_sync()
            t_mask_ms += elapsed_ms(e2s, e2e)

            # c) cross_scan_with_pruning
            e3s, e3e = new_events()
            e3s.record()
            xs = cross_scan_with_pruning(x_cur, prune_mask)  # (B,4,D,L_kept)
            e3e.record(); cuda_sync()
            t_scan_ms += elapsed_ms(e3s, e3e)

            # d) restore_pruned_crossscan
            e4s, e4e = new_events()
            e4s.record()
            y_full_2d = restore_pruned_crossscan(xs, prune_mask, H, W)  # (B,4,D,H,W)
            e4e.record(); cuda_sync()
            t_restore_ms += elapsed_ms(e4s, e4e)

            # e) merge_crossscan_directions
            e5s, e5e = new_events()
            e5s.record()
            y_merged = merge_crossscan_directions(y_full_2d, mode="mean")  # (B,D,H,W)
            e5e.record(); cuda_sync()
            t_merge_ms += elapsed_ms(e5s, e5e)

            # 作为下一层输入
            x_cur = y_merged

        # 完成一轮后重置
        pruner.reset()

    cuda_sync()
    wall_end = time.time()
    wall_ms_total = (wall_end - wall_start) * 1000.0

    # ================== Report ==================
    total_steps = num_repeats * num_layers
    sub_total = t_import_ms + t_mask_ms + t_scan_ms + t_restore_ms + t_merge_ms
    eps = 1e-9

    def pct(x): return 100.0 * x / max(sub_total, eps)

    print("==== 分步骤性能统计（CUDA 事件计时）====")
    print(f"批大小 B={B}, D={D}, HxW={H}x{W}  |  layers/rep={num_layers}, repeats={num_repeats}")
    print(f"总层数（计时步数）: {total_steps}")

    print("\n-- 分步骤累计耗时（ms） --")
    print(f"compute_importance        : {t_import_ms:.3f} ms  ({pct(t_import_ms):.2f}%)")
    print(f"build_prune_mask          : {t_mask_ms:.3f} ms  ({pct(t_mask_ms):.2f}%)")
    print(f"cross_scan_with_pruning   : {t_scan_ms:.3f} ms  ({pct(t_scan_ms):.2f}%)")
    print(f"restore_pruned_crossscan  : {t_restore_ms:.3f} ms  ({pct(t_restore_ms):.2f}%)")
    print(f"merge_crossscan_directions: {t_merge_ms:.3f} ms  ({pct(t_merge_ms):.2f}%)")
    print(f"小计(五段之和)             : {sub_total:.3f} ms")

    print("\n-- 平均每层耗时（ms/layer） --")
    print(f"compute_importance        : {t_import_ms/total_steps:.6f} ms")
    print(f"build_prune_mask          : {t_mask_ms/total_steps:.6f} ms")
    print(f"cross_scan_with_pruning   : {t_scan_ms/total_steps:.6f} ms")
    print(f"restore_pruned_crossscan  : {t_restore_ms/total_steps:.6f} ms")
    print(f"merge_crossscan_directions: {t_merge_ms/total_steps:.6f} ms")
    print(f"五段合计                   : {sub_total/total_steps:.6f} ms")

    print("\n-- 端到端墙钟（包含随机输入、Python循环、reset 等） --")
    print(f"overall wall time         : {wall_ms_total:.3f} ms")
    print(f"平均单层墙钟               : {wall_ms_total/total_steps:.6f} ms")

    print("\n注：overall 包含随机输入生成与 Python 开销；五段合计仅统计 5 个 GPU 子步骤。")