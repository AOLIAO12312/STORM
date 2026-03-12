import torch
import torch.nn.functional as F
from typing import Optional, Tuple


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

class TokenPruner2D:
    """
    2D 特征上的 token 评分 + 批量无循环的 ordered merging（统一 r，无 CLS）。
    - 输入始终是 (B, D, H, W)
    - merge_tokens_unified：只产生映射，不再返回 x_merged
    - merge_and_cross_scan：依据 mask+rep_map 真正执行合并，并输出四方向 [B,4,D,L_kept]
    """

    def __init__(
        self,
        score_type: str = "l2",             # 'l2' | 'var' | 'entropy'
        prune_ratio: float = 0.3,           # 每轮减少比例 [0,1)
        eps: float = 1e-6,
        distance: str = "cosine",           # 'cosine' | 'l1' | 'l2'
        choose: str = "max",                # 'max' | '3c' | '5c' | '7c' | '14c' | '21c'
        merge_mode: str = "mean",           # 'mean' | 'sum'
    ):
        assert 0.0 <= prune_ratio < 1.0
        assert score_type in {"l2", "var", "entropy"}
        assert distance in {"cosine", "l1", "l2"}
        assert choose in {"max", "3c", "5c", "7c", "14c", "21c"}
        assert merge_mode in {"mean", "sum"}

        self.score_type = score_type
        self.prune_ratio = prune_ratio
        self.eps = eps
        self.distance = distance
        self.choose = choose
        self.merge_mode = merge_mode

        # 1) __init__ 里新增一行状态（其他不变）
        self.prev_mask: Optional[torch.Tensor] = None  # (B,H,W)
        self.rep_map_global: Optional[torch.Tensor] = None  # (B,L) 累计：原位置 -> 当前列号

    @torch.no_grad()
    def merge_tokens_unified(
            self,
            x: torch.Tensor,  # (B,D,H,W)
            *,
            num_prune: Optional[int] = None,  # 统一 r；若 None，则 r=floor(K_alive_min*ratio) 且 <= floor(K_alive_min/2)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert x.dim() == 4, "x must be (B,D,H,W)"
        B, D, H, W = x.shape
        L = H * W
        device, dtypef = x.device, x.dtype

        # ---------- 1) 历史状态 ----------
        if self.prev_mask is None:
            self.prev_mask = torch.zeros((B, H, W), dtype=torch.bool, device=device)
        if getattr(self, "rep_map_global", None) is None:
            self.rep_map_global = torch.arange(L, device=device).view(1, L).expand(B, L).clone()

        # ---------- 2) 活跃原位置 ----------
        alive_pos = (~self.prev_mask).view(B, L)  # (B,L) bool
        rep_old = self.rep_map_global.long()  # (B,L)

        # ---------- 3) 聚合到当前列空间（sum & count） ----------
        # K_cur_max：仅一次同步
        K_cur_max = int(rep_old.amax(dim=1).amax().item()) + 1
        x_seq = x.view(B, D, L)

        x_sum = torch.zeros((B, D, K_cur_max), device=device, dtype=dtypef)
        x_sum.scatter_add_(2, rep_old.unsqueeze(1).expand(B, D, L), x_seq)

        cnt_all = torch.zeros((B, 1, K_cur_max), device=device, dtype=dtypef)
        cnt_all.scatter_add_(2, rep_old.unsqueeze(1), torch.ones((B, 1, L), device=device, dtype=dtypef))

        # ---------- 4) 活跃列 ----------
        alive_cnt = torch.zeros((B, 1, K_cur_max), device=device, dtype=torch.long)
        alive_cnt.scatter_add_(2, rep_old.unsqueeze(1), alive_pos.long().unsqueeze(1))
        alive_cols = (alive_cnt.squeeze(1) > 0)  # (B,K_cur_max) bool
        K_alive = alive_cols.sum(dim=1)  # (B,)
        if (K_alive < 2).any():  # 任一 batch 无法成对则不再合并
            size_merged = cnt_all  # (B,1,K_cur_max)
            mask_round = torch.zeros((B, H, W), device=device, dtype=torch.bool)
            return size_merged, mask_round, self.rep_map_global

        # ---------- 5) 取每个 batch 的活跃列索引（用 topk 代替全量排序） ----------
        K_alive_max = int(K_alive.max().item())
        score_alive = alive_cols.float()  # 1=alive, 0=dead
        # 选出每行前 K_alive_max 的列号（alive 全入选，dead 填充）
        _, idx_alive = score_alive.topk(K_alive_max, dim=1, largest=True, sorted=False)  # (B,K_alive_max)
        pos_rng = torch.arange(K_alive_max, device=device).unsqueeze(0)
        keep_alive = pos_rng < K_alive.unsqueeze(1)  # (B,K_alive_max)
        idx_alive = torch.where(keep_alive, idx_alive, torch.full_like(idx_alive, -1))

        # 收集活跃列特征（mean）
        gather_idx = idx_alive.clamp_min(0).unsqueeze(1).expand(B, D, K_alive_max)
        x_alive_sum = x_sum.gather(2, gather_idx)  # (B,D,K_alive_max)
        cnt_alive = cnt_all.gather(2, idx_alive.clamp_min(0).unsqueeze(1))  # (B,1,K_alive_max)
        pad_mask = (idx_alive < 0).unsqueeze(1)  # (B,1,K_alive_max)
        x_alive_sum = x_alive_sum.masked_fill(pad_mask.expand_as(x_alive_sum), 0)
        cnt_alive = cnt_alive.masked_fill(pad_mask, 0)
        x_alive_mean = x_alive_sum / cnt_alive.clamp_min(1.0)  # (B,D,K_alive_max)

        xk = x_alive_mean.transpose(1, 2).contiguous()  # (B,K_alive_max,D)

        # ---------- 6) 偶/奇配对 + 逐对相似度（O(B·m·D)） ----------
        K_eff = K_alive_max
        m = K_eff // 2
        two_m = 2 * m
        has_tail = (K_eff % 2 == 1)

        src = xk[:, :two_m:2, :]  # (B,m,D)
        dst = xk[:, 1:two_m:2, :]  # (B,m,D)

        # 分数（仅对齐对）：cosine / -L1 / -L2
        dist = getattr(self, "distance", "cosine")
        if dist == "cosine":
            a = torch.nn.functional.normalize(src, dim=-1, eps=1e-6)
            b = torch.nn.functional.normalize(dst, dim=-1, eps=1e-6)
            pair_score = (a * b).sum(dim=-1)  # (B,m)
        elif dist == "l1":
            pair_score = -(src - dst).abs().sum(dim=-1)  # (B,m)
        else:  # 'l2'
            pair_score = -torch.norm(src - dst, dim=-1)  # (B,m)

        # ---------- 7) 统一 r ----------
        K_alive_min = int(K_alive.min().item())
        m_min = K_alive_min // 2
        if num_prune is None:
            r_default = int(getattr(self, "prune_ratio", 0.0) * K_alive_min)
            r_target = int(K_alive_min - 1) if r_default <= 0 else r_default
            r = max(0, min(r_target, m_min))
        else:
            r = max(0, min(int(num_prune), m_min))
        if r == 0:
            size_merged = cnt_all
            mask_round = torch.zeros((B, H, W), device=device, dtype=torch.bool)
            return size_merged, mask_round, self.rep_map_global

        # 选前 r 个配对合并：src_idx_sel / src_idx_unm；dst_idx_sel=同序
        order = pair_score.argsort(dim=1, descending=True)  # (B,m)
        src_idx_sel = order[:, :r].unsqueeze(-1)  # (B,r,1)
        src_idx_unm = order[:, r:].unsqueeze(-1)  # (B,m-r,1)
        dst_idx_sel = src_idx_sel  # 对齐配对

        # ---------- 8) 活跃容器索引 -> 旧列号 ----------
        even_cols = torch.arange(0, two_m, 2, device=device).view(1, m)  # (1,m)
        odd_cols = torch.arange(1, two_m, 2, device=device).view(1, m)  # (1,m)

        even_old = idx_alive.gather(1, even_cols.expand(B, m))  # (B,m)
        odd_old = idx_alive.gather(1, odd_cols.expand(B, m))  # (B,m)

        src_cols_sel = even_old.gather(1, src_idx_sel.squeeze(-1)).unsqueeze(-1)  # (B,r,1)
        src_cols_unm = even_old.gather(1, src_idx_unm.squeeze(-1)).unsqueeze(-1)  # (B,m-r,1)
        dst_cols_all = odd_old.unsqueeze(-1)  # (B,m,1)
        dst_cols_sel = odd_old.gather(1, dst_idx_sel.squeeze(-1)).unsqueeze(-1)  # (B,r,1)

        # ---------- 9) 规模更新（sum） ----------
        cnt_vec = cnt_all.squeeze(1)  # (B,K_cur_max)
        size_unm_src = cnt_vec.gather(1, src_cols_unm.squeeze(-1)).unsqueeze(-1)  # (B,m-r,1)
        size_dst = cnt_vec.gather(1, dst_cols_all.squeeze(-1)).unsqueeze(-1)  # (B,m,1)
        add_src = cnt_vec.gather(1, src_cols_sel.squeeze(-1)).unsqueeze(-1)  # (B,r,1)
        size_dst.scatter_add_(1, dst_idx_sel, add_src)  # (B,m,1)

        # ---------- 10) 组装保留列（未并入的 src + 所有 dst + 尾巴）并“挤掉 -1” ----------
        pos_cat = torch.cat([src_cols_unm, dst_cols_all], dim=1)  # (B,(m-r)+m,1)
        size_cat = torch.cat([size_unm_src, size_dst], dim=1)  # (B,(m-r)+m,1)

        if has_tail:
            tail_idx = torch.tensor(two_m, device=device).view(1, 1).expand(B, 1)
            tail_old = idx_alive.gather(1, tail_idx).unsqueeze(-1)  # (B,1,1)
            tail_siz = cnt_vec.gather(1, tail_old.squeeze(-1)).unsqueeze(-1)  # (B,1,1)
            pos_cat = torch.cat([pos_cat, tail_old], dim=1)  # (B,K_new~,1)
            size_cat = torch.cat([size_cat, tail_siz], dim=1)  # (B,K_new~,1)

        valid_mask = (pos_cat.squeeze(-1) >= 0)  # (B,K_new~)
        valid_count = valid_mask.sum(dim=1)  # (B,)
        K_new_max = int(valid_count.max().item())

        # 用 topk 把有效位“挤到前面”（比分拣/全排序便宜）
        keep_score = valid_mask.float()
        _, order2 = keep_score.topk(K_new_max, dim=1, largest=True, sorted=True)  # (B,K_new_max)

        pos_keep = torch.gather(pos_cat.squeeze(-1), 1, order2)  # (B,K_new_max)
        siz_keep = torch.gather(size_cat.squeeze(-1), 1, order2).unsqueeze(-1)  # (B,K_new_max,1)

        keep_mask2 = (torch.arange(K_new_max, device=device).unsqueeze(0) < valid_count.unsqueeze(1))  # (B,K_new_max)
        pos_keep = torch.where(keep_mask2, pos_keep, torch.full_like(pos_keep, -1))
        siz_keep = torch.where(keep_mask2.unsqueeze(-1), siz_keep, torch.zeros_like(siz_keep))

        # ---------- 11) old_col -> new_col（批量重编号） ----------
        rep_cols = torch.full((B, K_cur_max), -1, device=device, dtype=torch.long)  # (B,K_cur_max)

        # 为无效位准备“安全索引”：把无效 pos 写到 0 位，且 new_id=-1（amax 不会覆盖有效写入）
        safe_index = pos_keep.clamp_min(0)
        new_ids = torch.arange(K_new_max, device=device).unsqueeze(0).expand(B, -1)  # (B,K_new_max)
        new_ids = torch.where(keep_mask2, new_ids, torch.full_like(new_ids, -1))

        # 写入 kept old_col -> new_col
        rep_cols.scatter_reduce_(1, safe_index, new_ids, reduce="amax", include_self=True)

        # 被并掉的 src old_col -> 其对应 dst 的 new_col
        dst_new_cols = rep_cols.gather(1, dst_cols_sel.squeeze(-1))  # (B,r)
        rep_cols.scatter_(1, src_cols_sel.squeeze(-1), dst_new_cols)  # old src -> dst_new

        # ---------- 12) 更新累计映射 & 本轮新增 mask ----------
        rep_map_old = self.rep_map_global  # (B,L)
        rep_map_new = rep_cols.gather(1, rep_map_old)  # (B,L)
        self.rep_map_global = rep_map_new

        src_cols_set = src_cols_sel.squeeze(-1)  # (B,r)
        new_mask_flat = (rep_map_old.unsqueeze(-1) == src_cols_set.unsqueeze(1)).any(dim=-1)  # (B,L)
        new_mask_flat &= (~self.prev_mask.view(B, L))
        mask_round = new_mask_flat.view(B, H, W)

        self.prev_mask = self.prev_mask | mask_round

        # ---------- 13) 输出 ----------
        size_merged = siz_keep.transpose(1, 2).contiguous()  # (B,1,K_new_max)
        return size_merged, self.prev_mask, rep_map_new

    @torch.no_grad()
    def merge_and_cross_scan(
            self,
            x: torch.Tensor,  # (B,D,H,W)
            mask: torch.Tensor,  # (B,H,W) True=非代表（被合并）  # 未用到，但保留接口
            rep_map: torch.Tensor,  # (B,L)   原位置 -> 列索引（0..K-1）
            H: int,
            W: int,
            *,
            return_sizes: bool = True
    ):
        """
        返回:
          - xs4:   (B,4,D,K)   四个扫描方向的合并序列
          - sizes4:(B,4,1,K)   对应列的聚合规模（若 return_sizes=True）
        优化:
          - 四方向并行，无 Python for 循环
        """
        assert x.dim() == 4
        B, D, Hx, Wx = x.shape
        assert Hx == H and Wx == W
        L = H * W
        device, dtypef = x.device, x.dtype

        # === 1) 按 rep_map 聚合 (sum/mean) ===
        K = int(rep_map.max().item()) + 1
        x_seq = x.view(B, D, L)
        idx = rep_map.unsqueeze(1).expand(B, D, L)  # (B,D,L)

        x_sum = torch.zeros((B, D, K), device=device, dtype=dtypef)
        x_sum.scatter_add_(2, idx, x_seq)

        cnt = torch.zeros((B, 1, K), device=device, dtype=dtypef)
        cnt.scatter_add_(2, rep_map.unsqueeze(1), torch.ones((B, 1, L), device=device, dtype=dtypef))

        if getattr(self, "merge_mode", "mean") == "mean":
            x_agg = x_sum / cnt.clamp_min(1.0)
        else:  # "sum"
            x_agg = x_sum

        # === 2) 并行计算四方向顺序 ===
        maps = _build_index_maps(H, W, device=device)  # 提供的函数，返回 1D 索引
        maps4 = torch.stack([maps['d0'], maps['d1'], maps['d2'], maps['d3']], dim=0)  # (4,L)
        maps4 = maps4.unsqueeze(0).expand(B, -1, -1)  # (B,4,L)

        # rep_dir: (B,4,L) —— 各方向扫描得到的列号序列
        rep_dir = rep_map.unsqueeze(1).expand(B, 4, L).gather(2, maps4)

        # 每方向“首出现位置”（没有出现的列会保持 L+1，排序时排到后面）
        first_pos = torch.full((B, 4, K), L + 1, device=device, dtype=torch.long)
        arangeL = torch.arange(L, device=device, dtype=torch.long).view(1, 1, L).expand(B, 4, L)
        if hasattr(first_pos, "scatter_reduce_"):
            first_pos.scatter_reduce_(2, rep_dir, arangeL, reduce="amin")
        else:
            first_pos = first_pos.scatter_reduce(2, rep_dir, arangeL, "amin", include_self=True)

        # order: (B,4,K) —— 每方向的列顺序（位置 -> 规范列号）
        order = first_pos.argsort(dim=2)

        # 依据列顺序 gather 成四方向序列
        xs4 = x_agg.unsqueeze(1).expand(B, 4, D, K).gather(3, order.unsqueeze(2).expand(B, 4, D, K))  # (B,4,D,K)

        if return_sizes:
            sizes4 = cnt.unsqueeze(1).expand(B, 4, 1, K).gather(3, order.unsqueeze(2))  # (B,4,1,K)  <-- 修正处
            return xs4, sizes4
        return xs4

    # # -------- 依据 mask+rep_map 真正执行合并 & Cross-Scan 四方向输出 --------
    # @torch.no_grad()
    # def merge_and_cross_scan(
    #     self,
    #     x: torch.Tensor,          # (B,D,H,W)
    #     mask: torch.Tensor,       # (B,H,W) True=非代表（被合并）
    #     rep_map: torch.Tensor,    # (B,L)   原位置 -> 列索引（0..K-1）
    #     H: int,
    #     W: int,
    #     *,
    #     return_sizes: bool = True
    # ):
    #     """
    #     返回:
    #       - xs4:   (B, 4, D, L_kept)   四个扫描方向的合并序列
    #       - sizes4:(B, 4, 1, L_kept)   对应列的聚合规模（若 return_sizes=True）
    #     说明：
    #       - 先按 rep_map 做 sum/mean 聚合得到 (B,D,K)
    #       - 再按四方向“首次出现”顺序为每个列（代表 token）排序，得到每方向列顺序
    #     """
    #     assert x.dim() == 4
    #     B, D, Hx, Wx = x.shape
    #     assert Hx == H and Wx == W
    #     L = H * W
    #     device = x.device
    #     dtypef = x.dtype
    #
    #     # 计算 K（列数=L_kept）
    #     # 统一 r 的前提下，各 batch 的 K 一致；这里取全局 max+1
    #     K = int(rep_map.max().item()) + 1
    #
    #     # === 1) 按 rep_map 对 (B,D,L) 做 sum/mean 聚合 ===
    #     x_seq = x.view(B, D, L)
    #     idx = rep_map.unsqueeze(1).expand(B, D, L)                   # (B,D,L) -> 列索引
    #     x_sum = torch.zeros((B, D, K), device=device, dtype=dtypef)
    #     x_sum.scatter_add_(2, idx, x_seq)                            # sum 聚合
    #
    #     cnt = torch.zeros((B, 1, K), device=device, dtype=dtypef)
    #     ones = torch.ones((B, 1, L), device=device, dtype=dtypef)
    #     cnt.scatter_add_(2, rep_map.unsqueeze(1), ones)              # 每列规模
    #
    #     if self.merge_mode == "mean":
    #         x_agg = x_sum / cnt.clamp_min(1.0)
    #     else:  # "sum"
    #         x_agg = x_sum
    #
    #     # === 2) 四方向列顺序：按“首次出现位置”排序（稳定、无循环）===
    #     maps = _build_index_maps(H, W, device=device)                # 提供的函数
    #     xs_dir = []
    #     sizes_dir = []
    #
    #     arangeL = torch.arange(L, device=device).view(1, L).expand(B, L)  # (B,L)
    #
    #     for key in ('d0', 'd1', 'd2', 'd3'):
    #         idx_dir = maps[key].view(1, L).expand(B, L)                    # (B,L)
    #         rep_dir = rep_map.gather(1, idx_dir)                           # (B,L) 列号按该方向扫描顺序
    #
    #         # 计算每个列号的“首次出现位置” t_first[c] = min{ p | rep_dir[p] == c }
    #         big = torch.full((B, K), L + 1, device=device, dtype=torch.long)      # 初始化为很大
    #         # 注意：scatter_reduce_ 的 reduce='amin' 需要 PyTorch 2.0+；若不支持，可改用 segment-min trick
    #         big = big.to(torch.long)
    #         pos_long = arangeL.to(torch.long)
    #         big.scatter_reduce_(1, rep_dir, pos_long, reduce='amin')              # (B,K) 每列的首出现位置
    #
    #         order = big.argsort(dim=1)                                            # (B,K) 小者在前 => 列顺序
    #         # 依据列顺序 gather
    #         xs_ordered   = x_agg.gather(2, order.unsqueeze(1).expand(B, D, K))    # (B,D,K)
    #         xs_dir.append(xs_ordered)
    #
    #         if return_sizes:
    #             sizes_ordered = cnt.gather(2, order.unsqueeze(1))                 # (B,1,K)
    #             sizes_dir.append(sizes_ordered)
    #
    #     xs4 = torch.stack(xs_dir, dim=1)              # (B,4,D,K)
    #     if return_sizes:
    #         sizes4 = torch.stack(sizes_dir, dim=1)    # (B,4,1,K)
    #         return xs4, sizes4
    #     return xs4

    @torch.no_grad()
    def restore_2d_from_cross_scans(
            self,
            xs4: torch.Tensor,  # (B, 4, D, K)
            rep_map: torch.Tensor,  # (B, L)
            H: int,
            W: int,
            *,
            reduce: str = "mean",  # 'mean' | 'sum' | 'max'
    ) -> torch.Tensor:
        """
        将四方向序列 (B,4,D,K) 合并还原到 2D 特征 (B,D,H,W)。
        - 向量化：方向维度一次性并行处理（无 for di in range(4) 循环）。
        - 步骤：求各方向首出现 -> 得到列顺序 -> 逆置换对齐 -> 聚合 -> 回填。
        """
        assert xs4.dim() == 4 and xs4.shape[1] == 4, "xs4 must be (B,4,D,K)"
        assert rep_map.dim() == 2
        B, _, D, K = xs4.shape
        L = H * W
        device = xs4.device

        # ---- 1) 构建方向索引 (B,4,L) ----
        maps = _build_index_maps(H, W, device=device)
        maps4 = torch.stack(
            (maps['d0'].view(-1), maps['d1'].view(-1), maps['d2'].view(-1), maps['d3'].view(-1)),
            dim=0
        ).unsqueeze(0).expand(B, -1, -1)  # (B,4,L)

        # ---- 2) 投影 rep_map -> 每个方向的扫描顺序 ----
        # rep_dir: (B,4,L)
        rep_dir = rep_map.unsqueeze(1).expand(B, 4, L).gather(2, maps4)

        # ---- 3) 每个方向的列顺序 ----
        first_pos = torch.full((B, 4, K), L + 1, device=device, dtype=torch.long)
        arangeL = torch.arange(L, device=device, dtype=torch.long).view(1, 1, L).expand(B, 4, L)
        if hasattr(first_pos, "scatter_reduce_"):
            first_pos.scatter_reduce_(2, rep_dir, arangeL, reduce="amin")
        else:
            first_pos = first_pos.scatter_reduce(2, rep_dir, arangeL, "amin", include_self=True)

        order = first_pos.argsort(dim=2)  # (B,4,K)
        inv_order = torch.empty_like(order)
        arangeK = torch.arange(K, device=device, dtype=torch.long).view(1, 1, K).expand_as(order)
        inv_order.scatter_(2, order, arangeK)  # (B,4,K)

        # ---- 4) 将 4 个方向的 (B,4,D,K) 对齐到规范列顺序 ----
        idx = inv_order.view(B, 4, 1, K).expand(B, 4, D, K)
        xs_canon = xs4.gather(3, idx)  # (B,4,D,K)

        # ---- 5) 聚合 ----
        if reduce == "mean":
            x_merged = xs_canon.mean(dim=1)  # (B,D,K)
        elif reduce == "sum":
            x_merged = xs_canon.sum(dim=1)  # (B,D,K)
        elif reduce == "max":
            x_merged = xs_canon.amax(dim=1)  # (B,D,K)
        else:
            raise ValueError(f"Unknown reduce='{reduce}'")

        # ---- 6) 回填到 2D ----
        out_seq = x_merged.gather(2, rep_map.unsqueeze(1).expand(B, D, L))  # (B,D,L)
        return out_seq.view(B, D, H, W)

    def reset(self):
        self.prev_mask = None
        self.rep_map_global = None


# if __name__ == '__main__':
#     import os
#     from matplotlib import pyplot as plt
#     # 确保输出文件夹存在
#     os.makedirs("./vis", exist_ok=True)
#     pruner = TokenPruner2D(score_type="l2", prune_ratio=0.0)  # ratio=0，强制用 num_prune
#     B, D, H, W = 128, 384, 14, 14
#     x = torch.randn(B, D, H, W, device='cuda:0')
#     # 循环 8 次
#     for step in range(8):
#         print(f"\n=== Iter {step + 1} ===")
#
#         # 2.1 生成 rep_map/mask
#         size_merged, mask, rep_map = pruner.merge_tokens_unified(x, num_prune=10)
#
#         # 可视化 mask[0]
#         plt.figure(figsize=(5, 5))
#         plt.imshow(mask[0].cpu().numpy(), cmap="gray")
#         plt.title(f"Iter {step + 1} Mask[0]")
#         plt.axis("off")
#         plt.savefig(f"./vis/mask_iter{step + 1}.png")
#         plt.close()
#
#         # 2.2 真正执行合并 + 四方向序列
#         xs4, sizes4 = pruner.merge_and_cross_scan(x, mask, rep_map, H, W)  # (B,4,D,L_kept)
#         print("xs4:", xs4.shape)
#
#         # 2.3 从四方向恢复 2D 特征
#         x = pruner.restore_2d_from_cross_scans(xs4, rep_map, H, W)  # (B,D,H,W)
#         print("x restored:", x.shape)
#
#         # 更新序列长度
#         L_kept = (rep_map.max() + 1).item()
#         print(f"L_kept after merge: {L_kept}")

if __name__ == "__main__":
    import torch
    import time

    # ================== Config ==================
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True

    B, D, H, W = 128, 384, 14, 14
    num_iters = 8
    num_repeats = 100
    num_warmup = 5  # 预热次数，不计入统计

    device = "cuda:0"

    # ================== Pruner ==================
    pruner = TokenPruner2D(score_type="l2", prune_ratio=0.0)  # ratio=0，强制用 num_prune


    # ================== Helpers ==================
    def ms():
        return time.time() * 1000.0


    def cuda_sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()


    def new_events():
        return (torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True))


    def record_duration_ms(start_event, end_event):
        return start_event.elapsed_time(end_event)  # ms


    # ================== Warmup ==================
    for _ in range(num_warmup):
        x = torch.randn(B, D, H, W, device=device)
        size_merged, mask, rep_map = pruner.merge_tokens_unified(x, num_prune=10)
        xs4, sizes4 = pruner.merge_and_cross_scan(x, mask, rep_map, H, W)
        x = pruner.restore_2d_from_cross_scans(xs4, rep_map, H, W)
        pruner.reset()
    cuda_sync()

    # ================== Timing accumulators ==================
    merge_ms_total = 0.0
    scan_ms_total = 0.0
    restore_ms_total = 0.0
    overall_ms_total = 0.0

    # ================== Main benchmark ==================
    cuda_sync()
    overall_start = ms()

    for repeat in range(num_repeats):
        x = torch.randn(B, D, H, W, device=device)

        for step in range(num_iters):
            # ---- merge_tokens_unified ----
            m_s, m_e = new_events()
            m_s.record()
            size_merged, mask, rep_map = pruner.merge_tokens_unified(x, num_prune=10)
            m_e.record()
            cuda_sync()
            merge_ms = record_duration_ms(m_s, m_e)

            # ---- merge_and_cross_scan ----
            s_s, s_e = new_events()
            s_s.record()
            xs4, sizes4 = pruner.merge_and_cross_scan(x, mask, rep_map, H, W)
            s_e.record()
            cuda_sync()
            scan_ms = record_duration_ms(s_s, s_e)

            # ---- restore_2d_from_cross_scans ----
            r_s, r_e = new_events()
            r_s.record()
            x = pruner.restore_2d_from_cross_scans(xs4, rep_map, H, W)
            r_e.record()
            cuda_sync()
            restore_ms = record_duration_ms(r_s, r_e)

            merge_ms_total += merge_ms
            scan_ms_total += scan_ms
            restore_ms_total += restore_ms

        pruner.reset()

    cuda_sync()
    overall_end = ms()
    overall_ms_total = overall_end - overall_start

    # ================== Report ==================
    total_steps = num_repeats * num_iters
    sub_total = merge_ms_total + scan_ms_total + restore_ms_total
    eps = 1e-9


    def pct(x):
        return 100.0 * x / max(sub_total, eps)


    print("==== 剪枝流水线性能统计（CUDA 事件计时）====")
    print(f"批大小 B={B}, D={D}, HxW={H}x{W}, iters/rep={num_iters}, repeats={num_repeats}")
    print(f"总步数（计时）：{total_steps}")

    print("\n-- 分步骤累计耗时（ms） --")
    print(f"merge_tokens_unified     : {merge_ms_total:.3f} ms  ({pct(merge_ms_total):.2f}%)")
    print(f"merge_and_cross_scan     : {scan_ms_total:.3f} ms  ({pct(scan_ms_total):.2f}%)")
    print(f"restore_2d_from_cross... : {restore_ms_total:.3f} ms  ({pct(restore_ms_total):.2f}%)")
    print(f"小计(三段之和)            : {sub_total:.3f} ms")

    print("\n-- 平均每步耗时（ms/step） --")
    print(f"merge_tokens_unified     : {merge_ms_total / total_steps:.6f} ms")
    print(f"merge_and_cross_scan     : {scan_ms_total / total_steps:.6f} ms")
    print(f"restore_2d_from_cross... : {restore_ms_total / total_steps:.6f} ms")
    print(f"三段合计                  : {sub_total / total_steps:.6f} ms")

    print("\n-- 整体墙钟时间（包含循环、张量生成、重置等） --")
    print(f"overall wall time        : {overall_ms_total:.3f} ms")
    print(f"平均单步墙钟             : {overall_ms_total / total_steps:.6f} ms")
    print("\n注：overall 包含张量随机生成、Python 开销与 reset() 等，三段小计只统计三个 GPU 子步骤。")

