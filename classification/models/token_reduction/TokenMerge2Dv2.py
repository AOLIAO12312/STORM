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
            num_prune: Optional[int] = None,  # 统一 r；若 None，则 r=floor(K_cur*ratio)，且 <= floor(K_cur/2)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        基于历史 prev_mask 的多轮合并（全向量化版本，无 per-batch Python 循环）：
          - 仅在上一轮未被丢弃的位置(~prev_mask)上继续做两两配对合并；
          - 返回累计后的：
              size_merged:(B,1,K_new_max) —— 本轮后每列聚合规模（按 batch 最大 K_new 对齐，pad=0）
              mask_round :(B,H,W)          —— 本轮新被并入的原位置（历史已 True 的不再重复标记）
              rep_map    :(B,L)            —— 原位置 -> 本轮后的列号(0..K_new[b]-1)，各 batch 独立编号
        """
        assert x.dim() == 4, "x must be (B,D,H,W)"
        B, D, H, W = x.shape
        L = H * W
        device, dtypef = x.device, x.dtype

        # ---------- 1) 初始化历史状态 ----------
        if self.prev_mask is None:
            self.prev_mask = torch.zeros((B, H, W), dtype=torch.bool, device=device)
        if getattr(self, "rep_map_global", None) is None:
            self.rep_map_global = torch.arange(L, device=device).view(1, L).expand(B, L).clone()

        # ---------- 2) 当前活跃原位置 ----------
        alive_pos = (~self.prev_mask).view(B, L)  # (B,L) bool

        # ---------- 3) 聚合到“当前列空间” ----------
        rep_old = self.rep_map_global  # (B,L)
        K_cur = int(rep_old.max().item()) + 1  # 注意：动态大小需要一次 .item() 同步

        x_seq = x.view(B, D, L)
        # sum 聚合
        x_sum = torch.zeros((B, D, K_cur), device=device, dtype=dtypef)
        x_sum.scatter_add_(2, rep_old.unsqueeze(1).expand(B, D, L), x_seq)
        # 计数（用于 mean）
        cnt_all = torch.zeros((B, 1, K_cur), device=device, dtype=dtypef)
        cnt_all.scatter_add_(2, rep_old.unsqueeze(1), torch.ones((B, 1, L), device=device, dtype=dtypef))

        # ---------- 4) 活跃列筛出 ----------
        alive_int = alive_pos.long()
        alive_cnt = torch.zeros((B, 1, K_cur), device=device, dtype=alive_int.dtype)
        alive_cnt.scatter_add_(2, rep_old.unsqueeze(1), alive_int.unsqueeze(1))
        alive_cols = (alive_cnt.squeeze(1) > 0)  # (B,K_cur) bool
        K_alive = alive_cols.sum(dim=1)  # (B,)

        # 若无可合并列（任意 batch 的 K_alive<2），直接返回
        if (K_alive < 2).any():
            x_agg = x_sum / cnt_all.clamp_min(1.0)
            size_merged = cnt_all  # (B,1,K_cur)
            mask_round = torch.zeros((B, H, W), device=device, dtype=torch.bool)
            return size_merged, mask_round, self.rep_map_global

        # ---------- 5) 构造每个 batch 的“活跃列顺序表” (无循环) ----------
        K_alive_max = int(K_alive.max().item())
        idx_all = torch.arange(K_cur, device=device).view(1, K_cur).expand(B, K_cur)
        # sort_key 把“活跃列”排前面
        sort_key = (~alive_cols).long() * (K_cur + 1) + idx_all
        order = sort_key.argsort(dim=1)  # (B,K_cur)
        # 直接截前 K_alive_max，再用位置掩码把超过各自 K_alive 的列置为 -1
        idx_alive = order[:, :K_alive_max]  # (B,K_alive_max)
        pos_rng = torch.arange(K_alive_max, device=device).unsqueeze(0)  # (1,K_alive_max)
        keep_alive = pos_rng < K_alive.unsqueeze(1)  # (B,K_alive_max)
        idx_alive = torch.where(keep_alive, idx_alive, torch.full_like(idx_alive, -1))

        # 收集活跃列特征（对 padding=-1 的位置置零）
        gather_idx = idx_alive.clamp_min(0).unsqueeze(1).expand(B, D, K_alive_max)  # (B,D,K_alive_max)
        x_alive = x_sum.gather(2, gather_idx)  # (B,D,K_alive_max)
        cnt_alive = cnt_all.gather(2, idx_alive.clamp_min(0).unsqueeze(1))  # (B,1,K_alive_max)
        pad_mask = (idx_alive < 0).unsqueeze(1)  # (B,1,K_alive_max)
        x_alive = x_alive.masked_fill(pad_mask.expand_as(x_alive), 0)
        cnt_alive = cnt_alive.masked_fill(pad_mask, 0)

        x_alive_mean = x_alive / cnt_alive.clamp_min(1.0)  # (B,D,K_alive_max)
        xk = x_alive_mean.transpose(1, 2).contiguous()  # (B,K_alive_max,D)

        # ---------- 6) 奇偶配对 + 相似度 ----------
        K_eff = K_alive_max
        m = K_eff // 2
        two_m = 2 * m
        has_tail = (K_eff % 2 == 1)

        src = xk[:, :two_m:2, :]  # (B,m,D)
        dst = xk[:, 1:two_m:2, :]  # (B,m,D)

        if self.distance == "cosine":
            a = src / (src.norm(dim=-1, keepdim=True).clamp_min(getattr(self, "eps", 1e-6)))
            b = dst / (dst.norm(dim=-1, keepdim=True).clamp_min(getattr(self, "eps", 1e-6)))
            scores = a @ b.transpose(-1, -2)  # (B,m,m)
        elif self.distance == "l1":
            scores = -torch.cdist(src, dst, p=1)  # (B,m,m)
        else:
            scores = -torch.cdist(src, dst, p=2)  # (B,m,m)

        node_max, node_idx = scores.max(dim=-1)  # (B,m), (B,m)

        # ---------- 7) 统一 r 的计算 ----------
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

        # 选 top-r 的 src（活跃容器内索引）
        edge_idx = node_max.argsort(dim=-1, descending=True).unsqueeze(-1)  # (B,m,1)
        src_idx_sel = edge_idx[:, :r, :]  # (B,r,1)
        src_idx_unm = edge_idx[:, r:, :]  # (B,m-r,1)
        dst_idx_sel = node_idx.gather(dim=-1, index=src_idx_sel.squeeze(-1)).unsqueeze(-1)  # (B,r,1)

        # ---------- 8) 活跃容器索引 -> 旧列号（无循环） ----------
        even_cols = torch.arange(0, two_m, 2, device=device).view(1, m, 1)  # (1,m,1)
        odd_cols = torch.arange(1, two_m, 2, device=device).view(1, m, 1)  # (1,m,1)

        even_old = idx_alive.gather(1, even_cols.expand(B, m, 1).squeeze(-1))  # (B,m)
        odd_old = idx_alive.gather(1, odd_cols.expand(B, m, 1).squeeze(-1))  # (B,m)

        src_cols_sel = even_old.gather(1, src_idx_sel.squeeze(-1)).unsqueeze(-1)  # (B,r,1)
        src_cols_unm = even_old.gather(1, src_idx_unm.squeeze(-1)).unsqueeze(-1)  # (B,m-r,1)
        dst_cols_all = odd_old.unsqueeze(-1)  # (B,m,1)
        dst_cols_sel = odd_old.gather(1, dst_idx_sel.squeeze(-1)).unsqueeze(-1)  # (B,r,1)

        # ---------- 9) 规模更新 ----------
        cnt_vec = cnt_all.squeeze(1)  # (B,K_cur)
        size_unm_src = cnt_vec.gather(1, src_cols_unm.squeeze(-1)).unsqueeze(-1)  # (B,m-r,1)
        size_dst = cnt_vec.gather(1, dst_cols_all.squeeze(-1)).unsqueeze(-1)  # (B,m,1)
        add_src = cnt_vec.gather(1, src_cols_sel.squeeze(-1)).unsqueeze(-1)  # (B,r,1)
        size_dst.scatter_add_(1, dst_idx_sel, add_src)  # (B,m,1)

        # ---------- 10) 组装保留列（未并入的 src + 所有 dst + 尾巴），并去 padding ----------
        pos_cat = torch.cat([src_cols_unm, dst_cols_all], dim=1)  # (B,(m-r)+m,1)
        size_cat = torch.cat([size_unm_src, size_dst], dim=1)  # (B,(m-r)+m,1)

        if has_tail:
            tail_idx = torch.tensor(two_m, device=device).view(1, 1, 1).expand(B, 1, 1)
            tail_old = idx_alive.gather(1, tail_idx.squeeze(-1)).unsqueeze(-1)  # (B,1,1)
            tail_siz = cnt_vec.gather(1, tail_old.squeeze(-1)).unsqueeze(-1)  # (B,1,1)
            pos_cat = torch.cat([pos_cat, tail_old], dim=1)  # (B,K_new~,1)
            size_cat = torch.cat([size_cat, tail_siz], dim=1)  # (B,K_new~,1)

        valid_mask = (pos_cat.squeeze(-1) >= 0)  # (B,K_new~)
        # 用排序把无效放后面，然后批量截取（无循环）
        sort_key2 = (~valid_mask).long() * (K_cur + 1) + pos_cat.squeeze(-1).clamp_min(0)
        order2 = sort_key2.argsort(dim=1)  # (B,K_new~)
        pos_keep_all = torch.take_along_dim(pos_cat.squeeze(-1), order2, dim=1)  # (B,K_new~)
        siz_keep_all = torch.take_along_dim(size_cat.squeeze(-1), order2, dim=1).unsqueeze(-1)  # (B,K_new~,1)

        valid_count = valid_mask.sum(dim=1)  # (B,)
        K_new_max = int(valid_count.max().item())
        pos_keep = pos_keep_all[:, :K_new_max]  # (B,K_new_max)
        siz_keep = siz_keep_all[:, :K_new_max, :]  # (B,K_new_max,1)

        keep_mask2 = (torch.arange(K_new_max, device=device).unsqueeze(0) < valid_count.unsqueeze(1))  # (B,K_new_max)
        pos_keep = torch.where(keep_mask2, pos_keep, torch.full_like(pos_keep, -1))
        siz_keep = torch.where(keep_mask2.unsqueeze(-1), siz_keep, torch.zeros_like(siz_keep))

        # ---------- 11) old_col -> new_col（批量重编号；用 scatter_reduce_ 去循环） ----------
        # rep_cols: (B,K_cur) 置为 -1
        rep_cols = torch.full((B, K_cur), -1, device=device, dtype=torch.long)
        # new 列号 = 在 pos_keep 的第二维位置（0..K_new[b]-1），对超过各自有效数的置 -1
        new_ids = torch.arange(K_new_max, device=device).unsqueeze(0).expand(B, -1)  # (B,K_new_max)
        new_ids = torch.where(keep_mask2, new_ids, torch.full_like(new_ids, -1))  # 无效置 -1
        # 将 (old_col -> new_col) 批量写入（取 amax 相当于忽略 -1，只保留有效 new_id）
        rep_cols.scatter_reduce_(1, pos_keep.clamp_min(0), new_ids, reduce="amax", include_self=True)  # (B,K_cur)

        # 把“被并掉的 src 旧列”映射到其对应 dst 的新列号
        dst_new_cols = rep_cols.gather(1, dst_cols_sel.squeeze(-1))  # (B,r)
        rep_cols.scatter_(1, src_cols_sel.squeeze(-1), dst_new_cols)  # old src -> dst_new

        # ---------- 12) 更新累计映射、构造本轮新增 mask ----------
        rep_map_old = self.rep_map_global  # (B,L)
        rep_map_new = rep_cols.gather(1, rep_map_old)  # (B,L)
        self.rep_map_global = rep_map_new

        src_cols_set = src_cols_sel.squeeze(-1)  # (B,r)
        new_mask_flat = (rep_map_old.unsqueeze(-1) == src_cols_set.unsqueeze(1)).any(dim=-1)  # (B,L)
        new_mask_flat = new_mask_flat & (~self.prev_mask.view(B, L))
        mask_round = new_mask_flat.view(B, H, W)

        # 叠加历史 mask
        self.prev_mask = self.prev_mask | mask_round

        # ---------- 13) 输出 ----------
        # 规模输出按 batch 的最大 K_new 对齐（pad=0），下游如需各自有效长度，用 valid_count 即可
        size_merged = siz_keep.transpose(1, 2).contiguous()  # (B,1,K_new_max)

        return size_merged, self.prev_mask, rep_map_new

    # -------- 依据 mask+rep_map 真正执行合并 & Cross-Scan 四方向输出 --------
    @torch.no_grad()
    def merge_and_cross_scan(
        self,
        x: torch.Tensor,          # (B,D,H,W)
        mask: torch.Tensor,       # (B,H,W) True=非代表（被合并）
        rep_map: torch.Tensor,    # (B,L)   原位置 -> 列索引（0..K-1）
        H: int,
        W: int,
        *,
        return_sizes: bool = True
    ):
        """
        返回:
          - xs4:   (B, 4, D, L_kept)   四个扫描方向的合并序列
          - sizes4:(B, 4, 1, L_kept)   对应列的聚合规模（若 return_sizes=True）
        说明：
          - 先按 rep_map 做 sum/mean 聚合得到 (B,D,K)
          - 再按四方向“首次出现”顺序为每个列（代表 token）排序，得到每方向列顺序
        """
        assert x.dim() == 4
        B, D, Hx, Wx = x.shape
        assert Hx == H and Wx == W
        L = H * W
        device = x.device
        dtypef = x.dtype

        # 计算 K（列数=L_kept）
        # 统一 r 的前提下，各 batch 的 K 一致；这里取全局 max+1
        K = int(rep_map.max().item()) + 1

        # === 1) 按 rep_map 对 (B,D,L) 做 sum/mean 聚合 ===
        x_seq = x.view(B, D, L)
        idx = rep_map.unsqueeze(1).expand(B, D, L)                   # (B,D,L) -> 列索引
        x_sum = torch.zeros((B, D, K), device=device, dtype=dtypef)
        x_sum.scatter_add_(2, idx, x_seq)                            # sum 聚合

        cnt = torch.zeros((B, 1, K), device=device, dtype=dtypef)
        ones = torch.ones((B, 1, L), device=device, dtype=dtypef)
        cnt.scatter_add_(2, rep_map.unsqueeze(1), ones)              # 每列规模

        if self.merge_mode == "mean":
            x_agg = x_sum / cnt.clamp_min(1.0)
        else:  # "sum"
            x_agg = x_sum

        # === 2) 四方向列顺序：按“首次出现位置”排序（稳定、无循环）===
        maps = _build_index_maps(H, W, device=device)                # 提供的函数
        xs_dir = []
        sizes_dir = []

        arangeL = torch.arange(L, device=device).view(1, L).expand(B, L)  # (B,L)

        for key in ('d0', 'd1', 'd2', 'd3'):
            idx_dir = maps[key].view(1, L).expand(B, L)                    # (B,L)
            rep_dir = rep_map.gather(1, idx_dir)                           # (B,L) 列号按该方向扫描顺序

            # 计算每个列号的“首次出现位置” t_first[c] = min{ p | rep_dir[p] == c }
            big = torch.full((B, K), L + 1, device=device, dtype=torch.long)      # 初始化为很大
            # 注意：scatter_reduce_ 的 reduce='amin' 需要 PyTorch 2.0+；若不支持，可改用 segment-min trick
            big = big.to(torch.long)
            pos_long = arangeL.to(torch.long)
            big.scatter_reduce_(1, rep_dir, pos_long, reduce='amin')              # (B,K) 每列的首出现位置

            order = big.argsort(dim=1)                                            # (B,K) 小者在前 => 列顺序
            # 依据列顺序 gather
            xs_ordered   = x_agg.gather(2, order.unsqueeze(1).expand(B, D, K))    # (B,D,K)
            xs_dir.append(xs_ordered)

            if return_sizes:
                sizes_ordered = cnt.gather(2, order.unsqueeze(1))                 # (B,1,K)
                sizes_dir.append(sizes_ordered)

        xs4 = torch.stack(xs_dir, dim=1)              # (B,4,D,K)
        if return_sizes:
            sizes4 = torch.stack(sizes_dir, dim=1)    # (B,4,1,K)
            return xs4, sizes4
        return xs4

    @torch.no_grad()
    def restore_2d_from_cross_scans(
            self,
            xs4: torch.Tensor,  # (B, 4, D, K) —— 四方向序列（方向内部是各自的列顺序）
            rep_map: torch.Tensor,  # (B, L)       —— 原位置 -> “规范列号”(0..K-1)
            H: int,
            W: int,
            *,
            reduce: str = "mean",  # 'mean' | 'sum' | 'max' 四方向聚合方式
    ) -> torch.Tensor:
        """
        将四方向序列 (B,4,D,K) 合并还原到 2D 特征 (B,D,H,W)。
        要点：
          - 方向间仅列顺序不同，先用首出现位置求出每个方向的列顺序 order；
          - 用逆置换把每个方向的 (B,D,K) 对齐到“规范列号”；
          - 对四方向在对齐后聚合，再依据 rep_map 回填到 2D。
        """
        assert xs4.dim() == 4 and xs4.shape[1] == 4, "xs4 must be (B,4,D,K)"
        assert rep_map.dim() == 2
        B, _, D, K = xs4.shape
        L = H * W
        device = xs4.device

        # 1) 构建四方向展平索引
        maps = _build_index_maps(H, W, device=device)  # {'d0','d1','d2','d3'}
        keys = ('d0', 'd1', 'd2', 'd3')

        # 2) 为每个方向计算列顺序 order，并把 xs4 对齐到“规范列号顺序”
        arangeL = torch.arange(L, device=device).view(1, L).expand(B, L)  # (B,L)
        arangeK = torch.arange(K, device=device).view(1, K).expand(B, K)  # (B,K)

        xs_canon_list = []  # 每个方向对齐到规范列号后的 (B,D,K)

        for di, key in enumerate(keys):
            idx_dir = maps[key].view(1, L).expand(B, L)  # (B,L)
            rep_dir = rep_map.gather(1, idx_dir)  # (B,L)，列号按该方向的扫描顺序排列

            # 每个列号的“首出现位置”（用 amin / argsort 拿到方向内的列顺序）
            # 初始化为很大，再做 amin 归约
            first_pos = torch.full((B, K), L + 1, device=device, dtype=torch.long)
            first_pos.scatter_reduce_(1, rep_dir, arangeL, reduce='amin')  # (B,K)

            order = first_pos.argsort(dim=1)  # (B,K) 方向内列顺序（位置->规范列号）
            # 逆置换：inv_order[b, canon_col] = 在该方向序列中的位置
            inv_order = torch.empty_like(order)
            inv_order.scatter_(1, order, arangeK)  # (B,K)

            # 将该方向的 (B,D,K) 还原到规范列顺序
            x_dir = xs4[:, di, :, :]  # (B,D,K)
            x_dir_canon = x_dir.gather(2, inv_order.unsqueeze(1).expand(B, D, K))  # (B,D,K)
            xs_canon_list.append(x_dir_canon)

        xs_canon = torch.stack(xs_canon_list, dim=1)  # (B,4,D,K)，四方向已在列上对齐

        # 3) 四方向聚合
        if reduce == "mean":
            x_merged = xs_canon.mean(dim=1)  # (B,D,K)
        elif reduce == "sum":
            x_merged = xs_canon.sum(dim=1)  # (B,D,K)
        elif reduce == "max":
            x_merged = xs_canon.max(dim=1).values  # (B,D,K)
        else:
            raise ValueError(f"Unknown reduce='{reduce}'")

        # 4) 用 rep_map 回填成 (B,D,L) 并 reshape 到 (B,D,H,W)
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
        size_merged, mask, rep_map = pruner.merge_tokens_unified(x, num_prune=0)
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
            size_merged, mask, rep_map = pruner.merge_tokens_unified(x, num_prune=0)
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

