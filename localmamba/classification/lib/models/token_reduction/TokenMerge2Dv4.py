import math
import torch
from torch import nn, Tensor

from typing import Tuple, Any
import torch.nn.functional as F

from .merge import RMeeTo_Merge


def pad_zeros(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 3, "x must be [B, D, L]"
    B, D, L = x.shape
    H = math.isqrt(L)
    L_orig = H * H if H * H == L else (H + 1) * (H + 1)
    assert L <= L_orig, "L must not exceed L_orig"
    # pad 在最后一个维度补 (0, L_orig - L)
    return F.pad(x, (0, L_orig - L))

def window_flatten_safe(core2d):
    """
    将 [B, C, H, W] 划分为 2x2 窗口展开；
    若 H/W 为奇数，则裁切主区域进行窗口化，并拼接剩余边界像素。
    返回 [B, C, H*W]
    """
    B, C, H, W = core2d.shape

    # 主体窗口区域大小（确保偶数）
    h2, w2 = H // 2, W // 2
    main_H, main_W = h2 * 2, w2 * 2

    # 主体部分
    core_main = core2d[:, :, :main_H, :main_W]
    y_main = (
        core_main.reshape(B, C, 2, h2, 2, w2)
        .permute(0, 1, 2, 4, 3, 5)
        .reshape(B, C, 4, -1)
        .reshape(B, C, -1)
    )

    # 剩余部分：右边列 + 下边行 + 右下角单点
    remain_blocks = []

    # 下边行（如果 H 是奇数）
    if H % 2 == 1:
        remain_blocks.append(core2d[:, :, H-1:H, :main_W].reshape(B, C, -1))
    # 右边列（如果 W 是奇数）
    if W % 2 == 1:
        remain_blocks.append(core2d[:, :, :main_H, W-1:W].reshape(B, C, -1))
    # 右下角单点（如果 H,W 都是奇数）
    if H % 2 == 1 and W % 2 == 1:
        remain_blocks.append(core2d[:, :, H-1:H, W-1:W].reshape(B, C, -1))

    # 拼接
    if remain_blocks:
        y_full = torch.cat([y_main] + remain_blocks, dim=2)
    else:
        y_full = y_main

    return y_full


def window_unflatten_safe(tokens, H, W):
    """
    tokens: [B, C, H*W] 来自 window_flatten_safe 的输出
    返回: [B, C, H, W]
    """
    B, C, L = tokens.shape
    assert L == H * W, f"Token数不匹配: got {L}, expected {H*W}"

    h2, w2 = H // 2, W // 2
    main_H, main_W = h2 * 2, w2 * 2
    n_main = main_H * main_W

    # --- 主体部分 ---
    main_tokens = tokens[:, :, :n_main]
    core_main = (
        main_tokens
        .reshape(B, C, 4, h2, w2)
        .reshape(B, C, 2, 2, h2, w2)
        .permute(0, 1, 2, 4, 3, 5)
        .reshape(B, C, main_H, main_W)
    )

    # --- 剩余部分 ---
    offset = n_main
    out = torch.zeros(B, C, H, W, device=tokens.device, dtype=tokens.dtype)
    out[:, :, :main_H, :main_W] = core_main

    # 下边行
    if H % 2 == 1:
        n = W - (W % 2)  # 对齐 main_W
        out[:, :, H-1:H, :n] = tokens[:, :, offset:offset + n].reshape(B, C, 1, n)
        offset += n

    # 右边列
    if W % 2 == 1:
        n = H - (H % 2)
        out[:, :, :n, W-1:W] = tokens[:, :, offset:offset + n].reshape(B, C, n, 1)
        offset += n

    # 右下角
    if H % 2 == 1 and W % 2 == 1:
        out[:, :, H-1, W-1] = tokens[:, :, offset].reshape(B, C)
        offset += 1

    return out.flatten(2)

class TokenMerge2Dv4(nn.Module):
    def __init__(self,
                 num_prune: int = 5,
                 if_prune: bool = False,
                 if_order: bool = True,
                 distance: str = 'cosine',
                 merge_mode: str = 'sum',
                 choose: str = 'max'):
        super().__init__()
        # 这里默认按“有CLS”的模式跑（我们会临时加一个假的 CLS）
        self.merge = RMeeTo_Merge(
            class_token=True,          # 关键：告诉 RMeeTo_Merge 序列里有一个“受保护”的 token
            num_prune=num_prune,
            if_prune=if_prune,
            if_order=True,
            distance=distance,
            metric='X',
            merge_mode=merge_mode,
            choose=choose
        )

    def change_num_prune(self,num_prune:int):
        self.merge.change_num_prune(num_prune)

    @torch.no_grad()
    def forward(self, x: torch.Tensor,residual:torch.Tensor,L_remain,num_prune:int) -> tuple[Tensor, None] | tuple[Any, Any]:
        """
        x: [B, D, L]  ->  return: [B, D, L_kept]
        """
        assert x.dim() == 3, "x must be [B, D, L]"
        B_orig,D_orig,L_orig = x.shape
        if L_remain is not None:
            x = x[:, :, :L_remain]
        if num_prune == 0:
            return x,residual
        B, D, L = x.shape
        device, dtype = x.device, x.dtype

        # --- 1) 变换到 [B, T, C] 以适配 RMeeTo_Merge 的接口 ---
        seq = x.permute(0, 2, 1).contiguous()  # [B, L, D]  (T=L, C=D)
        residual = residual.permute(0, 2, 1).contiguous()

        # --- 2) 临时构造一个“CLS”放在最前面，便于使用 RMeeTo_Merge 的保护/移除逻辑 ---
        # 也可以用 zeros 或者学到的 cls 向量；这里用序列均值更稳妥
        cls_token = seq.mean(dim=1, keepdim=True)            # [B, 1, D]
        seq_with_cls = torch.cat([cls_token, seq], dim=1)    # [B, L+1, D]
        residual_with_cls = torch.cat([cls_token, residual], dim=1)

        # metric 与特征一致即可（内部会做 normalize）
        metric = seq_with_cls
        B, L, C = metric.shape
        size = torch.ones(B, L, 1, device=metric.device)

        # --- 3) 生成 merge 函数（保护位置 0，即我们加的临时 CLS） ---
        merge_fn = self.merge(metric, token_position=0,num_prune=num_prune)

        # --- 4) 走带权合并（权重 size 若未知，内部用全1），返回合并后的序列和 CLS 的新位置 ---
        merged, sizes, cls_new_pos = self.merge.merge_wavg(merge_fn, seq_with_cls, size=size)
        merged_residual, _, _ = self.merge.merge_wavg(merge_fn,residual_with_cls,size=size)
        # merged: [B, T_kept, D]，cls_new_pos: int（每个 batch 相同的标量位置）

        # --- 5) 去掉临时 CLS ---
        T_kept = merged.shape[1]
        idx = torch.arange(T_kept, device=device)
        keep_mask = (idx != cls_new_pos)                     # [T_kept]
        merged_wo_cls = merged[:, keep_mask, :]              # [B, T_kept-1, D]
        residual_wo_cls = merged_residual[:, keep_mask, :]

        # --- 6) 转回 [B, D, L_kept] ---
        out_x = merged_wo_cls.permute(0, 2, 1).contiguous()    # [B, D, L_kept]
        out_residual = residual_wo_cls.permute(0, 2, 1).contiguous()
        return out_x, out_residual

    # TODO:完成函数
    # 输入[B,D,H,W] -> [B,D,L],并删去L序列中为0的尾部部分
    # 执行token merging(x: [B,D,L]) -> [B,D,L_kept] (L_kept < L)
    # 补充为[B,D,L],后续空位补充0 -> [B,D,H,W]
    # 高效cross scan -> [B,4,D,L]
    # 批量删除序列中为0的token -> [B,4,D,L_kept2],L_kept2可能小于L_kept(极小概率)

    @torch.no_grad()
    def cross_scan_fwd(self, x: torch.Tensor) -> torch.Tensor:
        """
        近似快速 cross-scan：
          - 输入:  x [B, C, L_kept]
          - 输出:  y [B, 4, C, L_kept]
          - 动态取 H = floor(L_kept / W)，只扫描 H*W 的部分，多余部分附加。
        """
        assert x.dim() == 3, "x must be [B, C, L_kept]"
        B, C, L_kept = x.shape
        device, dtype = x.device, x.dtype
        H = int(L_kept ** 0.5)
        W = H
        baseL = H * W  # 实际能扫描的长度
        is_even = (H % 2 == 0) and (W % 2 == 0)

        core = x[:, :, :baseL]  # [B,C,baseL]
        core2d = core.view(B, C, H, W)

        y0 = core2d.flatten(2, 3)  # 行优先
        y1 = core2d.transpose(2, 3).flatten(2, 3)  # 列优先

        if is_even:
            # 若为偶数则执行
            y2 = window_flatten_safe(core2d).flip(dims=[-1])
            y3 = window_flatten_safe(core2d.transpose(2, 3)).flip(dims=[-1])
        else:
            # 若为奇数则执行
            # 修改扫描方向
            y2 = torch.flip(y0, dims=[-1])  # 行反向
            y3 = torch.flip(y1, dims=[-1])  # 列反向

        y_core = torch.stack([y0, y1, y2, y3], dim=1)  # [B,4,C,baseL]

        if L_kept > baseL:
            extra = x[:, :, baseL:L_kept]  # [B,C,L_kept-baseL]
            extra4 = extra.unsqueeze(1).expand(B, 4, C, extra.shape[-1])
            y = torch.cat([y_core, extra4], dim=-1)
        else:
            y = y_core

        return y[:, :, :, :L_kept].contiguous()  # [B,4,C,L_kept]

    @torch.no_grad()
    def cross_scan_bwd(self, y: torch.Tensor) -> torch.Tensor:
        """
        反向融合 cross-scan:
          - 输入: y [B, 4, C, L_kept]（来自 cross_scan_fwd 的输出）
          - 输出: x [B, C, L_kept]
          - 动态 H = L_kept // W，尾部 token 单独拼接
        """
        assert y.dim() == 4, "y must be [B, 4, C, L_kept]"
        B, K, D, L_kept = y.shape
        assert K == 4, "必须是四方向 cross-scan 的结果"

        # 1) 动态 H, baseL, extraL
        H = int(L_kept ** 0.5)
        W = H
        baseL = H * W
        extraL = L_kept - baseL
        is_even = (H % 2 == 0) and (W % 2 == 0)

        # 2) 去掉尾部 extra 部分（保留 core）
        y_core = y[:, :, :, :baseL]  # [B,4,D,baseL]

        # 3) 合并对称方向 (0+2, 1+3)
        #   注意：fwd 里 y[:,2:4] 是 flip 出来的，所以这里要再 flip 回来
        if is_even:
            y0_out = y_core[:,0]
            y1_out = y_core[:,1].view(B, D, W, H).transpose(2, 3).contiguous().view(B, D, baseL)
            y2_out = window_unflatten_safe(y_core[:,2],H,W).flip(dims=[-1])
            y3_out = window_unflatten_safe(y_core[:,3],H,W).view(B, D, W, H).transpose(2, 3).contiguous().view(B, D, baseL).flip(dims=[-1])
            x_core = y0_out+y1_out+y2_out+y3_out
        else:
            y_merge = y_core[:, 0:2] + y_core[:, 2:4].flip(dims=[-1])  # [B,2,D,baseL]
            # 4) 转换为 [B,D,H,W] 结构
            y0 = y_merge[:, 0]  # 行方向
            y1 = y_merge[:, 1]  # 列方向
            x_core = (y0 + y1.view(B, D, W, H).transpose(2, 3).contiguous().view(B, D, baseL))

        if extraL > 0:
            extra = y[:, :, :, baseL:L_kept].sum(dim=1)  # [B,D,extraL]
            x = torch.cat([x_core, extra], dim=-1)
        else:
            x = x_core

        return x

if __name__ == "__main__":
    B, D, L = 32, 384, 196
    x = torch.randn(B, D, L).cuda()

    merger = TokenMerge2Dv4(
        num_prune=20,
        if_prune=False,
        if_order=True,
        distance='cosine',
        merge_mode='sum',  # 权重聚合方式
        choose='max'
    )

    for i in range(8):
        x = merger(x,196-20*i)

        xs = merger.cross_scan_fwd(x)

        x_out = merger.cross_scan_bwd(xs)

        # 调整为自动pad到方形，如12*12
        x = merger.pad_zeros(x_out)
        print(x.view(B, D, int(x.shape[-1] ** 0.5),int(x.shape[-1] ** 0.5)).shape)
        print(x.shape)
        print(xs.shape)
        print(x_out.shape)
        print()


