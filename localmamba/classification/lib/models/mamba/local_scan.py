
import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def triton_local_scan(
    x, # x point (B, C, H, W) or (B, C, L)
    y, # y point (B, C, H, W) or (B, C, L)
    K: tl.constexpr,  # window size
    flip: tl.constexpr, # whether to flip the tokens
    BC: tl.constexpr,  # number of channels in each program
    BH: tl.constexpr,  # number of heights in each program
    BW: tl.constexpr,  # number of width in each program
    DC: tl.constexpr,  # original channels
    DH: tl.constexpr,  # original height
    DW: tl.constexpr,  # original width
    NH: tl.constexpr,  # number of programs on height
    NW: tl.constexpr,  # number of programs on width
):
    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)  # program id of hw axis, c axis, batch axis
    i_h, i_w = (i_hw // NW), (i_hw % NW)  # program idx of h and w
    _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
    _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]  # [BH, BW]
    _for_C = min(DC - i_c * BC, BC)  # valid number of c in the program

    _tmp0 = i_c * BC * DH * DW  # start offset of this program
    _tmp1 = DC * DH * DW  # n_elements in one batch
    _tmp2 = _tmp0 + i_h * BH * DW  + tl.arange(0, BH)[:, None] * DW + i_w * BW + tl.arange(0, BW)[None, :]  # offsets of elements in this program
    
    p_x = x + i_b * _tmp1 + _tmp2

    _i = (tl.arange(0, BH) + BH * i_h)[:, None]
    _j = (tl.arange(0, BW) + BW * i_w)[None, :]
    _c_offset = ((DW // K) * (_i // K) + (_j // K)) * K * K + (_i % K) * K + _j % K
    if flip:
        _c_offset = DH * DW - _c_offset - 1

    p_y = y + i_b * _tmp1 + _tmp0 + _c_offset
    for idxc in range(_for_C):
        _idx = idxc * DH * DW
        _x = tl.load(p_x + _idx, mask=_mask_hw)
        tl.store(p_y + _idx, _x, mask=_mask_hw)
    tl.debug_barrier()


@triton.jit
def triton_local_reverse(
    x, # x point (B, C, H, W) or (B, C, L)
    y, # y point (B, C, H, W) or (B, C, L)
    K: tl.constexpr,  # window size
    flip: tl.constexpr,  # whether to flip the tokens
    BC: tl.constexpr,  # number of channels in each program
    BH: tl.constexpr,  # number of heights in each program
    BW: tl.constexpr,  # number of width in each program
    DC: tl.constexpr,  # original channels
    DH: tl.constexpr,  # original height
    DW: tl.constexpr,  # original width
    NH: tl.constexpr,  # number of programs on height
    NW: tl.constexpr,  # number of programs on width
):
    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)  # program id of hw axis, c axis, batch axis
    i_h, i_w = (i_hw // NW), (i_hw % NW)  # program idx of h and w
    _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
    _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]  # [BH, BW]
    _for_C = min(DC - i_c * BC, BC)  # valid number of c in the program

    _tmp0 = i_c * BC * DH * DW  # start offset of this program
    _tmp1 = DC * DH * DW  # n_elements in one batch
    _tmp2 = _tmp0 + i_h * BH * DW  + tl.arange(0, BH)[:, None] * DW + i_w * BW + tl.arange(0, BW)[None, :]  # offsets of elements in this program
    
    p_x = x + i_b * _tmp1 + _tmp2

    _i = (tl.arange(0, BH) + BH * i_h)[:, None]
    _j = (tl.arange(0, BW) + BW * i_w)[None, :]
    _o = _i * DW + _j

    _i = _o // (K * K) // (DW // K) * K + _o % (K * K) // K
    _j = _o // (K * K) % (DW // K) * K + _o % (K * K) % K
    _c_offset = _i * DW + _j
    if flip:
        _c_offset = DH * DW - _c_offset - 1

    p_y = y + i_b * _tmp1 + _tmp0 + _c_offset
    for idxc in range(_for_C):
        _idx = idxc * DH * DW
        _x = tl.load(p_x + _idx, mask=_mask_hw)
        tl.store(p_y + _idx, _x, mask=_mask_hw)
    tl.debug_barrier()


class LocalScanTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, K: int, flip: bool, H: int = None, W: int = None):
        ori_x = x
        B, C = x.shape[:2]
        if H is None or W is None:
            if len(x.shape) == 4:
                H, W = x.shape[-2:]
            elif len(x.shape) == 3:
                raise RuntimeError("x must be BCHW format to infer the H W")
        B, C, H, W = int(B), int(C), int(H), int(W)

        ctx.ori_shape = (B, C, H, W)
        # pad tensor to make it evenly divisble by window size
        x, (H, W) = pad_tensor(x, K, H, W)
        ctx.shape = (B, C, H, W)

        BC, BH, BW = min(triton.next_power_of_2(C), 1), min(triton.next_power_of_2(H), 64), min(triton.next_power_of_2(W), 64)
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
        ctx.K = K
        ctx.flip = flip

        if x.stride(-1) != 1:
            x = x.contiguous()

        if len(ori_x.shape) == 4:
            y = x.new_empty((B, C, H, W))
        elif len(ori_x.shape) == 3:
            y = x.new_empty((B, C, H * W))

        triton_local_scan[(NH * NW, NC, B)](x, y, K, flip, BC, BH, BW, C, H, W, NH, NW)
        return y
    
    @staticmethod
    def backward(ctx, y: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape

        if y.stride(-1) != 1:
            y = y.contiguous()
        if len(y.shape) == 4 or ctx.shape != ctx.ori_shape:
            x = y.new_empty((B, C, H, W))
        else:
            x = y.new_empty((B, C, H * W))

        triton_local_reverse[(NH * NW, NC, B)](y, x, ctx.K, ctx.flip, BC, BH, BW, C, H, W, NH, NW)

        if ctx.shape != ctx.ori_shape:
            _, _, ori_H, ori_W = ctx.ori_shape
            x = x[:, :, :ori_H, :ori_W]
            if len(y.shape) == 3:
                x = x.flatten(2)

        return x, None, None, None, None


class LocalReverseTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, K: int, flip: bool, H: int = None, W: int = None):
        B, C = x.shape[:2]
        if H is None or W is None:
            if len(x.shape) == 4:
                H, W = x.shape[-2:]
            elif len(x.shape) == 3:
                raise RuntimeError("x must be BCHW format to infer the H W")
        B, C, H, W = int(B), int(C), int(H), int(W)
        
        ctx.ori_shape = (B, C, H, W)
        # x may have been padded
        Hg, Wg = math.ceil(H / K), math.ceil(W / K)
        H, W = Hg * K, Wg * K
        ctx.shape = (B, C, H, W)

        BC, BH, BW = min(triton.next_power_of_2(C), 1), min(triton.next_power_of_2(H), 64), min(triton.next_power_of_2(W), 64)
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
        ctx.K = K
        ctx.flip = flip

        if x.stride(-1) != 1:
            x = x.contiguous()
        
        if len(x.shape) == 4 or ctx.ori_shape != ctx.shape:
            y = x.new_empty((B, C, H, W))
        else:
            y = x.new_empty((B, C, H * W))

        triton_local_reverse[(NH * NW, NC, B)](x, y, K, flip, BC, BH, BW, C, H, W, NH, NW)

        if ctx.ori_shape != ctx.shape:
            ori_H, ori_W = ctx.ori_shape[-2:]
            y = y[:, :, :ori_H, :ori_W]
            if len(x.shape) == 3:
                y = y.flatten(2)

        return y
    
    @staticmethod
    def backward(ctx, y: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.ori_shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape

        _is_y_BCHW = len(y.shape) == 4

        y, (H, W) = pad_tensor(y, ctx.K, H, W)

        if y.stride(-1) != 1:
            y = y.contiguous()

        if _is_y_BCHW:
            x = y.new_empty((B, C, H, W))
        else:
            x = y.new_empty((B, C, H * W))

        triton_local_scan[(NH * NW, NC, B)](y, x, ctx.K, ctx.flip, BC, BH, BW, C, H, W, NH, NW)

        return x, None, None, None, None



def pad_tensor(x, w, H, W):
    if H % w == 0 and W % w == 0:
        return x, (H, W)
    B, C = x.shape[:2]
    if len(x.shape) == 3:
        x = x.view(B, C, H, W)
    
    Hg, Wg = math.ceil(H / w), math.ceil(W / w)
    newH, newW = Hg * w, Wg * w
    x = F.pad(x, (0, newW - W, 0, newH - H))
    
    # We can skip flattening x back to BCL as the next operation
    # is triton_local_reverse / triton_local_scan, which supports
    # both BCHW and BCL inputs
    # if len(ori_x.shape) == 3:
    #     x = x.flatten(2)

    return x, (newH, newW)


"""PyTorch code for local scan and local reverse"""


import math
import torch
import torch.nn.functional as F

"""PyTorch code for local scan and local reverse (no extra tokens for padding)"""


def _build_scan_mask(H, W, w, flip=False, column_first=False, device=None):
    """
    构造一个 bool mask，表示在 window 扫描后的序列中，哪些位置是“真实 token”，哪些是 padding。
    mask 形状：[L_pad]，其中 L_pad = Hg * Wg * w * w
    True 表示对应的是原 HxW 里的有效 token。
    """
    Hg, Wg = math.ceil(H / w), math.ceil(W / w)
    newH, newW = Hg * w, Wg * w
    L_pad = newH * newW

    # 构造一个 [Hg, w, Wg, w] 的 index grid，表示每个位置在 padded HxW 网格里的线性 index
    idx = torch.arange(L_pad, device=device).view(Hg, w, Wg, w)

    # 按照 local_scan 中的 view + permute 顺序来展平
    if column_first:
        # 对应 local_scan / local_scan_bchw 中 column_first=True 的 permute
        # local_scan:  view(B, Hg, w, Wg, w, C).permute(0,5,3,1,4,2)
        # bchw 版本:   view(B, C, Hg, w, Wg, w).permute(0,1,4,2,5,3)
        # 去掉 batch / channel 后空间维度顺序等价于 idx.permute(2, 0, 1, 3)
        idx_seq = idx.permute(2, 0, 1, 3).reshape(-1)  # [L_pad]
    else:
        # 对应 local_scan / local_scan_bchw 中 column_first=False 的 permute
        # local_scan:  view(B, Hg, w, Wg, w, C).permute(0,5,1,3,2,4)
        # bchw 版本:   view(B, C, Hg, w, Wg, w).permute(0,1,2,4,3,5)
        # 空间维度顺序等价于 idx.permute(0, 2, 1, 3)
        idx_seq = idx.permute(0, 2, 1, 3).reshape(-1)  # [L_pad]

    # 如果有 flip，序列也会被反转，mask 也要同步反转
    if flip:
        idx_seq = idx_seq.flip(-1)

    # 原始 HxW 里的有效位置 index 在 [0, H*W)
    mask = idx_seq < (H * W)  # [L_pad] bool
    return mask  # True 位置对应有效 token


def local_scan(x, w=7, H=14, W=14, flip=False, column_first=False):
    """Local windowed scan in LocalMamba
    Input:
        x: [B, L, C], 其中 L = H * W
        H, W: original height and width
        column_first: column-wise scan first (the additional direction in VMamba)
    Return: [B, C, L]，L 始终为 H * W（不包含 padding token）
    """
    B, L, C = x.shape
    assert L == H * W, f"Expected L == H*W, got L={L}, H*W={H*W}"
    x = x.view(B, H, W, C)

    Hg, Wg = math.ceil(H / w), math.ceil(W / w)
    newH, newW = Hg * w, Wg * w

    # 如有需要，先 pad 到整 window
    if H % w != 0 or W % w != 0:
        x = F.pad(x, (0, 0, 0, newW - W, 0, newH - H))  # [B, newH, newW, C]

    # 按原始实现进行 window 重排
    if column_first:
        x = x.view(B, Hg, w, Wg, w, C).permute(0, 5, 3, 1, 4, 2).reshape(B, C, -1)
    else:
        x = x.view(B, Hg, w, Wg, w, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, -1)

    # flip 序列方向（多方向扫描用）
    if flip:
        x = x.flip([-1])

    # 如果没有 padding，直接返回
    if H % w == 0 and W % w == 0:
        return x  # [B, C, H*W]

    # 构造 mask，裁掉 padding 对应的位置，保证长度 = H * W
    mask = _build_scan_mask(H, W, w, flip=flip, column_first=column_first, device=x.device)  # [L_pad]
    x = x[:, :, mask]  # [B, C, H*W]
    return x


def local_scan_bchw(x, w=7, H=14, W=14, flip=False, column_first=False):
    """Local windowed scan in LocalMamba (BCHW version)
    Input:
        x: [B, C, H, W]
        H, W: original height and width
        column_first: column-wise scan first (the additional direction in VMamba)
    Return: [B, C, L]，L 始终为 H * W（不包含 padding token）
    """
    B, C, H_in, W_in = x.shape
    assert H_in == H and W_in == W, "H/W args must match x.shape"

    Hg, Wg = math.ceil(H / w), math.ceil(W / w)
    newH, newW = Hg * w, Wg * w

    # 如有需要，先 pad 到整 window
    if H % w != 0 or W % w != 0:
        x = F.pad(x, (0, newW - W, 0, newH - H))  # [B, C, newH, newW]

    # 按原始实现进行 window 重排
    if column_first:
        x = x.view(B, C, Hg, w, Wg, w).permute(0, 1, 4, 2, 5, 3).reshape(B, C, -1)
    else:
        x = x.view(B, C, Hg, w, Wg, w).permute(0, 1, 2, 4, 3, 5).reshape(B, C, -1)

    if flip:
        x = x.flip([-1])

    if H % w == 0 and W % w == 0:
        return x  # [B, C, H*W]

    # 裁掉 padding 位置
    mask = _build_scan_mask(H, W, w, flip=flip, column_first=column_first, device=x.device)
    x = x[:, :, mask]  # [B, C, H*W]
    return x


def local_reverse(x, w=7, H=14, W=14, flip=False, column_first=False):
    """Local windowed reverse scan in LocalMamba
    Input:
        x: [B, C, L]，这里的 L 必须等于 H * W（不包含 padding token）
        H, W: original height and width
        column_first: column-wise scan first (the additional direction in VMamba)
    Return: [B, C, L]，L = H * W
    """
    B, C, L = x.shape
    assert L == H * W, f"Expected L == H*W, got L={L}, H*W={H*W}"

    Hg, Wg = math.ceil(H / w), math.ceil(W / w)
    newH, newW = Hg * w, Wg * w
    L_pad = newH * newW

    # 无 padding 的简单情况：直接走原逻辑
    if H % w == 0 and W % w == 0:
        if flip:
            x = x.flip([-1])
        if column_first:
            x = x.view(B, C, Wg, Hg, w, w).permute(0, 1, 3, 5, 2, 4).reshape(B, C, L)
        else:
            x = x.view(B, C, Hg, Wg, w, w).permute(0, 1, 2, 4, 3, 5).reshape(B, C, L)
        return x  # [B, C, H*W]

    # 有 padding 的情况：
    # 1. 先把长度为 H*W 的有效序列塞回到长度为 L_pad 的 padded 序列位置上
    mask = _build_scan_mask(H, W, w, flip=flip, column_first=column_first, device=x.device)  # [L_pad]
    assert mask.sum().item() == H * W, "mask 中 True 的数量必须等于 H*W"

    x_full = x.new_zeros(B, C, L_pad)  # [B, C, L_pad]
    idx_valid = mask.nonzero(as_tuple=False).squeeze(-1)  # [H*W]
    x_full[:, :, idx_valid] = x  # 把有效 token 按原 scan 顺序放回

    # 2. 现在 x_full 相当于 local_scan 的完整输出（包含 padding 的），
    #    下面完全照原 local_reverse 的逻辑反变换 + 裁剪
    if flip:
        x_full = x_full.flip([-1])

    if column_first:
        x_full = x_full.view(B, C, Wg, Hg, w, w).permute(0, 1, 3, 5, 2, 4).reshape(B, C, newH, newW)
    else:
        x_full = x_full.view(B, C, Hg, Wg, w, w).permute(0, 1, 2, 4, 3, 5).reshape(B, C, newH, newW)

    x_full = x_full[:, :, :H, :W].reshape(B, C, -1)  # 裁掉 padding 空间，只留原始 HxW
    return x_full  # [B, C, H*W]
