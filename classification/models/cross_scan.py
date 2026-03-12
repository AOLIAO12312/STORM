import torch
from torch import nn

def even_odd_scan(x: torch.Tensor):
    """
    将输入的2D token展开为1维序列，并按奇偶位置分组拼接。
    Args:
        x: Tensor [B, C, H, W]
    Returns:
        x_seq: Tensor [B, C, H*W] (先偶后奇)
    """
    assert x.dim() == 4, f"x must be [B, C, H, W], got {x.shape}"
    B, C, H, W = x.shape
    L = H * W

    x_flat = x.flatten(2, 3)

    # 奇偶分组
    even_idx = torch.arange(0, L, 2, device=x.device)
    odd_idx = torch.arange(1, L, 2, device=x.device)
    perm = torch.cat([even_idx, odd_idx], dim=0)

    x_seq = torch.gather(x_flat, 2, perm.unsqueeze(0).unsqueeze(0).expand(B, C, L))
    return x_seq

def even_odd_unscan(x_seq: torch.Tensor):
    """
    根据序列长度反推奇偶位置，还原到 [B, C, H, W]
    Args:
        x_seq: Tensor [B, C, L]
        H, W:  原图尺寸
    Returns:
        x_rec: Tensor [B, C, H, W]
    """
    assert x_seq.dim() == 3, f"x_seq must be [B, C, L], got {x_seq.shape}"
    B, C, L = x_seq.shape

    # 前半段是偶数位，后半段是奇数位
    half = (L + 1) // 2  # ceil(L/2)

    x_rec_flat = torch.empty_like(x_seq)
    x_rec_flat[:, :, 0::2] = x_seq[:, :, :half]   # 偶数位置
    x_rec_flat[:, :, 1::2] = x_seq[:, :, half:]   # 奇数位置

    return x_rec_flat

# 全局缓存字典
_random_perm_cache = {}

def stable_random_perm(L, seed=42, device="cpu"):
    """
    生成稳定随机序列，并缓存
    Args:
        L (int): 序列长度
        seed (int): 随机种子
        device (str or torch.device): 生成设备
    Returns:
        perm: torch.LongTensor [L] 稳定随机序列
    """
    global _random_perm_cache
    key = (L, seed, device)  # 缓存键

    # 检查缓存
    if key in _random_perm_cache:
        return _random_perm_cache[key]

    # 未缓存，生成随机序列
    idx = torch.arange(L, device=device)
    rand_val = torch.sin(idx.float() * 12.9898 + seed * 78.233) * 43758.5453
    rand_val = rand_val - rand_val.floor()
    perm = rand_val.argsort()

    # 保存到缓存
    _random_perm_cache[key] = perm

    return perm

def random_scan(x: torch.Tensor, seed: int = 42):
    """
    对输入张量执行随机扫描（随机排序）
    Args:
        x: Tensor [B, C, H, W]
        seed: 可选，随机种子（固定结果时使用）
    Returns:
        x_rand: Tensor [B, C, H*W] 随机打乱后的序列
        perm:   Tensor [H*W]       随机序列索引（可用于反向还原）
    """
    assert x.dim() == 4, f"x must be [B, C, H, W], got {x.shape}"
    B, C, H, W = x.shape
    L = H * W

    # 设定随机种子（可复现性）
    if seed is not None:
        torch.manual_seed(seed)

    # 生成随机索引序列
    perm = stable_random_perm(L,seed,device=x.device)

    # 展平为 [B, C, L]
    x_flat = x.flatten(2, 3)

    # 扩展索引维度以适配 gather
    perm_expand = perm.unsqueeze(0).unsqueeze(0).expand(B, C, L)

    # 根据随机索引进行 gather
    x_rand = torch.gather(x_flat, dim=2, index=perm_expand)

    return x_rand

def random_scan_inverse(x_rand: torch.Tensor,seed: int = 42) -> torch.Tensor:
    """
    将随机扫描后的序列还原到原始顺序
    Args:
        x_rand: [B, C, H*W] 随机打乱后的序列
        perm:   [H*W]       随机打乱的索引（来自 random_scan）
    Returns:
        x_rec:  [B, C, H*W] 还原后的序列
    """
    assert x_rand.dim() == 3, f"x_rand must be [B, C, L], got {x_rand.shape}"
    B, C, L = x_rand.shape
    perm = stable_random_perm(L, seed,device=x_rand.device)
    assert perm.numel() == L, f"perm length {perm.numel()} != {L}"

    # 计算反向索引
    inv_perm = torch.argsort(perm)

    # 扩展以匹配维度
    inv_perm_expand = inv_perm.unsqueeze(0).unsqueeze(0).expand(B, C, L)

    # 根据逆序 gather 回原顺序
    x_rec = torch.gather(x_rand, dim=2, index=inv_perm_expand)

    return x_rec

def window_scan(x: torch.Tensor, window_size: int, scan_mode='normal') -> torch.Tensor:
    """
    对输入 [B, C, H, W] 张量进行窗口化扫描（行优先顺序）
    所有窗口flatten后再整体逆序拼接。
    """
    assert x.dim() == 4, "Input must be [B, C, H, W]"
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype

    patches = []  # 存储窗口flatten结果

    for h in range(0, H, window_size):
        h_end = min(h + window_size, H)
        for w in range(0, W, window_size):
            w_end = min(w + window_size, W)
            window = x[:, :, h:h_end, w:w_end]  # [B, C, h_size, w_size]

            if scan_mode == "D2_D4":
                # D2_D4 扫描方案：先转置 height/width，再翻转
                window = window.transpose(2, 3)  # [B, C, w_size, h_size]
                window_flat = window.flatten(2, 3).flip(-1)  # flatten + 翻转
            else:
                window_flat = window.flatten(2, 3)
            patches.append(window_flat)

    # 🔁 所有patch逆序排列
    patches = patches[::-1]

    # 拼接成整体序列
    y = torch.cat(patches, dim=-1)  # [B, C, H*W]
    return y.contiguous()


def window_unscan(y: torch.Tensor, window_size: int, scan_mode: str = "normal") -> torch.Tensor:
    """
    将窗口扫描后的数据 [B, C, H*W] 还原为原始形状 [B, C, H, W]
    自动匹配逆序拼接。
    """
    assert y.dim() == 3, "Input must be [B, C, H*W]"
    B, C, L = y.shape

    # 推测原始 H, W（假设输入是方形）
    H = int(L ** 0.5)
    W = H
    out = torch.zeros((B, C, H, W), device=y.device, dtype=y.dtype)

    # 计算所有窗口顺序，与扫描一致，但这里要逆序恢复
    window_positions = []
    for i in range(0, H, window_size):
        for j in range(0, W, window_size):
            h_end = min(i + window_size, H)
            w_end = min(j + window_size, W)
            window_positions.append((i, h_end, j, w_end))

    # 🔁 扫描时逆序拼接，所以这里要逆序取出还原
    window_positions = window_positions[::-1]

    idx = 0
    for (i, h_end, j, w_end) in window_positions:
        h_cur = h_end - i
        w_cur = w_end - j
        patch_size = h_cur * w_cur

        patch = y[:, :, idx:idx + patch_size]
        idx += patch_size

        if scan_mode == "D2_D4":
            patch = patch.flip(-1)
            patch = patch.view(B, C, w_cur, h_cur)
            patch = patch.transpose(2, 3)
        else:
            patch = patch.view(B, C, h_cur, w_cur)

        out[:, :, i:h_end, j:w_end] = patch

    return out.contiguous()

class CrossScanner(nn.Module):
    """
    Cross-Scan 工具类
    ------------------------
    功能:
      - cross_scan_fwd(): 正向扫描 (行/列 + 翻转)
      - cross_scan_bwd(): 反向融合还原

    输入输出格式:
      forward():
          输入:  x [B, C, H, W]
          输出:  y [B, 4, C, H*W]
      backward():
          输入:  y [B, 4, C, H, W]
          输出:  x [B, C, H*W]
    """

    def __init__(self,scan_mode = 0,window_size = 1,window_ratio = None,prune_channel = False):
        # scan_mode
        # 0:基准扫描方法
        # 1:窗口化扫描方法（窗口大小配置）
        # 2:随机顺序扫描
        self.scan_mode = scan_mode
        self.window_size = window_size
        self.window_ratio = window_ratio
        self.random_seed = 42
        self.prune_channel = prune_channel
        self.rand_channel = [True,True,True,True]
        super().__init__()

    def update_window_size(self, window_size):
        self.window_size = window_size
    # ----------------------------
    # 正向 Cross-Scan 扫描函数
    # ----------------------------
    def cross_scan_fwd(self, x: torch.Tensor) -> torch.Tensor:
        """
        正向扫描:
          行优先、列优先、行反向、列反向。
          输入:  x [B, C, H, W]
          输出:  y [B, 4, C, H*W]
        """
        assert x.dim() == 4, "Input must be [B, C, H, W]"
        B, C, H, W = x.shape

        # 注意：这里不要用 new_empty，否则梯度不连续
        # 改为 zeros_like 以确保连续性 + 参与计算图
        y = x.new_zeros((B, 4, C, H * W))

        if self.scan_mode == 0:
            # 行扫描
            y[:, 0, :, :] = x.flatten(2, 3).contiguous()
            # 列扫描
            y[:, 1, :, :] = x.transpose(2, 3).contiguous().flatten(2, 3)
            # 行反向、列反向
            y[:, 2:4, :, :] = torch.flip(y[:, 0:2, :, :], dims=[-1]).contiguous()

        elif self.scan_mode == 1:
            if self.window_ratio is not None:
                self.update_window_size(int(H * self.window_ratio))
            y[:, 0, :, :] = x.flatten(2, 3).contiguous()
            y[:, 1, :, :] = x.transpose(2, 3).contiguous().flatten(2, 3)
            y[:, 2, :, :] = window_scan(x.contiguous(), window_size=self.window_size,scan_mode="D2_D4").contiguous()
            y[:, 3, :, :] = window_scan(x.transpose(2, 3).contiguous(), window_size=self.window_size,scan_mode="D2_D4").contiguous()

        elif self.scan_mode == 2:
            if self.rand_channel[0]:
                y[:, 0, :, :] = random_scan(x,seed=self.random_seed)
            else:
                y[:, 0, :, :] = x.flatten(2, 3).contiguous()

            if self.rand_channel[1]:
                y[:, 1, :, :] = random_scan(x.transpose(2, 3).contiguous(),seed=self.random_seed)
            else:
                y[:, 1, :, :] = x.transpose(2, 3).contiguous().flatten(2, 3)

            if self.rand_channel[2]:
                y[:, 2, :, :] = random_scan(x.flatten(2, 3).flip(dims=[-1]).view(B,C,H,W),seed=self.random_seed)
            else:
                y[:, 2, :, :] = x.flatten(2, 3).flip(dims=[-1])  # 存原始 x，完全可逆

            if self.rand_channel[3]:
                y[:, 3, :, :] = random_scan(x.transpose(2, 3).flatten(2, 3).flip(dims=[-1]).view(B,C,H,W),seed=self.random_seed)
            else:
                y[:, 3, :, :] = x.transpose(2, 3).flatten(2, 3).flip(dims=[-1])  # 冗余可选（增鲁棒性/对称结构）
        elif self.scan_mode == 3:
            y[:, 0, :, :] = even_odd_scan(x).contiguous()
            y[:, 1, :, :] = even_odd_scan(x.transpose(2, 3)).contiguous()
            y[:, 2, :, :] = even_odd_scan(x.flatten(2, 3).flip(dims=[-1]).view(B,C,H,W))
            y[:, 3, :, :] = even_odd_scan(x.transpose(2, 3).flatten(2, 3).flip(dims=[-1]).view(B, C, H, W))
        if self.prune_channel:
            # 测试通道信息有效性
            y[:, 0:2, :, :] = 0
        return y.contiguous()

    # ----------------------------
    # 反向 Cross-Scan 还原函数
    # ----------------------------
    def cross_scan_bwd(self, y: torch.Tensor) -> torch.Tensor:
        """
        反向还原:
          将 cross_scan_fwd 的输出反向融合回单一特征序列。
          输入:  y [B, 4, C, H, W]
          输出:  x [B, C, H*W]
        """
        assert y.dim() == 5, "Input must be [B, 4, C, H, W]"
        B, K, D, H, W = y.shape
        assert K == 4, "K must be 4 directions (row/col + flips)"

        y = y.contiguous()

        if self.scan_mode == 0:
            y = y.view(B, K, D, -1).contiguous()
            y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).contiguous().view(B, 2, D, -1)
            y = (
                    y[:, 0]
                    + y[:, 1]
                    .contiguous()
                    .view(B, -1, W, H)
                    .transpose(dim0=2, dim1=3)
                    .contiguous()
                    .view(B, D, -1)
            )

        elif self.scan_mode == 1:
            if self.window_ratio is not None:
                self.update_window_size(int(H * self.window_ratio))
            y = y.view(B, K, D, -1).contiguous()

            y_0 = y[:, 0, :, :].contiguous()
            y_1 = (
                y[:, 1, :, :]
                .contiguous()
                .view(B, -1, W, H)
                .transpose(dim0=2, dim1=3)
                .contiguous()
                .view(B, D, -1)
            )
            y_2 = (
                window_unscan(y[:, 2, :, :].contiguous(), self.window_size,scan_mode="D2_D4")
                .contiguous()
                .view(B, D, -1)
            )
            y_3 = (
                window_unscan(y[:, 3, :, :].contiguous(), self.window_size,scan_mode="D2_D4")
                .contiguous()
                .view(B, -1, W, H)
                .transpose(dim0=2, dim1=3)
                .contiguous()
                .view(B, D, -1)
            )
            y = y_0 + y_1 + y_2 + y_3

        elif self.scan_mode == 2:
            y = y.view(B, K, D, -1).contiguous()

            if self.rand_channel[0]:
                y_0 = random_scan_inverse(y[:, 0, :, :],seed=self.random_seed)
            else:
                y_0 = y[:, 0, :, :]

            if self.rand_channel[1]:
                y_1 = random_scan_inverse(y[:, 1, :, :], seed=self.random_seed).view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
            else:
                y_1 = y[:, 1, :, :].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)

            if self.rand_channel[2]:
                y_2 = random_scan_inverse(y[:, 2, :, :], seed=self.random_seed).flip(dims=[-1]).contiguous()
            else:
                y_2 = y[:, 2, :, :].flip(dims=[-1]).contiguous()

            if self.rand_channel[3]:
                y_3 = random_scan_inverse(y[:, 3, :, :], seed=self.random_seed).flip(dims=[-1]).view(B, -1, W, H).transpose(2, 3).contiguous().view(B, D, -1)
            else:
                y_3 = y[:, 3, :, :].flip(dims=[-1]).view(B, -1, W, H).transpose(2, 3).contiguous().view(B, D, -1)
            y = y_0 + y_1 + y_2 + y_3

        elif self.scan_mode == 3:
            y = y.view(B, K, D, -1).contiguous()
            y_0 = even_odd_unscan(y[:, 0, :, :])
            y_1 = even_odd_unscan(y[:, 1, :, :]).view(B, -1, W, H).transpose(dim0=2,dim1=3).contiguous().view(B, D, -1)
            y_2 = even_odd_unscan(y[:, 2, :, :]).flip(dims=[-1]).contiguous()
            y_3 = even_odd_unscan(y[:, 3, :, :]).flip(dims=[-1]).view(B, -1, W, H).transpose(
                2, 3).contiguous().view(B, D, -1)
            y = y_0 + y_1 + y_2 + y_3

        return y.contiguous()

if __name__ == "__main__":
    scanner = CrossScanner(scan_mode=3,window_size=0,window_ratio=0.5)

    B, C, H, W = 1, 96, 7, 7
    x = torch.randn(B, C, H, W)
    # 正向扫描
    y = scanner.cross_scan_fwd(x)
    # 反向还原
    x_rec = scanner.cross_scan_bwd(y.view(B, 4, -1, H, W))

    print("输入 x:", x.shape)
    print("cross-scan 输出 y:", y.shape)
    print("还原后 x_rec:", x_rec.shape)
    print("误差:", (x - x_rec.view(B, C, H, W) / 4).abs().mean().item())

