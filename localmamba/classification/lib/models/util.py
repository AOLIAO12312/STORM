import torch

def prune_tokens_by_index(x: torch.Tensor, sorted_idx: torch.Tensor) -> torch.Tensor:
    """
    根据 sorted_idx 从原始 x 中保留指定 token。

    Args:
        x: torch.Tensor, shape [B, D, L_orig]
        sorted_idx: torch.Tensor, shape [B, L_keep, 1] 或 [B, L_keep]
    Returns:
        pruned_x: torch.Tensor, shape [B, D, L_keep]
    """
    B, D, L_orig = x.shape

    # 若 sorted_idx 是 [B, L_keep, 1]，去掉多余维度
    if sorted_idx.dim() == 3 and sorted_idx.shape[-1] == 1:
        sorted_idx = sorted_idx.squeeze(-1)  # [B, L_keep]

    # 检查维度匹配
    assert sorted_idx.shape[0] == B, f"Batch size 不匹配: {sorted_idx.shape[0]} vs {B}"

    # 扩展索引到 [B, D, L_keep]
    sorted_idx = sorted_idx.unsqueeze(1).expand(-1, D, -1)  # [B, D, L_keep]

    # 按索引取出需要保留的 token
    pruned_x = torch.gather(x, dim=2, index=sorted_idx)

    return pruned_x


def random_prune_tokens(x: torch.Tensor, num_prune: int, seed: int = None):
    """
    在 [B, C, L] (L=H*W) 中随机删除 num_prune 个 token。
    所有 batch 共用相同的随机删除模式。
    返回剪枝后的张量，以及保留的 sorted_idx（升序），包装为 [B, L_keep, 1]。

    Args:
        x (torch.Tensor): 输入张量 [B, C, L]
        num_prune (int): 要删除的 token 数
        seed (int, optional): 随机种子，保证可复现

    Returns:
        x_pruned (torch.Tensor): 删除后的张量 [B, C, L - num_prune]
        sorted_idx (torch.Tensor): 保留索引 [B, L - num_prune, 1]，升序排列
    """
    assert x.dim() == 3, f"Input must be [B, C, L], got {x.shape}"
    B, C, L = x.shape
    assert num_prune < L, "num_prune must be smaller than sequence length"

    device = x.device
    if seed is not None:
        torch.manual_seed(seed)

    # 随机生成要删除的 token 索引
    prune_idx = torch.randperm(L, device=device)[:num_prune]
    mask = torch.ones(L, dtype=torch.bool, device=device)
    mask[prune_idx] = False

    # 获取保留索引（升序）
    sorted_idx = torch.nonzero(mask, as_tuple=True)[0]  # [L_keep]

    # 剪枝
    x_pruned = x.index_select(dim=2, index=sorted_idx)

    # 包装 sorted_idx 为 [B, L_keep, 1]（每个 batch 共用）
    sorted_idx = sorted_idx.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1).contiguous()

    return x_pruned, sorted_idx


def pool_downsample(x, mode='avg'):
    """
    将 F.interpolate(x, size=[H//2, W//2], mode="nearest")
    替换为池化操作版本。

    支持 mode:
        - 'avg' : AvgPool2d
        - 'max' : MaxPool2d
        - 'nearest' : 用 nearest 插值保持一致行为
        - 'lp' : L2 pooling (可选扩展)
    """
    B, C, H, W = x.shape
    if mode == 'avg':
        pool = nn.AvgPool2d(kernel_size=2, stride=2)
        return pool(x)
    elif mode == 'max':
        pool = nn.MaxPool2d(kernel_size=2, stride=2)
        return pool(x)
    elif mode == 'nearest':
        # 如果希望和 interpolate 一样，就保留插值写法
        return F.interpolate(x, size=[H // 2, W // 2], mode="nearest")
    elif mode == 'lp':
        # L2 Pooling (等价于对平方求平均再开根号)
        pool = nn.LPPool2d(norm_type=2, kernel_size=2, stride=2)
        return pool(x)
    else:
        raise ValueError("mode must be 'avg', 'max', 'nearest', or 'lp'")


def pad_zeros(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 3, "x must be [B, D, L]"
    B, D, L = x.shape
    H = math.isqrt(L)
    L_orig = H * H if H * H == L else (H + 1) * (H + 1)
    assert L <= L_orig, "L must not exceed L_orig"
    # pad 在最后一个维度补 (0, L_orig - L)
    return F.pad(x, (0, L_orig - L))