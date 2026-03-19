import torch

def prune_tokens_by_index(x: torch.Tensor, sorted_idx: torch.Tensor) -> torch.Tensor:
    """
    keep token in x by sorted index

    Args:
        x: torch.Tensor, shape [B, D, L_orig]
        sorted_idx: torch.Tensor, shape [B, L_keep, 1] or [B, L_keep]
    Returns:
        pruned_x: torch.Tensor, shape [B, D, L_keep]
    """
    B, D, L_orig = x.shape

    if sorted_idx.dim() == 3 and sorted_idx.shape[-1] == 1:
        sorted_idx = sorted_idx.squeeze(-1)  # [B, L_keep]

    assert sorted_idx.shape[0] == B, f"Batch size not match: {sorted_idx.shape[0]} vs {B}"
    sorted_idx = sorted_idx.unsqueeze(1).expand(-1, D, -1)  # [B, D, L_keep]
    pruned_x = torch.gather(x, dim=2, index=sorted_idx)

    return pruned_x


def random_prune_tokens(x: torch.Tensor, num_prune: int, seed: int = None):
    assert x.dim() == 3, f"Input must be [B, C, L], got {x.shape}"
    B, C, L = x.shape
    assert num_prune < L, "num_prune must be smaller than sequence length"

    device = x.device
    if seed is not None:
        torch.manual_seed(seed)
    prune_idx = torch.randperm(L, device=device)[:num_prune]
    mask = torch.ones(L, dtype=torch.bool, device=device)
    mask[prune_idx] = False
    sorted_idx = torch.nonzero(mask, as_tuple=True)[0]
    x_pruned = x.index_select(dim=2, index=sorted_idx)
    sorted_idx = sorted_idx.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1).contiguous()
    return x_pruned, sorted_idx


def pool_downsample(x, mode='avg'):
    B, C, H, W = x.shape
    if mode == 'avg':
        pool = nn.AvgPool2d(kernel_size=2, stride=2)
        return pool(x)
    elif mode == 'max':
        pool = nn.MaxPool2d(kernel_size=2, stride=2)
        return pool(x)
    elif mode == 'nearest':
        return F.interpolate(x, size=[H // 2, W // 2], mode="nearest")
    elif mode == 'lp':
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
    return F.pad(x, (0, L_orig - L))