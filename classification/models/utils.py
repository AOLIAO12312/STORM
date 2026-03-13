import torch
def prune_tokens_by_index(x: torch.Tensor, sorted_idx: torch.Tensor) -> torch.Tensor:
    """
    Keep the appointed token indicated by sorted_idx

    Args:
        x: torch.Tensor, shape [B, D, L_orig]
        sorted_idx: torch.Tensor, shape [B, L_keep, 1] 或 [B, L_keep]
    Returns:
        pruned_x: torch.Tensor, shape [B, D, L_keep]
    """
    B, D, L_orig = x.shape
    if sorted_idx.dim() == 3 and sorted_idx.shape[-1] == 1:
        sorted_idx = sorted_idx.squeeze(-1)  # [B, L_keep]
    assert sorted_idx.shape[0] == B, f"Batch size 不匹配: {sorted_idx.shape[0]} vs {B}"
    sorted_idx = sorted_idx.unsqueeze(1).expand(-1, D, -1)  # [B, D, L_keep]
    pruned_x = torch.gather(x, dim=2, index=sorted_idx)
    return pruned_x


def random_prune_tokens(x: torch.Tensor, num_prune: int, seed: int = None):
    """
    Randomly remove num_prune token

    Args:
        x (torch.Tensor): Input tensor [B, C, L]
        num_prune (int): num of token to be pruned
        seed (int, optional): random seed
    Returns:
        x_pruned (torch.Tensor): return tensor [B, C, L - num_prune]
        sorted_idx (torch.Tensor): keep the index [B, L - num_prune, 1], asc
    """
    assert x.dim() == 3, f"Input must be [B, C, L], got {x.shape}"
    B, C, L = x.shape
    assert num_prune < L, "num_prune must be smaller than sequence length"
    device = x.device
    if seed is not None:
        torch.manual_seed(seed)
    prune_idx = torch.randperm(L, device=device)[:num_prune]
    mask = torch.ones(L, dtype=torch.bool, device=device)
    mask[prune_idx] = False
    sorted_idx = torch.nonzero(mask, as_tuple=True)[0]  # [L_keep]
    x_pruned = x.index_select(dim=2, index=sorted_idx)
    sorted_idx = sorted_idx.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1).contiguous()
    return x_pruned, sorted_idx


def pool_downsample(x, mode='avg'):
    """
    Replace F.interpolate(x, size=[H//2, W//2], mode="nearest") to pooling

    Supported mode:
        - 'avg' : AvgPool2d
        - 'max' : MaxPool2d
        - 'nearest' :use nearest to keep consistent behavior
        - 'lp' : L2 pooling
    """
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