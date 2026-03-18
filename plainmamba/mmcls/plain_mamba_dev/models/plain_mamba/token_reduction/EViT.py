import torch
import torch.nn as nn


def complement_idx(idx, dim):
    """
    Compute the complement indices given a set of indices.
    Args:
        idx: [B, selected] - selected indices
        dim: total dimension size
    Returns:
        complement indices [B, dim - selected]
    """
    # Create a range tensor [0, 1, ..., dim-1]
    device = idx.device
    batch_size = idx.shape[0]
    full_range = torch.arange(dim, device=device).unsqueeze(0).expand(batch_size, -1)  # [B, dim]

    # Create mask for each element in full_range to check if it's in idx
    expanded_idx = idx.unsqueeze(1)  # [B, 1, selected]
    expanded_range = full_range.unsqueeze(2)  # [B, dim, 1]

    # Compare all combinations and find matches
    mask = (expanded_range != expanded_idx).all(dim=2)  # [B, dim]

    # Gather complement indices
    complement_indices = torch.masked_select(full_range, mask).reshape(batch_size, -1)

    return complement_indices


class EViTTokenPruning(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, residual, num_prune):
        """
        Args:
            hidden_states: [B, D, L] where L is number of tokens, D is feature dimension
            residual: [B, D, L] same shape as hidden_states
            num_prune: int, number of tokens to prune
        Returns:
            pruned_hidden_states: [B, D, L-num_prune]
            pruned_residual: [B, D, L-num_prune]
        """
        B, D, L = hidden_states.shape

        # Transpose input from [B, D, L] to [B, L, D] to match original function format
        hidden_states_transposed = hidden_states.transpose(1, 2)  # [B, L, D]
        residual_transposed = residual.transpose(1, 2)  # [B, L, D]

        # Now apply the original pruning logic
        N = L  # Number of tokens is L in the original format

        # Use residual as importance scores (this can be changed to other scoring methods)
        importance_scores = residual_transposed.mean(dim=-1)  # Average across feature dimension to get [B, L]

        # Calculate how many tokens to keep
        num_keep = N - num_prune

        # Get top-k important token indices
        _, top_indices = torch.topk(importance_scores, num_keep, dim=1, largest=True)  # [B, num_keep]

        # Sort indices to maintain order
        top_indices_sorted, _ = torch.sort(top_indices, dim=-1)  # [B, num_keep]

        # Expand indices for gathering
        top_indices_expanded = top_indices_sorted.unsqueeze(-1).expand(-1, -1, D)  # [B, num_keep, D]

        # Gather the important tokens from both hidden states and residual
        pruned_hidden_states = torch.gather(hidden_states_transposed, 1, top_indices_expanded)  # [B, num_keep, D]
        pruned_residual = torch.gather(residual_transposed, 1, top_indices_expanded)  # [B, num_keep, D]

        # Transpose back to [B, D, L-num_prune] format
        pruned_hidden_states = pruned_hidden_states.transpose(1, 2)  # [B, D, L-num_prune]
        pruned_residual = pruned_residual.transpose(1, 2)  # [B, D, L-num_prune]

        return pruned_hidden_states, pruned_residual


# Example usage:
if __name__ == "__main__":
    # Create an instance of the module
    evit_pruning = EViTTokenPruning()

    # Test with example inputs [B, D, L]
    B, D, L = 4, 256, 197  # Example dimensions
    hidden_states = torch.randn(B, D, L)
    residual = torch.randn(B, D, L)
    num_prune = 50

    # Perform pruning
    pruned_hidden, pruned_residual = evit_pruning(hidden_states, residual, num_prune)

    print(f"Original shapes: hidden_states={hidden_states.shape}, residual={residual.shape}")
    print(f"Pruned shapes: hidden_states={pruned_hidden.shape}, residual={pruned_residual.shape}")
    print(f"Expected output shape: [B, D, L-num_prune] = [{B}, {D}, {L - num_prune}]")
