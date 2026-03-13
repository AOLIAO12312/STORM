import torch
import torch.nn as nn
import math


class HSA(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, residual, num_prune):
        importance_scores = hidden_states.sum(dim=1).detach()  # [B, D, L] -> [B, L]
        B, D, L = hidden_states.shape
        num_keep_node = L - num_prune
        num_keep_node = max(1, min(num_keep_node, L))
        _, top_indices = importance_scores.topk(num_keep_node, dim=1, largest=True)
        top_indices_sorted, _ = torch.sort(top_indices, dim=-1)
        top_indices_expanded = top_indices_sorted.unsqueeze(1).expand(-1, D, -1)  # [B, 1, num_keep_node] -> [B, D, num_keep_node]
        hidden_states_pruned = torch.gather(hidden_states, 2, top_indices_expanded)
        residual_pruned = torch.gather(residual, 2, top_indices_expanded)
        return hidden_states_pruned, residual_pruned


class DynamicPruningLayer(nn.Module):
    def __init__(self, layer_num, total_layers=24):
        super().__init__()
        self.layer_num = layer_num
        self.total_layers = total_layers
        self.hsa_module = HSA()

    def forward(self, hidden_states, residual,num_prune):
        return self.hsa_module(hidden_states, residual, num_prune)


if __name__ == "__main__":
    batch_size, seq_len, embed_dim = 2, 197, 768
    hidden_states = torch.randn(batch_size, embed_dim, seq_len)
    residual = torch.randn(batch_size, embed_dim, seq_len)
    hsa = HSA()
    pruned_hidden, pruned_residual = hsa(
        hidden_states, residual, num_prune = 50
    )

    print(f"原始形状 - Hidden: {hidden_states.shape}, Residual: {residual.shape}")
    print(f"剪枝后形状 - Hidden: {pruned_hidden.shape}, Residual: {pruned_residual.shape}")
    print(f"序列长度变化: {seq_len} -> {pruned_hidden.shape[2]}")
