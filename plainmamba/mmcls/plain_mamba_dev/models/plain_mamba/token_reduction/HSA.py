import torch
import torch.nn as nn
import math


class HSA(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, residual, num_prune):
        """
        通用动态Token剪枝，适用于[B, D, L]格式输入

        Args:
            hidden_states: [B, D, L] - 输入隐藏状态 (L为序列长度)
            residual: [B, D, L] - 残差连接
            num_prune: int - 要剪枝的token数量
        """
        # 计算重要性分数 - 基于residual在D维度上的重要程度
        importance_scores = hidden_states.sum(dim=1).detach()  # [B, D, L] -> [B, L]

        B, D, L = hidden_states.shape  # 获取原始序列长度 (L为序列维度)

        # 计算需要保留的token数量
        num_keep_node = L - num_prune

        # 确保保留数量有效
        num_keep_node = max(1, min(num_keep_node, L))  # 至少保留1个，最多保留全部

        # 获取最重要的token索引
        _, top_indices = importance_scores.topk(num_keep_node, dim=1, largest=True)
        top_indices_sorted, _ = torch.sort(top_indices, dim=-1)

        # 使用gather操作提取对应的hidden states和residual
        # 因为要在第2维(L维)进行gather，需要扩展索引
        top_indices_expanded = top_indices_sorted.unsqueeze(1).expand(-1, D,
                                                                      -1)  # [B, 1, num_keep_node] -> [B, D, num_keep_node]

        hidden_states_pruned = torch.gather(hidden_states, 2, top_indices_expanded)  # 在dim=2 (L维) gather
        residual_pruned = torch.gather(residual, 2, top_indices_expanded)  # 在dim=2 (L维) gather

        return hidden_states_pruned, residual_pruned


class DynamicPruningLayer(nn.Module):
    """完整的动态剪枝层"""

    def __init__(self, layer_num, total_layers=24):
        super().__init__()
        self.layer_num = layer_num
        self.total_layers = total_layers
        self.hsa_module = HSA()

    def forward(self, hidden_states, residual,num_prune):
        """
        在特定层执行动态剪枝
        """
        return self.hsa_module(hidden_states, residual, num_prune)


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    batch_size, seq_len, embed_dim = 2, 197, 768
    hidden_states = torch.randn(batch_size, embed_dim, seq_len)
    residual = torch.randn(batch_size, embed_dim, seq_len)

    # 创建HSA模块
    hsa = HSA()

    # 执行剪枝
    pruned_hidden, pruned_residual = hsa(
        hidden_states, residual, num_prune = 50
    )

    print(f"原始形状 - Hidden: {hidden_states.shape}, Residual: {residual.shape}")
    print(f"剪枝后形状 - Hidden: {pruned_hidden.shape}, Residual: {pruned_residual.shape}")
    print(f"序列长度变化: {seq_len} -> {pruned_hidden.shape[2]}")
