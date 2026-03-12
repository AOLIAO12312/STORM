# import math
#
# def compute_prune_per_layer(L: int, n: int, D: float) -> int:
#     """计算每层需要剪枝的 token 数量 p (四舍五入取整)"""
#     p_float = (2 * D * L) / (n + 1)
#     return round(p_float)
#
# def compute_final_length(L: int, n: int, p: int) -> int:
#     """计算剪枝后的最终 token 长度"""
#     return L - p * n
#
# # -------------------------------
# # 配置
# num_stage_layer = [2, 2, 8, 2]
# num_prune_ratio = 0.1
# stage_size = [56]
# num_prune = []
#
#
# for i, n_layer in enumerate(num_stage_layer):
#     H = stage_size[i]
#     L = H * H
#     # 计算每层剪枝数量
#     p = compute_prune_per_layer(L, n_layer, num_prune_ratio)
#     num_prune.append(p)
#
#     # 剪枝后的最终长度
#     L_final_prune = compute_final_length(L, n_layer, p)
#
#
#     H = math.isqrt(L_final_prune)
#     L_final = H * H if H * H == L else (H + 1) * (H + 1)
#     stage_size.append(math.isqrt(L_final) // 2)
#
#     print(f"Stage {i}: 输入L={L}, sqrt≈{H}, 每层剪 {p}, 剩余 {L_final}")
#
#     # 更新下一 stage 的输入 L
#     L = L_final
#
# print("最终 stage_size =", stage_size)
# print("最终 num_prune =", num_prune)
