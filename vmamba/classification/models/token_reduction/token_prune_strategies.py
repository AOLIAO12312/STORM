import math


def compute_stage_flops(prune_strategy, input_size=56):
    """
    根据剪枝策略计算每个stage的真实计算量（按扫描后剪枝逻辑）

    Args:
        prune_strategy (list[list[int]]): 每个stage的剪枝量列表。
        input_size (int): 初始输入尺寸。

    Returns:
        dict: {
            "stage_flops": 每个stage的计算量列表,
            "stage_outputs": 每个stage输出的尺寸,
            "total_flops": 总计算量
        }
    """
    current_size = input_size
    stage_flops = []
    stage_outputs = []

    for stage_idx, stage in enumerate(prune_strategy):
        stage_total = 0
        current_area = current_size ** 2

        for prune_val in stage:
            # 当前 block 的计算量 = 当前输入的像素数
            stage_total += current_area
            # 执行剪枝后，更新输出像素数（保证平方数）
            new_area = max(current_area - prune_val, 4)
            new_size = int(math.isqrt(new_area))
            current_area = new_size ** 2  # 强制对齐到平方数

        # 记录结果
        stage_flops.append(stage_total)
        out_size = int(math.isqrt(current_area))
        stage_outputs.append(out_size)

        # 下一个 stage 输入尺寸减半
        current_size = max(1, out_size // 2)

    total_flops = sum(stage_flops)
    return {
        "stage_flops": stage_flops,
        "stage_outputs": stage_outputs,
        "total_flops": total_flops
    }


# === 示例 ===
if __name__ == "__main__":
    prune_strategy_0 = [[0, 0], [0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0]]
    prune_strategy_1 = [[111, 109], [53, 0], [0, 25, 0, 23, 0, 21, 0, 0], [0, 0]]
    prune_strategy_2 = [[220, 212], [51, 49], [0, 23, 0, 21, 0, 0, 0, 0], [0, 0]]
    prune_strategy_3 = [[432, 400], [92, 84], [0, 19, 0, 17, 0, 0, 0, 0], [0, 0]]
    prune_strategy_4 = [[636, 564], [84, 76], [0, 0, 0, 17, 0, 0, 0, 0], [0, 0]]
    prune_strategy_5 = [[832, 704], [0, 0], [0, 19, 0, 17, 0, 0, 0, 0], [0, 0]]
    prune_strategy_6 = [[1020, 820], [35, 33], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0]]
    prune_strategy_7 = [[1372, 740], [0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0]]
    prune_strategy_8 = [[1536, 700], [0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0]]
    prune_strategy_9 = [[2352, 0], [0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [-40, 0]]

    prune_strategy_0_sb = [[0, 0], [0, 0], [0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0], [0, 0]]

    prune_strategy_tome = [[1840, 0], [0, 0], [17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0]]
    result = compute_stage_flops(prune_strategy_tome, input_size=56)

    print("\n剪枝策略计算结果:")
    for i, (f, o) in enumerate(zip(result["stage_flops"], result["stage_outputs"])):
        print(f"Stage{i}: 计算量 = {f:,}, 输出 = {o}×{o}")
    print(f"\n总计算量: {result['total_flops']:,}")
