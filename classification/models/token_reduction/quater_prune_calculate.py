def calculate_params(strategy):
    # 定义每个 stage 的原始参数量
    stages = [
        [3136, 3136],  # stage 0
        [784, 784],  # stage 1
        [196] * 8,  # stage 2
        [49, 49]  # stage 3
    ]

    total_params = 0

    # 遍历 stages 和 strategy 进行参数累加
    for stage_params, stage_strategy in zip(stages, strategy):
        for param, prune_flag in zip(stage_params, stage_strategy):
            if prune_flag == 0:
                total_params += param
            elif prune_flag == 1:
                total_params += param // 4  # 剪枝后为原始参数的1/4
            else:
                raise ValueError("Strategy values must be 0 or 1.")

    return total_params


if __name__ == "__main__":
    # 不剪枝BaseLine
    quater_prune_strategy_0 = [[0, 0], [0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0]]
    # 论文方法
    quater_prune_strategy_1 = [[0, 1], [0, 1], [0, 1, 0, 0, 1, 0, 0, 1], [0, 0]]
    # 更激进
    quater_prune_strategy_2 = [[0, 1], [0, 1], [0, 1, 0, 1, 0, 1, 0, 1], [0, 0]]
    # 保留浅层
    quater_prune_strategy_3 = [[0, 0], [0, 0], [0, 1, 0, 1, 0, 1, 0, 1], [0, 0]]
    # 较轻量减
    quater_prune_strategy_4 = [[0, 1], [0, 1], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0]]
    # 每Stage只剪第2个及以后block
    quater_prune_strategy_5 = [[0, 1], [0, 1], [0, 1, 1, 1, 1, 1, 1, 1], [0, 0]]
    # 极限剪_0
    quater_prune_strategy_6 = [[0, 1], [1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [0, 0]]
    # 极限剪_1
    quater_prune_strategy_7 = [[1, 1], [1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [0, 0]]

    # 计算参数量
    total_0 = calculate_params(quater_prune_strategy_0)
    total_1 = calculate_params(quater_prune_strategy_7)

    print(f"Total params for strategy: {total_1}")
    print(f"Reduction Ratio: {1 - total_1/total_0}")
