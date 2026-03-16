import math


def find_optimal_a(a, b, min_gcd_threshold=None):
    """
    在保持a变化最小的前提下，尽可能提高a和b的最大公约数

    参数:
    a, b: 输入的正整数，a > b
    min_gcd_threshold: 最小可接受的GCD阈值，如果当前GCD已经达到或超过这个值，就不修改a

    返回:
    (new_a, gcd_value, change_amount): 优化后的a值，最大公约数，a的变化量
    """

    # 验证输入
    if a <= b:
        raise ValueError("a必须大于b")
    if a <= 0 or b <= 0:
        raise ValueError("a和b必须是正整数")

    # 计算原始GCD
    original_gcd = math.gcd(a, b)

    # 如果设置了最小阈值且当前GCD已经达到要求，直接返回
    if min_gcd_threshold is not None and original_gcd >= min_gcd_threshold:
        return a, original_gcd, 0

    print(f"原始情况: a={a}, b={b}, GCD={original_gcd}")

    best_a = a
    best_gcd = original_gcd
    min_change = float('inf')

    # 策略1: 寻找比当前GCD更大的公约数，同时最小化a的增加
    # 我们可以从稍大于a的数开始尝试，但为了效率，设置一个合理的上限
    max_increase = max(b * 2, a * 0.5)  # 最多增加b*2或a的一半，避免无限增长

    # 记录我们尝试过的GCD值，从大到小尝试
    candidate_gcds = []

    # 生成可能的候选GCD：所有能整除b的较大因子
    b_factors = []
    for i in range(1, int(math.sqrt(b)) + 1):
        if b % i == 0:
            b_factors.append(i)
            if i != b // i:
                b_factors.append(b // i)

    # 只考虑比当前GCD大的因子
    candidate_gcds = [g for g in b_factors if g > original_gcd]
    candidate_gcds.sort(reverse=True)  # 从大到小尝试

    # 如果没有更大的公约数候选，我们可能需要考虑其他方法
    if not candidate_gcds:
        print("没有找到比当前GCD更大的公约数候选")
        return a, original_gcd, 0

    # 对于每个候选GCD，找到最小的a' >= a使得gcd(a', b) >= 候选GCD
    for target_gcd in candidate_gcds:
        # 要满足gcd(a', b) >= target_gcd，a'必须是target_gcd的倍数
        # 找到大于等于a的最小的target_gcd的倍数
        if a % target_gcd == 0:
            candidate_a = a
        else:
            candidate_a = ((a // target_gcd) + 1) * target_gcd

        # 检查这个candidate_a是否真的能得到至少target_gcd的GCD
        actual_gcd = math.gcd(candidate_a, b)

        if actual_gcd >= target_gcd:
            change = candidate_a - a
            # 我们选择变化最小且GCD最大的方案
            if change < min_change or (change == min_change and actual_gcd > best_gcd):
                best_a = candidate_a
                best_gcd = actual_gcd
                min_change = change

    # 如果找到了更好的方案
    if min_change > 0:
        print(f"优化结果: a={best_a} (增加了{min_change}), b={b}, GCD={best_gcd}")
        return best_a, best_gcd, min_change
    else:
        print("无需修改a，当前GCD已经是最优或无法进一步优化")
        return a, original_gcd, 0


def find_optimal_a_with_limit(a, b, max_increase_ratio=0.5, min_gcd_threshold=None):
    """
    带有限制条件的版本：限制a的最大增加比例

    参数:
    max_increase_ratio: a最多可以增加到原来的多少倍 (0-1之间的小数表示增加的比例)
    """
    if max_increase_ratio < 0:
        raise ValueError("max_increase_ratio不能为负数")

    original_gcd = math.gcd(a, b)

    # 如果设置了最小阈值且当前GCD已经达到要求，直接返回
    if min_gcd_threshold is not None and original_gcd >= min_gcd_threshold:
        return a, original_gcd, 0

    max_allowed_a = int(a * (1 + max_increase_ratio))

    print(f"原始情况: a={a}, b={b}, GCD={original_gcd}, 最大允许a={max_allowed_a}")

    best_a = a
    best_gcd = original_gcd
    min_change = float('inf')

    # 生成b的所有因子
    b_factors = []
    for i in range(1, int(math.sqrt(b)) + 1):
        if b % i == 0:
            b_factors.append(i)
            if i != b // i:
                b_factors.append(b // i)

    # 只考虑比当前GCD大且在限制范围内的因子
    candidate_gcds = [g for g in b_factors if g > original_gcd]
    candidate_gcds.sort(reverse=True)

    for target_gcd in candidate_gcds:
        # 找到大于等于a的最小的target_gcd的倍数，但不能超过最大限制
        if a % target_gcd == 0:
            candidate_a = a
        else:
            candidate_a = ((a // target_gcd) + 1) * target_gcd

        if candidate_a > max_allowed_a:
            continue

        actual_gcd = math.gcd(candidate_a, b)

        if actual_gcd >= target_gcd:
            change = candidate_a - a
            if change < min_change or (change == min_change and actual_gcd > best_gcd):
                best_a = candidate_a
                best_gcd = actual_gcd
                min_change = change

    if min_change > 0:
        print(f"优化结果: a={best_a} (增加了{min_change}), b={b}, GCD={best_gcd}")
        return best_a, best_gcd, min_change
    else:
        print("在限制范围内无法进一步提高GCD")
        return a, original_gcd, 0


# 测试函数
def test_function():
    """测试一些例子"""
    test_cases = [
        (10, 6),  # 普通情况
        (17, 5),  # 互质情况
        (100, 40),  # 已有较大公约数
        (15, 9),  # 中等公约数
        (23, 6),  # 质数情况
    ]

    for a, b in test_cases:
        print(f"\n{'=' * 50}")
        try:
            result = find_optimal_a(a, b)
            # 也可以测试带限制的版本
            # result = find_optimal_a_with_limit(a, b, max_increase_ratio=0.3)
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    # 交互式使用
    print("请输入两个正整数a和b (a > b):")
    try:
        a = int(input("a = "))
        b = int(input("b = "))

        if a <= b:
            print("错误：a必须大于b")
        else:
            result = find_optimal_a(a, b)
            # 或者使用带限制的版本：
            # result = find_optimal_a_with_limit(a, b, max_increase_ratio=0.3)

    except ValueError:
        print("请输入有效的整数！")

    # 运行测试用例
    print("\n\n运行测试用例:")
    test_function()