import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# Data
# ------------------------
# 剪枝率 RR（%）
rr = np.array([0.0, 24.8, 39.1, 53.3, 54.9, 64.2])

random = np.array([81.6, 76.2, 70.8, 62.8, 57.8, 51.0])
nearest = np.array([81.6, 77.7, 72.4, 65.0, 64.7, 54.0])
tome   = np.array([81.6, 76.7, 72.9, np.nan, np.nan, np.nan])  # 无结果用 nan
tome2d = np.array([81.6, 79.7, 77.0, 71.9, 71.1, 58.7])
conv_tome2d = np.array([81.6, 78.7, 75.8, 73.2, 73.2, 67.1])

methods = {
    "Random": random,
    "Nearest": nearest,
    "ToMe": tome,
    "ToMe2D": tome2d,
    "conv_tome2d": conv_tome2d,
}

markers = {
    "Random": "o",
    "Nearest": "s",
    "ToMe": "^",
    "ToMe2D": "D",
    "conv_tome2d": "P",  # 五角形
}

# ------------------------
# Plot
# ------------------------
plt.figure(figsize=(8, 5))

for name, acc in methods.items():
    plt.plot(rr, acc, marker=markers[name], label=name)
    # 标注每个有效数据点
    for x, y in zip(rr, acc):
        if np.isnan(y):
            continue
        plt.text(
            x,
            y + 0.4,               # 往上偏一点避免遮挡 marker
            f"{y:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

plt.xlabel("RR (Pruning Rate, %)")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs. Pruning Rate for Different Pruning Schemes in PlainMamba")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()

plt.show()
