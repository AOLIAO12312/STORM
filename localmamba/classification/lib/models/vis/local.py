import matplotlib.pyplot as plt
import numpy as np

# ------------------------
# Data
# ------------------------
rr = np.array([0.0, 24.8, 39.1, 53.3, 54.9, 64.2])

random = np.array([81.1, 76.5, 72.4, 63.9, 62.2, 48.0])
nearest = np.array([81.1, 76.8, 70.0, 62.1, 62.4, 46.5])
tome = np.array([81.1, 77.4, 74.2, 67.3, np.nan, np.nan])  # '-' 用 NaN 占位
tome2d = np.array([81.1, 77.9, 75.3, 69.7, 68.6, 59.6])
conv_tome2d = np.array([81.1, 77.0, 73.1, 67.7, 67.7, 56.0])

methods = {
    "random": random,
    "nearest": nearest,
    "tome": tome,
    "tome2d": tome2d,
    "conv_tome2d": conv_tome2d,
}

markers = {
    "random": "o",
    "nearest": "s",
    "tome": "^",
    "tome2d": "D",
    "conv_tome2d": "v",
}

# ------------------------
# Plot
# ------------------------
plt.figure(figsize=(7, 5))

for name, values in methods.items():
    plt.plot(
        rr, values,
        marker=markers.get(name, "o"),
        linewidth=2,
        label=name
    )
    # 标注每个有效点的数值
    for x, y in zip(rr, values):
        if np.isnan(y):
            continue
        plt.text(
            x, y + 0.3,                           # 0.3 作为竖直偏移，避免遮住点
            f"{y:.1f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

plt.xlabel("RR (%)", fontsize=12)
plt.ylabel("Top-1 Accuracy (%)", fontsize=12)
plt.title("Pruning Performance vs RR in LocalVim", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
