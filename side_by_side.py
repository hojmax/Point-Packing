import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_points import get_loss, ALPHA
from matplotlib import font_manager

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 14,
    }
)

paths = [
    "points/optimized_points_5612.9111.npz",
    "points/sweep_optimized_points_5613.0376.npz",
    "points/optimized_points_5613.1704.npz",
]
colors = ["#C16060", "#608BC1", "#60C18B"]

n_cols = 4
n_rows = (len(paths) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), dpi=300)
axes = axes.flatten()

for i, path in enumerate(paths):
    points = np.load(path)
    points_array = points["points"]

    p_x = torch.from_numpy(points_array[:, 0]).requires_grad_()
    p_y = torch.from_numpy(points_array[:, 1]).requires_grad_()

    loss = get_loss(p_x, p_y, ALPHA, 800, 800)

    axes[i].scatter(
        points_array[:, 0], points_array[:, 1], alpha=1, s=7, color=colors[i]
    )
    axes[i].text(400, 735, r"$\lambda = " + f"{loss.item():.3f}$", ha="center")
    axes[i].set_xlim(0, 800)
    axes[i].set_ylim(0, 800)
    axes[i].axis("off")

for j in range(i + 1, len(axes)):
    axes[j].remove()

plt.tight_layout()
plt.savefig("side_by_side.png", bbox_inches="tight")
