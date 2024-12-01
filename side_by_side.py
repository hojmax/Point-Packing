import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_points import get_loss, ALPHA
from matplotlib import font_manager

# Update font and plot settings for aesthetics
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
    }
)

paths = [
    "points/optimized_points_5612.9111.npz",
    "points/sweep_optimized_points_5613.0376.npz",
    "points/optimized_points_5613.1704.npz",
    "points/optimized_points_5612.9180.npz",
]
colors = [
    "#D72638",
    "#3B8EA5",
    "#3BB273",
    "#F49F05",
]  # Vibrant, visually contrasting colors

# Calculate rows and columns for the layout
n_cols = len(paths)  # Force a single-row layout for a cleaner presentation
fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6), dpi=300)

if n_cols == 1:
    axes = [axes]  # Ensure consistency in case of a single plot

# Iterate over paths and generate plots
for i, path in enumerate(paths):
    points = np.load(path)
    points_array = points["points"]

    p_x = torch.from_numpy(points_array[:, 0]).requires_grad_()
    p_y = torch.from_numpy(points_array[:, 1]).requires_grad_()

    loss = get_loss(p_x, p_y, ALPHA, 800, 800)

    axes[i].scatter(
        points_array[:, 0], points_array[:, 1], alpha=0.9, s=8, color=colors[i]
    )
    axes[i].text(
        400,
        735,
        r"$\lambda = " + f"{loss.item():.3f}$",
        ha="center",
        fontsize=16,
        color="black",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )
    axes[i].set_xlim(0, 800)
    axes[i].set_ylim(0, 800)
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    axes[i].spines["top"].set_visible(False)
    axes[i].spines["right"].set_visible(False)
    axes[i].spines["left"].set_visible(False)
    axes[i].spines["bottom"].set_visible(False)

# Remove any unused axes (only if axes > plots)
for j in range(i + 1, len(axes)):
    axes[j].remove()

# Add padding and save the plot
plt.tight_layout(pad=2)
plt.savefig("side_by_side_improved.png", bbox_inches="tight", transparent=True)
