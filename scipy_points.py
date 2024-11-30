import numpy as np
import scipy.optimize
import scipy.spatial.distance
import torch

W = 800
H = 800
N = 1000
ALPHA = 250


def get_loss_np(vars, alpha, w, h, n):
    x = vars[:n]
    y = vars[n:]
    positions = np.column_stack((x, y))  # Shape (N, 2)
    # Compute pairwise distances
    distances = scipy.spatial.distance.pdist(positions)
    reciprocal_sum = np.sum(1.0 / distances)
    # Border terms
    border = np.sum(1 / x + 1 / y + 1 / (w - x) + 1 / (h - y))
    loss = alpha * border + reciprocal_sum
    return loss


def get_points_np():
    path = "points/optimized_points_5612.9111.npz"
    points = np.load(path)
    points_array = points["points"]
    p_x = points_array[:, 0]
    p_y = points_array[:, 1]
    vars = np.concatenate([p_x, p_y])
    return vars


# Define parameters
w = W
h = H
n = N
alpha = ALPHA

# Initial variables
vars0 = get_points_np()
print("Initial loss: ", get_loss_np(vars0, alpha, w, h, n))

# Bounds for each variable
bounds = [(1, w - 1)] * n + [(1, h - 1)] * n

# Minimize the loss function using L-BFGS-B algorithm
result = scipy.optimize.minimize(
    get_loss_np, vars0, args=(alpha, w, h, n), bounds=bounds
)

# Extract optimized x and y
vars_opt = result.x
x_opt = vars_opt[:n]
y_opt = vars_opt[n:]

print("Optimization successful:", result.success)
print("Final loss:", result.fun)
